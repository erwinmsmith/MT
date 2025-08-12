import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from .base_feature_topics_encoder import FeatureTopicsEncoder

class ProteinFeatureTopicsEncoder(FeatureTopicsEncoder):
    """
    Specialized variational encoder for protein feature topic distributions.
    Optimized for protein expression characteristics and functional relationships.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 n_topics: int, dropout: float = 0.1,
                 prior_mean: float = 0.0, prior_std: float = 1.0,
                 use_modality_specific_priors: bool = True,
                 protein_specific_features: bool = True,
                 functional_relationship_layers: int = 2, complex_formation: bool = True):
        """
        Initialize protein feature topics encoder.
        
        Args:
            input_dim: Input feature dimension (from heterogeneous GNN)
            hidden_dims: List of hidden layer dimensions
            n_topics: Number of topics (latent dimension)
            dropout: Dropout rate
            prior_mean: Prior mean for the latent distribution
            prior_std: Prior standard deviation for the latent distribution
            use_modality_specific_priors: Whether to use protein-specific priors
            protein_specific_features: Whether to use protein-specific processing
        """
        # Initialize base class with protein modality
        # Store protein-specific parameters first
        self.protein_specific_features = protein_specific_features
        self.functional_relationship_layers = functional_relationship_layers
        self.complex_formation = complex_formation
        
        super(ProteinFeatureTopicsEncoder, self).__init__(
            modality='protein',
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            n_topics=n_topics,
            dropout=dropout,
            prior_mean=prior_mean,
            prior_std=prior_std,
            use_modality_specific_priors=use_modality_specific_priors
        )
        
        if protein_specific_features:
            # Protein-specific processing layers
            self.functional_layer = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.LayerNorm(input_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5)
            )
            
            # Protein-protein interaction attention
            self.ppi_attention = nn.Sequential(
                nn.Linear(input_dim, input_dim // 4),
                nn.GELU(),
                nn.Linear(input_dim // 4, 1),
                nn.Sigmoid()
            )
            
            # Protein complex formation simulation
            self.complex_encoder = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim // 2, input_dim)
            )
            
            # Localization-based processing (surface, intracellular, secreted)
            self.localization_encoder = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.LayerNorm(input_dim),
                nn.GELU()
            )
            
            # Expression level normalization (proteins have wide dynamic range)
            self.expression_normalizer = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.LayerNorm(input_dim),
                nn.Tanh()  # Normalize expression levels
            )
        
        # Protein-specific sparsity (intermediate between genes and peaks)
        self.protein_sparsity_strength = nn.Parameter(torch.tensor(1.5))
        
        # Functional group weights (simulate different protein functions)
        self.functional_weights = nn.Parameter(torch.randn(3, input_dim))  # 3 functional groups
        
        # Initialize protein-specific parameters
        self._initialize_protein_specific_weights()
    
    def _initialize_protein_specific_weights(self):
        """Initialize protein-specific weights."""
        if self.protein_specific_features:
            # Initialize with focus on functional relationships
            for module in [self.functional_layer, self.ppi_attention, 
                          self.complex_encoder, self.localization_encoder, 
                          self.expression_normalizer]:
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight, gain=0.8)
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0)
        
        # Initialize functional weights
        nn.init.normal_(self.functional_weights, mean=0, std=0.1)
    
    def _build_modality_preprocessing(self) -> nn.Module:
        """Build protein-specific preprocessing layers."""
        if self.protein_specific_features:
            return nn.Sequential(
                # Robust normalization for protein expression
                nn.LayerNorm(self.input_dim),
                nn.Linear(self.input_dim, self.input_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                # Expression level stabilization
                nn.Linear(self.input_dim, self.input_dim),
                nn.LayerNorm(self.input_dim)
            )
        else:
            return super()._build_modality_preprocessing()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with protein-specific processing.
        
        Args:
            x: Protein feature tensor (n_proteins, input_dim)
            
        Returns:
            Tuple of (mean, log_variance) tensors
        """
        if self.protein_specific_features:
            # Functional processing
            x_func = self.functional_layer(x)
            
            # Protein-protein interaction attention
            ppi_attention = self.ppi_attention(x)
            x_ppi = x * ppi_attention
            
            # Complex formation processing
            x_complex = self.complex_encoder(x_ppi)
            
            # Localization processing
            x_loc = self.localization_encoder(x)
            
            # Expression normalization
            x_norm = self.expression_normalizer(x + x_func)
            
            # Functional group processing
            functional_effects = []
            for i in range(self.functional_weights.shape[0]):
                effect = torch.sum(x * self.functional_weights[i], dim=-1, keepdim=True)
                functional_effects.append(effect)
            functional_effect = torch.cat(functional_effects, dim=-1)
            functional_effect = F.softmax(functional_effect, dim=-1)
            
            # Apply functional weighting
            weighted_x = torch.zeros_like(x)
            for i in range(self.functional_weights.shape[0]):
                weighted_x += functional_effect[:, i:i+1] * (x * self.functional_weights[i])
            
            # Combine all protein-specific features
            x = x + 0.2 * x_func + 0.3 * x_complex + 0.2 * x_loc + 0.1 * x_norm + 0.2 * weighted_x
        
        # Apply base class forward pass
        return super().forward(x)
    
    def sample_topics(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Sample topic distribution with protein-specific characteristics.
        
        Args:
            mean: Mean of the distribution
            logvar: Log variance of the distribution
            
        Returns:
            Topic distribution (probabilities)
        """
        # Sample from the distribution
        sampled = self.reparameterize(mean, logvar)
        
        # Apply protein-specific sparsity (moderate level)
        sparsity_factor = torch.clamp(self.protein_sparsity_strength, min=0.5, max=3.0)
        topic_logits = sampled * sparsity_factor
        
        # Apply functional group modulation
        if self.protein_specific_features:
            # Enhance topic assignments based on functional similarity
            # Ensure dimensional compatibility
            func_weights_normalized = F.normalize(self.functional_weights.mean(0).unsqueeze(0), p=2, dim=1)
            sampled_normalized = F.normalize(sampled, p=2, dim=1)
            
            # Check dimensions and adjust if needed
            if func_weights_normalized.size(-1) == sampled_normalized.size(-1):
                functional_similarity = torch.mm(
                    func_weights_normalized,
                    sampled_normalized.t()
                )
                topic_logits = topic_logits + 0.1 * functional_similarity.t()
            else:
                # Skip functional modulation if dimensions don't match
                pass
        
        # Apply softmax to get valid probability distribution
        topic_distribution = F.softmax(topic_logits, dim=-1)
        
        return topic_distribution
    
    def compute_protein_specific_loss(self, topic_distribution: torch.Tensor,
                                     protein_features: torch.Tensor) -> torch.Tensor:
        """
        Compute protein-specific regularization loss.
        
        Args:
            topic_distribution: Protein topic distribution
            protein_features: Original protein features
            
        Returns:
            Protein-specific loss
        """
        # Encourage functional coherence in topic assignments
        
        # Moderate sparsity loss (proteins can belong to multiple functional categories)
        sparsity_loss = torch.mean(torch.sum(topic_distribution, dim=-1))
        
        # Functional coherence loss (proteins with similar functions should have similar topics)
        if self.protein_specific_features:
            protein_similarity = torch.mm(F.normalize(protein_features, p=2, dim=1),
                                        F.normalize(protein_features, p=2, dim=1).t())
            topic_consistency = torch.mm(topic_distribution, topic_distribution.t())
            coherence_loss = F.mse_loss(protein_similarity, topic_consistency)
        else:
            coherence_loss = torch.tensor(0.0, device=topic_distribution.device)
        
        # Functional diversity loss (encourage proteins to cover diverse functions)
        topic_means = torch.mean(topic_distribution, dim=0)
        diversity_loss = -torch.sum(topic_means * torch.log(topic_means + 1e-8))
        
        # Complex formation loss (proteins in complexes should share some topics)
        complex_loss = torch.tensor(0.0, device=topic_distribution.device)
        if self.protein_specific_features and topic_distribution.shape[0] > 1:
            # Simulate protein complex relationships
            complex_similarity = torch.mm(topic_distribution, topic_distribution.t())
            complex_target = torch.eye(topic_distribution.shape[0], device=topic_distribution.device) * 0.5
            complex_loss = F.mse_loss(complex_similarity, complex_target)
        
        # Combine losses (balanced for protein characteristics)
        total_loss = (0.2 * sparsity_loss + 0.4 * coherence_loss + 
                     0.2 * diversity_loss + 0.2 * complex_loss)
        
        return total_loss
