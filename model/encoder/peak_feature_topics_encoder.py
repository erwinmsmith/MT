import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from .base_feature_topics_encoder import FeatureTopicsEncoder

class PeakFeatureTopicsEncoder(FeatureTopicsEncoder):
    """
    Specialized variational encoder for peak (chromatin accessibility) feature topic distributions.
    Optimized for chromatin accessibility characteristics and regulatory elements.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 n_topics: int, dropout: float = 0.1,
                 prior_mean: float = 0.0, prior_std: float = 1.0,
                 use_modality_specific_priors: bool = True,
                 peak_specific_features: bool = True,
                 chromatin_state_layers: int = 2, accessibility_enhancement: bool = True):
        """
        Initialize peak feature topics encoder.
        
        Args:
            input_dim: Input feature dimension (from heterogeneous GNN)
            hidden_dims: List of hidden layer dimensions
            n_topics: Number of topics (latent dimension)
            dropout: Dropout rate
            prior_mean: Prior mean for the latent distribution
            prior_std: Prior standard deviation for the latent distribution
            use_modality_specific_priors: Whether to use peak-specific priors
            peak_specific_features: Whether to use peak-specific processing
        """
        # Initialize base class with peak modality
        # Store peak-specific parameters first
        self.peak_specific_features = peak_specific_features
        self.chromatin_state_layers = chromatin_state_layers
        self.accessibility_enhancement = accessibility_enhancement
        
        super(PeakFeatureTopicsEncoder, self).__init__(
            modality='peak',
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            n_topics=n_topics,
            dropout=dropout,
            prior_mean=prior_mean,
            prior_std=prior_std,
            use_modality_specific_priors=use_modality_specific_priors
        )
        
        if peak_specific_features:
            # Peak-specific processing layers
            self.accessibility_layer = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(inplace=True),  # ReLU for binary-like accessibility
                nn.Dropout(dropout * 0.5)
            )
            
            # Regulatory element attention (peaks often regulate multiple genes)
            self.regulatory_attention = nn.Sequential(
                nn.Linear(input_dim, input_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(input_dim // 4, 1),
                nn.Sigmoid()
            )
            
            # Chromatin state simulation
            self.chromatin_state_encoder = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(input_dim // 2, input_dim)
            )
            
            # Binary enhancement for accessibility patterns
            self.binary_enhancement = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.Sigmoid()  # Enhance binary-like patterns
            )
        
        # Peak-specific sparsity (peaks are often very sparse)
        self.peak_sparsity_strength = nn.Parameter(torch.tensor(3.0))  # Higher than genes
        
        # Binary threshold for peak accessibility
        self.accessibility_threshold = nn.Parameter(torch.tensor(0.5))
        
        # Initialize peak-specific parameters
        self._initialize_peak_specific_weights()
    
    def _initialize_peak_specific_weights(self):
        """Initialize peak-specific weights."""
        if self.peak_specific_features:
            # Initialize with emphasis on binary patterns
            for module in [self.accessibility_layer, self.regulatory_attention, 
                          self.chromatin_state_encoder, self.binary_enhancement]:
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight, gain=0.5)
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0)
    
    def _build_modality_preprocessing(self) -> nn.Module:
        """Build peak-specific preprocessing layers."""
        if self.peak_specific_features:
            return nn.Sequential(
                # Binary normalization for accessibility
                nn.Linear(self.input_dim, self.input_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                # Peak accessibility enhancement
                nn.Linear(self.input_dim, self.input_dim),
                nn.Sigmoid()  # Enhance binary accessibility patterns
            )
        else:
            return super()._build_modality_preprocessing()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with peak-specific processing.
        
        Args:
            x: Peak feature tensor (n_peaks, input_dim)
            
        Returns:
            Tuple of (mean, log_variance) tensors
        """
        if self.peak_specific_features:
            # Accessibility processing
            x_acc = self.accessibility_layer(x)
            
            # Regulatory attention (peaks regulate multiple targets)
            reg_attention = self.regulatory_attention(x)
            x_regulated = x * reg_attention
            
            # Chromatin state processing
            x_chromatin = self.chromatin_state_encoder(x_regulated)
            
            # Binary enhancement
            x_binary = self.binary_enhancement(x + x_acc)
            
            # Combine all peak-specific features
            x = x + 0.2 * x_acc + 0.3 * x_chromatin + 0.2 * x_binary
            
            # Apply accessibility threshold
            x = x * (torch.sigmoid((x - self.accessibility_threshold) * 10))
        
        # Apply base class forward pass
        return super().forward(x)
    
    def sample_topics(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Sample topic distribution with peak-specific sparsity.
        
        Args:
            mean: Mean of the distribution
            logvar: Log variance of the distribution
            
        Returns:
            Topic distribution (probabilities)
        """
        # Sample from the distribution
        sampled = self.reparameterize(mean, logvar)
        
        # Apply peak-specific sparsity (highest among modalities)
        sparsity_factor = torch.clamp(self.peak_sparsity_strength, min=1.0, max=10.0)
        topic_logits = sampled * sparsity_factor
        
        # Apply accessibility-based masking
        accessibility_mask = torch.sigmoid(topic_logits - self.accessibility_threshold)
        topic_logits = topic_logits * accessibility_mask
        
        # Apply softmax to get valid probability distribution
        topic_distribution = F.softmax(topic_logits, dim=-1)
        
        return topic_distribution
    
    def compute_peak_specific_loss(self, topic_distribution: torch.Tensor,
                                  peak_features: torch.Tensor) -> torch.Tensor:
        """
        Compute peak-specific regularization loss.
        
        Args:
            topic_distribution: Peak topic distribution
            peak_features: Original peak features
            
        Returns:
            Peak-specific loss
        """
        # Encourage very sparse topic assignments for peaks (accessibility is binary)
        
        # High sparsity loss (peaks should have very sparse topic assignments)
        sparsity_loss = torch.mean(torch.sum(topic_distribution, dim=-1))
        
        # Binary consistency loss (encourage binary-like topic assignments)
        binary_loss = torch.mean(topic_distribution * (1 - topic_distribution))
        
        # Regulatory diversity loss (different peaks should regulate different pathways)
        topic_entropy = -torch.sum(topic_distribution * torch.log(topic_distribution + 1e-8), dim=-1)
        diversity_loss = -torch.mean(topic_entropy)  # Encourage low entropy (sparse)
        
        # Accessibility consistency (similar accessibility patterns → similar topics)
        if self.peak_specific_features:
            peak_similarity = torch.mm(F.normalize(peak_features, p=2, dim=1),
                                     F.normalize(peak_features, p=2, dim=1).t())
            topic_consistency = torch.mm(topic_distribution, topic_distribution.t())
            consistency_loss = F.mse_loss(peak_similarity, topic_consistency)
        else:
            consistency_loss = torch.tensor(0.0, device=topic_distribution.device)
        
        # Combine losses (emphasize sparsity for peaks)
        total_loss = 0.5 * sparsity_loss + 0.3 * binary_loss + 0.1 * diversity_loss + 0.1 * consistency_loss
        
        return total_loss
