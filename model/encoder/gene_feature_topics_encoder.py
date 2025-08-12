import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from .base_feature_topics_encoder import FeatureTopicsEncoder

class GeneFeatureTopicsEncoder(FeatureTopicsEncoder):
    """
    Specialized variational encoder for gene feature topic distributions.
    Optimized for gene expression characteristics and regulatory relationships.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 n_topics: int, dropout: float = 0.1,
                 prior_mean: float = 0.0, prior_std: float = 1.0,
                 use_modality_specific_priors: bool = True,
                 gene_specific_features: bool = True,
                 gene_regulation_layers: int = 2, pathway_enrichment: bool = True):
        """
        Initialize gene feature topics encoder.
        
        Args:
            input_dim: Input feature dimension (from heterogeneous GNN)
            hidden_dims: List of hidden layer dimensions
            n_topics: Number of topics (latent dimension)
            dropout: Dropout rate
            prior_mean: Prior mean for the latent distribution
            prior_std: Prior standard deviation for the latent distribution
            use_modality_specific_priors: Whether to use gene-specific priors
            gene_specific_features: Whether to use gene-specific processing
        """
        # Initialize base class with gene modality
        # Store gene-specific parameters first
        self.gene_specific_features = gene_specific_features
        self.gene_regulation_layers = gene_regulation_layers
        self.pathway_enrichment = pathway_enrichment
        
        super(GeneFeatureTopicsEncoder, self).__init__(
            modality='gene',
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            n_topics=n_topics,
            dropout=dropout,
            prior_mean=prior_mean,
            prior_std=prior_std,
            use_modality_specific_priors=use_modality_specific_priors
        )
        
        if gene_specific_features:
            # Gene-specific processing layers
            self.gene_regulation_layer = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.LayerNorm(input_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5)
            )
            
            # Gene expression level attention
            self.expression_attention = nn.Sequential(
                nn.Linear(input_dim, input_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(input_dim // 4, 1),
                nn.Sigmoid()
            )
            
            # Pathway enrichment simulation
            self.pathway_encoder = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim // 2, input_dim)
            )
        
        # Gene-specific sparsity (genes often have sparse topic assignments)
        self.gene_sparsity_strength = nn.Parameter(torch.tensor(2.0))  # Higher than default
        
        # Initialize gene-specific parameters
        self._initialize_gene_specific_weights()
    
    def _initialize_gene_specific_weights(self):
        """Initialize gene-specific weights."""
        if self.gene_specific_features:
            # Initialize with smaller weights for stability
            for module in [self.gene_regulation_layer, self.expression_attention, self.pathway_encoder]:
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight, gain=0.5)
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0)
    
    def _build_modality_preprocessing(self) -> nn.Module:
        """Build gene-specific preprocessing layers."""
        if self.gene_specific_features:
            return nn.Sequential(
                nn.LayerNorm(self.input_dim),
                nn.Linear(self.input_dim, self.input_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                # Gene expression variability normalization
                nn.BatchNorm1d(self.input_dim) if self.input_dim > 1 else nn.Identity()
            )
        else:
            return super()._build_modality_preprocessing()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with gene-specific processing.
        
        Args:
            x: Gene feature tensor (n_genes, input_dim)
            
        Returns:
            Tuple of (mean, log_variance) tensors
        """
        if self.gene_specific_features:
            # Gene regulation processing
            x_reg = self.gene_regulation_layer(x)
            
            # Expression level attention
            attention_weights = self.expression_attention(x)
            x_attended = x * attention_weights
            
            # Pathway enrichment
            x_pathway = self.pathway_encoder(x_attended)
            
            # Combine all gene-specific features
            x = x + 0.3 * x_reg + 0.2 * x_pathway
        
        # Apply base class forward pass
        return super().forward(x)
    
    def sample_topics(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Sample topic distribution with gene-specific sparsity.
        
        Args:
            mean: Mean of the distribution
            logvar: Log variance of the distribution
            
        Returns:
            Topic distribution (probabilities)
        """
        # Sample from the distribution
        sampled = self.reparameterize(mean, logvar)
        
        # Apply gene-specific sparsity (higher than base class)
        sparsity_factor = torch.clamp(self.gene_sparsity_strength, min=0.5, max=5.0)
        topic_logits = sampled * sparsity_factor
        
        # Apply softmax to get valid probability distribution
        topic_distribution = F.softmax(topic_logits, dim=-1)
        
        return topic_distribution
    
    def compute_gene_specific_loss(self, topic_distribution: torch.Tensor, 
                                  gene_features: torch.Tensor) -> torch.Tensor:
        """
        Compute gene-specific regularization loss.
        
        Args:
            topic_distribution: Gene topic distribution
            gene_features: Original gene features
            
        Returns:
            Gene-specific loss
        """
        # Encourage sparse but diverse topic assignments for genes
        # Genes should have sparse topic assignments but cover diverse biological functions
        
        # Sparsity loss (L1 on topic distribution)
        sparsity_loss = torch.mean(torch.sum(topic_distribution, dim=-1))
        
        # Diversity loss (encourage different genes to have different topic patterns)
        topic_similarity = torch.mm(topic_distribution, topic_distribution.t())
        diversity_loss = torch.mean(topic_similarity) - torch.mean(torch.diag(topic_similarity))
        
        # Feature consistency loss (similar genes should have similar topic patterns)
        if self.gene_specific_features:
            feature_similarity = torch.mm(F.normalize(gene_features, p=2, dim=1),
                                        F.normalize(gene_features, p=2, dim=1).t())
            topic_consistency = torch.mm(topic_distribution, topic_distribution.t())
            consistency_loss = F.mse_loss(feature_similarity, topic_consistency)
        else:
            consistency_loss = torch.tensor(0.0, device=topic_distribution.device)
        
        # Combine losses
        total_loss = 0.3 * sparsity_loss + 0.2 * diversity_loss + 0.1 * consistency_loss
        
        return total_loss
