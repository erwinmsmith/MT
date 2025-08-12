import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

class FeatureTopicsEncoder(nn.Module):
    """
    Dedicated variational encoder for generating feature topic distributions (B_d).
    This encoder produces topic distributions for features in each modality.
    """
    
    def __init__(self, modality: str, input_dim: int, hidden_dims: List[int], 
                 n_topics: int, dropout: float = 0.1,
                 prior_mean: float = 0.0, prior_std: float = 1.0,
                 use_modality_specific_priors: bool = True):
        """
        Initialize feature topics encoder for a specific modality.
        
        Args:
            modality: Modality name ('gene', 'peak', 'protein')
            input_dim: Input feature dimension (from heterogeneous GNN)
            hidden_dims: List of hidden layer dimensions
            n_topics: Number of topics (latent dimension)
            dropout: Dropout rate
            prior_mean: Prior mean for the latent distribution
            prior_std: Prior standard deviation for the latent distribution
            use_modality_specific_priors: Whether to use different priors for each modality
        """
        super(FeatureTopicsEncoder, self).__init__()
        
        self.modality = modality
        self.input_dim = input_dim
        self.n_topics = n_topics
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.use_modality_specific_priors = use_modality_specific_priors
        
        # Modality-specific preprocessing
        self.modality_preprocessing = self._build_modality_preprocessing()
        
        # Shared layers for processing input
        shared_layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            shared_layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),  # Layer norm better for features
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        
        self.shared_layers = nn.Sequential(*shared_layers)
        
        # Separate heads for mean and log variance
        final_dim = hidden_dims[-1] if hidden_dims else input_dim
        
        # Mean head with modality-specific initialization
        self.mean_head = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(final_dim // 2, n_topics)
        )
        
        # Log variance head
        self.logvar_head = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(final_dim // 2, n_topics)
        )
        
        # Modality-specific parameters
        if use_modality_specific_priors:
            self.modality_prior_mean = nn.Parameter(torch.tensor(prior_mean))
            self.modality_prior_logstd = nn.Parameter(torch.log(torch.tensor(prior_std)))
        
        # Topic sparsity parameter (encourages sparse topic assignments)
        self.sparsity_strength = nn.Parameter(torch.ones(1))
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_modality_preprocessing(self) -> nn.Module:
        """Build modality-specific preprocessing layers."""
        if self.modality == 'gene':
            # Gene features might benefit from additional normalization
            return nn.Sequential(
                nn.LayerNorm(self.input_dim),
                nn.Linear(self.input_dim, self.input_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
        elif self.modality == 'peak':
            # Peak features are often binary/sparse
            return nn.Sequential(
                nn.Linear(self.input_dim, self.input_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            )
        elif self.modality == 'protein':
            # Protein features might have different scales
            return nn.Sequential(
                nn.LayerNorm(self.input_dim),
                nn.Linear(self.input_dim, self.input_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
        else:
            return nn.Identity()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        
        # Modality-specific initialization
        if self.modality == 'gene':
            # Genes might have more diverse topic assignments
            nn.init.normal_(self.mean_head[-1].weight, mean=0, std=0.02)
        elif self.modality == 'peak':
            # Peaks might be more sparse
            nn.init.normal_(self.mean_head[-1].weight, mean=0, std=0.01)
        elif self.modality == 'protein':
            # Proteins might have intermediate sparsity
            nn.init.normal_(self.mean_head[-1].weight, mean=0, std=0.015)
        
        # Initialize mean head bias
        if self.use_modality_specific_priors:
            nn.init.constant_(self.mean_head[-1].bias, 0)
        else:
            nn.init.constant_(self.mean_head[-1].bias, self.prior_mean)
        
        # Initialize logvar head
        nn.init.normal_(self.logvar_head[-1].weight, mean=0, std=0.01)
        logvar_init = 2 * torch.log(torch.tensor(self.prior_std))
        nn.init.constant_(self.logvar_head[-1].bias, logvar_init)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get variational parameters.
        
        Args:
            x: Input tensor of shape (n_features, input_dim)
            
        Returns:
            Tuple of (mean, log_variance) tensors
        """
        # Modality-specific preprocessing
        x = self.modality_preprocessing(x)
        
        # Shared processing
        shared_output = self.shared_layers(x)
        
        # Get mean and log variance
        mean = self.mean_head(shared_output)
        logvar = self.logvar_head(shared_output)
        
        # Add modality-specific prior if enabled
        if self.use_modality_specific_priors:
            mean = mean + self.modality_prior_mean
        
        # Clamp log variance to avoid numerical issues
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        return mean, logvar
    
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for variational inference.
        
        Args:
            mean: Mean of the distribution
            logvar: Log variance of the distribution
            
        Returns:
            Sampled tensor from the distribution
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            return mean
    
    def sample_topics(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Sample topic distribution and apply softmax with sparsity.
        
        Args:
            mean: Mean of the distribution
            logvar: Log variance of the distribution
            
        Returns:
            Topic distribution (probabilities)
        """
        # Sample from the distribution
        sampled = self.reparameterize(mean, logvar)
        
        # Apply sparsity-inducing transformation
        # Higher sparsity_strength leads to more sparse topic assignments
        sparsity_factor = torch.clamp(self.sparsity_strength, min=0.1, max=10.0)
        topic_logits = sampled * sparsity_factor
        
        # Apply softmax to get valid probability distribution
        topic_distribution = F.softmax(topic_logits, dim=-1)
        
        return topic_distribution
    
    def compute_kl_divergence(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between the approximate posterior and prior.
        
        Args:
            mean: Mean of the approximate posterior
            logvar: Log variance of the approximate posterior
            
        Returns:
            KL divergence
        """
        if self.use_modality_specific_priors:
            prior_mean = self.modality_prior_mean
            prior_std = torch.exp(self.modality_prior_logstd)
        else:
            prior_mean = self.prior_mean
            prior_std = self.prior_std
        
        prior_var = prior_std ** 2
        
        # KL divergence between N(mean, exp(logvar)) and N(prior_mean, prior_var)
        kl_div = 0.5 * (
            torch.log(prior_var) - logvar +
            (torch.exp(logvar) + (mean - prior_mean) ** 2) / prior_var - 1
        )
        
        return kl_div.sum(dim=-1).mean()
    
    def compute_sparsity_loss(self, topic_distribution: torch.Tensor) -> torch.Tensor:
        """
        Compute sparsity loss to encourage sparse topic assignments.
        
        Args:
            topic_distribution: Topic distribution tensor
            
        Returns:
            Sparsity loss
        """
        # L1 loss on topic distribution to encourage sparsity
        sparsity_loss = torch.mean(torch.sum(topic_distribution, dim=-1))
        
        # Entropy loss to prevent collapse to single topic
        entropy = -torch.sum(topic_distribution * torch.log(topic_distribution + 1e-8), dim=-1)
        entropy_loss = -torch.mean(entropy)  # Negative because we want some entropy
        
        # Combine sparsity and entropy
        total_sparsity_loss = sparsity_loss + 0.1 * entropy_loss
        
        return total_sparsity_loss
    
    def get_topic_distribution(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get topic distribution along with variational parameters.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (topic_distribution, mean, logvar)
        """
        mean, logvar = self.forward(x)
        topic_distribution = self.sample_topics(mean, logvar)
        
        return topic_distribution, mean, logvar
