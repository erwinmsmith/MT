import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class CellTopicsEncoder(nn.Module):
    """
    Dedicated variational encoder for generating cell topic distributions (θ_d).
    This encoder produces the parameters for the variational distribution of cell topics.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 n_topics: int, dropout: float = 0.1,
                 prior_mean: float = 0.0, prior_std: float = 1.0,
                 use_batch_norm: bool = True):
        """
        Initialize cell topics encoder.
        
        Args:
            input_dim: Input feature dimension (from directed GNN)
            hidden_dims: List of hidden layer dimensions
            n_topics: Number of topics (latent dimension)
            dropout: Dropout rate
            prior_mean: Prior mean for the latent distribution
            prior_std: Prior standard deviation for the latent distribution
            use_batch_norm: Whether to use batch normalization
        """
        super(CellTopicsEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.n_topics = n_topics
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        
        # Shared layers for processing input
        shared_layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            shared_layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]) if use_batch_norm else nn.Identity(),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        
        self.shared_layers = nn.Sequential(*shared_layers)
        
        # Separate heads for mean and log variance
        final_dim = hidden_dims[-1] if hidden_dims else input_dim
        
        # Mean head
        self.mean_head = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),  # Less dropout for final layers
            nn.Linear(final_dim // 2, n_topics)
        )
        
        # Log variance head
        self.logvar_head = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(final_dim // 2, n_topics)
        )
        
        # Topic temperature parameter (learnable)
        self.topic_temperature = nn.Parameter(torch.ones(1))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        
        # Initialize mean head to produce values around prior mean
        nn.init.normal_(self.mean_head[-1].weight, mean=0, std=0.01)
        nn.init.constant_(self.mean_head[-1].bias, self.prior_mean)
        
        # Initialize logvar head to produce values around log(prior_std^2)
        nn.init.normal_(self.logvar_head[-1].weight, mean=0, std=0.01)
        nn.init.constant_(self.logvar_head[-1].bias, 2 * torch.log(torch.tensor(self.prior_std)))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get variational parameters.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (mean, log_variance) tensors
        """
        # Shared processing
        shared_output = self.shared_layers(x)
        
        # Get mean and log variance
        mean = self.mean_head(shared_output)
        logvar = self.logvar_head(shared_output)
        
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
        Sample topic distribution and apply softmax.
        
        Args:
            mean: Mean of the distribution
            logvar: Log variance of the distribution
            
        Returns:
            Topic distribution (probabilities)
        """
        # Sample from the distribution
        sampled = self.reparameterize(mean, logvar)
        
        # Apply temperature scaling and softmax
        topic_logits = sampled / torch.clamp(self.topic_temperature, min=0.1)
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
        # KL divergence between N(mean, exp(logvar)) and N(prior_mean, prior_std^2)
        prior_var = self.prior_std ** 2
        kl_div = 0.5 * (
            torch.log(torch.tensor(prior_var)) - logvar +
            (torch.exp(logvar) + (mean - self.prior_mean) ** 2) / prior_var - 1
        )
        
        return kl_div.sum(dim=-1).mean()
    
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
