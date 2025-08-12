import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class GeneEncoder(nn.Module):
    """
    Dedicated encoder for gene expression data (RNA-seq).
    Handles the specific characteristics of gene expression data.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 output_dim: int, dropout: float = 0.1, 
                 activation: str = 'relu', batch_norm: bool = True,
                 use_log_transform: bool = True):
        """
        Initialize gene encoder.
        
        Args:
            input_dim: Number of genes
            hidden_dims: List of hidden layer dimensions
            output_dim: Output embedding dimension
            dropout: Dropout rate
            activation: Activation function ('relu', 'gelu', 'leaky_relu')
            batch_norm: Whether to use batch normalization
            use_log_transform: Whether to apply log(1+x) transformation
        """
        super(GeneEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.use_log_transform = use_log_transform
        
        # Input normalization layer for gene expression
        self.input_norm = nn.BatchNorm1d(input_dim) if batch_norm else nn.Identity()
        
        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            # Linear layer
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            # Batch normalization (except for output layer)
            if batch_norm and i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            
            # Activation (except for output layer)
            if i < len(dims) - 2:
                if activation == 'relu':
                    layers.append(nn.ReLU(inplace=True))
                elif activation == 'gelu':
                    layers.append(nn.GELU())
                elif activation == 'leaky_relu':
                    layers.append(nn.LeakyReLU(0.2, inplace=True))
                else:
                    raise ValueError(f"Unsupported activation: {activation}")
                
                # Dropout
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.encoder = nn.Sequential(*layers)
        
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Gene expression tensor of shape (batch_size, n_genes)
            
        Returns:
            Encoded tensor of shape (batch_size, output_dim)
        """
        # Apply log transformation if specified (common for gene expression)
        if self.use_log_transform:
            x = torch.log1p(x)  # log(1 + x)
        
        # Input normalization
        x = self.input_norm(x)
        
        # Apply encoder
        return self.encoder(x)
