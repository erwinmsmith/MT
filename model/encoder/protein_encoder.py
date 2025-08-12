import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class ProteinEncoder(nn.Module):
    """
    Dedicated encoder for protein expression data (CITEseq).
    Handles the specific characteristics of protein expression data.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 output_dim: int, dropout: float = 0.1, 
                 activation: str = 'relu', batch_norm: bool = True,
                 use_log_transform: bool = True, robust_scaling: bool = True):
        """
        Initialize protein encoder.
        
        Args:
            input_dim: Number of proteins
            hidden_dims: List of hidden layer dimensions
            output_dim: Output embedding dimension
            dropout: Dropout rate
            activation: Activation function ('relu', 'gelu', 'leaky_relu')
            batch_norm: Whether to use batch normalization
            use_log_transform: Whether to apply log transformation
            robust_scaling: Whether to use robust scaling for protein data
        """
        super(ProteinEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.use_log_transform = use_log_transform
        self.robust_scaling = robust_scaling
        
        # Input normalization layer for protein expression
        self.input_norm = nn.BatchNorm1d(input_dim) if batch_norm else nn.Identity()
        
        # Protein-specific preprocessing layer
        if robust_scaling:
            self.protein_preprocessing = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.LayerNorm(input_dim),  # More robust to outliers than BatchNorm
                nn.ReLU(inplace=True)
            )
        else:
            self.protein_preprocessing = nn.Identity()
        
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
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Protein expression tensor of shape (batch_size, n_proteins)
            
        Returns:
            Encoded tensor of shape (batch_size, output_dim)
        """
        # Apply log transformation if specified (common for protein expression)
        if self.use_log_transform:
            x = torch.log1p(x)  # log(1 + x)
        
        # Protein-specific preprocessing
        x = self.protein_preprocessing(x)
        
        # Input normalization
        x = self.input_norm(x)
        
        # Apply encoder
        return self.encoder(x)
