import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class FusedEmbeddingEncoder(nn.Module):
    """
    Dedicated encoder for processing fused embeddings from multimodal fusion.
    This encoder processes the output of the multimodal fusion block.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 output_dim: int, dropout: float = 0.1, 
                 activation: str = 'gelu', use_residual: bool = True,
                 use_attention: bool = True):
        """
        Initialize fused embedding encoder.
        
        Args:
            input_dim: Input embedding dimension from fusion block
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            dropout: Dropout rate
            activation: Activation function ('relu', 'gelu', 'leaky_relu')
            use_residual: Whether to use residual connections
            use_attention: Whether to use self-attention mechanism
        """
        super(FusedEmbeddingEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.use_residual = use_residual
        self.use_attention = use_attention
        
        # Self-attention mechanism for fused embeddings
        if use_attention:
            self.self_attention = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(input_dim)
        
        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            # Linear layer
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            # Layer normalization (better for transformers/attention)
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
            
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
        
        self.encoder_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        # Create layers with potential residual connections
        for i in range(len(dims) - 1):
            self.encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                self.layer_norms.append(nn.LayerNorm(dims[i + 1]))
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
    
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Fused embedding tensor of shape (batch_size, input_dim)
            
        Returns:
            Encoded tensor of shape (batch_size, output_dim)
        """
        # Apply self-attention if enabled
        if self.use_attention:
            # Add sequence dimension for attention
            x_seq = x.unsqueeze(1)  # (batch_size, 1, input_dim)
            
            # Self-attention
            attn_output, _ = self.self_attention(x_seq, x_seq, x_seq)
            attn_output = attn_output.squeeze(1)  # (batch_size, input_dim)
            
            # Residual connection and layer norm
            x = self.attention_norm(x + attn_output)
        
        # Apply encoder layers
        for i, layer in enumerate(self.encoder_layers):
            residual = x if self.use_residual and i > 0 else None
            
            x = layer(x)
            
            # Apply layer norm, activation, and dropout (except for output layer)
            if i < len(self.encoder_layers) - 1:
                x = self.layer_norms[i](x)
                x = self.activation(x)
                x = self.dropout(x)
                
                # Residual connection (if dimensions match)
                if (residual is not None and 
                    residual.shape == x.shape and 
                    i > 0):
                    x = x + residual
        
        return x
