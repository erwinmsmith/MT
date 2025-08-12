import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

class BaseGNN(nn.Module, ABC):
    """
    Base class for Graph Neural Networks.
    Provides common functionality for all GNN implementations.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_layers: int = 3, dropout: float = 0.1, 
                 activation: str = 'relu', use_residual: bool = True):
        """
        Initialize base GNN.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            n_layers: Number of GNN layers
            dropout: Dropout rate
            activation: Activation function
            use_residual: Whether to use residual connections
        """
        super(BaseGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout
        self.use_residual = use_residual
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Input and output projections
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalizations
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.constant_(self.input_projection.bias, 0)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.constant_(self.output_projection.bias, 0)
    
    @abstractmethod
    def graph_convolution(self, x: torch.Tensor, edge_info: Any, layer_idx: int) -> torch.Tensor:
        """
        Abstract method for graph convolution operation.
        Must be implemented by subclasses.
        """
        pass
    
    def forward(self, x: torch.Tensor, edge_info: Any) -> torch.Tensor:
        """
        Forward pass through GNN.
        
        Args:
            x: Node features
            edge_info: Edge information (adjacency matrix, edge indices, etc.)
            
        Returns:
            Updated node features
        """
        # Input projection
        x = self.input_projection(x)
        x = self.input_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Apply graph convolution layers
        for layer_idx in range(self.n_layers):
            residual = x if self.use_residual else None
            
            # Graph convolution
            x = self.graph_convolution(x, edge_info, layer_idx)
            
            # Residual connection
            if residual is not None and x.shape == residual.shape:
                x = x + residual
            
            # Activation and dropout (except for last layer)
            if layer_idx < self.n_layers - 1:
                x = self.activation(x)
                x = self.dropout(x)
        
        # Output projection
        x = self.output_projection(x)
        x = self.output_norm(x)
        
        return x
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get attention weights if available (for interpretability)."""
        return None
    
    def get_node_embeddings(self, x: torch.Tensor, edge_info: Any, layer_idx: int = -1) -> torch.Tensor:
        """Get node embeddings from a specific layer."""
        if layer_idx == -1:
            return self.forward(x, edge_info)
        else:
            # Forward pass up to specified layer
            x = self.input_projection(x)
            x = self.input_norm(x)
            x = self.activation(x)
            x = self.dropout(x)
            
            for i in range(min(layer_idx + 1, self.n_layers)):
                residual = x if self.use_residual else None
                x = self.graph_convolution(x, edge_info, i)
                
                if residual is not None and x.shape == residual.shape:
                    x = x + residual
                
                if i < self.n_layers - 1:
                    x = self.activation(x)
                    x = self.dropout(x)
            
            return x
