import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np

class DirectedGNN(nn.Module):
    """
    Directed Graph Neural Network for processing cell graphs.
    Handles directed edges and maintains directional information flow.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_layers: int = 3, dropout: float = 0.1, 
                 activation: str = 'relu', use_residual: bool = True):
        """
        Initialize directed GNN.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            n_layers: Number of GNN layers
            dropout: Dropout rate
            activation: Activation function
            use_residual: Whether to use residual connections
        """
        super(DirectedGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.use_residual = use_residual
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Directed GNN layers
        self.gnn_layers = nn.ModuleList([
            DirectedGNNLayer(hidden_dim, hidden_dim, dropout, activation)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        
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
    
    def forward(self, node_features: torch.Tensor, 
                adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through directed GNN.
        
        Args:
            node_features: Node features of shape (batch_size, n_nodes, input_dim)
            adjacency_matrix: Directed adjacency matrix of shape (batch_size, n_nodes, n_nodes)
            
        Returns:
            Updated node features of shape (batch_size, n_nodes, output_dim)
        """
        # Input projection
        x = self.input_projection(node_features)
        x = self.dropout(x)
        
        # Apply GNN layers
        for i, (gnn_layer, layer_norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            residual = x if self.use_residual else None
            
            # Apply GNN layer
            x = gnn_layer(x, adjacency_matrix)
            
            # Layer normalization
            x = layer_norm(x)
            
            # Residual connection
            if residual is not None and x.shape == residual.shape:
                x = x + residual
            
            # Dropout
            x = self.dropout(x)
        
        # Output projection
        output = self.output_projection(x)
        
        return output

class DirectedGNNLayer(nn.Module):
    """Single layer of directed GNN."""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 dropout: float = 0.1, activation: str = 'relu'):
        super(DirectedGNNLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Separate transformations for incoming and outgoing edges
        self.W_in = nn.Linear(input_dim, output_dim, bias=False)
        self.W_out = nn.Linear(input_dim, output_dim, bias=False)
        self.W_self = nn.Linear(input_dim, output_dim, bias=True)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of directed GNN layer.
        
        Args:
            x: Node features (batch_size, n_nodes, input_dim)
            adj_matrix: Adjacency matrix (batch_size, n_nodes, n_nodes)
            
        Returns:
            Updated node features (batch_size, n_nodes, output_dim)
        """
        batch_size, n_nodes, _ = x.shape
        
        # Self transformation
        x_self = self.W_self(x)
        
        # Incoming edge aggregation (transpose adjacency matrix)
        adj_in = adj_matrix.transpose(-2, -1)  # Transpose to get incoming edges
        x_in = self.W_in(x)
        x_in_agg = torch.bmm(adj_in, x_in)  # Aggregate from incoming neighbors
        
        # Outgoing edge aggregation
        x_out = self.W_out(x)
        x_out_agg = torch.bmm(adj_matrix, x_out)  # Aggregate to outgoing neighbors
        
        # Combine transformations
        output = x_self + x_in_agg + x_out_agg
        
        # Apply activation and dropout
        output = self.activation(output)
        output = self.dropout(output)
        
        return output

class TrajectoryInference(nn.Module):
    """
    Trajectory inference module for inferring cell-to-cell directed relationships.
    """
    
    def __init__(self, embed_dim: int, n_components: int = 50, 
                 kernel_type: str = 'gaussian', sigma: float = 1.0):
        """
        Initialize trajectory inference module.
        
        Args:
            embed_dim: Embedding dimension
            n_components: Number of diffusion components
            kernel_type: Type of kernel ('gaussian', 'cosine')
            sigma: Kernel bandwidth parameter
        """
        super(TrajectoryInference, self).__init__()
        
        self.embed_dim = embed_dim
        self.n_components = n_components
        self.kernel_type = kernel_type
        self.sigma = sigma
        
        # Learnable projection for trajectory embedding
        self.trajectory_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 4, n_components)
        )
    
    def compute_pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distances between cell embeddings.
        
        Args:
            embeddings: Cell embeddings (batch_size, n_cells, embed_dim)
            
        Returns:
            Distance matrix (batch_size, n_cells, n_cells)
        """
        # Compute pairwise squared Euclidean distances
        embeddings_expanded_1 = embeddings.unsqueeze(2)  # (batch_size, n_cells, 1, embed_dim)
        embeddings_expanded_2 = embeddings.unsqueeze(1)  # (batch_size, 1, n_cells, embed_dim)
        
        squared_distances = torch.sum((embeddings_expanded_1 - embeddings_expanded_2) ** 2, dim=-1)
        
        return squared_distances
    
    def create_kernel_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Create kernel matrix from embeddings.
        
        Args:
            embeddings: Cell embeddings (batch_size, n_cells, embed_dim)
            
        Returns:
            Kernel matrix (batch_size, n_cells, n_cells)
        """
        if self.kernel_type == 'gaussian':
            distances = self.compute_pairwise_distances(embeddings)
            kernel_matrix = torch.exp(-distances / (2 * self.sigma ** 2))
        elif self.kernel_type == 'cosine':
            # Normalize embeddings
            embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
            kernel_matrix = torch.bmm(embeddings_norm, embeddings_norm.transpose(-2, -1))
            kernel_matrix = (kernel_matrix + 1) / 2  # Scale to [0, 1]
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")
        
        return kernel_matrix
    
    def infer_trajectory(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Infer trajectory from cell embeddings.
        
        Args:
            embeddings: Cell embeddings (batch_size, n_cells, embed_dim)
            
        Returns:
            Directed adjacency matrix (batch_size, n_cells, n_cells)
        """
        batch_size, n_cells, _ = embeddings.shape
        
        # Project embeddings for trajectory inference
        trajectory_embeddings = self.trajectory_projection(embeddings)
        
        # Create kernel matrix
        kernel_matrix = self.create_kernel_matrix(trajectory_embeddings)
        
        # Convert to directed adjacency matrix using asymmetric transformation
        # Use a learnable asymmetric transformation
        directed_matrix = self._create_directed_matrix(kernel_matrix)
        
        return directed_matrix
    
    def _create_directed_matrix(self, kernel_matrix: torch.Tensor) -> torch.Tensor:
        """
        Convert symmetric kernel matrix to directed adjacency matrix.
        
        Args:
            kernel_matrix: Symmetric kernel matrix
            
        Returns:
            Directed adjacency matrix
        """
        batch_size, n_cells, _ = kernel_matrix.shape
        
        # Apply row-wise softmax to create directed probabilities
        # This ensures each row sums to 1, representing transition probabilities
        directed_matrix = F.softmax(kernel_matrix / self.sigma, dim=-1)
        
        # Optional: Add small epsilon to avoid completely sparse matrices
        epsilon = 1e-6
        directed_matrix = directed_matrix + epsilon
        directed_matrix = directed_matrix / directed_matrix.sum(dim=-1, keepdim=True)
        
        return directed_matrix
