import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer for learning attention weights between nodes.
    """
    
    def __init__(self, input_dim: int, output_dim: int, n_heads: int = 1,
                 dropout: float = 0.1, use_bias: bool = True,
                 attention_dropout: float = 0.1):
        """
        Initialize Graph Attention Layer.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
            use_bias: Whether to use bias in linear layers
            attention_dropout: Dropout rate for attention weights
        """
        super(GraphAttentionLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.head_dim = output_dim // n_heads
        
        assert output_dim % n_heads == 0, "output_dim must be divisible by n_heads"
        
        # Linear transformations for queries, keys, and values
        self.query_projection = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.key_projection = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.value_projection = nn.Linear(input_dim, output_dim, bias=use_bias)
        
        # Attention mechanism
        self.attention_weights = nn.Parameter(torch.randn(n_heads, 2 * self.head_dim))
        
        # Output projection
        self.output_projection = nn.Linear(output_dim, output_dim, bias=use_bias)
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in [self.query_projection, self.key_projection, 
                      self.value_projection, self.output_projection]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
        # Initialize attention weights
        nn.init.xavier_uniform_(self.attention_weights)
    
    def forward(self, x: torch.Tensor, adjacency_matrix: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through graph attention layer.
        
        Args:
            x: Node features (batch_size, n_nodes, input_dim)
            adjacency_matrix: Adjacency matrix (batch_size, n_nodes, n_nodes)
            mask: Optional mask for attention (batch_size, n_nodes, n_nodes)
            
        Returns:
            Tuple of (updated_features, attention_weights)
        """
        batch_size, n_nodes, _ = x.shape
        
        # Linear transformations
        queries = self.query_projection(x)  # (batch_size, n_nodes, output_dim)
        keys = self.key_projection(x)       # (batch_size, n_nodes, output_dim)
        values = self.value_projection(x)   # (batch_size, n_nodes, output_dim)
        
        # Reshape for multi-head attention
        queries = queries.view(batch_size, n_nodes, self.n_heads, self.head_dim)
        keys = keys.view(batch_size, n_nodes, self.n_heads, self.head_dim)
        values = values.view(batch_size, n_nodes, self.n_heads, self.head_dim)
        
        # Compute attention scores
        attention_scores = self._compute_attention_scores(queries, keys)
        
        # Apply adjacency matrix mask (only attend to connected nodes)
        adjacency_mask = adjacency_matrix.unsqueeze(2).expand(-1, -1, self.n_heads, -1)
        attention_scores = attention_scores.masked_fill(adjacency_mask == 0, float('-inf'))
        
        # Apply additional mask if provided
        if mask is not None:
            mask = mask.unsqueeze(2).expand(-1, -1, self.n_heads, -1)
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, values)  # (batch_size, n_nodes, n_heads, head_dim)
        
        # Concatenate heads
        attended_values = attended_values.contiguous().view(batch_size, n_nodes, self.output_dim)
        
        # Output projection
        output = self.output_projection(attended_values)
        output = self.dropout(output)
        
        # Residual connection and layer normalization
        if x.shape[-1] == output.shape[-1]:
            output = self.layer_norm(output + x)
        else:
            output = self.layer_norm(output)
        
        # Return mean attention weights across heads for visualization
        mean_attention_weights = attention_weights.mean(dim=2)
        
        return output, mean_attention_weights
    
    def _compute_attention_scores(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        Compute attention scores using learnable attention mechanism.
        
        Args:
            queries: Query vectors (batch_size, n_nodes, n_heads, head_dim)
            keys: Key vectors (batch_size, n_nodes, n_heads, head_dim)
            
        Returns:
            Attention scores (batch_size, n_nodes, n_heads, n_nodes)
        """
        batch_size, n_nodes, n_heads, head_dim = queries.shape
        
        # Prepare for pairwise attention computation
        queries_expanded = queries.unsqueeze(3).expand(-1, -1, -1, n_nodes, -1)
        keys_expanded = keys.unsqueeze(1).expand(-1, n_nodes, -1, -1, -1)
        
        # Concatenate queries and keys
        attention_input = torch.cat([queries_expanded, keys_expanded], dim=-1)
        
        # Compute attention scores using learnable weights
        attention_scores = torch.matmul(attention_input, self.attention_weights.unsqueeze(0).unsqueeze(0).unsqueeze(0))
        
        # Apply activation
        attention_scores = F.leaky_relu(attention_scores, negative_slope=0.2)
        
        return attention_scores

class MultiHeadGraphAttention(nn.Module):
    """
    Multi-head graph attention with separate attention mechanisms per head.
    """
    
    def __init__(self, input_dim: int, output_dim: int, n_heads: int = 8,
                 dropout: float = 0.1, use_residual: bool = True):
        """
        Initialize Multi-head Graph Attention.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
            use_residual: Whether to use residual connections
        """
        super(MultiHeadGraphAttention, self).__init__()
        
        self.n_heads = n_heads
        self.use_residual = use_residual
        
        # Multiple attention heads
        self.attention_heads = nn.ModuleList([
            GraphAttentionLayer(input_dim, output_dim // n_heads, n_heads=1, dropout=dropout)
            for _ in range(n_heads)
        ])
        
        # Output combination
        self.output_combination = nn.Linear(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, adjacency_matrix: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through multi-head graph attention.
        
        Args:
            x: Node features
            adjacency_matrix: Adjacency matrix
            mask: Optional attention mask
            
        Returns:
            Tuple of (updated_features, attention_weights)
        """
        head_outputs = []
        head_attentions = []
        
        # Apply each attention head
        for attention_head in self.attention_heads:
            head_output, head_attention = attention_head(x, adjacency_matrix, mask)
            head_outputs.append(head_output)
            head_attentions.append(head_attention)
        
        # Concatenate head outputs
        combined_output = torch.cat(head_outputs, dim=-1)
        
        # Final transformation
        output = self.output_combination(combined_output)
        output = self.dropout(output)
        
        # Residual connection and normalization
        if self.use_residual and x.shape[-1] == output.shape[-1]:
            output = self.layer_norm(output + x)
        else:
            output = self.layer_norm(output)
        
        # Average attention weights across heads
        mean_attention = torch.stack(head_attentions).mean(dim=0)
        
        return output, mean_attention
