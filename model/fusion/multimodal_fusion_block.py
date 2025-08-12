import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import math

class MultimodalFusionBlock(nn.Module):
    """
    Multi-head attention based fusion block for combining embeddings from different modalities.
    Implements a transformer-like architecture for modality fusion.
    """
    
    def __init__(self, embed_dim: int, n_heads: int = 8, n_layers: int = 3,
                 dropout: float = 0.1, feedforward_dim: int = 512,
                 activation: str = 'gelu'):
        """
        Initialize multimodal fusion block.
        
        Args:
            embed_dim: Embedding dimension for each modality
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout rate
            feedforward_dim: Dimension of feedforward layers
            activation: Activation function
        """
        super(MultimodalFusionBlock, self).__init__()
        
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Positional embeddings for different modalities
        self.modality_embeddings = nn.Parameter(torch.randn(3, embed_dim))  # gene, peak, protein
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(embed_dim, n_heads, feedforward_dim, dropout, activation)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        nn.init.normal_(self.modality_embeddings, std=0.02)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, modality_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for multimodal fusion.
        
        Args:
            modality_embeddings: Dictionary with keys 'gene', 'peak', 'protein'
                                Each tensor has shape (batch_size, embed_dim)
        
        Returns:
            Fused embedding tensor of shape (batch_size, embed_dim)
        """
        batch_size = list(modality_embeddings.values())[0].size(0)
        
        # Stack modality embeddings and add positional embeddings
        modality_list = ['gene', 'peak', 'protein']
        embeddings = []
        
        for i, modality in enumerate(modality_list):
            if modality in modality_embeddings:
                # Add modality-specific positional embedding
                emb = modality_embeddings[modality] + self.modality_embeddings[i]
                embeddings.append(emb)
        
        if not embeddings:
            raise ValueError("No modality embeddings provided")
        
        # Stack to create sequence: (batch_size, n_modalities, embed_dim)
        x = torch.stack(embeddings, dim=1)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Aggregate across modalities (mean pooling)
        fused_embedding = torch.mean(x, dim=1)  # (batch_size, embed_dim)
        
        # Apply output projection and layer norm
        output = self.layer_norm(self.output_projection(fused_embedding))
        
        return output

class TransformerLayer(nn.Module):
    """Single transformer layer for multimodal fusion."""
    
    def __init__(self, embed_dim: int, n_heads: int, feedforward_dim: int,
                 dropout: float = 0.1, activation: str = 'gelu'):
        super(TransformerLayer, self).__init__()
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.GELU() if activation == 'gelu' else nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Output tensor of same shape
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feedforward with residual connection
        ff_output = self.feedforward(x)
        x = self.norm2(x + ff_output)
        
        return x

class AdaptiveFusionBlock(nn.Module):
    """
    Adaptive fusion block that learns modality-specific attention weights.
    """
    
    def __init__(self, embed_dim: int, n_modalities: int = 3, dropout: float = 0.1):
        """
        Initialize adaptive fusion block.
        
        Args:
            embed_dim: Embedding dimension
            n_modalities: Number of modalities
            dropout: Dropout rate
        """
        super(AdaptiveFusionBlock, self).__init__()
        
        self.embed_dim = embed_dim
        self.n_modalities = n_modalities
        
        # Attention mechanism for modality weighting
        self.attention_weights = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Projection layers for each modality
        self.modality_projections = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(n_modalities)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, modality_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with adaptive weighting.
        
        Args:
            modality_embeddings: Dictionary of modality embeddings
            
        Returns:
            Fused embedding tensor
        """
        modality_list = ['gene', 'peak', 'protein']
        projected_embeddings = []
        attention_weights = []
        
        for i, modality in enumerate(modality_list):
            if modality in modality_embeddings:
                # Project modality embedding
                projected = self.modality_projections[i](modality_embeddings[modality])
                projected_embeddings.append(projected)
                
                # Compute attention weight for this modality
                weight = self.attention_weights(projected)
                attention_weights.append(weight)
        
        if not projected_embeddings:
            raise ValueError("No modality embeddings provided")
        
        # Stack embeddings and weights
        embeddings = torch.stack(projected_embeddings, dim=1)  # (batch_size, n_modalities, embed_dim)
        weights = torch.stack(attention_weights, dim=1)  # (batch_size, n_modalities, 1)
        
        # Normalize weights
        weights = F.softmax(weights, dim=1)
        
        # Weighted fusion
        fused_embedding = torch.sum(embeddings * weights, dim=1)  # (batch_size, embed_dim)
        
        # Apply output projection
        output = self.output_projection(self.dropout(fused_embedding))
        
        return output
