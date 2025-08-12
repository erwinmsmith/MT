import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from .base_fusion import BaseFusion

class AttentionFusion(BaseFusion):
    """
    Attention-based fusion mechanism for multimodal data.
    Uses self-attention and cross-attention for modality fusion.
    """
    
    def __init__(self, modalities: List[str], embed_dim: int, 
                 n_heads: int = 8, n_layers: int = 3,
                 dropout: float = 0.1, use_cross_attention: bool = True):
        """
        Initialize attention fusion module.
        
        Args:
            modalities: List of modality names
            embed_dim: Embedding dimension
            n_heads: Number of attention heads
            n_layers: Number of attention layers
            dropout: Dropout rate
            use_cross_attention: Whether to use cross-attention between modalities
        """
        super(AttentionFusion, self).__init__(modalities, embed_dim, dropout)
        
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.use_cross_attention = use_cross_attention
        
        # Self-attention layers
        self.self_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim, n_heads, dropout=dropout, batch_first=True
            ) for _ in range(n_layers)
        ])
        
        # Cross-attention layers (if enabled)
        if use_cross_attention:
            self.cross_attention_layers = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim, n_heads, dropout=dropout, batch_first=True
                ) for _ in range(n_layers)
            ])
        
        # Layer normalizations
        self.attention_layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(n_layers)
        ])
        
        if use_cross_attention:
            self.cross_attention_layer_norms = nn.ModuleList([
                nn.LayerNorm(embed_dim) for _ in range(n_layers)
            ])
        
        # Feedforward networks
        self.feedforward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout)
            ) for _ in range(n_layers)
        ])
        
        self.ff_layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(n_layers)
        ])
        
        # Final aggregation
        self.final_aggregation = nn.Sequential(
            nn.Linear(embed_dim * len(modalities), embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Attention weights storage for interpretability
        self.attention_weights = []
    
    def fuse_modalities(self, modality_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse modalities using attention mechanism.
        
        Args:
            modality_embeddings: Dictionary of preprocessed modality embeddings
            
        Returns:
            Fused embedding
        """
        # Clear previous attention weights
        self.attention_weights.clear()
        
        # Stack modality embeddings
        modality_list = []
        for modality in self.modalities:
            if modality in modality_embeddings:
                modality_list.append(modality_embeddings[modality])
        
        if not modality_list:
            raise ValueError("No valid modality embeddings provided")
        
        # Stack to create sequence: (batch_size, n_modalities, embed_dim)
        x = torch.stack(modality_list, dim=1)
        batch_size = x.shape[0]
        
        # Apply attention layers
        for i in range(self.n_layers):
            # Self-attention
            residual = x
            attn_output, attn_weights = self.self_attention_layers[i](x, x, x)
            x = self.attention_layer_norms[i](x + attn_output)
            
            # Store attention weights for interpretability
            self.attention_weights.append(attn_weights.detach())
            
            # Cross-attention (if enabled)
            if self.use_cross_attention:
                cross_residual = x
                cross_attn_output, _ = self.cross_attention_layers[i](x, x, x)
                x = self.cross_attention_layer_norms[i](x + cross_attn_output)
            
            # Feedforward
            ff_residual = x
            ff_output = self.feedforward_layers[i](x)
            x = self.ff_layer_norms[i](x + ff_output)
        
        # Final aggregation
        # Option 1: Mean pooling across modalities
        pooled = torch.mean(x, dim=1)  # (batch_size, embed_dim)
        
        # Option 2: Concatenate and project (alternative)
        # concatenated = x.view(batch_size, -1)  # (batch_size, n_modalities * embed_dim)
        # final_output = self.final_aggregation(concatenated)
        
        return pooled
    
    def get_modality_weights(self) -> Optional[torch.Tensor]:
        """Get modality attention weights for interpretability."""
        if self.attention_weights:
            # Return the last layer's attention weights
            return self.attention_weights[-1]
        return None
    
    def get_attention_patterns(self) -> List[torch.Tensor]:
        """Get attention patterns from all layers."""
        return self.attention_weights.copy()

class AdaptiveAttentionFusion(BaseFusion):
    """
    Adaptive attention fusion that learns modality-specific attention patterns.
    """
    
    def __init__(self, modalities: List[str], embed_dim: int,
                 n_heads: int = 8, dropout: float = 0.1,
                 temperature: float = 1.0):
        """
        Initialize adaptive attention fusion.
        
        Args:
            modalities: List of modality names
            embed_dim: Embedding dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
            temperature: Temperature for attention softmax
        """
        super(AdaptiveAttentionFusion, self).__init__(modalities, embed_dim, dropout)
        
        self.n_heads = n_heads
        self.temperature = temperature
        
        # Adaptive attention mechanism
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        
        # Modality-specific attention weights
        self.modality_attention = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim // 4, 1),
                nn.Sigmoid()
            ) for modality in modalities
        })
        
        # Temperature parameter (learnable)
        self.adaptive_temperature = nn.Parameter(torch.tensor(temperature))
        
        # Final projection
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        
        # Store attention weights
        self.computed_attention_weights = {}
    
    def fuse_modalities(self, modality_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse modalities using adaptive attention.
        
        Args:
            modality_embeddings: Dictionary of preprocessed modality embeddings
            
        Returns:
            Fused embedding
        """
        # Clear previous weights
        self.computed_attention_weights.clear()
        
        modality_list = []
        attention_weights = []
        
        for modality in self.modalities:
            if modality in modality_embeddings:
                embedding = modality_embeddings[modality]
                modality_list.append(embedding)
                
                # Compute modality-specific attention weight
                attention_weight = self.modality_attention[modality](embedding)
                attention_weights.append(attention_weight)
                
                # Store for interpretability
                self.computed_attention_weights[modality] = attention_weight.detach()
        
        if not modality_list:
            raise ValueError("No valid modality embeddings provided")
        
        # Stack embeddings and weights
        stacked_embeddings = torch.stack(modality_list, dim=1)  # (batch_size, n_modalities, embed_dim)
        stacked_weights = torch.stack(attention_weights, dim=1)  # (batch_size, n_modalities, 1)
        
        # Apply temperature scaling
        temp = torch.clamp(self.adaptive_temperature, min=0.1, max=10.0)
        scaled_weights = stacked_weights / temp
        
        # Normalize attention weights
        normalized_weights = F.softmax(scaled_weights, dim=1)
        
        # Apply attention
        attended_embeddings = stacked_embeddings * normalized_weights
        
        # Sum across modalities
        fused_embedding = torch.sum(attended_embeddings, dim=1)  # (batch_size, embed_dim)
        
        # Final projection
        output = self.output_projection(fused_embedding)
        
        return output
    
    def get_modality_weights(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get computed modality attention weights."""
        return self.computed_attention_weights if self.computed_attention_weights else None
