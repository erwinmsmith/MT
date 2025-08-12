import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

class BaseFusion(nn.Module, ABC):
    """
    Base class for multimodal fusion mechanisms.
    Provides common functionality for all fusion implementations.
    """
    
    def __init__(self, modalities: List[str], embed_dim: int, 
                 dropout: float = 0.1, use_layer_norm: bool = True):
        """
        Initialize base fusion module.
        
        Args:
            modalities: List of modality names
            embed_dim: Embedding dimension for each modality
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
        """
        super(BaseFusion, self).__init__()
        
        self.modalities = modalities
        self.embed_dim = embed_dim
        self.n_modalities = len(modalities)
        self.dropout_rate = dropout
        self.use_layer_norm = use_layer_norm
        
        # Modality-specific preprocessing
        self.modality_preprocessors = nn.ModuleDict({
            modality: self._build_modality_preprocessor(modality)
            for modality in modalities
        })
        
        # Modality embeddings (learnable positional embeddings)
        self.modality_embeddings = nn.ParameterDict({
            modality: nn.Parameter(torch.randn(embed_dim))
            for modality in modalities
        })
        
        # Common components
        self.dropout = nn.Dropout(dropout)
        
        if use_layer_norm:
            self.input_layer_norms = nn.ModuleDict({
                modality: nn.LayerNorm(embed_dim)
                for modality in modalities
            })
            self.output_layer_norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_modality_preprocessor(self, modality: str) -> nn.Module:
        """Build modality-specific preprocessor."""
        if modality == 'gene':
            return nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate * 0.5)
            )
        elif modality == 'peak':
            return nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate * 0.5)
            )
        elif modality == 'protein':
            return nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate * 0.5)
            )
        else:
            return nn.Identity()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for modality in self.modalities:
            nn.init.normal_(self.modality_embeddings[modality], std=0.02)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def preprocess_modalities(self, modality_embeddings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Preprocess modality embeddings.
        
        Args:
            modality_embeddings: Dictionary of modality embeddings
            
        Returns:
            Preprocessed modality embeddings
        """
        processed = {}
        
        for modality, embedding in modality_embeddings.items():
            if modality in self.modalities:
                # Apply modality-specific preprocessing
                processed_embedding = self.modality_preprocessors[modality](embedding)
                
                # Add modality positional embedding
                processed_embedding = processed_embedding + self.modality_embeddings[modality]
                
                # Apply layer normalization
                if self.use_layer_norm:
                    processed_embedding = self.input_layer_norms[modality](processed_embedding)
                
                # Apply dropout
                processed_embedding = self.dropout(processed_embedding)
                
                processed[modality] = processed_embedding
        
        return processed
    
    @abstractmethod
    def fuse_modalities(self, modality_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Abstract method for fusing modality embeddings.
        Must be implemented by subclasses.
        """
        pass
    
    def forward(self, modality_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through fusion module.
        
        Args:
            modality_embeddings: Dictionary of modality embeddings
            
        Returns:
            Fused embedding
        """
        # Preprocess modalities
        processed_embeddings = self.preprocess_modalities(modality_embeddings)
        
        # Fuse modalities (implemented by subclasses)
        fused_embedding = self.fuse_modalities(processed_embeddings)
        
        # Apply output layer normalization
        if self.use_layer_norm:
            fused_embedding = self.output_layer_norm(fused_embedding)
        
        return fused_embedding
    
    def get_modality_weights(self) -> Optional[torch.Tensor]:
        """Get modality attention weights if available (for interpretability)."""
        return None
    
    def get_fusion_statistics(self, modality_embeddings: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Get fusion statistics for analysis."""
        stats = {
            'n_modalities': len(modality_embeddings),
            'modalities': list(modality_embeddings.keys()),
            'embedding_shapes': {k: v.shape for k, v in modality_embeddings.items()}
        }
        
        # Compute pairwise similarities
        modality_list = list(modality_embeddings.keys())
        similarities = {}
        for i, mod1 in enumerate(modality_list):
            for j, mod2 in enumerate(modality_list[i+1:], i+1):
                emb1 = F.normalize(modality_embeddings[mod1], p=2, dim=-1)
                emb2 = F.normalize(modality_embeddings[mod2], p=2, dim=-1)
                sim = F.cosine_similarity(emb1, emb2, dim=-1).mean()
                similarities[f'{mod1}_{mod2}'] = float(sim)
        
        stats['pairwise_similarities'] = similarities
        
        return stats
