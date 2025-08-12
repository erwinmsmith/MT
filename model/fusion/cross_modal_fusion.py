import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from .base_fusion import BaseFusion

class CrossModalFusion(BaseFusion):
    """
    Cross-modal fusion mechanism that explicitly models interactions between modalities.
    Uses bilinear pooling and cross-modal attention for comprehensive fusion.
    """
    
    def __init__(self, modalities: List[str], embed_dim: int,
                 fusion_dim: int = 512, dropout: float = 0.1,
                 use_bilinear_pooling: bool = True):
        """
        Initialize cross-modal fusion module.
        
        Args:
            modalities: List of modality names
            embed_dim: Embedding dimension
            fusion_dim: Intermediate fusion dimension
            dropout: Dropout rate
            use_bilinear_pooling: Whether to use bilinear pooling
        """
        super(CrossModalFusion, self).__init__(modalities, embed_dim, dropout)
        
        self.fusion_dim = fusion_dim
        self.use_bilinear_pooling = use_bilinear_pooling
        
        # Cross-modal interaction matrices
        self.cross_modal_interactions = nn.ModuleDict()
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities[i+1:], i+1):
                interaction_name = f"{mod1}_{mod2}"
                self.cross_modal_interactions[interaction_name] = CrossModalInteraction(
                    embed_dim, fusion_dim, dropout, use_bilinear_pooling
                )
        
        # Self-interaction modules (for individual modality processing)
        self.self_interactions = nn.ModuleDict({
            modality: SelfInteraction(embed_dim, fusion_dim, dropout)
            for modality in modalities
        })
        
        # Fusion aggregation
        n_interactions = len(self.cross_modal_interactions) + len(modalities)
        self.fusion_aggregator = nn.Sequential(
            nn.Linear(fusion_dim * n_interactions, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Gating mechanism for controlling interaction strength
        self.interaction_gates = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 1),
                nn.Sigmoid()
            ) for name in self.cross_modal_interactions.keys()
        })
        
        # Store interaction results for analysis
        self.interaction_results = {}
    
    def fuse_modalities(self, modality_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse modalities using cross-modal interactions.
        
        Args:
            modality_embeddings: Dictionary of preprocessed modality embeddings
            
        Returns:
            Fused embedding
        """
        # Clear previous results
        self.interaction_results.clear()
        
        fusion_components = []
        
        # Self-interactions (individual modality processing)
        for modality in self.modalities:
            if modality in modality_embeddings:
                self_interaction = self.self_interactions[modality](
                    modality_embeddings[modality]
                )
                fusion_components.append(self_interaction)
                self.interaction_results[f"{modality}_self"] = self_interaction.detach()
        
        # Cross-modal interactions
        modality_list = [mod for mod in self.modalities if mod in modality_embeddings]
        
        for i, mod1 in enumerate(modality_list):
            for j, mod2 in enumerate(modality_list[i+1:], i+1):
                interaction_name = f"{mod1}_{mod2}"
                
                if interaction_name in self.cross_modal_interactions:
                    emb1 = modality_embeddings[mod1]
                    emb2 = modality_embeddings[mod2]
                    
                    # Compute cross-modal interaction
                    cross_interaction = self.cross_modal_interactions[interaction_name](emb1, emb2)
                    
                    # Apply gating
                    gate_input = torch.cat([emb1, emb2], dim=-1)
                    gate_weight = self.interaction_gates[interaction_name](gate_input)
                    gated_interaction = cross_interaction * gate_weight
                    
                    fusion_components.append(gated_interaction)
                    self.interaction_results[interaction_name] = gated_interaction.detach()
        
        if not fusion_components:
            raise ValueError("No valid interactions computed")
        
        # Concatenate and aggregate all interaction results
        concatenated_interactions = torch.cat(fusion_components, dim=-1)
        fused_embedding = self.fusion_aggregator(concatenated_interactions)
        
        return fused_embedding
    
    def get_interaction_analysis(self) -> Dict[str, torch.Tensor]:
        """Get analysis of cross-modal interactions."""
        return self.interaction_results.copy()

class CrossModalInteraction(nn.Module):
    """
    Individual cross-modal interaction module between two modalities.
    """
    
    def __init__(self, embed_dim: int, fusion_dim: int, dropout: float = 0.1,
                 use_bilinear_pooling: bool = True):
        """
        Initialize cross-modal interaction.
        
        Args:
            embed_dim: Input embedding dimension
            fusion_dim: Output fusion dimension
            dropout: Dropout rate
            use_bilinear_pooling: Whether to use bilinear pooling
        """
        super(CrossModalInteraction, self).__init__()
        
        self.embed_dim = embed_dim
        self.fusion_dim = fusion_dim
        self.use_bilinear_pooling = use_bilinear_pooling
        
        if use_bilinear_pooling:
            # Bilinear pooling matrix
            self.bilinear_matrix = nn.Parameter(torch.randn(embed_dim, embed_dim))
        
        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        
        # Fusion networks
        self.fusion_network = nn.Sequential(
            nn.Linear(embed_dim * 3, fusion_dim),  # 3 = bilinear + 2 * original
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # Hadamard product enhancement
        self.hadamard_processor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
    
    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-modal interaction between two embeddings.
        
        Args:
            emb1: First modality embedding (batch_size, embed_dim)
            emb2: Second modality embedding (batch_size, embed_dim)
            
        Returns:
            Cross-modal interaction result (batch_size, fusion_dim)
        """
        batch_size = emb1.shape[0]
        
        # Bilinear pooling
        if self.use_bilinear_pooling:
            # emb1^T * W * emb2
            bilinear_result = torch.matmul(
                torch.matmul(emb1.unsqueeze(1), self.bilinear_matrix),
                emb2.unsqueeze(-1)
            ).squeeze()  # (batch_size, embed_dim)
        else:
            bilinear_result = emb1 * emb2  # Element-wise product
        
        # Cross-attention (emb1 attends to emb2)
        emb1_seq = emb1.unsqueeze(1)  # (batch_size, 1, embed_dim)
        emb2_seq = emb2.unsqueeze(1)  # (batch_size, 1, embed_dim)
        
        attended_emb1, _ = self.cross_attention(emb1_seq, emb2_seq, emb2_seq)
        attended_emb1 = attended_emb1.squeeze(1)  # (batch_size, embed_dim)
        
        # Hadamard product with processing
        hadamard_product = self.hadamard_processor(emb1 * emb2)
        
        # Combine all interaction types
        combined_features = torch.cat([
            bilinear_result,
            attended_emb1,
            hadamard_product
        ], dim=-1)
        
        # Final fusion
        interaction_result = self.fusion_network(combined_features)
        
        return interaction_result

class SelfInteraction(nn.Module):
    """
    Self-interaction module for individual modality processing.
    """
    
    def __init__(self, embed_dim: int, fusion_dim: int, dropout: float = 0.1):
        """
        Initialize self-interaction module.
        
        Args:
            embed_dim: Input embedding dimension
            fusion_dim: Output fusion dimension
            dropout: Dropout rate
        """
        super(SelfInteraction, self).__init__()
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        
        self.self_enhancement = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
    
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute self-interaction for a single modality.
        
        Args:
            embedding: Modality embedding (batch_size, embed_dim)
            
        Returns:
            Self-interaction result (batch_size, fusion_dim)
        """
        # Self-attention
        emb_seq = embedding.unsqueeze(1)  # (batch_size, 1, embed_dim)
        attended_emb, _ = self.self_attention(emb_seq, emb_seq, emb_seq)
        attended_emb = attended_emb.squeeze(1)  # (batch_size, embed_dim)
        
        # Self-enhancement
        enhanced_emb = self.self_enhancement(attended_emb + embedding)
        
        return enhanced_emb
