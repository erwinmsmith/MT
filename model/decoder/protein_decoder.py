import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base_decoder import BaseDecoder

class ProteinDecoder(BaseDecoder):
    """
    Specialized decoder for protein expression data.
    Handles protein abundance with functional relationship modeling.
    """
    
    def __init__(self, n_topics: int, topic_embedding_dim: int,
                 feature_embedding_dim: int, n_proteins: int,
                 use_log_normal: bool = True,
                 functional_groups: int = 3):
        """
        Initialize protein decoder.
        
        Args:
            n_topics: Number of topics
            topic_embedding_dim: Topic embedding dimension
            feature_embedding_dim: Feature embedding dimension
            n_proteins: Number of proteins
            use_log_normal: Whether to use log-normal modeling for protein abundance
            functional_groups: Number of functional protein groups
        """
        super(ProteinDecoder, self).__init__(
            modality='protein',
            n_topics=n_topics,
            topic_embedding_dim=topic_embedding_dim,
            feature_embedding_dim=feature_embedding_dim,
            n_features=n_proteins
        )
        
        self.use_log_normal = use_log_normal
        self.functional_groups = functional_groups
        
        if use_log_normal:
            # Log-normal parameters
            self.log_std = nn.Parameter(torch.ones(n_proteins))
        
        # Protein-specific processing layers
        self.protein_processor = nn.Sequential(
            nn.Linear(n_proteins, n_proteins),
            nn.LayerNorm(n_proteins),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Functional group modeling
        self.functional_group_weights = nn.Parameter(
            torch.randn(functional_groups, n_proteins)
        )
        
        # Protein complex formation modeling
        self.complex_formation = nn.Sequential(
            nn.Linear(n_proteins, n_proteins // 2),
            nn.GELU(),
            nn.Linear(n_proteins // 2, n_proteins),
            nn.Softplus()  # Positive abundance
        )
        
        # Expression level normalization
        self.expression_normalizer = nn.Sequential(
            nn.Linear(n_proteins, n_proteins),
            nn.LayerNorm(n_proteins),
            nn.Tanh()
        )
        
        # Localization-based processing
        self.localization_processor = nn.Sequential(
            nn.Linear(n_proteins, n_proteins),
            nn.LayerNorm(n_proteins),
            nn.ReLU(inplace=True)
        )
    
    def apply_modality_specific_processing(self, reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Apply protein-specific post-processing.
        
        Args:
            reconstructed: Raw reconstructed data
            
        Returns:
            Processed protein abundance data
        """
        # Ensure positive values (protein abundance is non-negative)
        reconstructed = torch.clamp(reconstructed, min=1e-6)
        
        # Apply protein-specific processing
        processed = self.protein_processor(reconstructed)
        
        # Functional group processing
        functional_effects = []
        for i in range(self.functional_groups):
            effect = torch.sum(reconstructed * self.functional_group_weights[i], 
                             dim=-1, keepdim=True)
            functional_effects.append(effect)
        
        functional_effect = torch.cat(functional_effects, dim=-1)
        functional_weights = F.softmax(functional_effect, dim=-1)
        
        weighted_reconstruction = torch.zeros_like(reconstructed)
        for i in range(self.functional_groups):
            weighted_reconstruction += (functional_weights[:, i:i+1] * 
                                      reconstructed * self.functional_group_weights[i])
        
        # Complex formation modeling
        complex_enhanced = self.complex_formation(processed + weighted_reconstruction)
        
        # Expression normalization
        normalized = self.expression_normalizer(reconstructed + processed)
        
        # Localization processing
        localization_processed = self.localization_processor(reconstructed)
        
        # Combine all components
        final_abundance = (reconstructed + 0.2 * processed + 0.3 * complex_enhanced + 
                          0.1 * normalized + 0.2 * localization_processed + 
                          0.2 * weighted_reconstruction)
        
        # Final clamping to ensure realistic protein levels
        final_abundance = torch.clamp(final_abundance, min=1e-6, max=100.0)
        
        return final_abundance
    
    def compute_reconstruction_loss(self, original: torch.Tensor, 
                                  reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Compute protein abundance reconstruction loss.
        
        Args:
            original: Original protein abundance data
            reconstructed: Reconstructed protein abundance data
            
        Returns:
            Reconstruction loss
        """
        if self.use_log_normal:
            # Log-normal loss for protein abundance
            log_std = torch.clamp(self.log_std, min=1e-6, max=10.0)
            
            # Log-normal likelihood
            log_original = torch.log(torch.clamp(original, min=1e-6))
            log_reconstructed = torch.log(torch.clamp(reconstructed, min=1e-6))
            
            # Gaussian loss in log space
            loss = F.gaussian_nll_loss(log_reconstructed, log_original, 
                                     log_std.pow(2), reduction='mean')
        else:
            # MSE loss for protein abundance
            loss = F.mse_loss(reconstructed, original, reduction='mean')
        
        # Functional coherence regularization
        functional_coherence_loss = 0.0
        for i in range(self.functional_groups):
            group_proteins = self.functional_group_weights[i]
            # Encourage proteins in the same group to have similar patterns
            group_coherence = torch.var(reconstructed * group_proteins, dim=0).mean()
            functional_coherence_loss += group_coherence
        
        functional_coherence_loss = 0.01 * functional_coherence_loss / self.functional_groups
        
        total_loss = loss + functional_coherence_loss
        
        return total_loss
    
    def get_protein_topic_abundance(self, topic_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get protein-topic abundance matrix for interpretation.
        
        Args:
            topic_embeddings: Topic embeddings
            
        Returns:
            Protein-topic abundance (n_proteins, n_topics)
        """
        topic_feature_weights = self.get_topic_feature_weights(topic_embeddings)
        abundance_weights = torch.clamp(topic_feature_weights, min=0)
        return abundance_weights.T  # (n_proteins, n_topics)
    
    def get_abundant_proteins_per_topic(self, topic_embeddings: torch.Tensor,
                                      protein_names: Optional[list] = None,
                                      top_k: int = 15) -> dict:
        """
        Get top abundant proteins for each topic.
        
        Args:
            topic_embeddings: Topic embeddings
            protein_names: List of protein names
            top_k: Number of top proteins to return per topic
            
        Returns:
            Dictionary mapping topic indices to abundant protein indices/names
        """
        protein_topic_abundance = self.get_protein_topic_abundance(topic_embeddings)
        
        abundant_proteins_per_topic = {}
        for topic_idx in range(self.n_topics):
            topic_abundance = protein_topic_abundance[:, topic_idx]
            top_protein_indices = torch.topk(topic_abundance, k=top_k).indices
            
            if protein_names is not None:
                top_protein_names = [protein_names[idx] for idx in top_protein_indices]
                abundant_proteins_per_topic[topic_idx] = top_protein_names
            else:
                abundant_proteins_per_topic[topic_idx] = top_protein_indices.tolist()
        
        return abundant_proteins_per_topic
    
    def get_functional_group_analysis(self, topic_embeddings: torch.Tensor) -> dict:
        """
        Get functional group analysis for each topic.
        
        Args:
            topic_embeddings: Topic embeddings
            
        Returns:
            Dictionary with functional group statistics per topic
        """
        protein_abundance = self.get_protein_topic_abundance(topic_embeddings)
        
        functional_analysis = {}
        for topic_idx in range(self.n_topics):
            abundance = protein_abundance[:, topic_idx]
            
            # Analyze each functional group
            group_activities = []
            for i in range(self.functional_groups):
                group_weights = self.functional_group_weights[i]
                group_activity = torch.sum(abundance * torch.abs(group_weights))
                group_activities.append(float(group_activity))
            
            functional_analysis[topic_idx] = {
                'group_activities': group_activities,
                'dominant_group': int(torch.argmax(torch.tensor(group_activities))),
                'mean_abundance': float(torch.mean(abundance)),
                'max_abundance': float(torch.max(abundance)),
                'n_active_proteins': int(torch.sum(abundance > 0.1))
            }
        
        return functional_analysis
