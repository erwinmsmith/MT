import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base_decoder import BaseDecoder

class PeakDecoder(BaseDecoder):
    """
    Specialized decoder for chromatin accessibility (peak) data.
    Handles binary/sparse accessibility patterns with appropriate loss functions.
    """
    
    def __init__(self, n_topics: int, topic_embedding_dim: int,
                 feature_embedding_dim: int, n_peaks: int,
                 accessibility_threshold: float = 0.5,
                 sparsity_weight: float = 0.1):
        """
        Initialize peak decoder.
        
        Args:
            n_topics: Number of topics
            topic_embedding_dim: Topic embedding dimension
            feature_embedding_dim: Feature embedding dimension
            n_peaks: Number of peaks
            accessibility_threshold: Threshold for peak accessibility
            sparsity_weight: Weight for sparsity regularization
        """
        super(PeakDecoder, self).__init__(
            modality='peak',
            n_topics=n_topics,
            topic_embedding_dim=topic_embedding_dim,
            feature_embedding_dim=feature_embedding_dim,
            n_features=n_peaks
        )
        
        self.accessibility_threshold = nn.Parameter(torch.tensor(accessibility_threshold))
        self.sparsity_weight = sparsity_weight
        
        # Peak-specific processing layers
        self.accessibility_processor = nn.Sequential(
            nn.Linear(n_peaks, n_peaks),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Binary enhancement layers
        self.binary_enhancer = nn.Sequential(
            nn.Linear(n_peaks, n_peaks // 2),
            nn.ReLU(inplace=True),
            nn.Linear(n_peaks // 2, n_peaks),
            nn.Sigmoid()  # Binary-like output
        )
        
        # Chromatin state modeling
        self.chromatin_state_processor = nn.Sequential(
            nn.Linear(n_peaks, n_peaks),
            nn.LayerNorm(n_peaks),
            nn.ReLU(inplace=True)
        )
    
    def apply_modality_specific_processing(self, reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Apply peak-specific post-processing.
        
        Args:
            reconstructed: Raw reconstructed data
            
        Returns:
            Processed peak accessibility data
        """
        # Apply accessibility processing
        processed = self.accessibility_processor(reconstructed)
        
        # Chromatin state processing
        chromatin_processed = self.chromatin_state_processor(reconstructed)
        
        # Binary enhancement
        binary_enhanced = self.binary_enhancer(processed + chromatin_processed)
        
        # Combine with original
        combined = reconstructed + 0.3 * processed + 0.2 * binary_enhanced
        
        # Apply accessibility threshold
        threshold = torch.clamp(self.accessibility_threshold, min=0.1, max=0.9)
        accessibility_mask = torch.sigmoid((combined - threshold) * 10)
        
        # Final accessibility pattern
        final_accessibility = combined * accessibility_mask
        
        # Apply sigmoid to get probabilities
        final_accessibility = torch.sigmoid(final_accessibility)
        
        return final_accessibility
    
    def compute_reconstruction_loss(self, original: torch.Tensor, 
                                  reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Compute peak accessibility reconstruction loss.
        
        Args:
            original: Original peak accessibility data
            reconstructed: Reconstructed peak accessibility data
            
        Returns:
            Reconstruction loss
        """
        # Binary cross-entropy loss for accessibility
        bce_loss = F.binary_cross_entropy(reconstructed, original, reduction='mean')
        
        # Sparsity regularization (peaks should be sparse)
        sparsity_loss = self.sparsity_weight * torch.mean(reconstructed)
        
        # Accessibility pattern consistency
        accessibility_consistency = torch.mean(
            torch.abs(reconstructed - torch.round(reconstructed))
        )
        
        total_loss = bce_loss + sparsity_loss + 0.1 * accessibility_consistency
        
        return total_loss
    
    def get_peak_topic_accessibility(self, topic_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get peak-topic accessibility matrix for interpretation.
        
        Args:
            topic_embeddings: Topic embeddings
            
        Returns:
            Peak-topic accessibility (n_peaks, n_topics)
        """
        topic_feature_weights = self.get_topic_feature_weights(topic_embeddings)
        accessibility_weights = torch.sigmoid(topic_feature_weights)
        return accessibility_weights.T  # (n_peaks, n_topics)
    
    def get_accessible_peaks_per_topic(self, topic_embeddings: torch.Tensor,
                                     peak_names: Optional[list] = None,
                                     top_k: int = 20,
                                     min_accessibility: float = 0.5) -> dict:
        """
        Get top accessible peaks for each topic.
        
        Args:
            topic_embeddings: Topic embeddings
            peak_names: List of peak names
            top_k: Number of top peaks to return per topic
            min_accessibility: Minimum accessibility threshold
            
        Returns:
            Dictionary mapping topic indices to accessible peak indices/names
        """
        peak_topic_accessibility = self.get_peak_topic_accessibility(topic_embeddings)
        
        accessible_peaks_per_topic = {}
        for topic_idx in range(self.n_topics):
            topic_accessibility = peak_topic_accessibility[:, topic_idx]
            
            # Filter by minimum accessibility
            accessible_mask = topic_accessibility > min_accessibility
            accessible_indices = torch.where(accessible_mask)[0]
            
            if len(accessible_indices) > 0:
                # Get top accessible peaks
                accessible_scores = topic_accessibility[accessible_indices]
                if len(accessible_indices) > top_k:
                    top_accessible = torch.topk(accessible_scores, k=top_k)
                    final_indices = accessible_indices[top_accessible.indices]
                else:
                    final_indices = accessible_indices
                
                if peak_names is not None:
                    accessible_peak_names = [peak_names[idx] for idx in final_indices]
                    accessible_peaks_per_topic[topic_idx] = accessible_peak_names
                else:
                    accessible_peaks_per_topic[topic_idx] = final_indices.tolist()
            else:
                accessible_peaks_per_topic[topic_idx] = []
        
        return accessible_peaks_per_topic
    
    def get_chromatin_state_summary(self, topic_embeddings: torch.Tensor) -> dict:
        """
        Get chromatin state summary for each topic.
        
        Args:
            topic_embeddings: Topic embeddings
            
        Returns:
            Dictionary with chromatin state statistics per topic
        """
        peak_accessibility = self.get_peak_topic_accessibility(topic_embeddings)
        
        chromatin_summary = {}
        for topic_idx in range(self.n_topics):
            accessibility = peak_accessibility[:, topic_idx]
            
            chromatin_summary[topic_idx] = {
                'mean_accessibility': float(torch.mean(accessibility)),
                'accessibility_std': float(torch.std(accessibility)),
                'n_accessible_peaks': int(torch.sum(accessibility > 0.5)),
                'sparsity': float(torch.mean(accessibility < 0.1)),
                'max_accessibility': float(torch.max(accessibility))
            }
        
        return chromatin_summary
