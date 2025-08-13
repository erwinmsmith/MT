import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from abc import ABC, abstractmethod

class BaseDecoder(nn.Module, ABC):
    """
    Base decoder class for modality-specific decoders.
    Provides common functionality for all decoders.
    """
    
    def __init__(self, modality: str, n_topics: int, topic_embedding_dim: int,
                 feature_embedding_dim: int, n_features: int):
        """
        Initialize base decoder.
        
        Args:
            modality: Modality name ('gene', 'peak', 'protein')
            n_topics: Number of topics
            topic_embedding_dim: Topic embedding dimension
            feature_embedding_dim: Feature embedding dimension
            n_features: Number of features for this modality
        """
        super(BaseDecoder, self).__init__()
        
        self.modality = modality
        self.n_topics = n_topics
        self.topic_embedding_dim = topic_embedding_dim
        self.feature_embedding_dim = feature_embedding_dim
        self.n_features = n_features
        
        # Transformation from topic embedding to feature embedding
        self.topic_to_feature_embedding = nn.Linear(
            topic_embedding_dim, feature_embedding_dim
        )
        
        # Additional transformation layer
        self.topic_to_feature_transform = nn.Linear(
            n_topics, n_features
        )
        
        # Embedding-to-feature matrix
        self.embedding_to_feature = nn.Parameter(
            torch.randn(feature_embedding_dim, n_features)
        )
        
        # Modality-specific scaling and bias
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(n_features))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize decoder parameters."""
        nn.init.xavier_uniform_(self.topic_to_feature_embedding.weight)
        nn.init.constant_(self.topic_to_feature_embedding.bias, 0)
        nn.init.xavier_uniform_(self.embedding_to_feature)
    
    @abstractmethod
    def apply_modality_specific_processing(self, reconstructed: torch.Tensor) -> torch.Tensor:
        """Apply modality-specific post-processing to reconstructed data."""
        pass
    
    @abstractmethod
    def compute_reconstruction_loss(self, original: torch.Tensor, 
                                  reconstructed: torch.Tensor) -> torch.Tensor:
        """Compute modality-specific reconstruction loss."""
        pass
    
    def forward(self, cell_topic_distribution: torch.Tensor,
                topic_embeddings: torch.Tensor,
                feature_topic_distribution: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            cell_topic_distribution: Cell topic distribution (batch_size, n_topics)
            topic_embeddings: Topic embeddings (n_topics, topic_embedding_dim)
            feature_topic_distribution: Feature topic distribution (n_features, n_topics)
            
        Returns:
            Reconstructed data (batch_size, n_features)
        """
        # Transform topic embeddings to feature embedding space
        feature_embeddings = self.topic_to_feature_embedding(topic_embeddings)
        feature_embeddings = self.dropout(feature_embeddings)
        
        # Compute topic-feature matrix
        if feature_topic_distribution is not None:
            # Use feature topic distribution to constrain the mapping
            # Check dimensions for proper matrix multiplication
            ft_dist_T = feature_topic_distribution.T  # (n_topics, n_features)
            
            # Ensure compatible dimensions
            if ft_dist_T.size(-1) == feature_embeddings.size(0):
                # Standard case: multiply (n_topics, n_features) x (n_features, feature_embedding_dim)
                weighted_embedding_to_feature = torch.matmul(
                    ft_dist_T,  # (n_topics, n_features)
                    feature_embeddings  # (n_features, feature_embedding_dim)
                )  # Result: (n_topics, feature_embedding_dim)
            elif ft_dist_T.size(0) == feature_embeddings.size(0):
                # Alternative case: multiply (n_topics, n_features) x (n_topics, feature_embedding_dim)
                weighted_embedding_to_feature = torch.matmul(
                    ft_dist_T.T,  # (n_features, n_topics)
                    feature_embeddings  # (n_topics, feature_embedding_dim)
                )  # Result: (n_features, feature_embedding_dim)
            else:
                # Fallback: use just the feature embeddings
                weighted_embedding_to_feature = feature_embeddings
            
            topic_feature_matrix = torch.matmul(
                weighted_embedding_to_feature.T,  # (feature_embedding_dim, n_features)
                self.embedding_to_feature.T  # (n_features, feature_embedding_dim)
            ).T  # Result: (n_topics, n_features)
        else:
            # Standard matrix multiplication
            topic_feature_matrix = torch.matmul(
                feature_embeddings, self.embedding_to_feature
            )  # (n_topics, n_features)
        
        # Reconstruct data: (batch_size, n_topics) × (n_topics, n_features)
        # Ensure proper matrix multiplication dimensions
        if cell_topic_distribution.size(-1) == topic_feature_matrix.size(0):
            reconstructed = torch.matmul(cell_topic_distribution, topic_feature_matrix)
        elif cell_topic_distribution.size(-1) == topic_feature_matrix.size(-1):
            # Transpose topic_feature_matrix if needed
            reconstructed = torch.matmul(cell_topic_distribution, topic_feature_matrix.T)
        else:
            # Fallback: use a learnable transformation
            if not hasattr(self, 'topic_to_feature_transform'):
                self.topic_to_feature_transform = nn.Linear(
                    cell_topic_distribution.size(-1), 
                    self.n_features
                ).to(cell_topic_distribution.device)
            
            reconstructed = self.topic_to_feature_transform(cell_topic_distribution)
        
        # Apply scaling and bias
        reconstructed = reconstructed * torch.clamp(self.scale, min=0.1) + self.bias
        
        # Apply modality-specific processing
        reconstructed = self.apply_modality_specific_processing(reconstructed)
        
        return reconstructed
    
    def get_topic_feature_weights(self, topic_embeddings: torch.Tensor) -> torch.Tensor:
        """Get topic-feature weights for interpretation."""
        feature_embeddings = self.topic_to_feature_embedding(topic_embeddings)
        topic_feature_matrix = torch.matmul(feature_embeddings, self.embedding_to_feature)
        return topic_feature_matrix
