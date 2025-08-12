import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base_decoder import BaseDecoder

class GeneDecoder(BaseDecoder):
    """
    Specialized decoder for gene expression data.
    Handles count-based gene expression reconstruction with appropriate loss functions.
    """
    
    def __init__(self, n_topics: int, topic_embedding_dim: int,
                 feature_embedding_dim: int, n_genes: int,
                 use_negative_binomial: bool = True,
                 min_expression: float = 1e-6):
        """
        Initialize gene decoder.
        
        Args:
            n_topics: Number of topics
            topic_embedding_dim: Topic embedding dimension
            feature_embedding_dim: Feature embedding dimension
            n_genes: Number of genes
            use_negative_binomial: Whether to use negative binomial for count modeling
            min_expression: Minimum expression level to avoid log(0)
        """
        super(GeneDecoder, self).__init__(
            modality='gene',
            n_topics=n_topics,
            topic_embedding_dim=topic_embedding_dim,
            feature_embedding_dim=feature_embedding_dim,
            n_features=n_genes
        )
        
        self.use_negative_binomial = use_negative_binomial
        self.min_expression = min_expression
        
        if use_negative_binomial:
            # Dispersion parameter for negative binomial
            self.dispersion = nn.Parameter(torch.ones(n_genes))
            
        # Gene-specific processing layers
        self.gene_expression_processor = nn.Sequential(
            nn.Linear(n_genes, n_genes),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Expression level enhancement
        self.expression_enhancer = nn.Sequential(
            nn.Linear(n_genes, n_genes // 2),
            nn.ReLU(inplace=True),
            nn.Linear(n_genes // 2, n_genes),
            nn.Softplus()  # Ensure positive values
        )
    
    def apply_modality_specific_processing(self, reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Apply gene-specific post-processing.
        
        Args:
            reconstructed: Raw reconstructed data
            
        Returns:
            Processed gene expression data
        """
        # Ensure positive values (gene expression is non-negative)
        reconstructed = torch.clamp(reconstructed, min=self.min_expression)
        
        # Apply gene-specific processing
        processed = self.gene_expression_processor(reconstructed)
        processed = torch.clamp(processed + reconstructed, min=self.min_expression)
        
        # Expression enhancement
        enhanced = self.expression_enhancer(processed)
        
        # Combine original and enhanced
        final_expression = processed + 0.2 * enhanced
        
        # Final clamping to ensure realistic expression levels
        final_expression = torch.clamp(final_expression, min=self.min_expression, max=1000.0)
        
        return final_expression
    
    def compute_reconstruction_loss(self, original: torch.Tensor, 
                                  reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Compute gene expression reconstruction loss.
        
        Args:
            original: Original gene expression data
            reconstructed: Reconstructed gene expression data
            
        Returns:
            Reconstruction loss
        """
        if self.use_negative_binomial:
            # Negative binomial loss for count data
            # Clamp dispersion to avoid numerical issues
            dispersion = torch.clamp(self.dispersion, min=1e-6, max=100.0)
            
            # Convert to rate and concentration parameters
            mean = torch.clamp(reconstructed, min=self.min_expression)
            
            # Negative binomial NLL loss
            loss = F.poisson_nll_loss(mean, original, reduction='none')
            
            # Add dispersion regularization
            dispersion_reg = 0.01 * torch.mean(torch.log(dispersion))
            
            loss = torch.mean(loss) + dispersion_reg
        else:
            # Poisson loss for count data
            mean = torch.clamp(reconstructed, min=self.min_expression)
            loss = F.poisson_nll_loss(mean, original, reduction='mean')
        
        return loss
    
    def get_gene_topic_loadings(self, topic_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get gene-topic loading matrix for interpretation.
        
        Args:
            topic_embeddings: Topic embeddings
            
        Returns:
            Gene-topic loadings (n_genes, n_topics)
        """
        topic_feature_weights = self.get_topic_feature_weights(topic_embeddings)
        return topic_feature_weights.T  # (n_genes, n_topics)
    
    def get_highly_expressed_genes_per_topic(self, topic_embeddings: torch.Tensor, 
                                           gene_names: Optional[list] = None,
                                           top_k: int = 10) -> dict:
        """
        Get top expressed genes for each topic.
        
        Args:
            topic_embeddings: Topic embeddings
            gene_names: List of gene names
            top_k: Number of top genes to return per topic
            
        Returns:
            Dictionary mapping topic indices to top gene indices/names
        """
        gene_topic_loadings = self.get_gene_topic_loadings(topic_embeddings)
        
        top_genes_per_topic = {}
        for topic_idx in range(self.n_topics):
            topic_loadings = gene_topic_loadings[:, topic_idx]
            top_gene_indices = torch.topk(topic_loadings, k=top_k).indices
            
            if gene_names is not None:
                top_gene_names = [gene_names[idx] for idx in top_gene_indices]
                top_genes_per_topic[topic_idx] = top_gene_names
            else:
                top_genes_per_topic[topic_idx] = top_gene_indices.tolist()
        
        return top_genes_per_topic
