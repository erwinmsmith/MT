import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from .gene_decoder import GeneDecoder
from .peak_decoder import PeakDecoder
from .protein_decoder import ProteinDecoder

class MultiOmicsDecoder(nn.Module):
    """
    Comprehensive multi-omics decoder that orchestrates modality-specific decoders.
    
    This decoder:
    1. Uses specialized decoders for each modality
    2. Coordinates topic embeddings across modalities
    3. Integrates feature topic distributions from featuregraph pathway
    4. Provides comprehensive reconstruction and interpretation
    """
    
    def __init__(self, config: Dict):
        """
        Initialize multi-omics decoder.
        
        Args:
            config: Configuration dictionary
        """
        super(MultiOmicsDecoder, self).__init__()
        
        self.config = config
        data_config = config.get('data', {})
        decoder_config = config.get('model', {}).get('decoder', {})
        
        # Dimensions
        self.n_topics = data_config.get('n_topics', 20)
        self.n_genes = data_config.get('n_genes', 3000)
        self.n_peaks = data_config.get('n_peaks', 5000)
        self.n_proteins = data_config.get('n_proteins', 1000)
        
        self.topic_embedding_dim = decoder_config.get('topic_embedding_dim', 256)
        self.feature_embedding_dim = decoder_config.get('feature_embedding_dim', 256)
        
        # Shared topic embedding matrix
        self.topic_embeddings = nn.Parameter(
            torch.randn(self.n_topics, self.topic_embedding_dim)
        )
        
        # Modality-specific decoders
        self.gene_decoder = GeneDecoder(
            n_topics=self.n_topics,
            topic_embedding_dim=self.topic_embedding_dim,
            feature_embedding_dim=self.feature_embedding_dim,
            n_genes=self.n_genes,
            use_negative_binomial=decoder_config.get('use_negative_binomial', True)
        )
        
        self.peak_decoder = PeakDecoder(
            n_topics=self.n_topics,
            topic_embedding_dim=self.topic_embedding_dim,
            feature_embedding_dim=self.feature_embedding_dim,
            n_peaks=self.n_peaks,
            accessibility_threshold=decoder_config.get('accessibility_threshold', 0.5),
            sparsity_weight=decoder_config.get('sparsity_weight', 0.1)
        )
        
        self.protein_decoder = ProteinDecoder(
            n_topics=self.n_topics,
            topic_embedding_dim=self.topic_embedding_dim,
            feature_embedding_dim=self.feature_embedding_dim,
            n_proteins=self.n_proteins,
            use_log_normal=decoder_config.get('use_log_normal', True),
            functional_groups=decoder_config.get('functional_groups', 3)
        )
        
        # Cross-modality coordination
        self.modality_coordination = nn.ModuleDict({
            'gene_peak': nn.Linear(self.topic_embedding_dim, self.topic_embedding_dim),
            'gene_protein': nn.Linear(self.topic_embedding_dim, self.topic_embedding_dim),
            'peak_protein': nn.Linear(self.topic_embedding_dim, self.topic_embedding_dim)
        })
        
        # Initialize parameters
        self._initialize_parameters()
        
        # Store intermediate results
        self.intermediate_results = {}
    
    def _initialize_parameters(self):
        """Initialize decoder parameters."""
        nn.init.xavier_uniform_(self.topic_embeddings)
        
        for module in self.modality_coordination.values():
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def coordinate_topic_embeddings(self) -> Dict[str, torch.Tensor]:
        """
        Create modality-specific topic embeddings through cross-modality coordination.
        
        Returns:
            Dictionary of modality-specific topic embeddings
        """
        base_embeddings = self.topic_embeddings
        
        # Create modality-specific embeddings
        gene_embeddings = base_embeddings + 0.1 * self.modality_coordination['gene_peak'](base_embeddings)
        peak_embeddings = base_embeddings + 0.1 * self.modality_coordination['gene_peak'](base_embeddings)
        protein_embeddings = base_embeddings + 0.1 * self.modality_coordination['gene_protein'](base_embeddings)
        
        # Cross-modality influences
        gene_embeddings = gene_embeddings + 0.05 * self.modality_coordination['gene_protein'](base_embeddings)
        protein_embeddings = protein_embeddings + 0.05 * self.modality_coordination['peak_protein'](base_embeddings)
        peak_embeddings = peak_embeddings + 0.05 * self.modality_coordination['peak_protein'](base_embeddings)
        
        return {
            'gene': gene_embeddings,
            'peak': peak_embeddings,
            'protein': protein_embeddings
        }
    
    def forward(self, cell_topic_distribution: torch.Tensor,
                feature_topic_distributions: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-omics decoder.
        
        Args:
            cell_topic_distribution: Cell topic distribution from cellgraph (batch_size, n_topics)
            feature_topic_distributions: Feature topic distributions from featuregraph
            
        Returns:
            Dictionary of reconstructed data for each modality
        """
        # Get modality-specific topic embeddings
        modality_embeddings = self.coordinate_topic_embeddings()
        
        # Reconstruct each modality
        reconstructed_data = {}
        
        # Gene reconstruction
        if 'gene' in feature_topic_distributions:
            gene_feature_topics, _, _ = feature_topic_distributions['gene']
            reconstructed_data['gene'] = self.gene_decoder(
                cell_topic_distribution,
                modality_embeddings['gene'],
                gene_feature_topics
            )
        
        # Peak reconstruction  
        if 'peak' in feature_topic_distributions:
            peak_feature_topics, _, _ = feature_topic_distributions['peak']
            reconstructed_data['peak'] = self.peak_decoder(
                cell_topic_distribution,
                modality_embeddings['peak'],
                peak_feature_topics
            )
        
        # Protein reconstruction
        if 'protein' in feature_topic_distributions:
            protein_feature_topics, _, _ = feature_topic_distributions['protein']
            reconstructed_data['protein'] = self.protein_decoder(
                cell_topic_distribution,
                modality_embeddings['protein'],
                protein_feature_topics
            )
        
        # Store intermediate results
        self.intermediate_results = {
            'modality_embeddings': modality_embeddings,
            'reconstructed_data': reconstructed_data
        }
        
        return reconstructed_data
    
    def compute_reconstruction_losses(self, original_data: Dict[str, torch.Tensor],
                                    reconstructed_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute modality-specific reconstruction losses.
        
        Args:
            original_data: Original data dictionary
            reconstructed_data: Reconstructed data dictionary
            
        Returns:
            Dictionary of reconstruction losses
        """
        losses = {}
        
        if 'gene' in reconstructed_data and 'gene' in original_data:
            losses['gene'] = self.gene_decoder.compute_reconstruction_loss(
                original_data['gene'], reconstructed_data['gene']
            )
        
        if 'peak' in reconstructed_data and 'peak' in original_data:
            losses['peak'] = self.peak_decoder.compute_reconstruction_loss(
                original_data['peak'], reconstructed_data['peak']
            )
        
        if 'protein' in reconstructed_data and 'protein' in original_data:
            losses['protein'] = self.protein_decoder.compute_reconstruction_loss(
                original_data['protein'], reconstructed_data['protein']
            )
        
        return losses
    
    def get_comprehensive_interpretation(self, feature_names: Optional[Dict[str, list]] = None) -> Dict[str, any]:
        """
        Get comprehensive interpretation of the learned model.
        
        Args:
            feature_names: Dictionary mapping modalities to feature names
            
        Returns:
            Comprehensive interpretation dictionary
        """
        modality_embeddings = self.coordinate_topic_embeddings()
        interpretation = {}
        
        # Gene interpretation
        if hasattr(self, 'gene_decoder'):
            gene_names = feature_names.get('gene') if feature_names else None
            interpretation['gene'] = {
                'topic_loadings': self.gene_decoder.get_gene_topic_loadings(modality_embeddings['gene']),
                'top_genes_per_topic': self.gene_decoder.get_highly_expressed_genes_per_topic(
                    modality_embeddings['gene'], gene_names
                )
            }
        
        # Peak interpretation
        if hasattr(self, 'peak_decoder'):
            peak_names = feature_names.get('peak') if feature_names else None
            interpretation['peak'] = {
                'topic_accessibility': self.peak_decoder.get_peak_topic_accessibility(modality_embeddings['peak']),
                'accessible_peaks_per_topic': self.peak_decoder.get_accessible_peaks_per_topic(
                    modality_embeddings['peak'], peak_names
                ),
                'chromatin_state_summary': self.peak_decoder.get_chromatin_state_summary(modality_embeddings['peak'])
            }
        
        # Protein interpretation
        if hasattr(self, 'protein_decoder'):
            protein_names = feature_names.get('protein') if feature_names else None
            interpretation['protein'] = {
                'topic_abundance': self.protein_decoder.get_protein_topic_abundance(modality_embeddings['protein']),
                'abundant_proteins_per_topic': self.protein_decoder.get_abundant_proteins_per_topic(
                    modality_embeddings['protein'], protein_names
                ),
                'functional_group_analysis': self.protein_decoder.get_functional_group_analysis(modality_embeddings['protein'])
            }
        
        # Cross-modality analysis
        interpretation['cross_modality'] = self._analyze_cross_modality_relationships(modality_embeddings)
        
        return interpretation
    
    def _analyze_cross_modality_relationships(self, modality_embeddings: Dict[str, torch.Tensor]) -> Dict[str, any]:
        """Analyze relationships between modalities."""
        cross_modality = {}
        
        # Topic embedding similarities
        for mod1 in modality_embeddings:
            for mod2 in modality_embeddings:
                if mod1 < mod2:  # Avoid duplicate pairs
                    emb1 = modality_embeddings[mod1]
                    emb2 = modality_embeddings[mod2]
                    
                    # Cosine similarity between topic embeddings
                    similarity = F.cosine_similarity(
                        emb1.unsqueeze(1), emb2.unsqueeze(0), dim=2
                    )
                    
                    cross_modality[f'{mod1}_{mod2}_similarity'] = similarity.detach()
        
        return cross_modality
    
    def get_topic_embeddings(self) -> torch.Tensor:
        """Get shared topic embeddings."""
        return self.topic_embeddings
    
    def get_modality_specific_embeddings(self) -> Dict[str, torch.Tensor]:
        """Get modality-specific topic embeddings."""
        return self.coordinate_topic_embeddings()
    
    def get_embedding_to_feature_matrices(self) -> Dict[str, torch.Tensor]:
        """Get embedding-to-feature matrices for each modality."""
        return {
            'gene': self.gene_decoder.embedding_to_feature,
            'peak': self.peak_decoder.embedding_to_feature,
            'protein': self.protein_decoder.embedding_to_feature
        }
    
    def get_intermediate_results(self) -> Dict[str, any]:
        """Get intermediate results for analysis."""
        return self.intermediate_results.copy()
    
    def clear_intermediate_results(self):
        """Clear stored intermediate results."""
        self.intermediate_results.clear()
