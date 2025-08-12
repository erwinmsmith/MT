import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import networkx as nx
from .encoder import (
    GeneFeatureTopicsEncoder,
    PeakFeatureTopicsEncoder,
    ProteinFeatureTopicsEncoder
)
from .gnn import HeterogeneousDirectedGNN
from .gnn.heterogeneous_directed_gnn import HeterogeneousGraphProcessor

class FeatureGraphPathway(nn.Module):
    """
    Feature graph pathway for multi-omics topic modeling.
    
    This pathway:
    1. Constructs heterogeneous directed graph from features and prior knowledge
    2. Applies heterogeneous directed GNN to get feature embeddings
    3. Generates topic distributions for each modality (B_d)
    """
    
    def __init__(self, config: Dict, prior_knowledge: Dict):
        """
        Initialize feature graph pathway.
        
        Args:
            config: Configuration dictionary
            prior_knowledge: Prior knowledge including embeddings and graphs
        """
        super(FeatureGraphPathway, self).__init__()
        
        self.config = config
        self.prior_knowledge = prior_knowledge
        
        # Extract configuration
        data_config = config.get('data', {})
        model_config = config.get('model', {})
        featuregraph_config = model_config.get('featuregraph', {})
        
        # Dimensions
        self.n_genes = data_config.get('n_genes', 3000)
        self.n_peaks = data_config.get('n_peaks', 5000)
        self.n_proteins = data_config.get('n_proteins', 1000)
        self.n_topics = data_config.get('n_topics', 20)
        self.embedding_dim = data_config.get('embedding_dim', 256)
        
        # Node and edge types
        self.node_types = ['gene', 'peak', 'protein']
        self.edge_types = ['gene2peak', 'gene2protein', 'peak2protein', 'protein2protein']
        
        # Input dimensions for each node type (from prior knowledge embeddings)
        self.input_dims = {
            'gene': 256,  # From foundation model embeddings
            'peak': 256,  # Computed from gene embeddings through mapping
            'protein': 256  # From foundation model embeddings
        }
        
        # Graph processor
        self.graph_processor = HeterogeneousGraphProcessor(
            node_types=self.node_types,
            edge_types=self.edge_types
        )
        
        # Heterogeneous directed GNN
        gnn_config = featuregraph_config.get('heterogeneous_gnn', {})
        self.heterogeneous_gnn = HeterogeneousDirectedGNN(
            node_types=self.node_types,
            edge_types=self.edge_types,
            input_dims=self.input_dims,
            hidden_dim=gnn_config.get('hidden_dim', 256),
            output_dim=gnn_config.get('output_dim', self.embedding_dim),
            n_layers=gnn_config.get('n_layers', 3),
            n_heads=gnn_config.get('n_heads', 4),
            dropout=gnn_config.get('dropout', 0.1),
            activation=gnn_config.get('activation', 'relu'),
            use_residual=gnn_config.get('use_residual', True)
        )
        
        # Feature encoders for generating topic distributions
        feature_encoders_config = featuregraph_config.get('feature_encoders', {})
        
        # Gene feature encoder (specialized for gene characteristics)
        gene_encoder_config = feature_encoders_config.get('gene', {})
        self.gene_feature_encoder = GeneFeatureTopicsEncoder(
            input_dim=gene_encoder_config.get('input_dim', self.embedding_dim),
            hidden_dims=gene_encoder_config.get('hidden_dims', [256, 128]),
            n_topics=gene_encoder_config.get('n_topics', self.n_topics),
            dropout=gene_encoder_config.get('dropout', 0.1),
            prior_mean=gene_encoder_config.get('prior_mean', 0.0),
            prior_std=gene_encoder_config.get('prior_std', 1.0),
            use_modality_specific_priors=gene_encoder_config.get('use_modality_specific_priors', True),
            gene_specific_features=gene_encoder_config.get('gene_specific_features', True)
        )
        
        # Peak feature encoder (specialized for chromatin accessibility)
        peak_encoder_config = feature_encoders_config.get('peak', {})
        self.peak_feature_encoder = PeakFeatureTopicsEncoder(
            input_dim=peak_encoder_config.get('input_dim', self.embedding_dim),
            hidden_dims=peak_encoder_config.get('hidden_dims', [256, 128]),
            n_topics=peak_encoder_config.get('n_topics', self.n_topics),
            dropout=peak_encoder_config.get('dropout', 0.1),
            prior_mean=peak_encoder_config.get('prior_mean', 0.0),
            prior_std=peak_encoder_config.get('prior_std', 1.0),
            use_modality_specific_priors=peak_encoder_config.get('use_modality_specific_priors', True),
            peak_specific_features=peak_encoder_config.get('peak_specific_features', True)
        )
        
        # Protein feature encoder (specialized for protein expression)
        protein_encoder_config = feature_encoders_config.get('protein', {})
        self.protein_feature_encoder = ProteinFeatureTopicsEncoder(
            input_dim=protein_encoder_config.get('input_dim', self.embedding_dim),
            hidden_dims=protein_encoder_config.get('hidden_dims', [256, 128]),
            n_topics=protein_encoder_config.get('n_topics', self.n_topics),
            dropout=protein_encoder_config.get('dropout', 0.1),
            prior_mean=protein_encoder_config.get('prior_mean', 0.0),
            prior_std=protein_encoder_config.get('prior_std', 1.0),
            use_modality_specific_priors=protein_encoder_config.get('use_modality_specific_priors', True),
            protein_specific_features=protein_encoder_config.get('protein_specific_features', True)
        )
        
        # Initialize feature embeddings from prior knowledge
        self._initialize_feature_embeddings()
        
        # Store intermediate results
        self.intermediate_results = {}
    
    def _initialize_feature_embeddings(self):
        """Initialize feature embeddings from prior knowledge."""
        device = next(self.parameters()).device
        
        # Gene embeddings (from foundation models like GenePT)
        if 'gene_embeddings' in self.prior_knowledge:
            gene_embeddings = torch.tensor(
                self.prior_knowledge['gene_embeddings'], 
                dtype=torch.float32, device=device
            )
        else:
            gene_embeddings = torch.randn(self.n_genes, 256, device=device)
        
        # Protein embeddings (from foundation models)
        if 'protein_embeddings' in self.prior_knowledge:
            protein_embeddings = torch.tensor(
                self.prior_knowledge['protein_embeddings'],
                dtype=torch.float32, device=device
            )
        else:
            protein_embeddings = torch.randn(self.n_proteins, 256, device=device)
        
        # Peak embeddings (derived from gene embeddings through mapping)
        peak_embeddings = self._create_peak_embeddings_from_genes(gene_embeddings)
        
        # Register as buffers (not trainable parameters)
        self.register_buffer('gene_embeddings', gene_embeddings)
        self.register_buffer('protein_embeddings', protein_embeddings)
        self.register_buffer('peak_embeddings', peak_embeddings)
    
    def _create_peak_embeddings_from_genes(self, gene_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Create peak embeddings from gene embeddings using gene2peak mapping.
        
        Args:
            gene_embeddings: Gene embeddings tensor
            
        Returns:
            Peak embeddings tensor
        """
        device = gene_embeddings.device
        peak_embeddings = torch.zeros(self.n_peaks, 256, device=device)
        
        # Use gene2peak mapping from prior knowledge
        if 'gene2peak_mapping' in self.prior_knowledge:
            gene2peak_mapping = self.prior_knowledge['gene2peak_mapping']
            
            # Aggregate gene embeddings to create peak embeddings
            peak_weights = torch.zeros(self.n_peaks, device=device)
            
            for gene_idx, peak_idx, weight in gene2peak_mapping:
                if gene_idx < self.n_genes and peak_idx < self.n_peaks:
                    peak_embeddings[peak_idx] += weight * gene_embeddings[gene_idx]
                    peak_weights[peak_idx] += weight
            
            # Normalize by accumulated weights
            mask = peak_weights > 0
            peak_embeddings[mask] = peak_embeddings[mask] / peak_weights[mask].unsqueeze(1)
            
            # For peaks without any gene mapping, use random embeddings
            unmapped_mask = peak_weights == 0
            if unmapped_mask.sum() > 0:
                peak_embeddings[unmapped_mask] = torch.randn(
                    unmapped_mask.sum(), 256, device=device
                )
        else:
            # If no mapping available, use random embeddings
            peak_embeddings = torch.randn(self.n_peaks, 256, device=device)
        
        return peak_embeddings
    
    def get_node_features(self) -> Dict[str, torch.Tensor]:
        """
        Get node features for the heterogeneous graph.
        
        Returns:
            Dictionary mapping node types to feature tensors
        """
        return {
            'gene': self.gene_embeddings,
            'peak': self.peak_embeddings,
            'protein': self.protein_embeddings
        }
    
    def construct_heterogeneous_graph(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Construct heterogeneous graph from prior knowledge.
        
        Returns:
            Tuple of (edge_indices, edge_weights)
        """
        device = next(self.parameters()).device
        
        if 'heterogeneous_graph' in self.prior_knowledge:
            # Use provided heterogeneous graph
            graph = self.prior_knowledge['heterogeneous_graph']
            node_features = self.get_node_features()
            
            edge_indices, edge_weights = self.graph_processor.process_networkx_graph(
                graph, node_features, device
            )
        else:
            # Create graph from individual mappings
            edge_indices, edge_weights = self._create_graph_from_mappings(device)
        
        # Store intermediate results
        self.intermediate_results['edge_indices'] = edge_indices
        self.intermediate_results['edge_weights'] = edge_weights
        
        return edge_indices, edge_weights
    
    def _create_graph_from_mappings(self, device: torch.device) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Create graph from individual mapping files.
        
        Args:
            device: Target device
            
        Returns:
            Tuple of (edge_indices, edge_weights)
        """
        edge_indices = {}
        edge_weights = {}
        
        # Gene2Peak edges
        if 'gene2peak_mapping' in self.prior_knowledge:
            gene2peak = self.prior_knowledge['gene2peak_mapping']
            source_idx = [gene_idx for gene_idx, _, _ in gene2peak if gene_idx < self.n_genes]
            target_idx = [peak_idx for _, peak_idx, _ in gene2peak if peak_idx < self.n_peaks]
            weights = [weight for gene_idx, peak_idx, weight in gene2peak 
                      if gene_idx < self.n_genes and peak_idx < self.n_peaks]
            
            if source_idx:
                edge_indices['gene2peak'] = torch.tensor([source_idx, target_idx], 
                                                       dtype=torch.long, device=device)
                edge_weights['gene2peak'] = torch.tensor(weights, dtype=torch.float, device=device)
        
        # Gene2Protein edges
        if 'gene2protein_mapping' in self.prior_knowledge:
            gene2protein = self.prior_knowledge['gene2protein_mapping']
            source_idx = [gene_idx for gene_idx, _, _ in gene2protein if gene_idx < self.n_genes]
            target_idx = [protein_idx for _, protein_idx, _ in gene2protein if protein_idx < self.n_proteins]
            weights = [weight for gene_idx, protein_idx, weight in gene2protein 
                      if gene_idx < self.n_genes and protein_idx < self.n_proteins]
            
            if source_idx:
                edge_indices['gene2protein'] = torch.tensor([source_idx, target_idx], 
                                                          dtype=torch.long, device=device)
                edge_weights['gene2protein'] = torch.tensor(weights, dtype=torch.float, device=device)
        
        # Peak2Protein edges
        if 'peak2protein_mapping' in self.prior_knowledge:
            peak2protein = self.prior_knowledge['peak2protein_mapping']
            source_idx = [peak_idx for peak_idx, _, _ in peak2protein if peak_idx < self.n_peaks]
            target_idx = [protein_idx for _, protein_idx, _ in peak2protein if protein_idx < self.n_proteins]
            weights = [weight for peak_idx, protein_idx, weight in peak2protein 
                      if peak_idx < self.n_peaks and protein_idx < self.n_proteins]
            
            if source_idx:
                edge_indices['peak2protein'] = torch.tensor([source_idx, target_idx], 
                                                          dtype=torch.long, device=device)
                edge_weights['peak2protein'] = torch.tensor(weights, dtype=torch.float, device=device)
        
        # Protein2Protein edges (self-interactions)
        # Create some random protein-protein interactions for demonstration
        n_ppi = min(100, self.n_proteins // 3)
        source_idx = torch.randint(0, self.n_proteins, (n_ppi,), device=device)
        target_idx = torch.randint(0, self.n_proteins, (n_ppi,), device=device)
        # Remove self-loops
        mask = source_idx != target_idx
        source_idx = source_idx[mask]
        target_idx = target_idx[mask]
        
        if len(source_idx) > 0:
            edge_indices['protein2protein'] = torch.stack([source_idx, target_idx])
            edge_weights['protein2protein'] = torch.rand(len(source_idx), device=device) * 0.5 + 0.2
        
        return edge_indices, edge_weights
    
    def apply_heterogeneous_gnn(self, node_features: Dict[str, torch.Tensor],
                               edge_indices: Dict[str, torch.Tensor],
                               edge_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply heterogeneous directed GNN to update feature embeddings.
        
        Args:
            node_features: Dictionary of node features
            edge_indices: Dictionary of edge indices
            edge_weights: Dictionary of edge weights
            
        Returns:
            Updated feature embeddings
        """
        updated_features = self.heterogeneous_gnn(node_features, edge_indices, edge_weights)
        
        # Store intermediate results
        self.intermediate_results['gnn_features'] = updated_features
        
        return updated_features
    
    def generate_modality_topic_distributions(self, 
                                             gnn_features: Dict[str, torch.Tensor]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Generate topic distributions for each modality.
        
        Args:
            gnn_features: GNN-updated feature embeddings
            
        Returns:
            Dictionary mapping modalities to (topic_distribution, mean, logvar)
        """
        results = {}
        
        # Gene modality
        if 'gene' in gnn_features:
            gene_topics, gene_mean, gene_logvar = self.gene_feature_encoder.get_topic_distribution(gnn_features['gene'])
            results['gene'] = (gene_topics, gene_mean, gene_logvar)
        
        # Peak modality
        if 'peak' in gnn_features:
            peak_topics, peak_mean, peak_logvar = self.peak_feature_encoder.get_topic_distribution(gnn_features['peak'])
            results['peak'] = (peak_topics, peak_mean, peak_logvar)
        
        # Protein modality
        if 'protein' in gnn_features:
            protein_topics, protein_mean, protein_logvar = self.protein_feature_encoder.get_topic_distribution(gnn_features['protein'])
            results['protein'] = (protein_topics, protein_mean, protein_logvar)
        
        # Store intermediate results
        self.intermediate_results['modality_topic_distributions'] = results
        
        return results
    
    def forward(self) -> Dict[str, any]:
        """
        Forward pass through feature graph pathway.
        
        Returns:
            Dictionary containing pathway outputs
        """
        # Step 1: Get node features from prior knowledge
        node_features = self.get_node_features()
        
        # Step 2: Construct heterogeneous graph
        edge_indices, edge_weights = self.construct_heterogeneous_graph()
        
        # Step 3: Apply heterogeneous directed GNN
        gnn_features = self.apply_heterogeneous_gnn(node_features, edge_indices, edge_weights)
        
        # Step 4: Generate topic distributions for each modality
        modality_topic_distributions = self.generate_modality_topic_distributions(gnn_features)
        
        return {
            'node_features': node_features,
            'edge_indices': edge_indices,
            'edge_weights': edge_weights,
            'gnn_features': gnn_features,
            'modality_topic_distributions': modality_topic_distributions
        }
    
    def get_intermediate_results(self) -> Dict[str, any]:
        """Get intermediate results for analysis."""
        return self.intermediate_results.copy()
    
    def clear_intermediate_results(self):
        """Clear stored intermediate results."""
        self.intermediate_results.clear()
