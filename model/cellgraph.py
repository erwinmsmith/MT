import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from .encoder import (
    GeneEncoder,
    PeakEncoder,
    ProteinEncoder,
    FusedEmbeddingEncoder,
    CellTopicsEncoder
)
from .fusion import MultimodalFusionBlock
from .gnn import DirectedGNN
from .gnn import TrajectoryInference

class CellGraphPathway(nn.Module):
    """
    Cell graph pathway for multi-omics topic modeling.
    
    This pathway:
    1. Encodes each modality separately
    2. Fuses modalities using multimodal fusion block
    3. Performs trajectory inference to create cell-cell adjacency matrix
    4. Applies directed GNN on the cell graph
    5. Generates topic distributions for cells (θ_d)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize cell graph pathway.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super(CellGraphPathway, self).__init__()
        
        self.config = config
        model_config = config.get('model', {})
        
        # Extract dimensions
        self.n_topics = config.get('data', {}).get('n_topics', 20)
        self.embedding_dim = config.get('data', {}).get('embedding_dim', 256)
        
        # Initialize modality encoders
        self.gene_encoder = GeneEncoder(
            **model_config.get('gene_encoder', {
                'input_dim': 3000,
                'hidden_dims': [1024, 512, 256],
                'output_dim': 256,
                'dropout': 0.1
            })
        )
        
        self.atac_encoder = PeakEncoder(
            **model_config.get('atac_encoder', {
                'input_dim': 5000,
                'hidden_dims': [1024, 512, 256],
                'output_dim': 256,
                'dropout': 0.1
            })
        )
        
        self.protein_encoder = ProteinEncoder(
            **model_config.get('protein_encoder', {
                'input_dim': 1000,
                'hidden_dims': [512, 256],
                'output_dim': 256,
                'dropout': 0.1
            })
        )
        
        # Multimodal fusion block
        self.fusion_block = MultimodalFusionBlock(
            **model_config.get('fusion_block', {
                'embed_dim': 256,
                'n_heads': 8,
                'n_layers': 3,
                'dropout': 0.1,
                'feedforward_dim': 512
            })
        )
        
        # Trajectory inference for creating cell-cell adjacency matrix
        trajectory_config = config.get('trajectory', {})
        self.trajectory_inference = TrajectoryInference(
            embed_dim=self.embedding_dim,
            n_components=trajectory_config.get('n_components', 50),
            kernel_type=trajectory_config.get('kernel_type', 'gaussian'),
            sigma=trajectory_config.get('sigma', 1.0)
        )
        
        # Directed GNN for processing cell graph
        cellgraph_config = model_config.get('cellgraph', {})
        directed_gnn_config = cellgraph_config.get('directed_gnn', {})
        self.directed_gnn = DirectedGNN(
            input_dim=self.embedding_dim,
            hidden_dim=directed_gnn_config.get('hidden_dim', 256),
            output_dim=self.embedding_dim,
            n_layers=directed_gnn_config.get('n_layers', 3),
            dropout=directed_gnn_config.get('dropout', 0.1),
            activation=directed_gnn_config.get('activation', 'relu'),
            use_residual=directed_gnn_config.get('use_residual', True)
        )
        
        # Cell encoder for generating topic distribution parameters
        cell_encoder_config = cellgraph_config.get('cell_topics_encoder', {})
        self.cell_encoder = CellTopicsEncoder(
            input_dim=cell_encoder_config.get('input_dim', self.embedding_dim),
            hidden_dims=cell_encoder_config.get('hidden_dims', [256, 128]),
            n_topics=cell_encoder_config.get('n_topics', self.n_topics),
            dropout=cell_encoder_config.get('dropout', 0.1),
            prior_mean=cell_encoder_config.get('prior_mean', 0.0),
            prior_std=cell_encoder_config.get('prior_std', 1.0),
            use_batch_norm=cell_encoder_config.get('use_batch_norm', True)
        )
        
        # Store intermediate results for analysis
        self.intermediate_results = {}
    
    def encode_modalities(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encode each modality separately.
        
        Args:
            batch: Dictionary containing modality data
            
        Returns:
            Dictionary of encoded modality embeddings
        """
        modality_embeddings = {}
        
        if 'gene' in batch:
            modality_embeddings['gene'] = self.gene_encoder(batch['gene'])
        
        if 'peak' in batch:
            modality_embeddings['peak'] = self.atac_encoder(batch['peak'])
        
        if 'protein' in batch:
            modality_embeddings['protein'] = self.protein_encoder(batch['protein'])
        
        # Store intermediate results
        self.intermediate_results['modality_embeddings'] = modality_embeddings
        
        return modality_embeddings
    
    def fuse_modalities(self, modality_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse modality embeddings using multimodal fusion block.
        
        Args:
            modality_embeddings: Dictionary of modality embeddings
            
        Returns:
            Fused cell embeddings
        """
        fused_embeddings = self.fusion_block(modality_embeddings)
        
        # Store intermediate results
        self.intermediate_results['fused_embeddings'] = fused_embeddings
        
        return fused_embeddings
    
    def infer_cell_graph(self, cell_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Infer cell-to-cell directed adjacency matrix.
        
        Args:
            cell_embeddings: Fused cell embeddings (batch_size, embed_dim)
            
        Returns:
            Directed adjacency matrix (batch_size, batch_size)
        """
        # Add batch dimension for trajectory inference if needed
        if len(cell_embeddings.shape) == 2:
            cell_embeddings = cell_embeddings.unsqueeze(0)  # (1, n_cells, embed_dim)
        
        adjacency_matrix = self.trajectory_inference.infer_trajectory(cell_embeddings)
        
        # Remove batch dimension if it was added
        if adjacency_matrix.shape[0] == 1:
            adjacency_matrix = adjacency_matrix.squeeze(0)  # (n_cells, n_cells)
        
        # Store intermediate results
        self.intermediate_results['adjacency_matrix'] = adjacency_matrix
        
        return adjacency_matrix
    
    def apply_directed_gnn(self, cell_embeddings: torch.Tensor, 
                          adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        Apply directed GNN to update cell embeddings.
        
        Args:
            cell_embeddings: Cell embeddings (batch_size, embed_dim)
            adjacency_matrix: Directed adjacency matrix (batch_size, batch_size)
            
        Returns:
            Updated cell embeddings (batch_size, embed_dim)
        """
        # Prepare inputs for directed GNN
        if len(cell_embeddings.shape) == 2:
            cell_embeddings = cell_embeddings.unsqueeze(0)  # (1, n_cells, embed_dim)
        if len(adjacency_matrix.shape) == 2:
            adjacency_matrix = adjacency_matrix.unsqueeze(0)  # (1, n_cells, n_cells)
        
        # Apply directed GNN
        updated_embeddings = self.directed_gnn(cell_embeddings, adjacency_matrix)
        
        # Remove batch dimension if it was added
        if updated_embeddings.shape[0] == 1:
            updated_embeddings = updated_embeddings.squeeze(0)  # (n_cells, embed_dim)
        
        # Store intermediate results
        self.intermediate_results['gnn_embeddings'] = updated_embeddings
        
        return updated_embeddings
    
    def generate_topic_distribution(self, cell_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate topic distribution for cells.
        
        Args:
            cell_embeddings: Cell embeddings (batch_size, embed_dim)
            
        Returns:
            Tuple of (topic_distribution, mean, logvar)
        """
        # Get topic distribution and variational parameters
        topic_distribution, mean, logvar = self.cell_encoder.get_topic_distribution(cell_embeddings)
        
        # Store intermediate results
        self.intermediate_results['topic_mean'] = mean
        self.intermediate_results['topic_logvar'] = logvar
        self.intermediate_results['topic_distribution'] = topic_distribution
        
        return topic_distribution, mean, logvar
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through cell graph pathway.
        
        Args:
            batch: Dictionary containing modality data
            
        Returns:
            Dictionary containing pathway outputs
        """
        # Step 1: Encode each modality
        modality_embeddings = self.encode_modalities(batch)
        
        # Step 2: Fuse modalities
        fused_embeddings = self.fuse_modalities(modality_embeddings)
        
        # Step 3: Infer cell-to-cell adjacency matrix
        adjacency_matrix = self.infer_cell_graph(fused_embeddings)
        
        # Step 4: Apply directed GNN
        gnn_embeddings = self.apply_directed_gnn(fused_embeddings, adjacency_matrix)
        
        # Step 5: Generate topic distribution
        topic_distribution, topic_mean, topic_logvar = self.generate_topic_distribution(gnn_embeddings)
        
        return {
            'modality_embeddings': modality_embeddings,
            'fused_embeddings': fused_embeddings,
            'adjacency_matrix': adjacency_matrix,
            'gnn_embeddings': gnn_embeddings,
            'topic_distribution': topic_distribution,
            'topic_mean': topic_mean,
            'topic_logvar': topic_logvar
        }
    
    def get_intermediate_results(self) -> Dict[str, torch.Tensor]:
        """Get intermediate results for analysis."""
        return self.intermediate_results.copy()
    
    def clear_intermediate_results(self):
        """Clear stored intermediate results."""
        self.intermediate_results.clear()
