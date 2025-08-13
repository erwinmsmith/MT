import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data.graph_structure_manager import GraphStructureManager
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
        trajectory_kwargs = {
            'embed_dim': self.embedding_dim,
            'n_components': trajectory_config.get('n_components', 50),
            'kernel_type': trajectory_config.get('kernel_type', 'gaussian'),
            'k_neighbors': trajectory_config.get('k_neighbors', 30),
            'method': trajectory_config.get('method', 'diffusion')
        }
        
        # Only add root_cell_idx if the method is diffusion
        if trajectory_config.get('method', 'diffusion') == 'diffusion':
            trajectory_kwargs['root_cell_idx'] = trajectory_config.get('root_cell_idx', 0)
        
        
        self.trajectory_inference = TrajectoryInference(**trajectory_kwargs)
        
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
        
        # Graph structure manager for persistent storage
        self.graph_manager = GraphStructureManager(data_dir="data/dataset")
        
        # Cache for fixed adjacency matrix (once computed, it stays fixed)
        self._fixed_adjacency_matrix = None
        self._adjacency_computed = False
        self._cached_batch_size = None
        
        # Flag to disable graph computation during pretraining
        self._disable_graph_computation = False
        self._config_for_graph = config  # Store config for graph structure identification
    
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
    
    def infer_cell_graph(self, cell_embeddings: torch.Tensor, batch_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract cell-to-cell adjacency matrix for current batch from the global graph.
        The global graph should be precomputed using all cells.
        
        Args:
            cell_embeddings: Fused cell embeddings (batch_size, embed_dim) - used for device info
            batch_indices: Indices of cells in current batch relative to full dataset
            
        Returns:
            Directed adjacency matrix subset for current batch (batch_size, batch_size)
        """
        batch_size = cell_embeddings.shape[0]
        device = cell_embeddings.device
        
        # Priority 1: Use full graph structure if available
        n_total_cells = self._config_for_graph.get('data', {}).get('n_cells', 2000)
        if (self._adjacency_computed and 
            self._fixed_adjacency_matrix is not None and
            self._fixed_adjacency_matrix.shape[0] == n_total_cells):
            
            # Ensure fixed adjacency matrix is on correct device
            if self._fixed_adjacency_matrix.device != device:
                self._fixed_adjacency_matrix = self._fixed_adjacency_matrix.to(device)
            
            if batch_indices is not None:
                # Extract subgraph using actual batch indices
                batch_adjacency = self._fixed_adjacency_matrix[batch_indices][:, batch_indices]
            else:
                # Fallback: use first batch_size cells (for compatibility)
                batch_adjacency = self._fixed_adjacency_matrix[:batch_size, :batch_size]
            
            # Ensure batch adjacency is on correct device
            batch_adjacency = batch_adjacency.to(device)
            
            # Store intermediate results
            self.intermediate_results['adjacency_matrix'] = batch_adjacency
            
            # Only print this message once per training session
            if not hasattr(self, '_precomputed_graph_message_shown'):
                print(f"Using precomputed global graph (will extract {batch_adjacency.shape} subgraphs for batches)")
                self._precomputed_graph_message_shown = True
            
            return batch_adjacency
        
        # Priority 2: Try to load full graph structure from disk (only once)
        if not hasattr(self, '_disk_check_done'):
            self._disk_check_done = True
            graph_data = self.graph_manager.load_graph_structure(self._config_for_graph)
            
            if graph_data is not None:
                loaded_adjacency_matrix, metadata = graph_data
                loaded_adjacency_matrix = loaded_adjacency_matrix.to(device)
                
                print(f"Graph structure loaded from disk (shape: {loaded_adjacency_matrix.shape})")
                print(f"Source: {metadata.get('source', 'Unknown')}")
                
                loaded_size = loaded_adjacency_matrix.shape[0]
                
                if loaded_size == n_total_cells:
                    # This is a full graph - cache it and return subset
                    self._fixed_adjacency_matrix = loaded_adjacency_matrix
                    self._adjacency_computed = True
                    
                    if batch_indices is not None:
                        batch_adjacency = loaded_adjacency_matrix[batch_indices][:, batch_indices]
                    else:
                        batch_adjacency = loaded_adjacency_matrix[:batch_size, :batch_size]
                    
                    # Ensure batch adjacency is on correct device
                    batch_adjacency = batch_adjacency.to(device)
                    
                    self.intermediate_results['adjacency_matrix'] = batch_adjacency
                    
                    # Only print this message once per training session
                    if not hasattr(self, '_precomputed_graph_message_shown'):
                        print(f"Using loaded global graph (will extract {batch_adjacency.shape} subgraphs for batches)")
                        self._precomputed_graph_message_shown = True
                    
                    return batch_adjacency
                else:
                    print(f"Loaded graph size ({loaded_size}) doesn't match expected total cells ({n_total_cells})")
               
        # Add batch dimension for trajectory inference if needed
        if len(cell_embeddings.shape) == 2:
            cell_embeddings_for_traj = cell_embeddings.unsqueeze(0)
        else:
            cell_embeddings_for_traj = cell_embeddings
        
        # Compute adjacency matrix using trajectory inference
        adjacency_matrix = self.trajectory_inference.infer_trajectory(cell_embeddings_for_traj)
        
        # Remove batch dimension if it was added
        if adjacency_matrix.shape[0] == 1:
            adjacency_matrix = adjacency_matrix.squeeze(0)
        
        print(f"Fallback graph computed: shape={adjacency_matrix.shape}, sparsity={float((adjacency_matrix > 1e-6).float().mean()):.4f}")
        
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
        
        # Step 3: Infer cell-to-cell adjacency matrix (skip if disabled)
        if self._disable_graph_computation:
            # During pretraining, skip graph computation and GNN
            gnn_embeddings = fused_embeddings  # Use fused embeddings directly
            adjacency_matrix = None
        else:
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
    
    def reset_graph_structure(self):
        """Reset the fixed adjacency matrix to allow recomputation."""
        self._fixed_adjacency_matrix = None
        self._adjacency_computed = False
        self._cached_batch_size = None
        if hasattr(self, '_batch_adjacency_cache'):
            self._batch_adjacency_cache.clear()
        if hasattr(self, '_disk_check_done'):
            self._disk_check_done = False
        print("Graph structure reset, adjacency matrix will be recomputed on next forward pass")
    
    def is_graph_fixed(self) -> bool:
        """Check if the graph structure is currently fixed."""
        return self._adjacency_computed and self._fixed_adjacency_matrix is not None
    
    def compute_full_graph_structure(self, data_matrices: dict, 
                                   cell_metadata=None, feature_metadata=None, 
                                   chunk_size: int = 400, 
                                   overlap_ratio: float = 0.2):
        """
        Compute graph structure with CPU-first strategy, fallback to chunked sampling.
        
        Strategy:
        1. First try CPU computation of full graph (no memory limit)
        2. If CPU fails, use GPU chunked sampling strategy
        
        Args:
            data_matrices: Dictionary of all data matrices (all cells)
            cell_metadata: Cell metadata  
            feature_metadata: Feature metadata
            chunk_size: Size of each chunk for computation (fallback only)
            overlap_ratio: Overlap ratio between chunks (fallback only)
        """

        
        device = next(self.parameters()).device
        n_cells = list(data_matrices.values())[0].shape[0]
        
        # Strategy 1: Try CPU-based full computation first

        try:
            full_adjacency = self._compute_full_graph_cpu(data_matrices, device)
            
            if full_adjacency is not None:
                # Store as fixed graph structure
                self._fixed_adjacency_matrix = full_adjacency.detach()
                self._adjacency_computed = True
                

                
                # Save to disk
                self._save_graph_structure(full_adjacency, 'cpu_full_computation')
                return full_adjacency
                
        except Exception as e:
            print(f" CPU computation failed: {e}")
        
        # Strategy 2: Fallback to chunked GPU computation
        print(f"Strategy 2: Fallback to GPU chunked computation (chunk_size={chunk_size}, overlap_ratio={overlap_ratio})...")
        try:
            # Encode all cells in small batches first
            print(f"Step 1: Encoding all {n_cells} cells...")
            all_embeddings = self._encode_all_cells_efficiently(data_matrices, device, store_on_cpu=True)
            
            # Compute graph using chunked strategy
            print(f"Step 2: Computing global graph using chunks of size {chunk_size}...")
            full_adjacency = self._compute_chunked_graph(all_embeddings, chunk_size, overlap_ratio, device)
            
            # Store as fixed graph structure
            self._fixed_adjacency_matrix = full_adjacency.detach()
            self._adjacency_computed = True
            
            print(f" Chunked computation successful: shape={full_adjacency.shape}, sparsity={float((full_adjacency > 1e-6).float().mean()):.4f}")
            
            # Save to disk
            self._save_graph_structure(full_adjacency, 'chunked_precomputation', chunk_size, overlap_ratio)
            return full_adjacency
            
        except Exception as e:
            print(f" Chunked computation also failed: {e}")
            raise RuntimeError("Both CPU and chunked GPU computation failed")
    
    def _compute_full_graph_cpu(self, data_matrices: dict, gpu_device: torch.device) -> torch.Tensor:
        """
        Attempt to compute full graph using CPU to avoid memory limitations.
        
        Args:
            data_matrices: Dictionary of all data matrices
            gpu_device: Original GPU device for model parameters
            
        Returns:
            Full adjacency matrix or None if failed
        """
        n_cells = list(data_matrices.values())[0].shape[0]
        
        original_device = {}  # Initialize outside try block
        try:
            # Encode all cells in batches on GPU, store embeddings on CPU

            all_embeddings_cpu = self._encode_all_cells_efficiently(data_matrices, gpu_device, store_on_cpu=True)
            
            # Move model components to CPU temporarily

            for name, param in self.trajectory_inference.named_parameters():
                original_device[name] = param.device
                param.data = param.data.cpu()
            
            # Compute trajectory on CPU

            all_embeddings_cpu_batch = all_embeddings_cpu.unsqueeze(0)  # Add batch dimension
            
            with torch.no_grad():
                cpu_adjacency = self.trajectory_inference.infer_trajectory(all_embeddings_cpu_batch)
            
            if cpu_adjacency.shape[0] == 1:
                cpu_adjacency = cpu_adjacency.squeeze(0)
            
            # Move model components back to GPU

            for name, param in self.trajectory_inference.named_parameters():
                param.data = param.data.to(original_device[name])
            
            # Move result to GPU
            gpu_adjacency = cpu_adjacency.to(gpu_device)
            

            return gpu_adjacency
            
        except Exception as e:
            # Ensure model is back on GPU even if failed

            try:
                for name, param in self.trajectory_inference.named_parameters():
                    if name in original_device:
                        param.data = param.data.to(original_device[name])
            except:
                pass
            raise e
    
    def _save_graph_structure(self, adjacency_matrix: torch.Tensor, source: str, 
                            chunk_size: int = None, overlap_ratio: float = None):
        """Helper method to save graph structure with metadata."""
        try:
            additional_info = {
                'source': source,
                'n_cells': adjacency_matrix.shape[0],
                'creation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            if chunk_size is not None:
                additional_info['chunk_size'] = chunk_size
            if overlap_ratio is not None:
                additional_info['overlap_ratio'] = overlap_ratio
            
            saved = self.graph_manager.save_graph_structure(
                adjacency_matrix,
                self._config_for_graph,
                additional_info=additional_info
            )
            if saved:
                print(f"Full graph structure saved to disk: {self.graph_manager.graph_dir}")
        except Exception as e:
            print(f"Warning: Could not save full graph structure: {e}")
    
    def _encode_all_cells_efficiently(self, data_matrices: dict, device: torch.device, 
                                    batch_size: int = 256, store_on_cpu: bool = True) -> torch.Tensor:
        """
        Encode all cells efficiently in small batches.
        
        Args:
            data_matrices: Dictionary of data matrices
            device: Device for computation 
            batch_size: Batch size for encoding
            store_on_cpu: Whether to store results on CPU (default True for memory efficiency)
        """
        n_cells = list(data_matrices.values())[0].shape[0]
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, n_cells, batch_size):
                end_idx = min(i + batch_size, n_cells)
                
                # Create batch
                batch = {}
                for modality, matrix in data_matrices.items():
                    batch[modality] = torch.tensor(matrix[i:end_idx], dtype=torch.float32, device=device)
                
                # Encode
                modality_embeddings = self.encode_modalities(batch)
                fused_embeddings = self.fuse_modalities(modality_embeddings)
                
                # Store on CPU or keep on GPU
                if store_on_cpu:
                    all_embeddings.append(fused_embeddings.cpu())
                else:
                    all_embeddings.append(fused_embeddings)
                
                if (i // batch_size + 1) % 10 == 0:
                    print(f"  Encoded {end_idx}/{n_cells} cells...")
        
        return torch.cat(all_embeddings, dim=0)
    
    def _compute_chunked_graph(self, all_embeddings: torch.Tensor, 
                             chunk_size: int, overlap_ratio: float, 
                             device: torch.device) -> torch.Tensor:
        """
        Compute graph using overlapping chunks and intelligent merging.
        """
        n_cells = all_embeddings.shape[0]
        overlap_size = int(chunk_size * overlap_ratio)
        stride = chunk_size - overlap_size
        
        # Initialize full adjacency matrix
        full_adjacency = torch.zeros(n_cells, n_cells, device='cpu')
        weight_matrix = torch.zeros(n_cells, n_cells, device='cpu')  # For weighted averaging
        
        print(f"Computing {(n_cells - chunk_size) // stride + 1} overlapping chunks...")
        
        chunk_count = 0
        for start_idx in range(0, n_cells - chunk_size + 1, stride):
            end_idx = min(start_idx + chunk_size, n_cells)
            actual_chunk_size = end_idx - start_idx
            
            if actual_chunk_size < chunk_size // 2:  # Skip very small chunks
                continue
                
            chunk_count += 1
            print(f"  Processing chunk {chunk_count}: cells {start_idx}-{end_idx-1}")
            
            # Extract chunk embeddings
            chunk_embeddings = all_embeddings[start_idx:end_idx].to(device)
            
            # Compute trajectory for this chunk
            chunk_embeddings_traj = chunk_embeddings.unsqueeze(0)
            chunk_adjacency = self.trajectory_inference.infer_trajectory(chunk_embeddings_traj)
            
            if chunk_adjacency.shape[0] == 1:
                chunk_adjacency = chunk_adjacency.squeeze(0)
            
            # Move to CPU and add to full matrix with weights
            chunk_adjacency_cpu = chunk_adjacency.cpu()
            
            # Add to full matrix with distance-based weighting
            for i in range(actual_chunk_size):
                for j in range(actual_chunk_size):
                    global_i = start_idx + i
                    global_j = start_idx + j
                    
                    # Weight based on distance from chunk center for smooth blending
                    center = actual_chunk_size // 2
                    weight_i = 1.0 - abs(i - center) / (actual_chunk_size / 2)
                    weight_j = 1.0 - abs(j - center) / (actual_chunk_size / 2)
                    weight = min(weight_i, weight_j)
                    
                    full_adjacency[global_i, global_j] += chunk_adjacency_cpu[i, j] * weight
                    weight_matrix[global_i, global_j] += weight
            
            # Clear GPU memory
            del chunk_embeddings, chunk_adjacency, chunk_adjacency_cpu
            torch.cuda.empty_cache()
        
        # Handle the last chunk if it doesn't fit perfectly
        if n_cells % stride != 0:
            start_idx = n_cells - chunk_size
            if start_idx >= 0:
                chunk_count += 1
                print(f"  Processing final chunk {chunk_count}: cells {start_idx}-{n_cells-1}")
                
                chunk_embeddings = all_embeddings[start_idx:].to(device)
                chunk_embeddings_traj = chunk_embeddings.unsqueeze(0)
                chunk_adjacency = self.trajectory_inference.infer_trajectory(chunk_embeddings_traj)
                
                if chunk_adjacency.shape[0] == 1:
                    chunk_adjacency = chunk_adjacency.squeeze(0)
                
                chunk_adjacency_cpu = chunk_adjacency.cpu()
                actual_size = n_cells - start_idx
                
                for i in range(actual_size):
                    for j in range(actual_size):
                        global_i = start_idx + i
                        global_j = start_idx + j
                        
                        center = actual_size // 2
                        weight_i = 1.0 - abs(i - center) / (actual_size / 2) if actual_size > 1 else 1.0
                        weight_j = 1.0 - abs(j - center) / (actual_size / 2) if actual_size > 1 else 1.0
                        weight = min(weight_i, weight_j)
                        
                        full_adjacency[global_i, global_j] += chunk_adjacency_cpu[i, j] * weight
                        weight_matrix[global_i, global_j] += weight
        
        # Normalize by weights (weighted average)
        mask = weight_matrix > 0
        full_adjacency[mask] = full_adjacency[mask] / weight_matrix[mask]
        
        print(f"Graph merging completed from {chunk_count} chunks")
        
        return full_adjacency.to(device)
    
    def get_graph_info(self) -> Dict:
        """Get information about the current graph structure."""
        if not self.is_graph_fixed():
            return {"status": "not_computed"}
        
        adjacency_matrix = self._fixed_adjacency_matrix
        return {
            "status": "fixed",
            "batch_size": self._cached_batch_size,
            "shape": list(adjacency_matrix.shape),
            "sparsity": float((adjacency_matrix > 1e-6).float().mean()),
            "device": str(adjacency_matrix.device),
            "trajectory_method": self.trajectory_inference.method
        }
    
    def list_available_graphs(self) -> Dict:
        """List all available precomputed graph structures."""
        return self.graph_manager.list_available_graphs()
    
    def clear_graph_cache(self, config_hash: Optional[str] = None):
        """Clear cached graph structures."""
        self.graph_manager.clear_graph_cache(config_hash)
        if config_hash is None:
            # Also clear current cache
            self.reset_graph_structure()
    
    def get_batch_graph_info(self) -> Dict:
        """Get information about cached batch adjacency matrices."""
        if not hasattr(self, '_batch_adjacency_cache'):
            return {"cached_batch_sizes": []}
        
        info = {
            "cached_batch_sizes": [],
            "cache_details": {}
        }
        
        for cache_key, matrix in self._batch_adjacency_cache.items():
            batch_size = matrix.shape[0]
            info["cached_batch_sizes"].append(batch_size)
            info["cache_details"][cache_key] = {
                "shape": list(matrix.shape),
                "sparsity": float((matrix > 1e-6).float().mean()),
                "device": str(matrix.device)
            }
        
        return info
    
    def preload_graph_structure(self) -> bool:
        """
        Attempt to preload graph structure from disk before evaluation.
        
        Returns:
            True if graph structure was successfully loaded, False otherwise
        """
        if hasattr(self, '_disk_check_done') and self._disk_check_done:
            print("Graph structure loading already attempted")
            return self._adjacency_computed
        
        print("Attempting to preload graph structure...")
        
        try:
            self._disk_check_done = True
            graph_data = self.graph_manager.load_graph_structure(self._config_for_graph)
            
            if graph_data is not None:
                loaded_adjacency_matrix, metadata = graph_data
                
                print(f"Graph structure found and loaded (shape: {loaded_adjacency_matrix.shape})")
                print(f"Source: {metadata.get('source', 'Unknown')}")
                print(f"Created: {metadata.get('creation_timestamp', 'Unknown')}")
                
                # Check if this could be a full graph
                n_total_cells = self._config_for_graph.get('data', {}).get('n_cells', 2000)
                loaded_size = loaded_adjacency_matrix.shape[0]
                
                if loaded_size == n_total_cells:
                    # Store as full graph
                    self._fixed_adjacency_matrix = loaded_adjacency_matrix
                    self._adjacency_computed = True
                    print("Full graph structure successfully preloaded")
                    return True
                else:
                    # Store in batch cache with appropriate key
                    if not hasattr(self, '_batch_adjacency_cache'):
                        self._batch_adjacency_cache = {}
                    cache_key = f"batch_{loaded_size}"
                    self._batch_adjacency_cache[cache_key] = loaded_adjacency_matrix.detach()
                    print(f"Batch-sized graph structure preloaded for batch size {loaded_size}")
                    return True
            else:
                print("No precomputed graph structure found")
                return False
                
        except Exception as e:
            print(f"Error preloading graph structure: {e}")
            return False
