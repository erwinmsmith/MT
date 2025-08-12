import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np

class TrajectoryInference(nn.Module):
    """
    Trajectory inference module for inferring cell-to-cell directed relationships.
    Implements multiple trajectory inference methods.
    """
    
    def __init__(self, embed_dim: int, n_components: int = 50, 
                 kernel_type: str = 'gaussian', sigma: float = 1.0,
                 method: str = 'diffusion'):
        """
        Initialize trajectory inference module.
        
        Args:
            embed_dim: Embedding dimension
            n_components: Number of diffusion components
            kernel_type: Type of kernel ('gaussian', 'cosine', 'polynomial')
            sigma: Kernel bandwidth parameter
            method: Trajectory inference method ('diffusion', 'pseudotime', 'velocity')
        """
        super(TrajectoryInference, self).__init__()
        
        self.embed_dim = embed_dim
        self.n_components = n_components
        self.kernel_type = kernel_type
        self.sigma = sigma
        self.method = method
        
        # Learnable trajectory parameters
        self.trajectory_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, n_components)
        )
        
        # Diffusion parameters
        if method == 'diffusion':
            self.diffusion_weights = nn.Parameter(torch.randn(n_components))
            self.diffusion_bias = nn.Parameter(torch.zeros(1))
        
        # Pseudotime parameters
        elif method == 'pseudotime':
            self.pseudotime_projector = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim // 2, 1),
                nn.Sigmoid()
            )
        
        # Velocity parameters
        elif method == 'velocity':
            self.velocity_encoder = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )
        
        # Adaptive sigma learning
        self.adaptive_sigma = nn.Parameter(torch.tensor(sigma))
        
        # Directionality enhancement
        self.direction_enhancer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
    
    def compute_kernel_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute kernel matrix from embeddings.
        
        Args:
            embeddings: Cell embeddings (batch_size, n_cells, embed_dim)
            
        Returns:
            Kernel matrix (batch_size, n_cells, n_cells)
        """
        if self.kernel_type == 'gaussian':
            distances = self._compute_pairwise_distances(embeddings)
            adaptive_sigma = torch.clamp(self.adaptive_sigma, min=0.1, max=10.0)
            kernel_matrix = torch.exp(-distances / (2 * adaptive_sigma ** 2))
            
        elif self.kernel_type == 'cosine':
            embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
            kernel_matrix = torch.bmm(embeddings_norm, embeddings_norm.transpose(-2, -1))
            kernel_matrix = (kernel_matrix + 1) / 2  # Scale to [0, 1]
            
        elif self.kernel_type == 'polynomial':
            # Polynomial kernel: (1 + <x, y>)^d
            degree = 2
            dot_products = torch.bmm(embeddings, embeddings.transpose(-2, -1))
            kernel_matrix = torch.pow(1 + dot_products, degree)
            
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")
        
        return kernel_matrix
    
    def _compute_pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise squared Euclidean distances.
        
        Args:
            embeddings: Cell embeddings (batch_size, n_cells, embed_dim)
            
        Returns:
            Distance matrix (batch_size, n_cells, n_cells)
        """
        embeddings_expanded_1 = embeddings.unsqueeze(2)  # (batch_size, n_cells, 1, embed_dim)
        embeddings_expanded_2 = embeddings.unsqueeze(1)  # (batch_size, 1, n_cells, embed_dim)
        
        squared_distances = torch.sum((embeddings_expanded_1 - embeddings_expanded_2) ** 2, dim=-1)
        
        return squared_distances
    
    def diffusion_trajectory(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute trajectory using diffusion maps.
        
        Args:
            embeddings: Cell embeddings
            
        Returns:
            Directed adjacency matrix
        """
        # Project to diffusion space
        diffusion_coords = self.trajectory_projection(embeddings)
        
        # Compute kernel matrix in diffusion space
        kernel_matrix = self.compute_kernel_matrix(diffusion_coords)
        
        # Apply diffusion weights - ensure weights match kernel matrix dimensions
        diffusion_weights = F.softmax(self.diffusion_weights, dim=0)
        
        # Get the actual dimensions we need
        if kernel_matrix.dim() == 3:  # (batch_size, n_cells, n_cells)
            batch_size, n_cells, _ = kernel_matrix.shape
        else:  # (n_cells, n_cells)
            n_cells = kernel_matrix.size(0)
            batch_size = 1
        
        # Adjust diffusion weights to match the number of cells
        if len(diffusion_weights) != n_cells:
            # Create new weights that match the current batch size
            if len(diffusion_weights) > n_cells:
                diffusion_weights = diffusion_weights[:n_cells]
            else:
                # Repeat or interpolate weights to match required size
                repeat_factor = (n_cells + len(diffusion_weights) - 1) // len(diffusion_weights)
                diffusion_weights = diffusion_weights.repeat(repeat_factor)[:n_cells]
        
        # Apply weights properly based on kernel matrix dimensions
        if kernel_matrix.dim() == 3:
            weighted_kernel = kernel_matrix * diffusion_weights.view(1, 1, -1).expand(batch_size, 1, n_cells)
        else:
            weighted_kernel = kernel_matrix * diffusion_weights.view(1, -1)
        
        # Normalize to create transition matrix
        row_sums = torch.sum(weighted_kernel, dim=-1, keepdim=True)
        transition_matrix = weighted_kernel / (row_sums + 1e-8)
        
        return transition_matrix
    
    def pseudotime_trajectory(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute trajectory using pseudotime ordering.
        
        Args:
            embeddings: Cell embeddings
            
        Returns:
            Directed adjacency matrix
        """
        batch_size, n_cells, _ = embeddings.shape
        
        # Compute pseudotime for each cell
        pseudotimes = self.pseudotime_projector(embeddings)  # (batch_size, n_cells, 1)
        pseudotimes = pseudotimes.squeeze(-1)  # (batch_size, n_cells)
        
        # Create directed edges based on pseudotime ordering
        pseudotime_diff = pseudotimes.unsqueeze(2) - pseudotimes.unsqueeze(1)  # (batch_size, n_cells, n_cells)
        
        # Only connect cells that are close in pseudotime and in the forward direction
        time_mask = (pseudotime_diff > 0) & (pseudotime_diff < 0.1)  # Forward direction within threshold
        
        # Compute similarity in embedding space
        kernel_matrix = self.compute_kernel_matrix(embeddings)
        
        # Combine time ordering with similarity
        directed_matrix = kernel_matrix * time_mask.float()
        
        # Normalize
        row_sums = torch.sum(directed_matrix, dim=-1, keepdim=True)
        directed_matrix = directed_matrix / (row_sums + 1e-8)
        
        return directed_matrix
    
    def velocity_trajectory(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute trajectory using velocity-like approach.
        
        Args:
            embeddings: Cell embeddings
            
        Returns:
            Directed adjacency matrix
        """
        # Compute velocity vectors
        velocity_vectors = self.velocity_encoder(embeddings)
        
        # Compute direction enhancement
        direction_enhanced = self.direction_enhancer(embeddings)
        combined_embeddings = embeddings + 0.1 * direction_enhanced
        
        # Compute directional kernel
        # For each pair of cells, compute if velocity points towards the target
        batch_size, n_cells, embed_dim = embeddings.shape
        
        # Expand for pairwise computation
        source_embeddings = combined_embeddings.unsqueeze(2).expand(-1, -1, n_cells, -1)
        target_embeddings = combined_embeddings.unsqueeze(1).expand(-1, n_cells, -1, -1)
        source_velocities = velocity_vectors.unsqueeze(2).expand(-1, -1, n_cells, -1)
        
        # Compute direction from source to target
        directions = target_embeddings - source_embeddings
        
        # Compute alignment between velocity and direction
        alignment = F.cosine_similarity(source_velocities, directions, dim=-1)
        alignment = torch.clamp(alignment, min=0)  # Only positive alignments
        
        # Compute base similarity
        base_similarity = self.compute_kernel_matrix(combined_embeddings)
        
        # Combine alignment with similarity
        directed_matrix = base_similarity * alignment
        
        # Normalize
        row_sums = torch.sum(directed_matrix, dim=-1, keepdim=True)
        directed_matrix = directed_matrix / (row_sums + 1e-8)
        
        return directed_matrix
    
    def infer_trajectory(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Infer trajectory from cell embeddings using the specified method.
        
        Args:
            embeddings: Cell embeddings (batch_size, n_cells, embed_dim)
            
        Returns:
            Directed adjacency matrix (batch_size, n_cells, n_cells)
        """
        if self.method == 'diffusion':
            return self.diffusion_trajectory(embeddings)
        elif self.method == 'pseudotime':
            return self.pseudotime_trajectory(embeddings)
        elif self.method == 'velocity':
            return self.velocity_trajectory(embeddings)
        else:
            raise ValueError(f"Unsupported trajectory method: {self.method}")
    
    def get_trajectory_analysis(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get comprehensive trajectory analysis.
        
        Args:
            embeddings: Cell embeddings
            
        Returns:
            Dictionary with trajectory analysis results
        """
        directed_matrix = self.infer_trajectory(embeddings)
        
        analysis = {
            'directed_matrix': directed_matrix,
            'kernel_matrix': self.compute_kernel_matrix(embeddings),
        }
        
        if self.method == 'pseudotime':
            pseudotimes = self.pseudotime_projector(embeddings).squeeze(-1)
            analysis['pseudotimes'] = pseudotimes
            analysis['pseudotime_ordering'] = torch.argsort(pseudotimes, dim=-1)
        
        elif self.method == 'velocity':
            velocity_vectors = self.velocity_encoder(embeddings)
            analysis['velocity_vectors'] = velocity_vectors
            analysis['velocity_magnitude'] = torch.norm(velocity_vectors, dim=-1)
        
        elif self.method == 'diffusion':
            diffusion_coords = self.trajectory_projection(embeddings)
            analysis['diffusion_coordinates'] = diffusion_coords
        
        return analysis
