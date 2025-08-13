import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np

class TrajectoryInference(nn.Module):
    """
    Trajectory inference module for inferring directed cell-to-cell relationships.
    Implements improved trajectory inference methods based on diffusion maps, pseudotime, and velocity.
    """
    
    def __init__(self, embed_dim: int, n_components: int = 50, 
                 kernel_type: str = 'gaussian', k_neighbors: int = 30,
                 method: str = 'diffusion', root_cell_idx: Optional[int] = None):
        """
        Initialize trajectory inference module.
        
        Args:
            embed_dim: Embedding dimension
            n_components: Number of diffusion components
            kernel_type: Kernel type ('gaussian', 'cosine')
            k_neighbors: Number of k-nearest neighbors
            method: Trajectory inference method ('diffusion', 'pseudotime', 'velocity')
            root_cell_idx: Root cell index (only used for diffusion method)
        """
        super(TrajectoryInference, self).__init__()
        
        self.embed_dim = embed_dim
        self.n_components = n_components
        self.kernel_type = kernel_type
        self.k_neighbors = k_neighbors
        self.method = method
        
        # Root cell index is only needed for diffusion method
        if method == 'diffusion' and root_cell_idx is None:
            self.root_cell_idx = 0  # Default to first cell
        else:
            self.root_cell_idx = root_cell_idx
        
        # Trajectory projection network (used by diffusion method)
        if method == 'diffusion':
            self.trajectory_projection = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim // 2, embed_dim // 4),
                nn.GELU(),
                nn.Linear(embed_dim // 4, n_components)
            )
        
        # Pseudotime predictor (used by pseudotime and velocity methods)
        if method in ['pseudotime', 'velocity']:
            self.pseudotime_projector = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim // 2, 1),
                nn.Sigmoid()
            )
        
        # Velocity encoder (used by velocity method)
        if method == 'velocity':
            self.velocity_encoder = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )
        
        # Direction enhancer (used by velocity method)
        if method == 'velocity':
            self.direction_enhancer = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.Tanh(),
                nn.Linear(embed_dim // 2, embed_dim)
            )
    
    def _pairwise_sqdist(self, x):
        """Compute pairwise squared distances"""
        # x: (B,N,D) -> (B,N,N)
        x1 = x.unsqueeze(2)  # (B,N,1,D)
        x2 = x.unsqueeze(1)  # (B,1,N,D)
        return ((x1 - x2) ** 2).sum(-1)
    
    def _local_sigma(self, d2, k=15):
        """Compute local adaptive bandwidth"""
        # d2: (B,N,N) squared distance matrix
        B, N, _ = d2.shape
        # Exclude self-distances
        d2 = d2 + torch.eye(N, device=d2.device).unsqueeze(0) * 1e9
        # Ensure k doesn't exceed available neighbors
        k = min(k, N - 1)
        # k-th smallest distance
        vals, _ = torch.topk(d2, k, dim=-1, largest=False)
        sigma2 = vals[..., -1]  # (B,N)
        return torch.sqrt(torch.clamp(sigma2, min=1e-12))
    
    def build_knn_kernel(self, x, k=None, eps=1e-12):
        """Build kNN kernel matrix"""
        if k is None:
            k = self.k_neighbors
        
        # x: (B,N,D)
        d2 = self._pairwise_sqdist(x)  # (B,N,N)
        B, N, _ = d2.shape
        
        if self.kernel_type == 'gaussian':
            # Local adaptive bandwidth
            sigma_i = self._local_sigma(d2, k=max(5, k//2))  # (B,N)
            
            # Adaptive Gaussian kernel
            denom = (sigma_i.unsqueeze(-1) * sigma_i.unsqueeze(1)) + eps  # (B,N,N)
            K = torch.exp(-d2 / denom)
            
        elif self.kernel_type == 'cosine':
            # Cosine similarity kernel
            x_norm = F.normalize(x, p=2, dim=-1)
            K = torch.bmm(x_norm, x_norm.transpose(-2, -1))
            K = (K + 1) / 2  # Scale to [0,1]
            
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")
        
        # kNN sparse mask (mutual neighbors union)
        # Ensure k doesn't exceed available neighbors
        k = min(k, N - 1)
        _, nn_idx = torch.topk(d2, k, dim=-1, largest=False)  # (B,N,k)
        mask = torch.zeros_like(K)
        batch_idx = torch.arange(B, device=x.device)[:, None, None]
        row_idx = torch.arange(N, device=x.device)[None, :, None]
        mask[batch_idx, row_idx, nn_idx] = 1.0
        mask = torch.maximum(mask, mask.transpose(-2, -1))  # Mutual
        K = K * mask
        
        return K  # (B,N,N)
    
    def markov_transition(self, K, eps=1e-12):
        """Build Markov transition matrix"""
        D = K.sum(-1, keepdim=True).clamp_min(eps)  # (B,N,1)
        P = K / D
        return P, D.squeeze(-1)  # P: (B,N,N), D: (B,N)
    
    def diffusion_pseudotime_from_root(self, K, root_idx, n_components=None, t=1):
        """Standard DPT pseudotime calculation based on diffusion maps"""
        if n_components is None:
            n_components = self.n_components
        
        # Standard symmetric normalization and spectral decomposition
        P, D = self.markov_transition(K)  # (B,N,N), (B,N)
        B, N, _ = P.shape
        d_sqrt_inv = (1.0 / D.clamp_min(1e-12)).sqrt()  # (B,N)
        S = d_sqrt_inv.unsqueeze(-1) * K * d_sqrt_inv.unsqueeze(1)  # (B,N,N)
        
        # Symmetric matrix eigendecomposition
        evals, evecs = torch.linalg.eigh(S)  # (B,N), (B,N,N)
        # Descending order
        idx = torch.argsort(evals, dim=-1, descending=True)
        evals = torch.gather(evals, -1, idx)
        evecs = torch.gather(evecs, -1, idx.unsqueeze(1).expand(-1, N, -1))
        
        # Remove trivial component (first eigenvalue ≈ 1)
        m = min(n_components + 1, N)
        lam = evals[:, 1:m] ** t              # (B,m-1)
        U = evecs[:, :, 1:m]                  # (B,N,m-1)
        psi = (U.transpose(1,2) * d_sqrt_inv.unsqueeze(1)).transpose(1,2)  # (B,N,m-1)
        
        # Diffusion coordinates
        Psi_t = psi * lam.unsqueeze(1)  # (B,N,m-1)
        
        # Diffusion distance from root cell as pseudotime
        if isinstance(root_idx, int):
            root = torch.full((B,), root_idx, device=K.device, dtype=torch.long)
        else:
            root = root_idx
        
        root_embed = Psi_t[torch.arange(B, device=K.device), root]  # (B,m-1)
        dpt = ((Psi_t - root_embed.unsqueeze(1))**2).sum(-1).sqrt()  # (B,N)
        
        # Directed adjacency: only keep forward pseudotime, soft weights
        P_forward = P.clone()
        dt = dpt.unsqueeze(-1) - dpt.unsqueeze(1)  # (B,N,N): Δt from i->j
        tau = (dpt.max(dim=-1, keepdim=True).values.clamp_min(1e-6)) / 10.0
        soft = (dt > 0).float() * torch.exp(-(dt / tau) ** 2)
        A_dir = P_forward * soft
        
        # Row normalization
        A_dir = A_dir / A_dir.sum(-1, keepdim=True).clamp_min(1e-12)
        
        return A_dir, dpt, Psi_t
    
    def diffusion_trajectory(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute trajectory using improved diffusion maps.
        
        Args:
            embeddings: Cell embeddings
            
        Returns:
            Directed adjacency matrix
        """
        # Project to diffusion space
        diffusion_coords = self.trajectory_projection(embeddings)
        
        # Build kNN kernel matrix
        K = self.build_knn_kernel(diffusion_coords)
        
        # Use standard DPT to compute directed adjacency matrix
        A_dir, dpt, diffusion_embeds = self.diffusion_pseudotime_from_root(
            K, self.root_cell_idx
        )
        
        return A_dir
    
    def pseudotime_trajectory(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute trajectory using improved pseudotime ordering.
        
        Args:
            embeddings: Cell embeddings
            
        Returns:
            Directed adjacency matrix
        """
        batch_size, n_cells, _ = embeddings.shape
        
        # Compute pseudotime for each cell
        pseudotimes = self.pseudotime_projector(embeddings)  # (B,N,1)
        pseudotimes = pseudotimes.squeeze(-1)  # (B,N)
        
        # Build similarity-based kernel matrix
        K = self.build_knn_kernel(embeddings)
        
        # Soft directional weights based on pseudotime
        dt = pseudotimes.unsqueeze(-1) - pseudotimes.unsqueeze(1)  # (B,N,N)
        tau = (pseudotimes.max(dim=-1, keepdim=True).values.clamp_min(1e-6)) / 5.0
        soft_direction = (dt > 0).float() * torch.exp(-(dt / tau) ** 2)
        
        # Combine similarity and directionality
        directed_matrix = K * soft_direction
        
        # Normalize
        directed_matrix = directed_matrix / directed_matrix.sum(-1, keepdim=True).clamp_min(1e-12)
        
        return directed_matrix
    
    def velocity_trajectory(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute trajectory using improved velocity method.
        
        Args:
            embeddings: Cell embeddings
            
        Returns:
            Directed adjacency matrix
        """
        # First compute pseudotime as directional constraint
        pseudotimes = self.pseudotime_projector(embeddings).squeeze(-1)
        
        # Compute velocity vectors (based on neighborhood differences)
        batch_size, n_cells, embed_dim = embeddings.shape
        
        # Build kNN graph
        K = self.build_knn_kernel(embeddings)
        
        # Forward neighbors based on pseudotime
        dt = pseudotimes.unsqueeze(-1) - pseudotimes.unsqueeze(1)  # (B,N,N)
        forward_mask = (dt > 0).float()
        forward_neighbors = K * forward_mask
        
        # Neighborhood difference approximation of velocity
        velocity_vectors = torch.zeros_like(embeddings)
        for b in range(batch_size):
            for i in range(n_cells):
                neighbors = forward_neighbors[b, i] > 0
                if neighbors.sum() > 0:
                    neighbor_embeds = embeddings[b, neighbors]
                    weights = forward_neighbors[b, i, neighbors]
                    weights = weights / weights.sum()
                    velocity_vectors[b, i] = (weights.unsqueeze(-1) * 
                                            (neighbor_embeds - embeddings[b, i])).sum(0)
        
        # Compute directional alignment
        source_embeddings = embeddings.unsqueeze(2).expand(-1, -1, n_cells, -1)
        target_embeddings = embeddings.unsqueeze(1).expand(-1, n_cells, -1, -1)
        source_velocities = velocity_vectors.unsqueeze(2).expand(-1, -1, n_cells, -1)
        
        # Direction from source to target
        directions = target_embeddings - source_embeddings
        
        # Alignment between velocity and direction
        alignment = F.cosine_similarity(source_velocities, directions, dim=-1)
        alignment = torch.clamp(alignment, min=0)  # Only keep positive alignments
        
        # Combine similarity, alignment, and directionality
        base_similarity = K
        directed_matrix = base_similarity * alignment * forward_mask
        
        # Normalize
        directed_matrix = directed_matrix / directed_matrix.sum(-1, keepdim=True).clamp_min(1e-12)
        
        return directed_matrix
    
    def infer_trajectory(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Infer trajectory using specified method.
        
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
            Dictionary containing trajectory analysis results
        """
        directed_matrix = self.infer_trajectory(embeddings)
        
        analysis = {
            'directed_matrix': directed_matrix,
            'kernel_matrix': self.build_knn_kernel(embeddings),
        }
        
        if self.method == 'pseudotime':
            pseudotimes = self.pseudotime_projector(embeddings).squeeze(-1)
            analysis['pseudotimes'] = pseudotimes
            analysis['pseudotime_ordering'] = torch.argsort(pseudotimes, dim=-1)
        
        elif self.method == 'velocity':
            # Compute velocity vectors
            velocity_vectors = self.velocity_encoder(embeddings)
            analysis['velocity_vectors'] = velocity_vectors
            analysis['velocity_magnitude'] = torch.norm(velocity_vectors, dim=-1)
        
        elif self.method == 'diffusion':
            diffusion_coords = self.trajectory_projection(embeddings)
            analysis['diffusion_coordinates'] = diffusion_coords
            
            # Get diffusion analysis
            K = self.build_knn_kernel(diffusion_coords)
            _, dpt, diffusion_embeds = self.diffusion_pseudotime_from_root(
                K, self.root_cell_idx
            )
            analysis['diffusion_pseudotime'] = dpt
            analysis['diffusion_embeddings'] = diffusion_embeds
        
        return analysis