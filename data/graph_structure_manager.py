#!/usr/bin/env python3
"""
Graph Structure Manager for Multi-omics Topic Model

This module handles saving and loading of trajectory inference graph structures
to avoid recomputation during training.
"""

import os
import pickle
import torch
import numpy as np
from typing import Dict, Optional, Tuple
import hashlib

class GraphStructureManager:
    """
    Manager for saving and loading precomputed graph structures.
    """
    
    def __init__(self, data_dir: str = "data/dataset"):
        """
        Initialize graph structure manager.
        
        Args:
            data_dir: Directory to save/load graph structures
        """
        self.data_dir = data_dir
        self.graph_dir = os.path.join(data_dir, "graph_structures")
        os.makedirs(self.graph_dir, exist_ok=True)
    
    def _get_config_hash(self, config: Dict) -> str:
        """
        Generate a hash from trajectory configuration to identify unique structures.
        
        Args:
            config: Trajectory configuration dictionary
            
        Returns:
            Configuration hash string
        """
        # Extract relevant configuration parameters
        traj_config = config.get('trajectory', {})
        relevant_params = {
            'method': traj_config.get('method', 'diffusion'),
            'n_components': traj_config.get('n_components', 50),
            'kernel_type': traj_config.get('kernel_type', 'gaussian'),
            'k_neighbors': traj_config.get('k_neighbors', 30),
            'root_cell_idx': traj_config.get('root_cell_idx', 0),
            'n_cells': config.get('data', {}).get('n_cells', 2000),
            'embedding_dim': config.get('data', {}).get('embedding_dim', 256)
        }
        
        # Create hash from parameters
        config_str = str(sorted(relevant_params.items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:16]
    
    def _get_graph_path(self, config_hash: str) -> str:
        """Get the file path for graph structure."""
        return os.path.join(self.graph_dir, f"graph_structure_{config_hash}.pkl")
    
    def save_graph_structure(self, adjacency_matrix: torch.Tensor, 
                           config: Dict, 
                           additional_info: Optional[Dict] = None) -> str:
        """
        Save graph structure to disk.
        
        Args:
            adjacency_matrix: Precomputed adjacency matrix
            config: Configuration used to generate the graph
            additional_info: Additional information to save
            
        Returns:
            Configuration hash for identification
        """
        config_hash = self._get_config_hash(config)
        graph_path = self._get_graph_path(config_hash)
        
        # Prepare data to save
        graph_data = {
            'adjacency_matrix': adjacency_matrix.cpu(),  # Save on CPU
            'config_hash': config_hash,
            'trajectory_config': config.get('trajectory', {}),
            'data_config': {
                'n_cells': config.get('data', {}).get('n_cells', 2000),
                'embedding_dim': config.get('data', {}).get('embedding_dim', 256)
            },
            'creation_timestamp': np.datetime64('now').astype(str),
            'additional_info': additional_info or {}
        }
        
        # Save to file
        with open(graph_path, 'wb') as f:
            pickle.dump(graph_data, f)
        

        return config_hash
    
    def load_graph_structure(self, config: Dict) -> Optional[Tuple[torch.Tensor, Dict]]:
        """
        Load graph structure from disk if available.
        
        Args:
            config: Current configuration
            
        Returns:
            Tuple of (adjacency_matrix, metadata) if found, None otherwise
        """
        config_hash = self._get_config_hash(config)
        graph_path = self._get_graph_path(config_hash)
        
        if not os.path.exists(graph_path):
            print(f"No precomputed graph structure found for config hash: {config_hash}")
            return None
        
        try:
            with open(graph_path, 'rb') as f:
                graph_data = pickle.load(f)
            
            adjacency_matrix = graph_data['adjacency_matrix']
            
            # Validate dimensions
            expected_n_cells = config.get('data', {}).get('n_cells', 2000)
            if adjacency_matrix.shape[0] != expected_n_cells:
                print(f"Graph structure dimension mismatch: expected {expected_n_cells}, got {adjacency_matrix.shape[0]}")
                print(f"Removing invalid graph structure file: {graph_path}")
                
                # Remove the invalid file
                try:
                    os.remove(graph_path)
                    print(f"Invalid graph structure file removed successfully")
                except Exception as e:
                    print(f"Warning: Could not remove invalid file: {e}")
                
                return None
            
            print(f"Graph structure loaded: {graph_path}")
            print(f"Configuration hash: {config_hash}")
            print(f"Graph shape: {adjacency_matrix.shape}")
            print(f"Created: {graph_data.get('creation_timestamp', 'Unknown')}")
            
            # Return adjacency matrix and metadata
            metadata = {
                'config_hash': graph_data['config_hash'],
                'trajectory_config': graph_data['trajectory_config'],
                'creation_timestamp': graph_data.get('creation_timestamp'),
                'additional_info': graph_data.get('additional_info', {})
            }
            
            return adjacency_matrix, metadata
            
        except Exception as e:
            print(f"Error loading graph structure: {e}")
            return None
    
    def list_available_graphs(self) -> Dict[str, Dict]:
        """
        List all available graph structures.
        
        Returns:
            Dictionary mapping config hashes to metadata
        """
        available_graphs = {}
        
        for filename in os.listdir(self.graph_dir):
            if filename.startswith("graph_structure_") and filename.endswith(".pkl"):
                graph_path = os.path.join(self.graph_dir, filename)
                try:
                    with open(graph_path, 'rb') as f:
                        graph_data = pickle.load(f)
                    
                    config_hash = graph_data['config_hash']
                    available_graphs[config_hash] = {
                        'trajectory_config': graph_data['trajectory_config'],
                        'data_config': graph_data['data_config'],
                        'creation_timestamp': graph_data.get('creation_timestamp'),
                        'file_path': graph_path
                    }
                except Exception as e:
                    print(f"Error reading {graph_path}: {e}")
        
        return available_graphs
    
    def clear_graph_cache(self, config_hash: Optional[str] = None):
        """
        Clear cached graph structures.
        
        Args:
            config_hash: Specific hash to clear, or None to clear all
        """
        if config_hash:
            graph_path = self._get_graph_path(config_hash)
            if os.path.exists(graph_path):
                os.remove(graph_path)
                print(f"Cleared graph structure: {config_hash}")
        else:
            # Clear all graphs
            for filename in os.listdir(self.graph_dir):
                if filename.startswith("graph_structure_") and filename.endswith(".pkl"):
                    os.remove(os.path.join(self.graph_dir, filename))
            print("Cleared all graph structures")
