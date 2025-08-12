import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, Optional
from pathlib import Path
import pickle

class DatasetSimulator:
    """Simulate multi-omics dataset for testing and development."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.n_cells = config['n_cells']
        self.n_genes = config['n_genes']
        self.n_peaks = config['n_peaks']
        self.n_proteins = config['n_proteins']
        self.n_topics = config.get('n_topics', 20)
        
    def simulate_single_cell_data(self) -> Dict[str, np.ndarray]:
        """
        Simulate single-cell multi-omics data.
        
        Returns:
            Dictionary containing simulated data matrices
        """
        np.random.seed(42)
        
        # Simulate cell-gene matrix (RNA-seq)
        # Use negative binomial distribution to simulate count data
        gene_matrix = np.random.negative_binomial(
            n=5, p=0.3, size=(self.n_cells, self.n_genes)
        ).astype(np.float32)
        
        # Add some structure - cells can be grouped into different types
        n_cell_types = 5
        cells_per_type = self.n_cells // n_cell_types
        
        for i in range(n_cell_types):
            start_idx = i * cells_per_type
            end_idx = (i + 1) * cells_per_type if i < n_cell_types - 1 else self.n_cells
            
            # Some genes are highly expressed in specific cell types
            specific_genes = np.random.choice(
                self.n_genes, size=self.n_genes // 10, replace=False
            )
            gene_matrix[start_idx:end_idx, specific_genes] *= (2 + i)
        
        # Simulate cell-peak matrix (ATAC-seq)
        # Binary/count data for chromatin accessibility
        peak_matrix = np.random.binomial(
            n=1, p=0.1, size=(self.n_cells, self.n_peaks)
        ).astype(np.float32)
        
        # Add some correlation with gene expression
        for i in range(min(self.n_genes, self.n_peaks) // 2):
            # Some peaks correlate with nearby genes
            correlation_strength = np.random.uniform(0.3, 0.8)
            noise = np.random.normal(0, 0.1, self.n_cells)
            peak_matrix[:, i] = (
                correlation_strength * (gene_matrix[:, i] > np.median(gene_matrix[:, i])) + 
                noise
            )
            peak_matrix[:, i] = np.clip(peak_matrix[:, i], 0, 1)
        
        # Simulate cell-protein matrix (CITEseq/protein)
        # Usually lower dimensional than genes
        protein_matrix = np.random.lognormal(
            mean=1, sigma=0.5, size=(self.n_cells, self.n_proteins)
        ).astype(np.float32)
        
        # Add some correlation with gene expression
        for i in range(min(self.n_genes, self.n_proteins)):
            correlation_strength = np.random.uniform(0.4, 0.9)
            protein_matrix[:, i] = (
                correlation_strength * gene_matrix[:, i] / np.max(gene_matrix[:, i]) +
                (1 - correlation_strength) * protein_matrix[:, i]
            )
        
        return {
            'gene': gene_matrix,
            'peak': peak_matrix,
            'protein': protein_matrix
        }
    
    def create_cell_metadata(self) -> pd.DataFrame:
        """Create cell metadata for simulated data."""
        n_cell_types = 5
        cell_types = [f'CellType_{i}' for i in range(n_cell_types)]
        cells_per_type = self.n_cells // n_cell_types
        
        metadata = []
        for i, cell_type in enumerate(cell_types):
            start_idx = i * cells_per_type
            end_idx = (i + 1) * cells_per_type if i < n_cell_types - 1 else self.n_cells
            n_cells_this_type = end_idx - start_idx
            
            for j in range(n_cells_this_type):
                metadata.append({
                    'cell_id': f'Cell_{start_idx + j}',
                    'cell_type': cell_type,
                    'batch': f'Batch_{np.random.randint(0, 3)}',
                    'stage': f'Stage_{np.random.randint(0, 4)}'
                })
        
        return pd.DataFrame(metadata)
    
    def create_feature_metadata(self) -> Dict[str, pd.DataFrame]:
        """Create feature metadata for each modality."""
        # Gene metadata
        gene_metadata = pd.DataFrame({
            'gene_id': [f'Gene_{i}' for i in range(self.n_genes)],
            'gene_symbol': [f'GENE{i}' for i in range(self.n_genes)],
            'chromosome': [f'chr{np.random.randint(1, 23)}' for _ in range(self.n_genes)],
            'start_pos': np.random.randint(1000, 100000, self.n_genes),
            'end_pos': np.random.randint(100000, 200000, self.n_genes)
        })
        
        # Peak metadata
        peak_metadata = pd.DataFrame({
            'peak_id': [f'Peak_{i}' for i in range(self.n_peaks)],
            'chromosome': [f'chr{np.random.randint(1, 23)}' for _ in range(self.n_peaks)],
            'start_pos': np.random.randint(1000, 100000, self.n_peaks),
            'end_pos': np.random.randint(100000, 200000, self.n_peaks)
        })
        
        # Protein metadata
        protein_metadata = pd.DataFrame({
            'protein_id': [f'Protein_{i}' for i in range(self.n_proteins)],
            'protein_symbol': [f'PROT{i}' for i in range(self.n_proteins)],
            'protein_type': np.random.choice(['surface', 'intracellular', 'secreted'], 
                                           self.n_proteins)
        })
        
        return {
            'gene': gene_metadata,
            'peak': peak_metadata,
            'protein': protein_metadata
        }
    
    def save_simulated_data(self, save_dir: str):
        """Save simulated data to files."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Generate data
        data_matrices = self.simulate_single_cell_data()
        cell_metadata = self.create_cell_metadata()
        feature_metadata = self.create_feature_metadata()
        
        # Save data matrices
        for modality, matrix in data_matrices.items():
            np.save(save_path / f'{modality}_matrix.npy', matrix)
        
        # Save metadata
        cell_metadata.to_csv(save_path / 'cell_metadata.csv', index=False)
        
        for modality, metadata in feature_metadata.items():
            metadata.to_csv(save_path / f'{modality}_metadata.csv', index=False)
        
        # Save configuration
        with open(save_path / 'data_config.pkl', 'wb') as f:
            pickle.dump(self.config, f)
        
        print(f"Simulated data saved to {save_path}")
        return data_matrices, cell_metadata, feature_metadata
