import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import pickle

class MultiOmicsDataset(Dataset):
    """Dataset class for multi-omics data."""
    
    def __init__(self, data_matrices: Dict[str, np.ndarray], 
                 cell_metadata: Optional[pd.DataFrame] = None,
                 feature_metadata: Optional[Dict[str, pd.DataFrame]] = None,
                 transform: Optional[callable] = None):
        """
        Initialize multi-omics dataset.
        
        Args:
            data_matrices: Dictionary with keys 'gene', 'peak', 'protein' and matrix values
            cell_metadata: DataFrame with cell metadata
            feature_metadata: Dictionary of DataFrames with feature metadata for each modality
            transform: Optional transform to apply to data
        """
        self.data_matrices = data_matrices
        self.cell_metadata = cell_metadata
        self.feature_metadata = feature_metadata or {}
        self.transform = transform
        
        # Validate that all matrices have the same number of cells
        n_cells_list = [matrix.shape[0] for matrix in data_matrices.values()]
        if not all(n == n_cells_list[0] for n in n_cells_list):
            raise ValueError("All data matrices must have the same number of cells")
        
        self.n_cells = n_cells_list[0]
        self.modalities = list(data_matrices.keys())
        
    def __len__(self) -> int:
        return self.n_cells
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single cell's data across all modalities."""
        sample = {}
        
        for modality in self.modalities:
            data = self.data_matrices[modality][idx].copy()
            
            if self.transform:
                data = self.transform(data)
            
            sample[modality] = torch.FloatTensor(data)
        
        # Add cell index
        sample['cell_idx'] = torch.LongTensor([idx])
        
        # Add metadata if available
        if self.cell_metadata is not None:
            sample['cell_metadata'] = self.cell_metadata.iloc[idx].to_dict()
        
        return sample
    
    def get_modality_shapes(self) -> Dict[str, Tuple[int, int]]:
        """Get the shape of each modality's data matrix."""
        return {modality: matrix.shape for modality, matrix in self.data_matrices.items()}
    
    def get_feature_names(self, modality: str) -> Optional[List[str]]:
        """Get feature names for a specific modality."""
        if modality in self.feature_metadata:
            if 'gene_symbol' in self.feature_metadata[modality].columns:
                return self.feature_metadata[modality]['gene_symbol'].tolist()
            elif 'protein_symbol' in self.feature_metadata[modality].columns:
                return self.feature_metadata[modality]['protein_symbol'].tolist()
            elif f'{modality}_id' in self.feature_metadata[modality].columns:
                return self.feature_metadata[modality][f'{modality}_id'].tolist()
        return None

class MultiOmicsDataLoader:
    """Data loader for multi-omics data with support for simulation and loading."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = Path(config.get('data_dir', 'data/dataset'))
        self.batch_size = config.get('batch_size', 64)
        self.num_workers = config.get('num_workers', 4)
        
    def load_data(self) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Load multi-omics data from files.
        
        Returns:
            Tuple of (data_matrices, cell_metadata, feature_metadata)
        """
        data_matrices = {}
        
        # Load data matrices
        for modality in ['gene', 'peak', 'protein']:
            matrix_path = self.data_dir / f'{modality}_matrix.npy'
            if matrix_path.exists():
                data_matrices[modality] = np.load(matrix_path)
            else:
                print(f"Warning: {modality} matrix not found at {matrix_path}")
        
        # Load cell metadata
        cell_metadata_path = self.data_dir / 'cell_metadata.csv'
        cell_metadata = None
        if cell_metadata_path.exists():
            cell_metadata = pd.read_csv(cell_metadata_path)
        
        # Load feature metadata
        feature_metadata = {}
        for modality in data_matrices.keys():
            metadata_path = self.data_dir / f'{modality}_metadata.csv'
            if metadata_path.exists():
                feature_metadata[modality] = pd.read_csv(metadata_path)
        
        return data_matrices, cell_metadata, feature_metadata
    
    def create_data_loaders(self, data_matrices: Dict[str, np.ndarray],
                          cell_metadata: Optional[pd.DataFrame] = None,
                          feature_metadata: Optional[Dict[str, pd.DataFrame]] = None,
                          train_ratio: float = 0.8,
                          val_ratio: float = 0.1) -> Dict[str, DataLoader]:
        """
        Create train/validation/test data loaders.
        
        Args:
            data_matrices: Dictionary of data matrices
            cell_metadata: Cell metadata DataFrame
            feature_metadata: Feature metadata dictionary
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            
        Returns:
            Dictionary of DataLoaders for train/val/test splits
        """
        n_cells = list(data_matrices.values())[0].shape[0]
        
        # Create indices for splits
        indices = np.arange(n_cells)
        np.random.shuffle(indices)
        
        train_end = int(train_ratio * n_cells)
        val_end = int((train_ratio + val_ratio) * n_cells)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # Create datasets for each split
        datasets = {}
        for split_name, split_indices in [('train', train_indices), 
                                        ('val', val_indices), 
                                        ('test', test_indices)]:
            # Extract data for this split
            split_matrices = {}
            for modality, matrix in data_matrices.items():
                split_matrices[modality] = matrix[split_indices]
            
            split_cell_metadata = None
            if cell_metadata is not None:
                split_cell_metadata = cell_metadata.iloc[split_indices].reset_index(drop=True)
            
            datasets[split_name] = MultiOmicsDataset(
                data_matrices=split_matrices,
                cell_metadata=split_cell_metadata,
                feature_metadata=feature_metadata
            )
        
        # Create data loaders
        data_loaders = {}
        for split_name, dataset in datasets.items():
            shuffle = (split_name == 'train')
            data_loaders[split_name] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=(split_name == 'train')
            )
        
        return data_loaders
    
    def get_data_info(self, data_matrices: Dict[str, np.ndarray]) -> Dict[str, any]:
        """Get information about the loaded data."""
        info = {
            'n_cells': list(data_matrices.values())[0].shape[0],
            'modalities': list(data_matrices.keys()),
            'shapes': {}
        }
        
        for modality, matrix in data_matrices.items():
            info['shapes'][modality] = {
                'n_cells': matrix.shape[0],
                'n_features': matrix.shape[1],
                'dtype': str(matrix.dtype),
                'mean': float(np.mean(matrix)),
                'std': float(np.std(matrix)),
                'min': float(np.min(matrix)),
                'max': float(np.max(matrix))
            }
        
        return info

def collate_multiomics_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for multi-omics data.
    
    Args:
        batch: List of samples from MultiOmicsDataset
        
    Returns:
        Batched data dictionary
    """
    if not batch:
        return {}
    
    # Get all keys from the first sample
    keys = batch[0].keys()
    
    batched_data = {}
    for key in keys:
        if key == 'cell_metadata':
            # Handle metadata separately
            batched_data[key] = [sample[key] for sample in batch]
        else:
            # Stack tensors
            batched_data[key] = torch.stack([sample[key] for sample in batch])
    
    return batched_data
