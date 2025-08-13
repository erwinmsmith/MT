import yaml
import argparse
from typing import Dict, Any
from pathlib import Path

class ConfigLoader:
    """Configuration loader for the multi-omics topic model."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def get_config(self) -> Dict[str, Any]:
        """Get the full configuration dictionary."""
        return self.config
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.config.get('data', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.get('model', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config.get('training', {})
    
    def get_trajectory_config(self) -> Dict[str, Any]:
        """Get trajectory inference configuration."""
        return self.config.get('trajectory', {})
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict:
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, updates)
    
    def save_config(self, save_path: str = None):
        """Save current configuration to file."""
        if save_path is None:
            save_path = self.config_path
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Multi-omics Topic Model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/dataset',
                        help='Directory containing dataset')
    parser.add_argument('--prior_knowledge_dir', type=str, default='data/prior_knowledge',
                        help='Directory containing prior knowledge')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    
    # Model arguments
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--n_topics', type=int, default=20,
                        help='Number of topics')
    parser.add_argument('--enable_cellgraph', action='store_true', default=True,
                        help='Enable cellgraph pathway')
    parser.add_argument('--enable_featuregraph', action='store_true', default=True,
                        help='Enable featuregraph pathway')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--kl_weight', type=float, default=1.0,
                        help='Weight for KL divergence loss')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    
    # Mode arguments
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'simulate'],
                        help='Running mode')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint for evaluation/inference')
    
    return parser.parse_args()
