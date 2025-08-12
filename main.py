#!/usr/bin/env python3
"""
Multi-omics Topic Model - Main Interface

This script provides a command-line interface for training and evaluating 
the multi-omics topic model with both cellgraph and featuregraph pathways.

Usage:
    python main.py --mode train --config config/config.yaml
    python main.py --mode eval --config config/config.yaml --checkpoint checkpoints/best_model.pth
    python main.py --mode simulate --config config/config.yaml
"""

import argparse
import torch
import numpy as np
import random
from pathlib import Path
import sys
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.config_loader import ConfigLoader, parse_args
from data.dataset_simulator import DatasetSimulator
from data.dataloader import MultiOmicsDataLoader
from data.prior_knowledge import PriorKnowledgeLoader
from model.multiomics_topic_model import MultiOmicsTopicModel
from trainer import MultiOmicsTrainer

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def simulate_data(config_loader: ConfigLoader):
    """Simulate multi-omics data and prior knowledge."""
    print("=" * 60)
    print("SIMULATING MULTI-OMICS DATA")
    print("=" * 60)
    
    data_config = config_loader.get_data_config()
    
    # Create dataset simulator
    dataset_simulator = DatasetSimulator(data_config)
    
    # Simulate and save data
    print("Simulating multi-omics dataset...")
    data_matrices, cell_metadata, feature_metadata = dataset_simulator.save_simulated_data(
        data_config['data_dir']
    )
    
    print(f"Simulated data shapes:")
    for modality, matrix in data_matrices.items():
        print(f"  {modality}: {matrix.shape}")
    
    # Create prior knowledge loader and simulate prior knowledge
    prior_knowledge_loader = PriorKnowledgeLoader(data_config)
    
    print("Simulating prior knowledge...")
    prior_knowledge = prior_knowledge_loader.save_prior_knowledge(
        n_genes=data_config['n_genes'],
        n_peaks=data_config['n_peaks'],
        n_proteins=data_config['n_proteins'],
        embedding_dim=data_config['embedding_dim']
    )
    
    print("Data simulation completed successfully!")
    return data_matrices, cell_metadata, feature_metadata, prior_knowledge

def load_data(config_loader: ConfigLoader):
    """Load multi-omics data and prior knowledge."""
    print("=" * 60)
    print("LOADING MULTI-OMICS DATA")
    print("=" * 60)
    
    data_config = config_loader.get_data_config()
    
    # Load dataset
    data_loader = MultiOmicsDataLoader(data_config)
    
    try:
        data_matrices, cell_metadata, feature_metadata = data_loader.load_data()
        print("Successfully loaded existing data.")
    except FileNotFoundError:
        print("Data files not found. Simulating data...")
        data_matrices, cell_metadata, feature_metadata, _ = simulate_data(config_loader)
    
    # Display data information
    data_info = data_loader.get_data_info(data_matrices)
    print("\nData Information:")
    print(f"Number of cells: {data_info['n_cells']}")
    print(f"Modalities: {data_info['modalities']}")
    for modality, info in data_info['shapes'].items():
        print(f"  {modality}: {info['n_features']} features, "
              f"mean={info['mean']:.3f}, std={info['std']:.3f}")
    
    # Load prior knowledge
    prior_knowledge_loader = PriorKnowledgeLoader(data_config)
    
    try:
        prior_knowledge = prior_knowledge_loader.load_prior_knowledge()
        print("Successfully loaded prior knowledge.")
    except FileNotFoundError:
        print("Prior knowledge not found. Simulating...")
        prior_knowledge = prior_knowledge_loader.save_prior_knowledge(
            n_genes=data_config['n_genes'],
            n_peaks=data_config['n_peaks'],
            n_proteins=data_config['n_proteins'],
            embedding_dim=data_config['embedding_dim']
        )
    
    print(f"Prior knowledge components: {list(prior_knowledge.keys())}")
    
    return data_matrices, cell_metadata, feature_metadata, prior_knowledge, data_loader

def create_model(config_loader: ConfigLoader, prior_knowledge: dict, device: torch.device):
    """Create and initialize the multi-omics topic model."""
    print("=" * 60)
    print("CREATING MODEL")
    print("=" * 60)
    
    config = config_loader.get_config()
    
    # Create model
    model = MultiOmicsTopicModel(config, prior_knowledge)
    model.to(device)
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    return model

def train_model(config_loader: ConfigLoader, model: MultiOmicsTopicModel, 
                data_loader: MultiOmicsDataLoader, data_matrices: dict,
                cell_metadata, feature_metadata, device: torch.device):
    """Train the multi-omics topic model."""
    print("=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)
    
    # Create data loaders
    data_loaders = data_loader.create_data_loaders(
        data_matrices, cell_metadata, feature_metadata
    )
    
    print(f"Data splits:")
    for split, loader in data_loaders.items():
        print(f"  {split}: {len(loader.dataset)} samples, {len(loader)} batches")
    
    # Create trainer
    trainer = MultiOmicsTrainer(model, config_loader.get_config(), data_loaders, device)
    
    # Train model
    train_history = trainer.train()
    
    # Generate and save report
    report = trainer.generate_report()
    print("\n" + report)
    
    # Save report to file
    report_path = Path('checkpoints') / 'training_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Training report saved to {report_path}")
    
    return trainer, train_history

def evaluate_model(config_loader: ConfigLoader, checkpoint_path: str, 
                  data_loader: MultiOmicsDataLoader, data_matrices: dict,
                  cell_metadata, feature_metadata, prior_knowledge: dict, 
                  device: torch.device):
    """Evaluate a trained model."""
    print("=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)
    
    # Create model
    model = create_model(config_loader, prior_knowledge, device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {checkpoint_path}")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    # Create data loaders
    data_loaders = data_loader.create_data_loaders(
        data_matrices, cell_metadata, feature_metadata
    )
    
    # Create trainer for evaluation
    trainer = MultiOmicsTrainer(model, config_loader.get_config(), data_loaders, device)
    
    # Evaluate on test set
    test_results = trainer.evaluate(data_loaders['test'])
    
    print("\nTest Results:")
    for loss_name, loss_value in test_results['losses'].items():
        print(f"  {loss_name}: {loss_value:.4f}")
    
    # Generate samples
    print("\nGenerating samples...")
    samples = model.generate_samples(n_samples=10, device=device)
    
    print("Generated sample shapes:")
    for modality, sample in samples.items():
        print(f"  {modality}: {sample.shape}")
    
    # Get topic interpretations
    print("\nExtracting topic interpretations...")
    interpretations = model.get_topic_interpretations()
    
    print("Topic interpretation components:")
    for component, data in interpretations.items():
        if isinstance(data, torch.Tensor):
            print(f"  {component}: {data.shape}")
        elif isinstance(data, dict):
            print(f"  {component}: {list(data.keys())}")
    
    # Save evaluation results
    eval_results = {
        'test_losses': test_results['losses'],
        'checkpoint_path': checkpoint_path,
        'best_val_loss': checkpoint['best_val_loss'],
        'model_config': config_loader.get_config()
    }
    
    eval_path = Path('checkpoints') / 'evaluation_results.json'
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"Evaluation results saved to {eval_path}")
    
    return test_results, samples, interpretations

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config_loader = ConfigLoader(args.config)
    
    # Update config with command line arguments
    config_updates = {}
    if args.data_dir:
        config_updates['data'] = {'data_dir': args.data_dir}
    if args.batch_size:
        config_updates['data'] = config_updates.get('data', {})
        config_updates['data']['batch_size'] = args.batch_size
    if args.epochs:
        config_updates['training'] = {'epochs': args.epochs}
    if args.lr:
        config_updates['training'] = config_updates.get('training', {})
        config_updates['training']['learning_rate'] = args.lr
    
    if config_updates:
        config_loader.update_config(config_updates)
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    print(f"Configuration loaded from: {args.config}")
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Random seed: {args.seed}")
    
    # Execute based on mode
    if args.mode == 'simulate':
        simulate_data(config_loader)
    
    elif args.mode == 'train':
        # Load/simulate data
        data_matrices, cell_metadata, feature_metadata, prior_knowledge, data_loader = load_data(config_loader)
        
        # Create model
        model = create_model(config_loader, prior_knowledge, device)
        
        # Train model
        trainer, train_history = train_model(
            config_loader, model, data_loader, data_matrices,
            cell_metadata, feature_metadata, device
        )
        
        print("Training completed successfully!")
    
    elif args.mode == 'eval':
        if not hasattr(args, 'checkpoint') or not args.checkpoint:
            print("Error: Checkpoint path required for evaluation mode")
            print("Use: python main.py --mode eval --checkpoint path/to/checkpoint.pth")
            return
        
        # Load data
        data_matrices, cell_metadata, feature_metadata, prior_knowledge, data_loader = load_data(config_loader)
        
        # Evaluate model
        test_results, samples, interpretations = evaluate_model(
            config_loader, args.checkpoint, data_loader, data_matrices,
            cell_metadata, feature_metadata, prior_knowledge, device
        )
        
        print("Evaluation completed successfully!")
    
    else:
        print(f"Unknown mode: {args.mode}")
        print("Available modes: train, eval, simulate")

if __name__ == "__main__":
    # Add checkpoint argument to parser
    parser = argparse.ArgumentParser(description='Multi-omics Topic Model')
    parser.add_argument('--checkpoint', type=str, 
                        help='Path to checkpoint file for evaluation')
    
    # Parse known args to get checkpoint
    known_args, unknown_args = parser.parse_known_args()
    
    # Add checkpoint to args if provided
    args = parse_args()
    if known_args.checkpoint:
        args.checkpoint = known_args.checkpoint
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
