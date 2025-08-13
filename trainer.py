import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import pickle
from model.multiomics_topic_model import MultiOmicsTopicModel
from data.prior_knowledge import PriorKnowledgeLoader

class MultiOmicsTrainer:
    """
    Trainer for the multi-omics topic model.
    Handles training, validation, and testing with proper logging.
    """
    
    def __init__(self, model: MultiOmicsTopicModel, config: Dict, 
                 data_loaders: Dict[str, DataLoader], device: torch.device):
        """
        Initialize the trainer.
        
        Args:
            model: Multi-omics topic model
            config: Configuration dictionary
            data_loaders: Dictionary of data loaders (train, val, test)
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.data_loaders = data_loaders
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Extract training configuration
        self.training_config = config.get('training', {})
        self.epochs = self.training_config.get('epochs', 100)
        self.learning_rate = float(self.training_config.get('learning_rate', 0.001))
        self.weight_decay = float(self.training_config.get('weight_decay', 1e-5))
        self.gradient_clip = self.training_config.get('gradient_clip', 1.0)
        self.log_interval = self.training_config.get('log_interval', 10)
        self.save_interval = self.training_config.get('save_interval', 20)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=5
        )
        
        # Training history
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_reconstruction_loss': [],
            'val_reconstruction_loss': [],
            'train_kl_loss': [],
            'val_kl_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Create save directory
        self.save_dir = Path('checkpoints')
        self.save_dir.mkdir(exist_ok=True)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_losses = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'kl_loss': 0.0,
            'gene_loss': 0.0,
            'peak_loss': 0.0,
            'protein_loss': 0.0
        }
        
        num_batches = len(self.data_loaders['train'])
        
        for batch_idx, batch in enumerate(self.data_loaders['train']):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            results = self.model(batch, epoch=epoch)
            
            # Extract losses
            losses = results['losses']
            total_loss = losses['total_loss']
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.optimizer.step()
            
            # Accumulate losses
            epoch_losses['total_loss'] += total_loss.item()
            epoch_losses['reconstruction_loss'] += losses['total_reconstruction_loss'].item()
            epoch_losses['kl_loss'] += losses['total_kl_loss'].item()
            
            # Modality-specific losses
            recon_losses = losses['reconstruction_losses']
            if 'gene' in recon_losses:
                epoch_losses['gene_loss'] += recon_losses['gene'].item()
            if 'peak' in recon_losses:
                epoch_losses['peak_loss'] += recon_losses['peak'].item()
            if 'protein' in recon_losses:
                epoch_losses['protein_loss'] += recon_losses['protein'].item()
            
            # Log progress
            if batch_idx % self.log_interval == 0:
                # compute progress
                progress = (batch_idx + 1) / num_batches * 100
                print(f'Epoch {epoch}, Batch {batch_idx+1}/{num_batches} ({progress:.1f}%), '
                      f'Loss: {total_loss.item():.4f}, '
                      f'Recon: {losses["total_reconstruction_loss"].item():.4f}, '
                      f'KL: {losses["total_kl_loss"].item():.4f}, '
                      f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
                # reset --force
                import sys
                sys.stdout.flush()
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        epoch_losses = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'kl_loss': 0.0,
            'gene_loss': 0.0,
            'peak_loss': 0.0,
            'protein_loss': 0.0
        }
        
        num_batches = len(self.data_loaders['val'])
        
        with torch.no_grad():
            for batch in self.data_loaders['val']:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                results = self.model(batch, epoch=epoch)
                
                # Extract losses
                losses = results['losses']
                
                # Accumulate losses
                epoch_losses['total_loss'] += losses['total_loss'].item()
                epoch_losses['reconstruction_loss'] += losses['total_reconstruction_loss'].item()
                epoch_losses['kl_loss'] += losses['total_kl_loss'].item()
                
                # Modality-specific losses
                recon_losses = losses['reconstruction_losses']
                if 'gene' in recon_losses:
                    epoch_losses['gene_loss'] += recon_losses['gene'].item()
                if 'peak' in recon_losses:
                    epoch_losses['peak_loss'] += recon_losses['peak'].item()
                if 'protein' in recon_losses:
                    epoch_losses['protein_loss'] += recon_losses['protein'].item()
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def train(self) -> Dict[str, List]:
        """
        Main training loop.
        
        Returns:
            Training history
        """
        print(f"Starting training for {self.epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()
            
            # Training
            train_losses = self.train_epoch(epoch)
            
            # Validation
            val_losses = self.validate_epoch(epoch)
            
            # Update learning rate scheduler
            self.scheduler.step(val_losses['total_loss'])
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Update history
            self.train_history['epoch'].append(epoch)
            self.train_history['train_loss'].append(train_losses['total_loss'])
            self.train_history['val_loss'].append(val_losses['total_loss'])
            self.train_history['train_reconstruction_loss'].append(train_losses['reconstruction_loss'])
            self.train_history['val_reconstruction_loss'].append(val_losses['reconstruction_loss'])
            self.train_history['train_kl_loss'].append(train_losses['kl_loss'])
            self.train_history['val_kl_loss'].append(val_losses['kl_loss'])
            self.train_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.train_history['epoch_time'].append(epoch_time)
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{self.epochs} Summary:")
            print(f"Time: {epoch_time:.2f}s")
            print(f"Train Loss: {train_losses['total_loss']:.4f} "
                  f"(Recon: {train_losses['reconstruction_loss']:.4f}, "
                  f"KL: {train_losses['kl_loss']:.4f})")
            print(f"Val Loss: {val_losses['total_loss']:.4f} "
                  f"(Recon: {val_losses['reconstruction_loss']:.4f}, "
                  f"KL: {val_losses['kl_loss']:.4f})")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Print modality-specific losses
            modalities = ['gene', 'peak', 'protein']
            train_mod_str = " | ".join([f"{mod}: {train_losses[f'{mod}_loss']:.4f}" 
                                       for mod in modalities if f'{mod}_loss' in train_losses])
            val_mod_str = " | ".join([f"{mod}: {val_losses[f'{mod}_loss']:.4f}" 
                                     for mod in modalities if f'{mod}_loss' in val_losses])
            print(f"Train Modality Losses: {train_mod_str}")
            print(f"Val Modality Losses: {val_mod_str}")
            
            # Save best model
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)
                print(f"New best model saved! Val Loss: {self.best_val_loss:.4f}")
            
            # Save regular checkpoint
            if epoch % self.save_interval == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            print("-" * 80)
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
        
        # Save computed graph structures after training
        self._save_graph_structures_after_training()
        
        return self.train_history
    
    def pretrain_multimodal_fusion(self, pretrain_epochs: int, data_matrices: dict):
        """
        Pretrain the multimodal fusion components to get stable cell embeddings.
        This phase focuses on training encoders and fusion block without graph computation.
        
        Args:
            pretrain_epochs: Number of epochs for pretraining
            data_matrices: Full data matrices for computing stable embeddings
        """

        
        # Set model to training mode
        self.model.train()
        
        # Only train the cellgraph pathway encoders and fusion
        if hasattr(self.model, 'cellgraph_pathway'):
            cellgraph = self.model.cellgraph_pathway
            
            # Disable graph computation during pretraining
            original_graph_computation = True
            if hasattr(cellgraph, '_disable_graph_computation'):
                original_graph_computation = cellgraph._disable_graph_computation
            cellgraph._disable_graph_computation = True
            
            try:
                for epoch in range(1, pretrain_epochs + 1):
                    epoch_start_time = time.time()
                    epoch_losses = {'total_loss': 0.0, 'reconstruction_loss': 0.0}
                    num_batches = 0
                    
                    for batch_idx, batch_data in enumerate(self.data_loaders['train']):
                        # Move batch to device - handle both dict and tuple formats
                        if isinstance(batch_data, dict):
                            batch = batch_data
                        else:
                            # If batch_data is a tuple/list, assume it's (data_dict, indices)
                            batch = batch_data[0] if isinstance(batch_data, (tuple, list)) else batch_data
                        
                        # Ensure batch is moved to device
                        for modality in batch:
                            if hasattr(batch[modality], 'to'):
                                batch[modality] = batch[modality].to(self.device)
                        
                        # Forward pass (simplified, no graph computation)
                        self.optimizer.zero_grad()
                        
                        # Only compute up to fusion stage
                        modality_embeddings = cellgraph.encode_modalities(batch)
                        fused_embeddings = cellgraph.fuse_modalities(modality_embeddings)
                        
                        # Simple reconstruction loss to train encoders/fusion
                        # Use a simplified reconstruction target
                        reconstruction_losses = {}
                        total_recon_loss = 0
                        
                        for modality, target in batch.items():
                            # Skip non-modality keys
                            if modality in ['cell_idx', 'cell_metadata']:
                                continue
                                
                            # Simple linear projection for pretraining
                            if not hasattr(cellgraph, f'_pretrain_proj_{modality}'):
                                proj_layer = torch.nn.Linear(fused_embeddings.shape[-1], target.shape[-1]).to(self.device)
                                setattr(cellgraph, f'_pretrain_proj_{modality}', proj_layer)
                            
                            proj_layer = getattr(cellgraph, f'_pretrain_proj_{modality}')
                            reconstructed = proj_layer(fused_embeddings)
                            recon_loss = F.mse_loss(reconstructed, target)
                            reconstruction_losses[modality] = recon_loss
                            total_recon_loss += recon_loss
                        
                        # Backward pass
                        total_recon_loss.backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                        
                        # Update parameters
                        self.optimizer.step()
                        
                        # Accumulate losses
                        epoch_losses['total_loss'] += total_recon_loss.item()
                        epoch_losses['reconstruction_loss'] += total_recon_loss.item()
                        num_batches += 1
                        
                        if batch_idx % 20 == 0:  # Less frequent output
                            print(f"     Epoch {epoch}/{pretrain_epochs}, Batch {batch_idx+1}, Loss: {total_recon_loss.item():.4f}")
                    
                    # Average losses
                    for key in epoch_losses:
                        epoch_losses[key] /= num_batches
                    
                    epoch_time = time.time() - epoch_start_time
                    print(f"     Epoch {epoch}/{pretrain_epochs} completed: Loss={epoch_losses['total_loss']:.4f}, Time={epoch_time:.1f}s")
                
                # Clean up temporary projection layers
                for modality in ['gene', 'peak', 'protein']:
                    proj_attr = f'_pretrain_proj_{modality}'
                    if hasattr(cellgraph, proj_attr):
                        delattr(cellgraph, proj_attr)
                
            finally:
                # Restore graph computation
                cellgraph._disable_graph_computation = original_graph_computation
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'config': self.config
        }
        
        if is_best:
            save_path = self.save_dir / 'best_model.pth'
            print(f"Saving best model to {save_path}")
        else:
            save_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
            print(f"Saving checkpoint to {save_path}")
        
        torch.save(checkpoint, save_path)
        
        # Also save training history as JSON
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
    
    def _save_graph_structures_after_training(self):
        """
        Verify that full graph structure is saved after training completion.
        The graph should have been precomputed and saved before training started.
        """
        try:
            # Check if cell graph pathway exists and is enabled
            if not hasattr(self.model, 'cellgraph_pathway'):
                print("No cellgraph pathway found in model - skipping graph structure verification")
                return
            
            # Get the cell graph pathway from the model
            cell_graph = self.model.cellgraph_pathway
            
            # Check if full graph structure exists
            if cell_graph.is_graph_fixed():
                print("Full graph structure is available in memory")
                print(f"Graph shape: {cell_graph._fixed_adjacency_matrix.shape}")
                print(f"Graph sparsity: {float((cell_graph._fixed_adjacency_matrix > 1e-6).float().mean()):.4f}")
                
                # Try to save it if not already saved
                try:
                    saved = cell_graph.graph_manager.save_graph_structure(
                        cell_graph._fixed_adjacency_matrix,
                        cell_graph._config_for_graph,
                        additional_info={
                            'source': 'training_verification',
                            'n_cells': cell_graph._fixed_adjacency_matrix.shape[0],
                            'creation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                    )
                    if saved:
                        pass
                except Exception as e:
                    print(f"Graph structure already exists or could not save: {e}")
            else:
                print("Warning: No full graph structure found after training")
                print("This means trajectory inference used fallback batch-wise computation")
                
        except Exception as e:
            print(f"Warning: Could not verify graph structures: {e}")
            # Don't fail the training if graph verification fails
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_history = checkpoint['train_history']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: DataLoader for evaluation
            
        Returns:
            Evaluation results
        """
        self.model.eval()
        total_losses = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'kl_loss': 0.0,
            'gene_loss': 0.0,
            'peak_loss': 0.0,
            'protein_loss': 0.0
        }
        
        num_batches = len(data_loader)
        all_results = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                results = self.model(batch)
                all_results.append(results)
                
                # Extract losses
                losses = results['losses']
                
                # Accumulate losses
                total_losses['total_loss'] += losses['total_loss'].item()
                total_losses['reconstruction_loss'] += losses['total_reconstruction_loss'].item()
                total_losses['kl_loss'] += losses['total_kl_loss'].item()
                
                # Modality-specific losses
                recon_losses = losses['reconstruction_losses']
                if 'gene' in recon_losses:
                    total_losses['gene_loss'] += recon_losses['gene'].item()
                if 'peak' in recon_losses:
                    total_losses['peak_loss'] += recon_losses['peak'].item()
                if 'protein' in recon_losses:
                    total_losses['protein_loss'] += recon_losses['protein'].item()
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= num_batches
        
        return {
            'losses': total_losses,
            'detailed_results': all_results
        }
    
    def generate_report(self) -> str:
        """
        Generate a training report.
        
        Returns:
            Training report as string
        """
        report = []
        report.append("=" * 60)
        report.append("MULTI-OMICS TOPIC MODEL TRAINING REPORT")
        report.append("=" * 60)
        
        # Model configuration
        report.append("\nMODEL CONFIGURATION:")
        report.append(f"Number of topics: {self.config.get('data', {}).get('n_topics', 'N/A')}")
        report.append(f"Embedding dimension: {self.config.get('data', {}).get('embedding_dim', 'N/A')}")
        report.append(f"Enable cellgraph: {self.config.get('enable_cellgraph', True)}")
        report.append(f"Enable featuregraph: {self.config.get('enable_featuregraph', True)}")
        
        # Training configuration
        report.append("\nTRAINING CONFIGURATION:")
        report.append(f"Epochs: {self.epochs}")
        report.append(f"Learning rate: {self.learning_rate}")
        report.append(f"Weight decay: {self.weight_decay}")
        report.append(f"Gradient clipping: {self.gradient_clip}")
        
        # Training results
        if self.train_history['epoch']:
            report.append("\nTRAINING RESULTS:")
            report.append(f"Best epoch: {self.best_epoch}")
            report.append(f"Best validation loss: {self.best_val_loss:.4f}")
            report.append(f"Final training loss: {self.train_history['train_loss'][-1]:.4f}")
            report.append(f"Final validation loss: {self.train_history['val_loss'][-1]:.4f}")
            
            total_time = sum(self.train_history['epoch_time'])
            avg_time = np.mean(self.train_history['epoch_time'])
            report.append(f"Total training time: {total_time:.2f}s ({total_time/60:.2f}m)")
            report.append(f"Average epoch time: {avg_time:.2f}s")
        
        # Model statistics
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        report.append(f"\nMODEL STATISTICS:")
        report.append(f"Total parameters: {total_params:,}")
        report.append(f"Trainable parameters: {trainable_params:,}")
        
        report.append("=" * 60)
        
        return "\n".join(report)
