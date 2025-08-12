import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
from .cellgraph import CellGraphPathway
from .featuregraph import FeatureGraphPathway
from .decoder import MultiOmicsDecoder

class MultiOmicsTopicModel(nn.Module):
    """
    Complete multi-omics topic model integrating both cellgraph and featuregraph pathways.
    
    This model:
    1. Processes multi-omics data through cellgraph pathway to get cell topic distributions (θ_d)
    2. Processes feature relationships through featuregraph pathway to get feature topic distributions (B_d)
    3. Uses both pathways to reconstruct original data through the decoder
    4. Implements VAE framework with KL divergence and reconstruction losses
    """
    
    def __init__(self, config: Dict, prior_knowledge: Dict):
        """
        Initialize the complete multi-omics topic model.
        
        Args:
            config: Configuration dictionary
            prior_knowledge: Prior knowledge including embeddings and graphs
        """
        super(MultiOmicsTopicModel, self).__init__()
        
        self.config = config
        self.prior_knowledge = prior_knowledge
        
        # Extract training configuration
        self.training_config = config.get('training', {})
        self.kl_weight = self.training_config.get('kl_weight', 1.0)
        self.reconstruction_weight = self.training_config.get('reconstruction_weight', 1.0)
        
        # Check which pathways are enabled
        self.enable_cellgraph = config.get('enable_cellgraph', True)
        self.enable_featuregraph = config.get('enable_featuregraph', True)
        
        if not (self.enable_cellgraph or self.enable_featuregraph):
            raise ValueError("At least one pathway (cellgraph or featuregraph) must be enabled")
        
        # Initialize pathways
        if self.enable_cellgraph:
            self.cellgraph_pathway = CellGraphPathway(config)
        
        if self.enable_featuregraph:
            self.featuregraph_pathway = FeatureGraphPathway(config, prior_knowledge)
        
        # Decoder for reconstruction
        self.decoder = MultiOmicsDecoder(config)
        
        # Loss weights that can be learned
        self.modality_loss_weights = nn.ParameterDict({
            'gene': nn.Parameter(torch.ones(1)),
            'peak': nn.Parameter(torch.ones(1)),
            'protein': nn.Parameter(torch.ones(1))
        })
        
        # Store intermediate results for analysis
        self.intermediate_results = {}
    
    def forward(self, batch: Dict[str, torch.Tensor], 
                epoch: Optional[int] = None) -> Dict[str, Any]:
        """
        Forward pass through the complete model.
        
        Args:
            batch: Dictionary containing multi-omics data
            epoch: Current training epoch (for annealing)
            
        Returns:
            Dictionary containing all outputs and losses
        """
        results = {}
        
        # Clear previous intermediate results
        self.clear_intermediate_results()
        
        # Cellgraph pathway
        if self.enable_cellgraph:
            cellgraph_results = self.cellgraph_pathway(batch)
            results['cellgraph'] = cellgraph_results
            
            # Extract cell topic distribution
            cell_topic_distribution = cellgraph_results['topic_distribution']
            cell_topic_mean = cellgraph_results['topic_mean']
            cell_topic_logvar = cellgraph_results['topic_logvar']
        else:
            # If cellgraph is disabled, use uniform distribution
            batch_size = list(batch.values())[0].shape[0]
            n_topics = self.config.get('data', {}).get('n_topics', 20)
            cell_topic_distribution = torch.ones(batch_size, n_topics, device=next(self.parameters()).device) / n_topics
            cell_topic_mean = torch.zeros(batch_size, n_topics, device=next(self.parameters()).device)
            cell_topic_logvar = torch.zeros(batch_size, n_topics, device=next(self.parameters()).device)
        
        # Featuregraph pathway
        if self.enable_featuregraph:
            featuregraph_results = self.featuregraph_pathway()
            results['featuregraph'] = featuregraph_results
            
            # Extract feature topic distributions
            feature_topic_distributions = featuregraph_results['modality_topic_distributions']
        else:
            # If featuregraph is disabled, create dummy distributions
            feature_topic_distributions = self._create_dummy_feature_distributions()
        
        # Decoder - reconstruct data
        reconstructed_data = self.decoder(cell_topic_distribution, feature_topic_distributions)
        results['reconstructed_data'] = reconstructed_data
        
        # Compute losses
        losses = self.compute_losses(
            batch, reconstructed_data, 
            cell_topic_mean, cell_topic_logvar,
            feature_topic_distributions,
            epoch
        )
        results['losses'] = losses
        
        # Store intermediate results
        self.intermediate_results['cellgraph'] = results.get('cellgraph', {})
        self.intermediate_results['featuregraph'] = results.get('featuregraph', {})
        self.intermediate_results['decoder'] = self.decoder.get_intermediate_results()
        
        return results
    
    def _create_dummy_feature_distributions(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Create dummy feature distributions when featuregraph is disabled."""
        device = next(self.parameters()).device
        n_topics = self.config.get('data', {}).get('n_topics', 20)
        
        dummy_distributions = {}
        for modality, n_features in [('gene', self.config.get('data', {}).get('n_genes', 3000)),
                                   ('peak', self.config.get('data', {}).get('n_peaks', 5000)),
                                   ('protein', self.config.get('data', {}).get('n_proteins', 1000))]:
            # Uniform distribution
            topics = torch.ones(n_features, n_topics, device=device) / n_topics
            mean = torch.zeros(n_features, n_topics, device=device)
            logvar = torch.zeros(n_features, n_topics, device=device)
            dummy_distributions[modality] = (topics, mean, logvar)
        
        return dummy_distributions
    
    def compute_losses(self, original_data: Dict[str, torch.Tensor],
                      reconstructed_data: Dict[str, torch.Tensor],
                      cell_topic_mean: torch.Tensor, cell_topic_logvar: torch.Tensor,
                      feature_topic_distributions: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                      epoch: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Compute all losses for the VAE framework.
        
        Args:
            original_data: Original input data
            reconstructed_data: Reconstructed data from decoder
            cell_topic_mean: Mean of cell topic distribution
            cell_topic_logvar: Log variance of cell topic distribution
            feature_topic_distributions: Feature topic distributions
            epoch: Current epoch for annealing
            
        Returns:
            Dictionary of computed losses
        """
        losses = {}
        device = next(self.parameters()).device
        
        # Reconstruction losses for each modality
        reconstruction_losses = {}
        total_reconstruction_loss = torch.tensor(0.0, device=device)
        
        for modality in reconstructed_data.keys():
            if modality in original_data:
                # Choose loss type based on modality
                if modality == 'gene':
                    # Poisson loss for count data
                    recon_loss = F.poisson_nll_loss(
                        torch.clamp(reconstructed_data[modality], min=1e-8),
                        original_data[modality], 
                        reduction='mean'
                    )
                elif modality == 'peak':
                    # Binary cross-entropy for binary accessibility data
                    recon_loss = F.binary_cross_entropy_with_logits(
                        reconstructed_data[modality],
                        original_data[modality], 
                        reduction='mean'
                    )
                elif modality == 'protein':
                    # MSE loss for protein expression
                    recon_loss = F.mse_loss(
                        reconstructed_data[modality],
                        original_data[modality], 
                        reduction='mean'
                    )
                else:
                    # Default to MSE
                    recon_loss = F.mse_loss(
                        reconstructed_data[modality],
                        original_data[modality], 
                        reduction='mean'
                    )
                
                # Apply modality-specific weight
                weighted_loss = recon_loss * torch.clamp(self.modality_loss_weights[modality], min=0.1)
                reconstruction_losses[modality] = weighted_loss
                total_reconstruction_loss = total_reconstruction_loss + weighted_loss
        
        losses['reconstruction_losses'] = reconstruction_losses
        losses['total_reconstruction_loss'] = total_reconstruction_loss
        
        # KL divergence for cell topics (if cellgraph is enabled)
        if self.enable_cellgraph:
            cell_kl_loss = self._compute_kl_divergence(cell_topic_mean, cell_topic_logvar)
            losses['cell_kl_loss'] = cell_kl_loss
        else:
            losses['cell_kl_loss'] = torch.tensor(0.0, device=device)
        
        # KL divergence for feature topics (if featuregraph is enabled)
        feature_kl_losses = {}
        total_feature_kl_loss = torch.tensor(0.0, device=device)
        
        if self.enable_featuregraph:
            for modality, (_, mean, logvar) in feature_topic_distributions.items():
                feature_kl = self._compute_kl_divergence(mean, logvar)
                feature_kl_losses[modality] = feature_kl
                total_feature_kl_loss += feature_kl
        
        losses['feature_kl_losses'] = feature_kl_losses
        losses['total_feature_kl_loss'] = total_feature_kl_loss
        
        # Total KL loss
        total_kl_loss = losses['cell_kl_loss'] + losses['total_feature_kl_loss']
        losses['total_kl_loss'] = total_kl_loss
        
        # Apply KL annealing if epoch is provided
        kl_weight = self._get_kl_weight(epoch)
        losses['kl_weight'] = torch.tensor(kl_weight, device=device)
        
        # Total loss
        total_loss = (self.reconstruction_weight * total_reconstruction_loss + 
                     kl_weight * total_kl_loss)
        losses['total_loss'] = total_loss
        
        return losses
    
    def _compute_kl_divergence(self, mean: torch.Tensor, logvar: torch.Tensor,
                              prior_mean: float = 0.0, prior_std: float = 1.0) -> torch.Tensor:
        """
        Compute KL divergence between approximate posterior and prior.
        
        Args:
            mean: Mean of approximate posterior
            logvar: Log variance of approximate posterior
            prior_mean: Mean of prior distribution
            prior_std: Standard deviation of prior distribution
            
        Returns:
            KL divergence
        """
        prior_var = prior_std ** 2
        kl_div = 0.5 * (
            torch.log(torch.tensor(prior_var, device=mean.device)) - logvar +
            (torch.exp(logvar) + (mean - prior_mean) ** 2) / prior_var - 1
        )
        
        return kl_div.sum(dim=-1).mean()
    
    def _get_kl_weight(self, epoch: Optional[int]) -> float:
        """
        Get KL weight with optional annealing.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            KL weight
        """
        if epoch is None:
            return self.kl_weight
        
        # KL annealing: gradually increase KL weight
        warmup_epochs = self.training_config.get('warmup_epochs', 10)
        
        if epoch < warmup_epochs:
            # Linear annealing from 0 to kl_weight
            return self.kl_weight * (epoch / warmup_epochs)
        else:
            return self.kl_weight
    
    def generate_samples(self, n_samples: int, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """
        Generate new samples from the learned topic model.
        
        Args:
            n_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            Dictionary of generated samples for each modality
        """
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        with torch.no_grad():
            n_topics = self.config.get('data', {}).get('n_topics', 20)
            
            # Sample from prior distribution
            cell_topic_distribution = torch.randn(n_samples, n_topics, device=device)
            cell_topic_distribution = F.softmax(cell_topic_distribution, dim=-1)
            
            # Get feature topic distributions
            if self.enable_featuregraph:
                featuregraph_results = self.featuregraph_pathway()
                feature_topic_distributions = featuregraph_results['modality_topic_distributions']
            else:
                feature_topic_distributions = self._create_dummy_feature_distributions()
            
            # Generate samples through decoder
            generated_samples = self.decoder(cell_topic_distribution, feature_topic_distributions)
            
        return generated_samples
    
    def get_topic_interpretations(self) -> Dict[str, Any]:
        """
        Get interpretations of learned topics.
        
        Returns:
            Dictionary containing topic interpretations
        """
        interpretations = {}
        
        # Get topic embeddings from decoder
        topic_embeddings = self.decoder.get_topic_embeddings()
        interpretations['topic_embeddings'] = topic_embeddings.detach()
        
        # Get embedding-to-feature matrices
        embedding_to_feature = self.decoder.get_embedding_to_feature_matrices()
        interpretations['embedding_to_feature_matrices'] = {
            k: v.detach() for k, v in embedding_to_feature.items()
        }
        
        # Get feature topic distributions if available
        if self.enable_featuregraph:
            featuregraph_results = self.featuregraph_pathway()
            feature_distributions = featuregraph_results['modality_topic_distributions']
            interpretations['feature_topic_distributions'] = {
                modality: (dist[0].detach(), dist[1].detach(), dist[2].detach())
                for modality, dist in feature_distributions.items()
            }
        
        return interpretations
    
    def get_intermediate_results(self) -> Dict[str, Any]:
        """Get all intermediate results for analysis."""
        return self.intermediate_results.copy()
    
    def clear_intermediate_results(self):
        """Clear stored intermediate results."""
        self.intermediate_results.clear()
        if hasattr(self, 'cellgraph_pathway'):
            self.cellgraph_pathway.clear_intermediate_results()
        if hasattr(self, 'featuregraph_pathway'):
            self.featuregraph_pathway.clear_intermediate_results()
        self.decoder.clear_intermediate_results()
