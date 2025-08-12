import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import networkx as nx

class HeterogeneousDirectedGNN(nn.Module):
    """
    Heterogeneous Directed Graph Neural Network for processing feature graphs.
    Handles different node types (gene, peak, protein) and edge types.
    """
    
    def __init__(self, node_types: List[str], edge_types: List[str],
                 input_dims: Dict[str, int], hidden_dim: int, output_dim: int,
                 n_layers: int = 3, n_heads: int = 4, dropout: float = 0.1,
                 activation: str = 'relu', use_residual: bool = True):
        """
        Initialize heterogeneous directed GNN.
        
        Args:
            node_types: List of node types (e.g., ['gene', 'peak', 'protein'])
            edge_types: List of edge types (e.g., ['gene2peak', 'gene2protein', 'peak2protein'])
            input_dims: Dictionary mapping node types to input dimensions
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            n_layers: Number of GNN layers
            n_heads: Number of attention heads
            dropout: Dropout rate
            activation: Activation function
            use_residual: Whether to use residual connections
        """
        super(HeterogeneousDirectedGNN, self).__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.use_residual = use_residual
        
        # Node type embeddings
        self.node_type_embeddings = nn.ParameterDict({
            node_type: nn.Parameter(torch.randn(hidden_dim))
            for node_type in node_types
        })
        
        # Input projections for each node type
        self.input_projections = nn.ModuleDict({
            node_type: nn.Linear(input_dims[node_type], hidden_dim)
            for node_type in node_types
        })
        
        # Heterogeneous GNN layers
        self.gnn_layers = nn.ModuleList([
            HeterogeneousGNNLayer(
                node_types, edge_types, hidden_dim, n_heads, dropout, activation
            ) for _ in range(n_layers)
        ])
        
        # Output projections for each node type
        self.output_projections = nn.ModuleDict({
            node_type: nn.Linear(hidden_dim, output_dim)
            for node_type in node_types
        })
        
        # Layer normalizations
        self.layer_norms = nn.ModuleList([
            nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_dim)
                for node_type in node_types
            }) for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        # Initialize node type embeddings
        for node_type in self.node_types:
            nn.init.normal_(self.node_type_embeddings[node_type], std=0.02)
        
        # Initialize other modules
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, node_features: Dict[str, torch.Tensor],
                edge_indices: Dict[str, torch.Tensor],
                edge_weights: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through heterogeneous directed GNN.
        
        Args:
            node_features: Dictionary mapping node types to feature tensors
            edge_indices: Dictionary mapping edge types to edge index tensors
            edge_weights: Optional dictionary of edge weights
            
        Returns:
            Dictionary mapping node types to updated feature tensors
        """
        # Input projection and add node type embeddings
        x = {}
        for node_type in self.node_types:
            if node_type in node_features:
                projected = self.input_projections[node_type](node_features[node_type])
                # Add node type embedding
                type_embedding = self.node_type_embeddings[node_type].unsqueeze(0).expand_as(projected)
                x[node_type] = projected + type_embedding
                x[node_type] = self.dropout(x[node_type])
        
        # Apply GNN layers
        for i, (gnn_layer, layer_norm_dict) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            residual = {node_type: features.clone() for node_type, features in x.items()} if self.use_residual else None
            
            # Apply GNN layer
            x = gnn_layer(x, edge_indices, edge_weights)
            
            # Layer normalization
            for node_type in x.keys():
                x[node_type] = layer_norm_dict[node_type](x[node_type])
            
            # Residual connections
            if residual is not None:
                for node_type in x.keys():
                    if node_type in residual and x[node_type].shape == residual[node_type].shape:
                        x[node_type] = x[node_type] + residual[node_type]
            
            # Dropout
            for node_type in x.keys():
                x[node_type] = self.dropout(x[node_type])
        
        # Output projections
        output = {}
        for node_type in x.keys():
            output[node_type] = self.output_projections[node_type](x[node_type])
        
        return output

class HeterogeneousGNNLayer(nn.Module):
    """Single layer of heterogeneous directed GNN."""
    
    def __init__(self, node_types: List[str], edge_types: List[str],
                 hidden_dim: int, n_heads: int = 4, dropout: float = 0.1,
                 activation: str = 'relu'):
        super(HeterogeneousGNNLayer, self).__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        # Multi-head attention for each edge type
        self.edge_attentions = nn.ModuleDict({
            edge_type: MultiHeadDirectedAttention(
                hidden_dim, n_heads, dropout
            ) for edge_type in edge_types
        })
        
        # Self-attention for each node type
        self.self_attentions = nn.ModuleDict({
            node_type: nn.MultiheadAttention(
                hidden_dim, n_heads, dropout=dropout, batch_first=True
            ) for node_type in node_types
        })
        
        # Aggregation weights for combining different edge types
        self.edge_type_weights = nn.ParameterDict({
            node_type: nn.Parameter(torch.ones(len(edge_types)))
            for node_type in node_types
        })
        
        # Output projections
        self.output_projections = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU() if activation == 'gelu' else nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim)
            ) for node_type in node_types
        })
    
    def forward(self, node_features: Dict[str, torch.Tensor],
                edge_indices: Dict[str, torch.Tensor],
                edge_weights: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of heterogeneous GNN layer.
        
        Args:
            node_features: Dictionary of node features
            edge_indices: Dictionary of edge indices
            edge_weights: Optional dictionary of edge weights
            
        Returns:
            Updated node features
        """
        # Initialize output
        updated_features = {node_type: [] for node_type in self.node_types}
        
        # Process each edge type
        for edge_type in self.edge_types:
            if edge_type in edge_indices:
                edge_idx = edge_indices[edge_type]
                edge_weight = edge_weights.get(edge_type) if edge_weights else None
                
                # Apply attention for this edge type
                edge_output = self.edge_attentions[edge_type](
                    node_features, edge_idx, edge_weight, edge_type
                )
                
                # Collect outputs for target node types
                for node_type in self.node_types:
                    if node_type in edge_output:
                        updated_features[node_type].append(edge_output[node_type])
        
        # Aggregate and combine features
        final_features = {}
        for node_type in self.node_types:
            if node_type in node_features:
                # Self-attention
                self_attended, _ = self.self_attentions[node_type](
                    node_features[node_type].unsqueeze(1),
                    node_features[node_type].unsqueeze(1),
                    node_features[node_type].unsqueeze(1)
                )
                self_attended = self_attended.squeeze(1)
                
                # Combine with edge-based updates
                combined_features = self_attended
                if updated_features[node_type]:
                    # Weighted combination of different edge type contributions
                    edge_contributions = torch.stack(updated_features[node_type], dim=0)
                    weights = F.softmax(self.edge_type_weights[node_type][:len(updated_features[node_type])], dim=0)
                    weighted_edges = torch.sum(weights.view(-1, 1, 1) * edge_contributions, dim=0)
                    combined_features = combined_features + weighted_edges
                
                # Apply output projection
                final_features[node_type] = self.output_projections[node_type](combined_features)
        
        return final_features

class MultiHeadDirectedAttention(nn.Module):
    """Multi-head attention for directed edges in heterogeneous graphs."""
    
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float = 0.1):
        super(MultiHeadDirectedAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        # Query, Key, Value projections
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, node_features: Dict[str, torch.Tensor],
                edge_indices: torch.Tensor, edge_weights: Optional[torch.Tensor],
                edge_type: str) -> Dict[str, torch.Tensor]:
        """
        Apply multi-head attention for a specific edge type.
        
        Args:
            node_features: Dictionary of node features
            edge_indices: Edge indices (2, n_edges) - [source_nodes, target_nodes]
            edge_weights: Optional edge weights
            edge_type: Type of edge being processed
            
        Returns:
            Dictionary of updated node features
        """
        if edge_indices.size(1) == 0:  # No edges of this type
            return {}
        
        # Determine source and target node types from edge type
        source_type, target_type = self._parse_edge_type(edge_type)
        
        if source_type not in node_features or target_type not in node_features:
            return {}
        
        source_features = node_features[source_type]
        target_features = node_features[target_type]
        
        # Get source and target indices
        source_idx = edge_indices[0]
        target_idx = edge_indices[1]
        
        # Project to Q, K, V
        queries = self.query_projection(target_features)  # Target nodes query
        keys = self.key_projection(source_features)       # Source nodes provide keys
        values = self.value_projection(source_features)   # Source nodes provide values
        
        # Reshape for multi-head attention
        batch_size = queries.size(0)
        queries = queries.view(batch_size, self.n_heads, self.head_dim)
        keys = keys.view(keys.size(0), self.n_heads, self.head_dim)
        values = values.view(values.size(0), self.n_heads, self.head_dim)
        
        # Compute attention scores
        source_keys = keys[source_idx]  # (n_edges, n_heads, head_dim)
        source_values = values[source_idx]  # (n_edges, n_heads, head_dim)
        target_queries = queries[target_idx]  # (n_edges, n_heads, head_dim)
        
        # Attention computation
        attention_scores = torch.sum(target_queries * source_keys, dim=-1) * self.scale  # (n_edges, n_heads)
        
        # Apply edge weights if provided
        if edge_weights is not None:
            attention_scores = attention_scores * edge_weights.unsqueeze(-1)
        
        # Apply softmax (per target node)
        attention_weights = torch.zeros_like(attention_scores)
        for i in range(batch_size):
            mask = target_idx == i
            if mask.sum() > 0:
                attention_weights[mask] = F.softmax(attention_scores[mask], dim=0)
        
        # Apply attention to values
        attended_values = attention_weights.unsqueeze(-1) * source_values  # (n_edges, n_heads, head_dim)
        
        # Aggregate for each target node
        output_features = torch.zeros(batch_size, self.n_heads, self.head_dim, device=queries.device)
        for i in range(batch_size):
            mask = target_idx == i
            if mask.sum() > 0:
                output_features[i] = torch.sum(attended_values[mask], dim=0)
        
        # Reshape and apply output projection
        output_features = output_features.view(batch_size, self.hidden_dim)
        output_features = self.output_projection(output_features)
        output_features = self.dropout(output_features)
        
        return {target_type: output_features}
    
    def _parse_edge_type(self, edge_type: str) -> Tuple[str, str]:
        """Parse edge type to get source and target node types."""
        if '2' in edge_type:
            parts = edge_type.split('2')
            return parts[0], parts[1]
        else:
            # Default to same type for self-loops
            return edge_type, edge_type

class HeterogeneousGraphProcessor(nn.Module):
    """
    Processor for converting NetworkX heterogeneous graph to tensors.
    """
    
    def __init__(self, node_types: List[str], edge_types: List[str]):
        super(HeterogeneousGraphProcessor, self).__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
    
    def process_networkx_graph(self, graph: nx.DiGraph, 
                             node_features: Dict[str, torch.Tensor],
                             device: torch.device) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Convert NetworkX graph to tensor format.
        
        Args:
            graph: NetworkX directed graph
            node_features: Dictionary of node features
            device: Target device
            
        Returns:
            Tuple of (edge_indices, edge_weights)
        """
        edge_indices = {}
        edge_weights = {}
        
        # Create node ID mappings
        node_id_maps = {}
        for node_type in self.node_types:
            nodes_of_type = [n for n in graph.nodes() if graph.nodes[n].get('node_type') == node_type]
            node_id_maps[node_type] = {node: i for i, node in enumerate(nodes_of_type)}
        
        # Process edges by type
        for edge_type in self.edge_types:
            edges_of_type = [(u, v) for u, v, d in graph.edges(data=True) if d.get('edge_type') == edge_type]
            
            if edges_of_type:
                source_indices = []
                target_indices = []
                weights = []
                
                for source, target in edges_of_type:
                    source_type = graph.nodes[source]['node_type']
                    target_type = graph.nodes[target]['node_type']
                    
                    if source_type in node_id_maps and target_type in node_id_maps:
                        source_id = node_id_maps[source_type][source]
                        target_id = node_id_maps[target_type][target]
                        
                        source_indices.append(source_id)
                        target_indices.append(target_id)
                        
                        # Get edge weight
                        weight = graph.edges[source, target].get('weight', 1.0)
                        weights.append(weight)
                
                if source_indices:
                    edge_indices[edge_type] = torch.tensor([source_indices, target_indices], 
                                                         dtype=torch.long, device=device)
                    edge_weights[edge_type] = torch.tensor(weights, dtype=torch.float, device=device)
        
        return edge_indices, edge_weights
