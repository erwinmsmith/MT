import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle
import networkx as nx

class PriorKnowledgeLoader:
    """Load and manage prior knowledge for feature graph construction."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.prior_knowledge_dir = Path(config.get('prior_knowledge_dir', 'data/prior_knowledge'))
        self.prior_knowledge_dir.mkdir(parents=True, exist_ok=True)
        
    def simulate_gene_embeddings(self, n_genes: int, embedding_dim: int = 256) -> np.ndarray:
        """
        Simulate gene embeddings from foundation models like GenePT.
        
        Args:
            n_genes: Number of genes
            embedding_dim: Dimension of embeddings
            
        Returns:
            Gene embeddings matrix (n_genes x embedding_dim)
        """
        np.random.seed(42)
        # Simulate embeddings with some structure
        embeddings = np.random.normal(0, 1, (n_genes, embedding_dim)).astype(np.float32)
        
        # Add some clustering structure to make embeddings more realistic
        n_clusters = 10
        cluster_centers = np.random.normal(0, 2, (n_clusters, embedding_dim))
        
        genes_per_cluster = n_genes // n_clusters
        for i in range(n_clusters):
            start_idx = i * genes_per_cluster
            end_idx = (i + 1) * genes_per_cluster if i < n_clusters - 1 else n_genes
            
            # Add cluster center to embeddings
            embeddings[start_idx:end_idx] += cluster_centers[i] * 0.5
        
        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def simulate_protein_embeddings(self, n_proteins: int, embedding_dim: int = 256) -> np.ndarray:
        """
        Simulate protein embeddings from foundation models.
        
        Args:
            n_proteins: Number of proteins
            embedding_dim: Dimension of embeddings
            
        Returns:
            Protein embeddings matrix (n_proteins x embedding_dim)
        """
        np.random.seed(43)
        embeddings = np.random.normal(0, 1, (n_proteins, embedding_dim)).astype(np.float32)
        
        # Add functional grouping structure
        n_groups = 5
        group_centers = np.random.normal(0, 2, (n_groups, embedding_dim))
        
        proteins_per_group = n_proteins // n_groups
        for i in range(n_groups):
            start_idx = i * proteins_per_group
            end_idx = (i + 1) * proteins_per_group if i < n_groups - 1 else n_proteins
            
            embeddings[start_idx:end_idx] += group_centers[i] * 0.4
        
        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def create_gene2peak_mapping(self, n_genes: int, n_peaks: int) -> List[Tuple[int, int, float]]:
        """
        Create gene-to-peak mapping based on genomic proximity.
        
        Returns:
            List of (gene_idx, peak_idx, weight) tuples
        """
        np.random.seed(44)
        mappings = []
        
        # Each gene can be associated with multiple peaks
        for gene_idx in range(n_genes):
            # Number of peaks associated with this gene
            n_associated_peaks = np.random.poisson(3) + 1
            n_associated_peaks = min(n_associated_peaks, n_peaks)
            
            # Choose random peaks (in practice, these would be based on genomic distance)
            associated_peaks = np.random.choice(n_peaks, size=n_associated_peaks, replace=False)
            
            for peak_idx in associated_peaks:
                # Weight based on distance (simulated)
                weight = np.random.exponential(0.5)
                weight = min(weight, 2.0)  # Cap the weight
                mappings.append((gene_idx, peak_idx, weight))
        
        return mappings
    
    def create_gene2protein_mapping(self, n_genes: int, n_proteins: int) -> List[Tuple[int, int, float]]:
        """
        Create gene-to-protein mapping based on known gene-protein relationships.
        
        Returns:
            List of (gene_idx, protein_idx, weight) tuples
        """
        np.random.seed(45)
        mappings = []
        
        # Not all genes have corresponding proteins in the protein panel
        mapped_proteins = min(n_proteins, n_genes // 2)
        
        for i in range(mapped_proteins):
            gene_idx = i * 2 if i * 2 < n_genes else np.random.randint(n_genes)
            protein_idx = i
            # High confidence for direct gene-protein relationships
            weight = np.random.uniform(0.8, 1.0)
            mappings.append((gene_idx, protein_idx, weight))
        
        # Add some additional indirect relationships
        n_indirect = n_proteins // 4
        for _ in range(n_indirect):
            gene_idx = np.random.randint(n_genes)
            protein_idx = np.random.randint(n_proteins)
            weight = np.random.uniform(0.3, 0.7)
            mappings.append((gene_idx, protein_idx, weight))
        
        return mappings
    
    def create_peak2protein_mapping(self, n_peaks: int, n_proteins: int) -> List[Tuple[int, int, float]]:
        """
        Create peak-to-protein mapping based on regulatory relationships.
        
        Returns:
            List of (peak_idx, protein_idx, weight) tuples
        """
        np.random.seed(46)
        mappings = []
        
        # Peaks can regulate protein expression
        for protein_idx in range(n_proteins):
            # Number of peaks that might regulate this protein
            n_regulatory_peaks = np.random.poisson(2) + 1
            n_regulatory_peaks = min(n_regulatory_peaks, n_peaks)
            
            regulatory_peaks = np.random.choice(n_peaks, size=n_regulatory_peaks, replace=False)
            
            for peak_idx in regulatory_peaks:
                # Weight represents regulatory strength
                weight = np.random.exponential(0.3)
                weight = min(weight, 1.5)
                mappings.append((peak_idx, protein_idx, weight))
        
        return mappings
    
    def create_heterogeneous_graph(self, n_genes: int, n_peaks: int, n_proteins: int) -> nx.DiGraph:
        """
        Create a heterogeneous directed graph with different node types and edge types.
        
        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()
        
        # Add nodes with types
        for i in range(n_genes):
            G.add_node(f'gene_{i}', node_type='gene', feature_idx=i)
        
        for i in range(n_peaks):
            G.add_node(f'peak_{i}', node_type='peak', feature_idx=i)
        
        for i in range(n_proteins):
            G.add_node(f'protein_{i}', node_type='protein', feature_idx=i)
        
        # Add edges based on different meta paths
        
        # Gene -> Peak edges (gene regulation of chromatin accessibility)
        gene2peak_mappings = self.create_gene2peak_mapping(n_genes, n_peaks)
        for gene_idx, peak_idx, weight in gene2peak_mappings:
            G.add_edge(f'gene_{gene_idx}', f'peak_{peak_idx}', 
                      edge_type='gene2peak', weight=weight)
        
        # Gene -> Protein edges (translation)
        gene2protein_mappings = self.create_gene2protein_mapping(n_genes, n_proteins)
        for gene_idx, protein_idx, weight in gene2protein_mappings:
            G.add_edge(f'gene_{gene_idx}', f'protein_{protein_idx}', 
                      edge_type='gene2protein', weight=weight)
        
        # Peak -> Protein edges (chromatin regulation of protein expression)
        peak2protein_mappings = self.create_peak2protein_mapping(n_peaks, n_proteins)
        for peak_idx, protein_idx, weight in peak2protein_mappings:
            G.add_edge(f'peak_{peak_idx}', f'protein_{protein_idx}', 
                      edge_type='peak2protein', weight=weight)
        
        # Add some protein-protein interaction edges
        np.random.seed(47)
        n_ppi = n_proteins // 3
        for _ in range(n_ppi):
            protein1_idx = np.random.randint(n_proteins)
            protein2_idx = np.random.randint(n_proteins)
            if protein1_idx != protein2_idx:
                weight = np.random.uniform(0.2, 0.8)
                G.add_edge(f'protein_{protein1_idx}', f'protein_{protein2_idx}', 
                          edge_type='protein2protein', weight=weight)
        
        return G
    
    def save_prior_knowledge(self, n_genes: int, n_peaks: int, n_proteins: int, 
                           embedding_dim: int = 256):
        """Save all prior knowledge to files."""
        
        # Generate and save embeddings
        gene_embeddings = self.simulate_gene_embeddings(n_genes, embedding_dim)
        protein_embeddings = self.simulate_protein_embeddings(n_proteins, embedding_dim)
        
        np.save(self.prior_knowledge_dir / 'gene_embeddings.npy', gene_embeddings)
        np.save(self.prior_knowledge_dir / 'protein_embeddings.npy', protein_embeddings)
        
        # Create and save heterogeneous graph
        hetero_graph = self.create_heterogeneous_graph(n_genes, n_peaks, n_proteins)
        
        with open(self.prior_knowledge_dir / 'heterogeneous_graph.pkl', 'wb') as f:
            pickle.dump(hetero_graph, f)
        
        # Save individual mappings as well
        gene2peak = self.create_gene2peak_mapping(n_genes, n_peaks)
        gene2protein = self.create_gene2protein_mapping(n_genes, n_proteins)
        peak2protein = self.create_peak2protein_mapping(n_peaks, n_proteins)
        
        with open(self.prior_knowledge_dir / 'gene2peak_mapping.pkl', 'wb') as f:
            pickle.dump(gene2peak, f)
        
        with open(self.prior_knowledge_dir / 'gene2protein_mapping.pkl', 'wb') as f:
            pickle.dump(gene2protein, f)
        
        with open(self.prior_knowledge_dir / 'peak2protein_mapping.pkl', 'wb') as f:
            pickle.dump(peak2protein, f)
        
        print(f"Prior knowledge saved to {self.prior_knowledge_dir}")
        
        return {
            'gene_embeddings': gene_embeddings,
            'protein_embeddings': protein_embeddings,
            'heterogeneous_graph': hetero_graph,
            'gene2peak': gene2peak,
            'gene2protein': gene2protein,
            'peak2protein': peak2protein
        }
    
    def load_prior_knowledge(self) -> Dict:
        """Load all prior knowledge from files."""
        prior_knowledge = {}
        
        # Load embeddings
        gene_emb_path = self.prior_knowledge_dir / 'gene_embeddings.npy'
        if gene_emb_path.exists():
            prior_knowledge['gene_embeddings'] = np.load(gene_emb_path)
        
        protein_emb_path = self.prior_knowledge_dir / 'protein_embeddings.npy'
        if protein_emb_path.exists():
            prior_knowledge['protein_embeddings'] = np.load(protein_emb_path)
        
        # Load graph
        graph_path = self.prior_knowledge_dir / 'heterogeneous_graph.pkl'
        if graph_path.exists():
            with open(graph_path, 'rb') as f:
                prior_knowledge['heterogeneous_graph'] = pickle.load(f)
        
        # Load mappings
        for mapping_name in ['gene2peak_mapping', 'gene2protein_mapping', 'peak2protein_mapping']:
            mapping_path = self.prior_knowledge_dir / f'{mapping_name}.pkl'
            if mapping_path.exists():
                with open(mapping_path, 'rb') as f:
                    prior_knowledge[mapping_name] = pickle.load(f)
        
        return prior_knowledge
