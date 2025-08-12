# Multi-omics Topic Model

A sophisticated deep learning framework for single-cell multi-omics data analysis using topic modeling with dual pathways: **cellgraph** and **featuregraph**.

## Overview

This project implements a novel multi-omics topic model that integrates:

- **CellGraph Pathway**: Processes cell-level multi-omics data through modality-specific encoders, multimodal fusion, trajectory inference, and directed GNN to generate cell topic distributions (θ_d)
- **FeatureGraph Pathway**: Utilizes prior knowledge and heterogeneous directed GNN to process feature relationships and generate feature topic distributions (B_d)  
- **Integration Module**: Combines both pathways through a sophisticated decoder for data reconstruction using VAE framework

## Architecture

### CellGraph Pathway
1. **Modality Encoders**: Separate encoders for gene, peak, and protein data
2. **Multimodal Fusion Block**: Attention-based fusion of modality embeddings
3. **Trajectory Inference**: Constructs cell-to-cell directed adjacency matrix
4. **Directed GNN**: Updates cell embeddings using graph structure
5. **Cell Topics Encoder**: Generates cell topic distributions (θ_d)

### FeatureGraph Pathway
1. **Prior Knowledge Integration**: Foundation model embeddings and literature-derived mappings
2. **Heterogeneous Graph Construction**: Multi-type nodes (gene, peak, protein) with different edge types
3. **Heterogeneous Directed GNN**: Processes the feature graph
4. **Feature Topics Encoders**: Generate modality-specific topic distributions (B_d)

### Integration & Decoding
1. **Topic Embeddings**: Shared topic representation space
2. **Constrained Decoding**: Uses both θ_d and B_d to reconstruct original data
3. **VAE Framework**: KL divergence and reconstruction losses

## Project Structure

```
multiomics/
├── config/
│   ├── __init__.py
│   ├── config.yaml              # Main configuration file
│   └── config_loader.py         # Configuration management
├── data/
│   ├── __init__.py
│   ├── dataset_simulator.py     # Data simulation
│   ├── dataloader.py           # Data loading utilities
│   └── prior_knowledge.py      # Prior knowledge management
├── model/
│   ├── __init__.py
│   ├── encoder/                 # Dedicated encoder folder
│   │   ├── __init__.py
│   │   ├── gene_encoder.py      # Gene expression encoder
│   │   ├── peak_encoder.py      # Chromatin accessibility encoder  
│   │   ├── protein_encoder.py   # Protein expression encoder
│   │   ├── fused_embedding_encoder.py   # Fused embedding encoder
│   │   ├── cell_topics_encoder.py       # Cell topic distribution encoder
│   │   ├── base_feature_topics_encoder.py   # Base feature topics encoder
│   │   ├── gene_feature_topics_encoder.py   # Gene-specific feature encoder
│   │   ├── peak_feature_topics_encoder.py   # Peak-specific feature encoder
│   │   └── protein_feature_topics_encoder.py # Protein-specific feature encoder
│   ├── multimodel_fusion_block.py      # Multimodal fusion
│   ├── directed_GNN.py          # Directed graph neural network
│   ├── heterogeneous_directed_GNN.py   # Heterogeneous directed GNN
│   ├── cellgraph.py             # CellGraph pathway
│   ├── featuregraph.py          # FeatureGraph pathway
│   ├── decoder.py               # Multi-omics decoder
│   └── multiomics_topic_model.py       # Complete model
├── trainer.py                   # Training system
├── main.py                      # Main interface
├── requirements.txt             # Dependencies
├── plan                         # Project plan (Chinese)
├── readme                       # Project description (Chinese)
└── rule                         # Development rules (Chinese)
```

## Unique Encoder Architecture

### Organized Encoder Structure
All encoders are now organized in the `model/encoder/` folder for better modularity:

**Cell-level Encoders:**
- **GeneEncoder**: Handles gene expression with log transformation and gene-specific normalization
- **PeakEncoder**: Processes binary/sparse chromatin accessibility data with sigmoid transformation
- **ProteinEncoder**: Manages protein expression with robust scaling and log transformation
- **FusedEmbeddingEncoder**: Uses self-attention for processing multimodal fusion outputs
- **CellTopicsEncoder**: Variational encoder for cell topic distributions with temperature scaling

**Feature-level Encoders (Modality-Specific):**
- **GeneFeatureTopicsEncoder**: Specialized for gene regulatory relationships with pathway enrichment
- **PeakFeatureTopicsEncoder**: Optimized for chromatin accessibility with binary pattern enhancement
- **ProteinFeatureTopicsEncoder**: Designed for protein functional relationships and complex formation

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd multiomics
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Simulation
Generate simulated multi-omics data and prior knowledge:
```bash
python main.py --mode simulate --config config/config.yaml
```

### Training
Train the complete model:
```bash
python main.py --mode train --config config/config.yaml --epochs 100 --lr 0.001
```

### Evaluation
Evaluate a trained model:
```bash
python main.py --mode eval --config config/config.yaml --checkpoint checkpoints/best_model.pth
```

### Configuration Options

Key configuration parameters in `config/config.yaml`:

```yaml
data:
  n_topics: 20                    # Number of topics
  embedding_dim: 256              # Embedding dimension
  batch_size: 64                  # Training batch size

model:
  gene_encoder:                   # Gene encoder config
    hidden_dims: [1024, 512, 256]
    dropout: 0.1
  
  fusion_block:                   # Multimodal fusion config
    n_heads: 8
    n_layers: 3

training:
  epochs: 100                     # Training epochs
  learning_rate: 0.001            # Learning rate
  kl_weight: 1.0                  # KL divergence weight
```

### Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --mode {train,eval,simulate}    # Running mode
  --config PATH                   # Configuration file path
  --checkpoint PATH               # Model checkpoint path (for eval)
  --epochs INT                    # Number of training epochs
  --lr FLOAT                      # Learning rate
  --batch_size INT                # Batch size
  --device {cuda,cpu}             # Device to use
  --seed INT                      # Random seed
```

## Features

### Modular Design
- **Extensible**: Easy to add new modalities or modify existing components
- **Configurable**: Comprehensive configuration system for all hyperparameters
- **Exposed Interfaces**: All intermediate matrices and components are accessible

### Advanced Techniques
- **Heterogeneous GNN**: Handles different node and edge types in feature graphs
- **Directed GNN**: Maintains directional information in cell relationships
- **Attention Mechanisms**: Multi-head attention for modality fusion
- **Variational Framework**: VAE with KL divergence and reconstruction losses
- **Prior Knowledge Integration**: Incorporates foundation model embeddings and literature

### Training Features
- **Progress Monitoring**: Detailed logging with epoch, time, and loss information
- **Model Checkpointing**: Automatic saving of best models and regular checkpoints
- **Learning Rate Scheduling**: Adaptive learning rate with plateau detection
- **Gradient Clipping**: Prevents gradient explosion
- **KL Annealing**: Gradual increase of KL weight during training

## Model Outputs

The model provides comprehensive outputs:

1. **Topic Distributions**:
   - Cell topic distributions (θ_d): `(n_cells, n_topics)`
   - Feature topic distributions (B_d): `(n_features, n_topics)` per modality

2. **Embeddings**:
   - Cell embeddings at various stages
   - Feature embeddings from heterogeneous GNN
   - Topic embeddings: `(n_topics, embedding_dim)`

3. **Reconstructed Data**:
   - Reconstructed gene expression: `(n_cells, n_genes)`
   - Reconstructed peak accessibility: `(n_cells, n_peaks)`
   - Reconstructed protein expression: `(n_cells, n_proteins)`

4. **Graph Structures**:
   - Cell-to-cell adjacency matrix
   - Heterogeneous feature graph with multiple edge types

## Contributing

This project follows modular design principles. When adding new components:

1. Create separate encoder files for new modalities
2. Maintain the exposed interface pattern for intermediate results
3. Update configuration files for new hyperparameters
4. Add comprehensive documentation

## License

[Specify your license here]

## Citation

If you use this code in your research, please cite:

```bibtex
@article{multiomics_topic_model,
  title={Multi-omics Topic Model with Dual Pathways for Single-cell Analysis},
  author={[Your Name]},
  journal={[Journal Name]},
  year={[Year]}
}
```
