# ReHGNN: Reliability-Aware Heterogeneous Hypergraph Neural Recommender

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/pytorch-1.12+-orange.svg)](https://pytorch.org/)

## Overview

**ReHGNN** is a deep learning framework for intelligent server recommendation in distributed large-model deployment scenarios. As large-scale AI models increasingly rely on distributed infrastructure for reliable service delivery, selecting optimal server combinations becomes critical for balancing deployment quality, system reliability, and service latency. Traditional rule-based or heuristic methods struggle with the inherent heterogeneity of system entities (users, models, servers) and the high-order deployment relationships (e.g., one model deployed on multiple servers, one user invoking multiple models).

This project addresses these challenges by modeling the deployment problem as a heterogeneous hypergraph neural network task, where:
- **Heterogeneous nodes** represent users, models, and servers with distinct feature spaces
- **Hyperedges** explicitly capture group-level relationships (user-model interactions, model-server deployments, server topology)
- **Weakly-supervised learning** leverages historical reliable deployment records to implicitly optimize multiple system objectives

The framework achieves superior performance in deployment recommendation accuracy and exhibits strong generalization across diverse data distributions.

---

## Key Features

- **Unified Heterogeneous Modeling**: Seamlessly integrates multi-type entities (users, models, servers) into a single graph structure
- **High-Order Relationship Encoding**: Uses hypergraph structure to natively express "one-to-many" and "many-to-many" deployment patterns
- **Implicit Multi-Objective Optimization**: Learns from historical reliable deployments without explicit multi-objective weighting
- **Scalable Architecture**: Sparse hypergraph computations enable efficient training on large-scale systems
- **Robust Generalization**: Maintains stable performance across different regional focus distributions

---

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (for GPU acceleration)
- 16GB+ RAM (32GB+ recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-organization/ReHGNN.git
cd ReHGNN

# Install dependencies
pip install -r requirements.txt
```

### Minimal Working Example

**1. Prepare Data Directory Structure**

Organize your data as follows (use placeholder datasets for testing):

```
datasets/
├── <dataset_name>/
│   ├── user-train.csv          # User features (Lo, La, ServerID, W1, W2, Size)
│   ├── user-test.csv
│   ├── model.csv               # Model features (ModelType, ArenaScore, Modelsize, Modelresource)
│   ├── server.csv              # Server features (Lo, La, LinkBandwidth, ComputationCapacity, StorageCapacity)
│   ├── user-model-train.csv    # User-model interactions (wide format)
│   ├── user-model-test.csv
│   └── server topology.csv     # Server network topology (adjacency matrix)
```

**2. Configure Basic Settings**

Edit `config/config.yaml` to specify data paths and basic hyperparameters:

```yaml
# Data paths (update to your actual paths)
data_root: 'datasets/<dataset_name>'
result_root: 'results'

# Model architecture (adjust based on your data scale)
k_positive: 5-20              # Number of positive samples (deployment redundancy)
n_hid: 64-256                 # Hidden layer dimension
dropout: 0.0-0.2              # Dropout rate

# Training settings
lr: 1e-4 to 1e-3              # Learning rate
max_epochs: 100-500           # Maximum training epochs
eval_k_list: [1, 3, 5, 10, 20]  # Evaluation K values for metrics
```

**3. Train the Model**

```bash
# Basic training with default configuration
python train.py

# Advanced: Specify custom configuration
python train.py --config config/custom_config.yaml
```

**4. Run Inference**

```bash
# Perform inference on test set
python inference_model_placement.py --checkpoint results/<experiment_dir>/checkpoints/best_model.pth

# Compare with baseline methods
python inference_model_placement.py --checkpoint <model_path> --compare_baselines
```


### Obtaining Data

For research purposes, you may:
1. **Use public benchmark datasets** (e.g., adapt datasets from system/network research)
2. **Contact the authors** for collaboration opportunities (see Contact section)

### Pre-trained Models

Pre-trained model checkpoints are **not publicly released** to protect proprietary experimental configurations. However, you can train models from scratch following the Quick Start guide. Typical training time: 4-7 hours on a single RTX 3070 GPU for datasets with ~10K users, ~200 models, ~1.5K servers.

---

## Reproducing Experiments


## Environment & Compatibility

### Tested Environments

| Component       | Version        | Notes                          |
|-----------------|----------------|--------------------------------|
| Python          | 3.8, 3.9, 3.10 | 3.8 recommended                |
| PyTorch         | 1.12, 1.13, 2.0| GPU support required           |
| CUDA            | 11.0, 11.3, 11.7| Match with PyTorch version    |
| NumPy           | 1.21+          | For numerical operations       |
| Pandas          | 1.3+           | For data loading               |
| scikit-learn    | 1.0+           | For t-SNE visualization        |
| Matplotlib      | 3.5+           | For plotting                   |




## Contact & Support

For questions, bug reports, or collaboration inquiries:

- **Email**: [gymorsiback@tju.edu.cn]

---

## Acknowledgments

This work was supported by [Your Institution/Funding Source]. We thank [Collaborators/Contributors] for their valuable feedback and support.

Special thanks to the open-source community for providing foundational tools:
- [PyTorch](https://pytorch.org/) for deep learning framework
- [DeepHypergraph (DHG)](https://github.com/iMoonLab/DeepHypergraph) for hypergraph utilities
- [scikit-learn](https://scikit-learn.org/) for evaluation metrics

---

**Last Updated**: November 2024  
**Version**: 1.0.0
