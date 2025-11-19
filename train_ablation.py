"""
Ablation Study for HGNN Model Placement

This script conducts ablation experiments to evaluate the contribution of different
hypergraph components (hyperedge types) to the overall model performance.

Ablation Variants:
1. Full Model: All hyperedge types (user-model, model-server, server-server)
2. No User-Model: Remove user-model hyperedges
3. No Model-Server: Remove model-server hyperedges
4. No Server-Server: Remove server-server hyperedges
5. Only Cross-Type: Keep only heterogeneous edges (user-model + model-server)
6. No Hypergraph (MLP): Use feature embeddings without graph structure

Results are saved to: results/train_ablation/
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.HGNN import HGNN_ModelPlacement
from datasets.topk_placement_loader import TopKPlacementDataset
from utils.hypergraph_utils import construct_H_heterogeneous, generate_G_from_H
from utils.metrics import RecommendationMetrics


class MLPBaseline(nn.Module):
    """
    Simple MLP baseline without hypergraph structure
    Uses only node features for prediction
    """
    def __init__(self, in_ch, n_hid, num_users, num_models, num_servers, dropout=0.5):
        super(MLPBaseline, self).__init__()
        self.num_users = num_users
        self.num_models = num_models
        self.num_servers = num_servers
        self.dropout = dropout
        
        # Feature encoder (3 layers like HGNN)
        self.encoder = nn.Sequential(
            nn.Linear(in_ch, n_hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hid, n_hid)
        )
        
        # Placement predictor
        self.placement_predictor = nn.Bilinear(n_hid, n_hid, 1)
        
    def forward(self, x, G=None):
        """
        Args:
            x: Node features [num_total_nodes, in_ch]
            G: Unused (for interface compatibility)
            
        Returns:
            placement_scores: [num_models, num_servers]
        """
        # Encode features without graph structure
        x = self.encoder(x)
        
        # Extract embeddings
        model_start = self.num_users
        model_end = self.num_users + self.num_models
        server_start = model_end
        server_end = server_start + self.num_servers
        
        model_embeddings = x[model_start:model_end]
        server_embeddings = x[server_start:server_end]
        
        # Predict placement scores
        num_models = model_embeddings.size(0)
        num_servers = server_embeddings.size(0)
        
        model_emb_expanded = model_embeddings.unsqueeze(1).expand(num_models, num_servers, -1)
        server_emb_expanded = server_embeddings.unsqueeze(0).expand(num_models, num_servers, -1)
        
        scores = self.placement_predictor(
            model_emb_expanded.reshape(-1, model_embeddings.size(1)),
            server_emb_expanded.reshape(-1, server_embeddings.size(1))
        ).reshape(num_models, num_servers)
        
        return scores


class SoftmaxRankingLoss(nn.Module):
    """Softmax Ranking Loss for Top-K recommendation"""
    
    def __init__(self, k=5, temperature=0.1):
        super(SoftmaxRankingLoss, self).__init__()
        self.k = k
        self.temperature = temperature
    
    def forward(self, all_scores, positive_indices):
        """
        Args:
            all_scores: [num_models, num_servers]
            positive_indices: [num_models, k]
        """
        num_models = all_scores.size(0)
        losses = []
        
        for i in range(num_models):
            model_scores = all_scores[i]
            pos_idx = positive_indices[i]
            
            probs = torch.softmax(model_scores / self.temperature, dim=0)
            pos_probs = probs[pos_idx]
            loss = -torch.log(pos_probs + 1e-10).mean()
            losses.append(loss)
        
        return torch.stack(losses).mean()


def construct_H_ablation(dataset, ablation_type='full', k_neig=10):
    """
    Construct hypergraph with specific ablation configuration
    
    Args:
        dataset: TopKPlacementDataset instance
        ablation_type: Type of ablation
            - 'full': All hyperedge types
            - 'no_user_model': Remove user-model hyperedges
            - 'no_model_server': Remove model-server hyperedges
            - 'no_server_server': Remove server-server hyperedges
            - 'only_cross': Only heterogeneous edges (user-model + model-server)
        k_neig: K for KNN hyperedges
        
    Returns:
        H: Hypergraph incidence matrix
        G: Graph Laplacian matrix
    """
    num_users = dataset.num_users
    num_models = dataset.num_models
    num_servers = dataset.num_servers
    num_total = num_users + num_models + num_servers
    
    hyperedges = []
    edge_types = []
    
    print(f"\n  Ablation Type: {ablation_type}")
    
    # Flags for edge types
    include_user_model = ablation_type not in ['no_user_model']
    include_model_server = ablation_type not in ['no_model_server']
    include_server_server = ablation_type not in ['no_server_server', 'only_cross']
    
    # User-Model hyperedges
    if include_user_model:
        print("  + Including user-model hyperedges")
        user_model_groups = {}
        for _, row in dataset.user_model_df.iterrows():
            user_id = int(row['UserID']) - 1
            model_id = int(row['ModelID']) - 1
            
            if 0 <= user_id < num_users and 0 <= model_id < num_models:
                if user_id not in user_model_groups:
                    user_model_groups[user_id] = set()
                user_model_groups[user_id].add(model_id)
        
        for user_id, model_ids in user_model_groups.items():
            if len(model_ids) > 0:
                user_idx = user_id
                model_indices = [num_users + mid for mid in model_ids]
                edge_nodes = [user_idx] + model_indices
                hyperedges.append(edge_nodes)
                edge_types.append('user_model')
        
        print(f"    Created {edge_types.count('user_model')} user-model hyperedges")
    else:
        print("  - Excluding user-model hyperedges")
    
    # Model-Server hyperedges
    if include_model_server:
        print("  + Including model-server hyperedges")
        model_server_dict = {}
        for _, row in dataset.model_server_df.iterrows():
            model_id = int(row['ModelID']) - 1
            server_id = int(row['ServerID']) - 1
            
            if 0 <= model_id < num_models and 0 <= server_id < num_servers:
                if model_id not in model_server_dict:
                    model_server_dict[model_id] = []
                model_server_dict[model_id].append(server_id)
        
        for model_id, server_list in model_server_dict.items():
            model_idx = num_users + model_id
            server_indices = [num_users + num_models + sid for sid in server_list]
            edge_nodes = [model_idx] + server_indices
            hyperedges.append(edge_nodes)
            edge_types.append('model_server')
        
        print(f"    Created {edge_types.count('model_server')} model-server hyperedges")
    else:
        print("  - Excluding model-server hyperedges")
    
    # Server-Server hyperedges
    if include_server_server:
        print("  + Including server-server hyperedges")
        server_topology = dataset.topology
        for i in range(min(num_servers, server_topology.shape[0])):
            neighbors = []
            for j in range(min(num_servers, server_topology.shape[1])):
                if server_topology[i, j] > 0 and i != j:
                    neighbors.append(j)
            
            if len(neighbors) > 0:
                server_idx = num_users + num_models + i
                neighbor_indices = [num_users + num_models + n for n in neighbors]
                edge_nodes = [server_idx] + neighbor_indices
                hyperedges.append(edge_nodes)
                edge_types.append('server_server')
        
        print(f"    Created {edge_types.count('server_server')} server-server hyperedges")
    else:
        print("  - Excluding server-server hyperedges")
    
    # Build incidence matrix
    if len(hyperedges) == 0:
        print("  WARNING: No hyperedges created! Using identity matrix.")
        H = np.eye(num_total, dtype=np.float32)
    else:
        num_edges = len(hyperedges)
        H = np.zeros((num_total, num_edges), dtype=np.float32)
        
        for edge_idx, nodes in enumerate(hyperedges):
            for node_idx in nodes:
                if 0 <= node_idx < num_total:
                    H[node_idx, edge_idx] = 1.0
        
        print(f"  Hypergraph shape: {H.shape}")
        print(f"  Total hyperedges: {num_edges}")
    
    # Generate G from H
    print("  Generating graph Laplacian...")
    G = generate_G_from_H(H, variable_weight=False, use_gpu=False)
    
    return H, G


def train_one_epoch(model, features, G_tensor, positive_indices,
                   criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    all_scores = model(features, G_tensor)
    
    # Compute loss
    loss = criterion(all_scores, positive_indices)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()


def evaluate(model, features, G_tensor, dataset, metrics_calculator, device):
    """Evaluate model on training set"""
    model.eval()
    
    with torch.no_grad():
        all_scores = model(features, G_tensor)
    
    # Get positive servers for each model
    model_indices, positive_servers_global = dataset.get_evaluation_pairs()
    positive_servers_local = [
        [s - dataset.num_users - dataset.num_models for s in servers]
        for servers in positive_servers_global
    ]
    
    # Compute metrics
    metrics = metrics_calculator.compute_all_metrics(
        all_scores,
        positive_servers_local,
        prefix=""
    )
    
    return metrics


def run_ablation_experiment(ablation_type, config, device):
    """
    Run a single ablation experiment
    
    Args:
        ablation_type: Type of ablation
        config: Configuration dictionary
        device: torch device
        
    Returns:
        results: Dictionary with experiment results
    """
    print("\n" + "=" * 80)
    print(f"ABLATION EXPERIMENT: {ablation_type.upper()}")
    print("=" * 80)
    
    # Load dataset
    print("\nLoading dataset...")
    train_dataset = TopKPlacementDataset(split='train', k_positive=config['k_positive'])
    train_dataset.prepare()
    
    # Construct hypergraph for this ablation
    print("\nConstructing hypergraph...")
    if ablation_type == 'mlp':
        print("  MLP Baseline: No hypergraph structure")
        H = np.eye(train_dataset.num_users + train_dataset.num_models + train_dataset.num_servers)
        G = np.eye(train_dataset.num_users + train_dataset.num_models + train_dataset.num_servers)
    else:
        H, G = construct_H_ablation(train_dataset, ablation_type, k_neig=10)
    
    # Convert to tensors
    features = torch.FloatTensor(train_dataset.node_features).to(device)
    G_tensor = torch.FloatTensor(G).to(device)
    
    # Initialize model
    print("\nInitializing model...")
    if ablation_type == 'mlp':
        model = MLPBaseline(
            in_ch=features.shape[1],
            n_hid=config['n_hid'],
            num_users=train_dataset.num_users,
            num_models=train_dataset.num_models,
            num_servers=train_dataset.num_servers,
            dropout=config['dropout']
        ).to(device)
        print(f"  Model: MLP Baseline (no hypergraph)")
    else:
        model = HGNN_ModelPlacement(
            in_ch=features.shape[1],
            n_hid=config['n_hid'],
            num_users=train_dataset.num_users,
            num_models=train_dataset.num_models,
            num_servers=train_dataset.num_servers,
            dropout=config['dropout']
        ).to(device)
        print(f"  Model: HGNN with {ablation_type} hypergraph")
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {num_params:,}")
    
    # Prepare positive indices
    positive_indices = []
    for model_id in range(train_dataset.num_models):
        pos_servers = train_dataset.model_positive_servers[model_id]
        positive_indices.append(pos_servers)
    positive_indices = torch.LongTensor(positive_indices).to(device)
    
    # Training setup
    criterion = SoftmaxRankingLoss(k=config['k_positive'], temperature=0.05)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'],
                          weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=False
    )
    
    # Metrics calculator
    metrics_calculator = RecommendationMetrics(k_list=config['eval_k_list'])
    
    # Training loop
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    best_ndcg = 0.0
    best_epoch = 0
    best_metrics = {}
    training_log = []
    
    start_time = time.time()
    
    for epoch in range(config['max_epochs']):
        # Train
        epoch_loss = train_one_epoch(
            model, features, G_tensor, positive_indices,
            criterion, optimizer, device
        )
        
        # Evaluate
        if (epoch + 1) % config['print_freq'] == 0 or epoch == 0:
            metrics = evaluate(
                model, features, G_tensor, train_dataset,
                metrics_calculator, device
            )
            
            metrics['loss'] = epoch_loss
            metrics['epoch'] = epoch + 1
            metrics['learning_rate'] = optimizer.param_groups[0]['lr']
            training_log.append(metrics.copy())
            
            print(f"Epoch {epoch + 1:3d}/{config['max_epochs']} | "
                  f"Loss: {epoch_loss:.4f} | "
                  f"P@5: {metrics['precision@5']:.4f} | "
                  f"R@5: {metrics['recall@5']:.4f} | "
                  f"NDCG@5: {metrics['ndcg@5']:.4f} | "
                  f"F1@5: {metrics['f1@5']:.4f}")
            
            # Check for improvement
            if metrics['ndcg@5'] > best_ndcg:
                best_ndcg = metrics['ndcg@5']
                best_epoch = epoch + 1
                best_metrics = metrics.copy()
                print(f"  >> New best! NDCG@5: {best_ndcg:.4f}")
            
            # Update scheduler
            scheduler.step(metrics['ndcg@5'])
        
        # Clear cache
        if epoch % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    training_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print(f"EXPERIMENT COMPLETED: {ablation_type.upper()}")
    print("=" * 80)
    print(f"Best Epoch: {best_epoch}")
    print(f"Best NDCG@5: {best_ndcg:.4f}")
    print(f"Best Precision@5: {best_metrics['precision@5']:.4f}")
    print(f"Best Recall@5: {best_metrics['recall@5']:.4f}")
    print(f"Best F1@5: {best_metrics['f1@5']:.4f}")
    print(f"Training Time: {training_time:.2f}s")
    
    # Save model checkpoint
    checkpoint_dir = Path('results') / 'train_ablation' / 'checkpoints' / ablation_type
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / 'best_model.pth'
    
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_ndcg': best_ndcg,
        'best_metrics': best_metrics,
        'config': config,
        'ablation_type': ablation_type
    }, checkpoint_path)
    print(f"Model checkpoint saved to: {checkpoint_path}")
    
    # Results
    results = {
        'ablation_type': ablation_type,
        'config': config,
        'best_metrics': best_metrics,
        'best_epoch': best_epoch,
        'training_time': training_time,
        'training_log': training_log,
        'checkpoint_path': str(checkpoint_path)
    }
    
    return results


def load_baseline_metrics(baseline_path='results/hgnn_k10_h128_e500_20251104_155422'):
    """
    Load metrics from baseline (best) model for comparison
    
    Args:
        baseline_path: Path to baseline model directory
    
    Returns:
        dict with baseline metrics, or None if not found
    """
    baseline_path = Path(baseline_path)
    
    # Try to load from inference results
    inference_csv = baseline_path.parent.parent / 'inference' / 'unified_metrics_evaluation.csv'
    if inference_csv.exists():
        df = pd.read_csv(inference_csv)
        hgnn_row = df[df['method'] == 'HGNN']
        if not hgnn_row.empty:
            return {
                'precision@5': float(hgnn_row['precision@5'].values[0]),
                'recall@5': float(hgnn_row['recall@5'].values[0]),
                'f1@5': float(hgnn_row['f1@5'].values[0]),
                'ndcg@5': float(hgnn_row['ndcg@5'].values[0]),
                'hit_rate@5': float(hgnn_row['hit_rate@5'].values[0]),
                'source': str(inference_csv)
            }
    
    # Try to load from checkpoint
    checkpoint_path = baseline_path / 'checkpoints' / 'best_model.pth'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'best_metrics' in checkpoint:
            return {
                'ndcg@5': checkpoint['best_metrics'].get('ndcg@5', 0.0),
                'source': str(checkpoint_path)
            }
    
    return None


def main():
    print("\n" + "=" * 80)
    print("ABLATION STUDY FOR HGNN MODEL PLACEMENT")
    print("=" * 80)
    
    # Load baseline model configuration and metrics
    baseline_path = 'results/hgnn_k10_h128_e500_20251104_155422'
    print(f"\nBaseline model: {baseline_path}")
    
    baseline_metrics = load_baseline_metrics(baseline_path)
    if baseline_metrics:
        print("\nBaseline (Ours) metrics loaded:")
        for key, value in baseline_metrics.items():
            if key != 'source':
                print(f"  {key}: {value:.4f}")
        print(f"  Source: {baseline_metrics['source']}")
    else:
        print("\nWarning: Could not load baseline metrics (will still run ablation)")
    
    # Configuration (matching best model)
    config = {
        'k_positive': 10,
        'n_hid': 128,
        'dropout': 0.05,      # Best model used 0.05
        'lr': 0.0005,         # Best model used 0.0005
        'weight_decay': 0.0001,
        'max_epochs': 300,    # Increased for better convergence (baseline used 500)
        'print_freq': 20,
        'eval_k_list': [1, 3, 5, 10, 20]
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print("\nConfiguration (matching best model):")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Define ablation types
    ablation_types = [
        'full',              # All hyperedge types
        'no_user_model',     # Remove user-model edges
        'no_model_server',   # Remove model-server edges
        'no_server_server',  # Remove server-server edges
        'only_cross',        # Only cross-type edges (user-model + model-server)
        'mlp'                # MLP baseline (no hypergraph)
    ]
    
    ablation_names = {
        'full': 'Full Model (All Hyperedges)',
        'no_user_model': 'No User-Model Hyperedges',
        'no_model_server': 'No Model-Server Hyperedges',
        'no_server_server': 'No Server-Server Hyperedges',
        'only_cross': 'Only Cross-Type Hyperedges',
        'mlp': 'MLP Baseline (No Hypergraph)'
    }
    
    print("\nAblation experiments to run:")
    for i, abl_type in enumerate(ablation_types, 1):
        print(f"  {i}. {ablation_names[abl_type]}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('results') / 'train_ablation' / f'ablation_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Save baseline metrics for reference
    if baseline_metrics:
        baseline_file = output_dir / 'baseline_metrics.json'
        with open(baseline_file, 'w') as f:
            json.dump(baseline_metrics, f, indent=2)
        print(f"Baseline metrics saved to: {baseline_file}")
    
    # Run all ablation experiments
    all_results = {}
    
    for i, ablation_type in enumerate(ablation_types, 1):
        print("\n" + "=" * 80)
        print(f"RUNNING EXPERIMENT {i}/{len(ablation_types)}")
        print("=" * 80)
        
        try:
            results = run_ablation_experiment(ablation_type, config, device)
            all_results[ablation_type] = results
            
            # Save individual result
            result_file = output_dir / f'{ablation_type}_results.json'
            with open(result_file, 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                json_results = {}
                for key, value in results.items():
                    if key == 'training_log':
                        json_results[key] = [
                            {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                             for k, v in log.items()}
                            for log in value
                        ]
                    elif key == 'best_metrics':
                        json_results[key] = {
                            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                            for k, v in value.items()
                        }
                    else:
                        json_results[key] = value
                
                json.dump(json_results, f, indent=2)
            print(f"\nResults saved to: {result_file}")
            
        except Exception as e:
            print(f"\nERROR in {ablation_type} experiment:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
            all_results[ablation_type] = {'error': str(e)}
    
    # Generate comparison summary
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)
    
    summary = {}
    comparison_table = []
    
    for ablation_type in ablation_types:
        if ablation_type in all_results and 'best_metrics' in all_results[ablation_type]:
            result = all_results[ablation_type]
            summary[ablation_type] = {
                'name': ablation_names[ablation_type],
                'precision@5': result['best_metrics']['precision@5'],
                'recall@5': result['best_metrics']['recall@5'],
                'f1@5': result['best_metrics']['f1@5'],
                'ndcg@5': result['best_metrics']['ndcg@5'],
                'training_time': result['training_time']
            }
            comparison_table.append([
                ablation_names[ablation_type],
                f"{result['best_metrics']['precision@5']:.4f}",
                f"{result['best_metrics']['recall@5']:.4f}",
                f"{result['best_metrics']['f1@5']:.4f}",
                f"{result['best_metrics']['ndcg@5']:.4f}",
                f"{result['training_time']:.1f}s"
            ])
    
    # Print comparison table
    print("\nPerformance Comparison:")
    print("-" * 100)
    print(f"{'Ablation Type':<40} {'P@5':>8} {'R@5':>8} {'F1@5':>8} {'NDCG@5':>10} {'Time':>10}")
    print("-" * 100)
    for row in comparison_table:
        print(f"{row[0]:<40} {row[1]:>8} {row[2]:>8} {row[3]:>8} {row[4]:>10} {row[5]:>10}")
    print("-" * 100)
    
    # Save summary JSON
    summary_file = output_dir / 'ablation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")
    
    # Save summary CSV (for easy plotting)
    csv_data = []
    for ablation_type in ablation_types:
        if ablation_type in summary:
            row = {
                'ablation_type': ablation_type,
                'name': ablation_names[ablation_type],
                'precision@5': summary[ablation_type]['precision@5'],
                'recall@5': summary[ablation_type]['recall@5'],
                'f1@5': summary[ablation_type]['f1@5'],
                'ndcg@5': summary[ablation_type]['ndcg@5'],
                'training_time': summary[ablation_type]['training_time']
            }
            csv_data.append(row)
    
    df_summary = pd.DataFrame(csv_data)
    csv_file = output_dir / 'ablation_summary.csv'
    df_summary.to_csv(csv_file, index=False)
    print(f"Summary CSV saved to: {csv_file} (for easy plotting)")
    
    # Calculate relative performance (vs Full Model)
    if 'full' in summary:
        full_ndcg = summary['full']['ndcg@5']
        print("\nRelative Performance (vs Full Model):")
        print("-" * 60)
        print(f"{'Ablation Type':<40} {'NDCG@5 Change':>15}")
        print("-" * 60)
        for ablation_type in ablation_types:
            if ablation_type in summary and ablation_type != 'full':
                ndcg = summary[ablation_type]['ndcg@5']
                change = ((ndcg / full_ndcg) - 1) * 100
                print(f"{ablation_names[ablation_type]:<40} {change:>+14.2f}%")
        print("-" * 60)
    
    # Compare with baseline (best model)
    if baseline_metrics and 'full' in summary:
        print("\nComparison with Baseline (Best Model):")
        print("-" * 60)
        print(f"{'Metric':<20} {'Baseline':>12} {'Full Ablation':>15} {'Difference':>12}")
        print("-" * 60)
        for metric in ['precision@5', 'recall@5', 'f1@5', 'ndcg@5']:
            if metric in baseline_metrics:
                baseline_val = baseline_metrics[metric]
                ablation_val = summary['full'][metric]
                diff = ablation_val - baseline_val
                print(f"{metric:<20} {baseline_val:>12.4f} {ablation_val:>15.4f} {diff:>+12.4f}")
        print("-" * 60)
    
    print("\n" + "=" * 80)
    print("ABLATION STUDY COMPLETED")
    print("=" * 80)
    print(f"All results saved to: {output_dir}")
    print(f"\nFiles generated:")
    print(f"  - ablation_summary.json")
    print(f"  - ablation_summary.csv (for plotting)")
    if baseline_metrics:
        print(f"  - baseline_metrics.json")
    print(f"  - Individual results: {{ablation_type}}_results.json")


if __name__ == '__main__':
    main()

