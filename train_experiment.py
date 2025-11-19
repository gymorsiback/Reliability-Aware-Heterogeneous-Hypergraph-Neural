"""
Optimized Training Script with Complete Experiment Logging

Key improvements:
1. Pre-compute and cache graph Laplacian (avoid repeated computation)
2. Efficient batching strategy
3. Comprehensive metrics logging
4. Baseline comparisons
5. Multiple K values evaluation
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.HGNN import HGNN_ModelPlacement
from datasets.topk_placement_loader import TopKPlacementDataset
from utils.hypergraph_utils import construct_H_for_model_placement
from utils.metrics import RecommendationMetrics, compute_diversity_metrics
from utils.experiment_logger import ExperimentLogger, BaselineLogger


class SoftmaxRankingLoss(nn.Module):
    """
    Softmax Ranking Loss (ListNet-style)
    
    Treats ranking as classification: positives should have high softmax probability.
    Directly optimizes the ranking distribution.
    """
    
    def __init__(self, k=5, temperature=0.1):
        """
        Args:
            k: Number of positive samples
            temperature: Temperature for softmax (lower = sharper)
        """
        super(SoftmaxRankingLoss, self).__init__()
        self.k = k
        self.temperature = temperature
    
    def forward(self, all_scores, positive_indices):
        """
        Compute softmax ranking loss
        
        Args:
            all_scores: Scores for ALL servers [num_models, num_servers]
            positive_indices: Positive server indices for each model [num_models, k]
        
        Returns:
            Loss value (scalar)
        """
        num_models = all_scores.size(0)
        
        losses = []
        for i in range(num_models):
            model_scores = all_scores[i]  # [num_servers]
            pos_idx = positive_indices[i]  # [k] positive servers
            
            # Apply softmax over all servers (temperature scaling)
            probs = torch.softmax(model_scores / self.temperature, dim=0)
            
            # Get probabilities for positive servers
            pos_probs = probs[pos_idx]  # [k]
            
            # Loss: negative log probability of positives
            # We want positives to have high probability
            loss = -torch.log(pos_probs + 1e-10).mean()
            losses.append(loss)
        
        return torch.stack(losses).mean()


def train_one_epoch(
    model,
    features,
    G_tensor,
    positive_indices,
    criterion,
    optimizer,
    dataset,
    device
):
    """
    Train for one epoch with Top-K ranking loss
    
    Args:
        model: HGNN model
        features: Node features
        G_tensor: Graph Laplacian matrix
        positive_indices: Positive server indices [num_models, k]
        criterion: Loss function (TopKRankingLoss)
        optimizer: Optimizer
        dataset: Dataset object
        device: Device (cuda or cpu)
    
    Returns:
        Average epoch loss
    """
    model.train()
    optimizer.zero_grad()
    
    # Forward pass through full model (ALL models, ALL servers)
    all_scores = model(features, G_tensor)  # [num_models, num_servers]
    
    # Compute loss on all models at once
    loss = criterion(all_scores, positive_indices)
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    
    optimizer.step()
    
    return loss.item()


def evaluate(
    model,
    features,
    G_tensor,
    dataset,
    metrics_calculator,
    device
):
    """
    Evaluate model on dataset
    
    Args:
        model: HGNN model
        features: Node features
        G_tensor: Graph Laplacian matrix
        dataset: Dataset object
        metrics_calculator: Metrics calculator
        device: Device (cuda or cpu)
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    with torch.no_grad():
        all_scores = model(features, G_tensor)
        model_server_scores = all_scores[:, :]
        
        # Get positive indices
        model_indices, positive_servers_global = dataset.get_evaluation_pairs()
        positive_servers_local = [
            [s - dataset.num_users - dataset.num_models for s in servers]
            for servers in positive_servers_global
        ]
        
        # Debug: Check evaluation data (only first call)
        if not hasattr(evaluate, '_debug_done'):
            print(f"\n[DEBUG] Evaluation Check:")
            print(f"  Num models: {len(positive_servers_local)}")
            print(f"  Example positive servers (model 0): {positive_servers_local[0]}")
            print(f"  Score shape: {model_server_scores.shape}")
            print(f"  Example scores (model 0, top 10): {torch.topk(model_server_scores[0], k=10)[1].cpu().numpy()}")
            print(f"  Example scores values (model 0, top 10): {torch.topk(model_server_scores[0], k=10)[0].cpu().numpy()}")
            evaluate._debug_done = True
        
        # Compute all metrics
        metrics = metrics_calculator.compute_all_metrics(
            model_server_scores,
            positive_servers_local
        )
        
        # Compute diversity metrics
        # Get binary placements (top-5 for each model)
        pred_placements = torch.zeros_like(model_server_scores)
        for i in range(model_server_scores.shape[0]):
            top_k = torch.topk(model_server_scores[i], k=5)[1]
            pred_placements[i, top_k] = 1
        
        diversity_metrics = compute_diversity_metrics(
            pred_placements,
            dataset.num_servers
        )
        metrics.update(diversity_metrics)
    
    return metrics


def run_random_baseline(dataset, metrics_calculator, logger, k=5):
    """
    Run random baseline evaluation
    
    Args:
        dataset: Dataset object
        metrics_calculator: Metrics calculator
        logger: Baseline logger
        k: Number of servers to recommend
    
    Returns:
        Dictionary of baseline metrics
    """
    print("\n" + "=" * 80)
    print("RANDOM BASELINE")
    print("=" * 80)
    
    # Random predictions
    pred_scores = torch.rand(dataset.num_models, dataset.num_servers)
    
    # Get positive indices
    model_indices, positive_servers_global = dataset.get_evaluation_pairs()
    positive_servers_local = [
        [s - dataset.num_users - dataset.num_models for s in servers]
        for servers in positive_servers_global
    ]
    
    # Compute metrics
    metrics = metrics_calculator.compute_all_metrics(
        pred_scores,
        positive_servers_local,
        prefix="random_"
    )
    
    # Log results
    logger.log_results('train', metrics, metrics_calculator.k_list)
    
    print(f"Random Baseline - NDCG@5: {metrics['random_ndcg@5']:.4f}")
    return metrics


def run_popular_baseline(dataset, metrics_calculator, logger, k=5):
    """
    Run popularity-based baseline (connectivity only, no user info)
    
    This baseline can ONLY use network topology (connectivity),
    and CANNOT use user geographic distribution or model characteristics.
    
    Args:
        dataset: Dataset object
        metrics_calculator: Metrics calculator
        logger: Baseline logger
        k: Number of servers to recommend
    
    Returns:
        Dictionary of baseline metrics
    """
    print("\n" + "=" * 80)
    print("POPULAR BASELINE (Connectivity-based)")
    print("=" * 80)
    
    # Score servers by connectivity (degree centrality)
    # This is the ONLY information available to this baseline
    server_degree = dataset.topology.sum(axis=0)
    server_scores = torch.from_numpy(server_degree).float()
    
    # All models get the same ranking (cannot personalize)
    pred_scores = server_scores.unsqueeze(0).expand(dataset.num_models, -1)
    
    print(f"  Strategy: Rank all servers by degree centrality")
    print(f"  Limitation: Cannot consider user locations or model characteristics")
    
    # Get positive indices
    model_indices, positive_servers_global = dataset.get_evaluation_pairs()
    positive_servers_local = [
        [s - dataset.num_users - dataset.num_models for s in servers]
        for servers in positive_servers_global
    ]
    
    # Compute metrics
    metrics = metrics_calculator.compute_all_metrics(
        pred_scores,
        positive_servers_local,
        prefix="popular_"
    )
    
    # Log results
    logger.log_results('train', metrics, metrics_calculator.k_list)
    
    print(f"Popular Baseline - NDCG@5: {metrics['popular_ndcg@5']:.4f}")
    return metrics


def run_load_balanced_baseline(dataset, metrics_calculator, logger, k=5):
    """
    Run load-balanced baseline
    
    Assign models to least-loaded servers to achieve uniform distribution.
    This baseline aims for fair load distribution across all servers.
    
    Args:
        dataset: Dataset object
        metrics_calculator: Metrics calculator
        logger: Baseline logger
        k: Number of servers to recommend
    
    Returns:
        Dictionary of baseline metrics
    """
    print("\n" + "=" * 80)
    print("LOAD-BALANCED BASELINE")
    print("=" * 80)
    print(f"  Strategy: Assign models to least-loaded servers")
    print(f"  Goal: Achieve uniform load distribution")
    
    # Initialize server loads
    server_loads = np.zeros(dataset.num_servers)
    
    # For each model, score servers inversely to their current load
    pred_scores = torch.zeros(dataset.num_models, dataset.num_servers)
    for model_id in range(dataset.num_models):
        # Score = 1 / (load + 1), higher score for less loaded servers
        scores = 1.0 / (server_loads + 1.0)
        pred_scores[model_id] = torch.from_numpy(scores).float()
        
        # Update loads (simulate deploying to top-5)
        top_servers = np.argsort(-scores)[:5]
        for sid in top_servers:
            server_loads[sid] += 1
    
    # Get positive indices
    model_indices, positive_servers_global = dataset.get_evaluation_pairs()
    positive_servers_local = [
        [s - dataset.num_users - dataset.num_models for s in servers]
        for servers in positive_servers_global
    ]
    
    # Compute metrics
    metrics = metrics_calculator.compute_all_metrics(
        pred_scores,
        positive_servers_local,
        prefix="load_balanced_"
    )
    
    # Log results
    logger.log_results('train', metrics, metrics_calculator.k_list)
    
    print(f"Load-Balanced Baseline - NDCG@5: {metrics['load_balanced_ndcg@5']:.4f}")
    return metrics


def run_resource_matching_baseline(dataset, metrics_calculator, logger, k=5):
    """
    Run resource-matching baseline
    
    Match model resource requirements to server capacities.
    Servers with better resource fit for a model get higher scores.
    
    Args:
        dataset: Dataset object
        metrics_calculator: Metrics calculator
        logger: Baseline logger
        k: Number of servers to recommend
    
    Returns:
        Dictionary of baseline metrics
    """
    print("\n" + "=" * 80)
    print("RESOURCE-MATCHING BASELINE")
    print("=" * 80)
    print(f"  Strategy: Match model requirements to server capacities")
    
    # Extract features
    models_df = dataset.models_df
    servers_df = dataset.servers_df
    
    # Model requirements: [size, resource_need]
    model_requirements = models_df[['Modelsize', 'Modelresource']].values.astype(float)
    model_requirements = (model_requirements - model_requirements.mean(0)) / (model_requirements.std(0) + 1e-8)
    
    # Server capacities: [compute, storage]
    server_capacities = servers_df[['ComputationCapacity', 'StorageCapacity']].values.astype(float)
    server_capacities = (server_capacities - server_capacities.mean(0)) / (server_capacities.std(0) + 1e-8)
    
    # For each model, compute matching score with all servers
    pred_scores = torch.zeros(dataset.num_models, dataset.num_servers)
    for model_idx in range(dataset.num_models):
        model_req = model_requirements[model_idx]
        # Compute Euclidean distance (lower = better match)
        distances = np.linalg.norm(server_capacities - model_req, axis=1)
        # Convert to scores (lower distance = higher score)
        max_dist = distances.max() + 1e-10
        scores = 1.0 - (distances / max_dist)
        pred_scores[model_idx] = torch.from_numpy(scores).float()
    
    # Get positive indices
    model_indices, positive_servers_global = dataset.get_evaluation_pairs()
    positive_servers_local = [
        [s - dataset.num_users - dataset.num_models for s in servers]
        for servers in positive_servers_global
    ]
    
    # Compute metrics
    metrics = metrics_calculator.compute_all_metrics(
        pred_scores,
        positive_servers_local,
        prefix="resource_matching_"
    )
    
    # Log results
    logger.log_results('train', metrics, metrics_calculator.k_list)
    
    print(f"Resource-Matching Baseline - NDCG@5: {metrics['resource_matching_ndcg@5']:.4f}")
    return metrics


def run_user_aware_baseline(dataset, metrics_calculator, logger, k=5):
    """
    Run user-aware baseline (simple heuristic)
    
    For each model, compute average user location and select nearest servers.
    This is a simple heuristic that uses user information but no learning.
    
    Args:
        dataset: Dataset object
        metrics_calculator: Metrics calculator
        logger: Baseline logger
        k: Number of servers to recommend
    
    Returns:
        Dictionary of baseline metrics
    """
    print("\n" + "=" * 80)
    print("USER-AWARE BASELINE (Simple Heuristic)")
    print("=" * 80)
    
    print(f"  Strategy: For each model, select servers nearest to average user location")
    
    # Get user-model relationships
    user_model_groups = {}
    for _, row in dataset.user_model_df.iterrows():
        model_id = int(row['ModelID']) - 1
        user_id = int(row['UserID']) - 1
        if model_id not in user_model_groups:
            user_model_groups[model_id] = []
        user_model_groups[model_id].append(user_id)
    
    # Compute scores for each model
    pred_scores = torch.zeros(dataset.num_models, dataset.num_servers)
    user_locs = dataset.users_df[['Lo', 'La']].values
    server_locs = dataset.servers_df[['Lo', 'La']].values
    
    for model_id in range(dataset.num_models):
        if model_id in user_model_groups:
            user_ids = user_model_groups[model_id]
            # Compute average user location
            avg_user_loc = user_locs[user_ids].mean(axis=0)
            
            # Compute distance from each server to average user location
            distances = np.linalg.norm(server_locs - avg_user_loc, axis=1)
            # Convert to score (closer = higher score)
            max_dist = distances.max() + 1e-10
            scores = 1.0 - (distances / max_dist)
            pred_scores[model_id] = torch.from_numpy(scores).float()
        else:
            # No users, use uniform scores
            pred_scores[model_id] = torch.ones(dataset.num_servers) * 0.5
    
    # Get positive indices
    model_indices, positive_servers_global = dataset.get_evaluation_pairs()
    positive_servers_local = [
        [s - dataset.num_users - dataset.num_models for s in servers]
        for servers in positive_servers_global
    ]
    
    # Compute metrics
    metrics = metrics_calculator.compute_all_metrics(
        pred_scores,
        positive_servers_local,
        prefix="user_aware_"
    )
    
    # Log results
    logger.log_results('train', metrics, metrics_calculator.k_list)
    
    print(f"User-Aware Baseline - NDCG@5: {metrics['user_aware_ndcg@5']:.4f}")
    return metrics


def main():
    print("\n" + "=" * 80)
    print("OPTIMIZED HGNN TRAINING WITH EXPERIMENT LOGGING")
    print("=" * 80)
    
    # Configuration - ENHANCED for deep training and convergence analysis
    config = {
        'k_positive': 10,          # GT size = 10 servers (to make P@5 != R@5)
        'n_hid': 128,
        'dropout': 0.05,           # Further reduced for more learning capacity
        'lr': 0.0005,              # Lower LR for smoother, stable convergence
        'weight_decay': 0.0001,
        'max_epochs': 500,         # 500 epochs to fully observe convergence curve
        'batch_size': 128,  
        'num_negatives': 6,
        'margin': 1.0,
        'print_freq': 10,          # Print every 10 epochs for clearer trend visualization
        'patience': 9999,          # DISABLED early stopping - run all 500 epochs for complete data
        'warmup_epochs': 30,       # Longer warmup for very stable start
        'eval_k_list': [1, 3, 5, 10, 20]
    }
    
    # Experiment name
    exp_name = f"hgnn_k{config['k_positive']}_h{config['n_hid']}_e{config['max_epochs']}"
    
    # Initialize logger
    logger = ExperimentLogger(exp_name, config)
    
    # Print configuration
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Step 1: Load data
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)
    train_dataset = TopKPlacementDataset(split='train', k_positive=config['k_positive'])
    train_dataset.prepare()
    
    # Step 2: Construct hypergraph (this is the slow part)
    print("\n" + "=" * 80)
    print("STEP 2: CONSTRUCTING HYPERGRAPH (ONE-TIME GPU COMPUTATION)")
    print("=" * 80)
    use_gpu = torch.cuda.is_available()
    
    # Keep model-server edges as learning bridges
    # Although not perfectly aligned with GT, they provide essential
    # connectivity between model and server nodes for HGNN learning
    print("Note: Using original model-server mappings as learning bridges")
    
    graph_start_time = time.time()
    H, G, edge_info = construct_H_for_model_placement(train_dataset, k_neig=10, use_gpu=use_gpu)
    graph_time = time.time() - graph_start_time
    print(f"Graph construction time: {graph_time:.2f} seconds")
    
    # Convert to tensors and CACHE on GPU
    print("\nCaching tensors on GPU...")
    features = torch.FloatTensor(train_dataset.node_features).to(device)
    G_tensor = torch.FloatTensor(G).to(device)
    
    print(f"Tensor info:")
    print(f"  Features: {features.shape}")
    print(f"  Graph G: {G_tensor.shape}")
    
    # Step 3: Initialize model
    print("\n" + "=" * 80)
    print("STEP 3: INITIALIZING MODEL")
    print("=" * 80)
    model = HGNN_ModelPlacement(
        in_ch=features.shape[1],
        n_hid=config['n_hid'],
        num_users=train_dataset.num_users,
        num_models=train_dataset.num_models,
        num_servers=train_dataset.num_servers,
        dropout=config['dropout']
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {num_params:,} trainable parameters")
    
    # Initialize metrics calculator
    metrics_calculator = RecommendationMetrics(k_list=config['eval_k_list'])
    
    # Run baselines (5 different methods)
    baseline_logger_random = BaselineLogger("random")
    baseline_logger_popular = BaselineLogger("popular")
    baseline_logger_load_balanced = BaselineLogger("load_balanced")
    baseline_logger_resource = BaselineLogger("resource_matching")
    baseline_logger_user_aware = BaselineLogger("user_aware")
    
    random_metrics = run_random_baseline(train_dataset, metrics_calculator, baseline_logger_random)
    popular_metrics = run_popular_baseline(train_dataset, metrics_calculator, baseline_logger_popular)
    load_balanced_metrics = run_load_balanced_baseline(train_dataset, metrics_calculator, baseline_logger_load_balanced)
    resource_metrics = run_resource_matching_baseline(train_dataset, metrics_calculator, baseline_logger_resource)
    user_aware_metrics = run_user_aware_baseline(train_dataset, metrics_calculator, baseline_logger_user_aware)
    
    # Prepare positive indices for TopKRankingLoss
    positive_indices = []
    for model_id in range(train_dataset.num_models):
        pos_servers = train_dataset.model_positive_servers[model_id]
        positive_indices.append(pos_servers)
    positive_indices = torch.LongTensor(positive_indices).to(device)  # [num_models, k]
    
    # Training setup - SIMPLIFIED to avoid OOM
    criterion = SoftmaxRankingLoss(k=config['k_positive'], temperature=0.05)
    # Use simple Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], 
                          weight_decay=config['weight_decay'])
    # Use only ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=False
    )
    
    # Learning rate warmup function
    def adjust_learning_rate_warmup(optimizer, epoch, base_lr, warmup_epochs):
        """Gradually increase learning rate during warmup"""
        if epoch < warmup_epochs:
            lr = base_lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        return base_lr
    
    # Step 4: Training loop
    print("\n" + "=" * 80)
    print("STEP 4: TRAINING")
    print("=" * 80)
    
    # Diagnostic: Check initial predictions
    print("Initial model check:")
    model.eval()
    with torch.no_grad():
        init_scores = model(features, G_tensor)
        print(f"  Score matrix shape: {init_scores.shape}")
        print(f"  Score range: [{init_scores.min():.4f}, {init_scores.max():.4f}]")
        print(f"  Score mean: {init_scores.mean():.4f}, std: {init_scores.std():.4f}")
    model.train()
    
    best_ndcg = 0.0
    best_epoch = 0
    best_metrics = {}
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(config['max_epochs']):
        # Apply warmup schedule
        if epoch < config['warmup_epochs']:
            current_lr = adjust_learning_rate_warmup(
                optimizer, epoch, config['lr'], config['warmup_epochs']
            )
            if epoch == 0:
                print(f"Warmup phase: LR will gradually increase to {config['lr']}")
        
        # Train one epoch (no triplets needed, use all scores + positive indices)
        epoch_loss = train_one_epoch(
            model, features, G_tensor, positive_indices,
            criterion, optimizer,
            train_dataset, device
        )
        
        # Clear cache periodically
        if epoch % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        # Evaluation
        if (epoch + 1) % config['print_freq'] == 0 or epoch == 0:
            # Diagnostic: Check score evolution
            if epoch == 0 or (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    curr_scores = model(features, G_tensor)
                    print(f"  [Epoch {epoch+1}] Score stats - "
                          f"range: [{curr_scores.min():.4f}, {curr_scores.max():.4f}], "
                          f"mean: {curr_scores.mean():.4f}, std: {curr_scores.std():.4f}")
                model.train()
            
            metrics = evaluate(
                model, features, G_tensor, train_dataset,
                metrics_calculator, device
            )
            
            # Add loss to metrics
            metrics['loss'] = epoch_loss
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log to CSV
            logger.log_train_epoch(epoch + 1, metrics, current_lr)
            
            # Print progress
            print(f"Epoch {epoch + 1:3d}/{config['max_epochs']} | "
                  f"Loss: {epoch_loss:.4f} | "
                  f"P@5: {metrics['precision@5']:.4f} | "
                  f"R@5: {metrics['recall@5']:.4f} | "
                  f"NDCG@5: {metrics['ndcg@5']:.4f} | "
                  f"F1@5: {metrics['f1@5']:.4f}")
            
            # Check for improvement
            is_best = metrics['ndcg@5'] > best_ndcg
            if is_best:
                best_ndcg = metrics['ndcg@5']
                best_epoch = epoch + 1
                best_metrics = metrics.copy()
                patience_counter = 0
                print(f"  >> New best! NDCG@5: {best_ndcg:.4f}")
            else:
                patience_counter += config['print_freq']
            
            logger.save_checkpoint(epoch + 1, model, optimizer, metrics, is_best)
            
            # Update scheduler
            scheduler.step(metrics['ndcg@5'])
            
            # Early stopping check
            if patience_counter >= config['patience']:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
    
    # Training completed
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print(f"Training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Best NDCG@5: {best_ndcg:.4f} at epoch {best_epoch}")
    
    # Save final summary
    logger.save_final_summary(best_epoch, best_metrics, total_time)
    logger.print_summary_table(best_metrics)
    
    # Final comparison with baselines
    print("\n" + "=" * 80)
    print("COMPREHENSIVE COMPARISON WITH 5 BASELINES")
    print("=" * 80)
    
    # Collect all methods
    methods = [
        ("Random", random_metrics, "random_"),
        ("Popular", popular_metrics, "popular_"),
        ("Load-Balanced", load_balanced_metrics, "load_balanced_"),
        ("Resource-Matching", resource_metrics, "resource_matching_"),
        ("User-Aware", user_aware_metrics, "user_aware_"),
    ]
    
    # Print header
    print(f"\n{'Method':<20} {'P@5':>8} {'R@5':>8} {'F1@5':>8} {'NDCG@5':>8}")
    print("-" * 60)
    
    # Print baselines
    for name, metrics, prefix in methods:
        p5 = metrics.get(f'{prefix}precision@5', 0)
        r5 = metrics.get(f'{prefix}recall@5', 0)
        f1_5 = metrics.get(f'{prefix}f1@5', 0)
        ndcg5 = metrics.get(f'{prefix}ndcg@5', 0)
        print(f"{name:<20} {p5:>8.4f} {r5:>8.4f} {f1_5:>8.4f} {ndcg5:>8.4f}")
    
    # Print HGNN
    print("-" * 60)
    hgnn_p5 = best_metrics.get('precision@5', 0)
    hgnn_r5 = best_metrics.get('recall@5', 0)
    hgnn_f1_5 = best_metrics.get('f1@5', 0)
    print(f"{'HGNN (Ours)':<20} {hgnn_p5:>8.4f} {hgnn_r5:>8.4f} {hgnn_f1_5:>8.4f} {best_ndcg:>8.4f}")
    
    # Relative improvements
    print(f"\n{'NDCG@5 Relative Improvements:'}")
    print("-" * 60)
    for name, metrics, prefix in methods:
        baseline_ndcg = metrics.get(f'{prefix}ndcg@5', 0)
        if baseline_ndcg > 0:
            improvement = ((best_ndcg / baseline_ndcg) - 1) * 100
            print(f"  vs {name:<18}: {improvement:+7.1f}%")
    
    print("\n" + "=" * 80)
    print("ALL DONE!")
    print("=" * 80)
    print(f"\nResults saved in: {logger.exp_dir}")


if __name__ == '__main__':
    main()

