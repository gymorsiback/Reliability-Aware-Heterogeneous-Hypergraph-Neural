"""
Inference script for HGNN model placement recommendation

This script demonstrates how to use the trained HGNN model for real-world inference:
- Input: A set of models that need to be deployed
- Output: Top-K recommended servers for each model
- Use case: System administrator wants to deploy new models optimally

Usage:
    python inference_model_placement.py --checkpoint results/hgnn_k10_h128_e500_TIMESTAMP/checkpoints/best_model.pth
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

from models.HGNN import HGNN_ModelPlacement
from datasets.topk_placement_loader import TopKPlacementDataset
from utils.hypergraph_utils import construct_H_for_model_placement


class ModelPlacementInference:
    """
    Inference engine for model placement recommendation
    """
    
    def __init__(self, checkpoint_path, data_root='datasets/server', device='cuda'):
        """
        Initialize inference engine
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            data_root: Path to dataset directory
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.data_root = Path(data_root)
        
        # Load dataset
        print("=" * 80)
        print("LOADING DATASET")
        print("=" * 80)
        self.dataset = TopKPlacementDataset(split='train', k_positive=10, data_root=data_root)
        self.dataset.prepare()
        print(f"Loaded {self.dataset.num_models} models, {self.dataset.num_servers} servers, {self.dataset.num_users} users")
        
        # Construct hypergraph
        print("\n" + "=" * 80)
        print("CONSTRUCTING HYPERGRAPH")
        print("=" * 80)
        H, G, edge_info = construct_H_for_model_placement(self.dataset, use_gpu=(device=='cuda'))
        self.G_tensor = torch.from_numpy(G).float().to(self.device)
        print(f"Hypergraph constructed: {H.shape[0]} nodes, {H.shape[1]} hyperedges")
        
        # Prepare features
        self.features = torch.from_numpy(self.dataset.node_features).float().to(self.device)
        print(f"Node features: {self.features.shape}")
        
        # Load model
        print("\n" + "=" * 80)
        print("LOADING MODEL")
        print("=" * 80)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model = HGNN_ModelPlacement(
            in_ch=self.features.shape[1],
            n_hid=checkpoint['config']['n_hid'],
            num_users=self.dataset.num_users,
            num_models=self.dataset.num_models,
            num_servers=self.dataset.num_servers,
            dropout=0.0  # No dropout during inference
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from: {checkpoint_path}")
        print(f"Training epoch: {checkpoint['epoch']}")
        print(f"Best NDCG@5: {checkpoint.get('best_ndcg', 'N/A')}")
        
        # Cache for efficiency
        self._server_embeddings = None
        self._precompute_embeddings()
    
    def _precompute_embeddings(self):
        """
        Precompute server embeddings for fast inference
        """
        print("\n" + "=" * 80)
        print("PRECOMPUTING EMBEDDINGS")
        print("=" * 80)
        with torch.no_grad():
            all_embeddings = self.model.get_embeddings(self.features, self.G_tensor)
            
            # Extract server embeddings
            start_idx = self.dataset.num_users + self.dataset.num_models
            end_idx = start_idx + self.dataset.num_servers
            self._server_embeddings = all_embeddings[start_idx:end_idx]
            
        print(f"Server embeddings cached: {self._server_embeddings.shape}")
    
    def recommend_servers_for_model(self, model_id, top_k=5):
        """
        Recommend top-K servers for a given model
        
        Args:
            model_id: Model ID (0-indexed)
            top_k: Number of servers to recommend
        
        Returns:
            recommendations: List of (server_id, score, server_info) tuples
        """
        if model_id < 0 or model_id >= self.dataset.num_models:
            raise ValueError(f"Invalid model_id: {model_id}. Must be in [0, {self.dataset.num_models-1}]")
        
        with torch.no_grad():
            # Get model embedding
            all_embeddings = self.model.get_embeddings(self.features, self.G_tensor)
            model_start_idx = self.dataset.num_users
            model_embedding = all_embeddings[model_start_idx + model_id].unsqueeze(0)
            
            # Compute scores with all servers
            scores = self.model.predict_placements(model_embedding, self._server_embeddings)
            scores = scores.squeeze().cpu().numpy()
            
            # Get top-K servers
            top_k_indices = np.argsort(-scores)[:top_k]
            
            # Prepare recommendations with server info
            recommendations = []
            for rank, server_idx in enumerate(top_k_indices):
                server_info = self.dataset.servers_df.iloc[server_idx].to_dict()
                recommendations.append({
                    'rank': rank + 1,
                    'server_id': int(server_idx + 1),  # 1-indexed for display
                    'score': float(scores[server_idx]),
                    'location': (float(server_info['Lo']), float(server_info['La'])),
                    'compute_capacity': float(server_info['ComputationCapacity']),
                    'storage_capacity': float(server_info['StorageCapacity']),
                    'link_bandwidth': float(server_info['LinkBandwidth'])
                })
            
            return recommendations
    
    def recommend_batch(self, model_ids, top_k=5):
        """
        Recommend servers for multiple models at once
        
        Args:
            model_ids: List of model IDs
            top_k: Number of servers to recommend per model
        
        Returns:
            batch_recommendations: Dictionary mapping model_id -> recommendations
        """
        batch_recommendations = {}
        for model_id in model_ids:
            batch_recommendations[model_id] = self.recommend_servers_for_model(model_id, top_k)
        return batch_recommendations
    
    def get_model_info(self, model_id):
        """
        Get information about a model
        
        Args:
            model_id: Model ID (0-indexed)
        
        Returns:
            model_info: Dictionary with model details
        """
        model_info = self.dataset.models_df.iloc[model_id].to_dict()
        
        # Get associated users
        user_model_interactions = self.dataset.user_model_df
        associated_users = user_model_interactions[
            user_model_interactions['ModelID'] == (model_id + 1)
        ]['UserID'].values
        
        return {
            'model_id': int(model_id + 1),  # 1-indexed
            'model_type': model_info['ModelType'],
            'arena_score': float(model_info['ArenaScore']),
            'model_size': float(model_info['Modelsize']),
            'resource_requirement': float(model_info['Modelresource']),
            'num_users': len(associated_users),
            'user_ids': [int(uid) for uid in associated_users[:10]]  # Show first 10
        }


def compute_ranking_metrics(predictions, ground_truth_dict, k=5):
    """
    Compute unified ranking metrics (P@K, R@K, F1@K, NDCG@K) against Ground Truth
    
    Args:
        predictions: Dict {model_id: np.array of predicted server indices}
        ground_truth_dict: Dict {model_id: list of positive server indices}
        k: Top-K for evaluation
    
    Returns:
        metrics: Dict of averaged metrics
    """
    precisions = []
    recalls = []
    f1_scores = []
    ndcgs = []
    hit_rates = []
    
    for model_id, pred_servers in predictions.items():
        if model_id not in ground_truth_dict:
            continue
        
        gt_servers = set(ground_truth_dict[model_id]) if not isinstance(ground_truth_dict[model_id], set) else ground_truth_dict[model_id]
        pred_servers_set = set(pred_servers[:k].tolist() if torch.is_tensor(pred_servers) else pred_servers[:k])
        
        # Number of hits
        hits = len(gt_servers & pred_servers_set)
        
        # Precision@K
        precision = hits / k if k > 0 else 0.0
        
        # Recall@K
        recall = hits / len(gt_servers) if len(gt_servers) > 0 else 0.0
        
        # F1@K
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Hit Rate@K
        hit_rate = 1.0 if hits > 0 else 0.0
        
        # NDCG@K
        dcg = 0.0
        idcg = 0.0
        for i, server_id in enumerate(pred_servers[:k]):
            if int(server_id) in gt_servers:
                dcg += 1.0 / np.log2(i + 2)
        for i in range(min(k, len(gt_servers))):
            idcg += 1.0 / np.log2(i + 2)
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        ndcgs.append(ndcg)
        hit_rates.append(hit_rate)
    
    return {
        f'precision@{k}': np.mean(precisions),
        f'recall@{k}': np.mean(recalls),
        f'f1@{k}': np.mean(f1_scores),
        f'ndcg@{k}': np.mean(ndcgs),
        f'hit_rate@{k}': np.mean(hit_rates),
        'num_models_evaluated': len(precisions)
    }


def compute_baseline_recommendations(dataset, model_id, top_k=5, method='random'):
    """
    Compute baseline recommendations for comparison
    
    Args:
        dataset: Dataset object
        model_id: Model ID (0-indexed)
        top_k: Number of servers to recommend
        method: 'random', 'popular', 'user_aware', 'resource_matching'
    
    Returns:
        recommendations: List of (server_id, score, server_info) tuples
    """
    num_servers = dataset.num_servers
    servers_df = dataset.servers_df
    
    if method == 'random':
        # Random baseline: Random scores
        scores = np.random.random(num_servers)
    
    elif method == 'popular':
        # Popular baseline: Server degree centrality (connectivity)
        scores = dataset.topology.sum(axis=0)
    
    elif method == 'user_aware':
        # User-Aware baseline: Distance to average user location
        user_model_groups = {}
        for _, row in dataset.user_model_df.iterrows():
            mid = int(row['ModelID']) - 1
            uid = int(row['UserID']) - 1
            if mid not in user_model_groups:
                user_model_groups[mid] = []
            user_model_groups[mid].append(uid)
        
        if model_id in user_model_groups:
            user_ids = user_model_groups[model_id]
            user_locs = dataset.users_df.iloc[user_ids][['Lo', 'La']].values
            avg_user_loc = user_locs.mean(axis=0)
            
            server_locs = servers_df[['Lo', 'La']].values
            distances = np.linalg.norm(server_locs - avg_user_loc, axis=1)
            scores = 1.0 - (distances / (distances.max() + 1e-10))
        else:
            scores = np.ones(num_servers) * 0.5
    
    elif method == 'resource_matching':
        # Resource-Matching baseline: Match model requirements to server capacities
        model_req = dataset.models_df.iloc[model_id][['Modelsize', 'Modelresource']].values.astype(float)
        model_req = (model_req - dataset.models_df[['Modelsize', 'Modelresource']].mean().values) / \
                    (dataset.models_df[['Modelsize', 'Modelresource']].std().values + 1e-8)
        
        server_caps = servers_df[['ComputationCapacity', 'StorageCapacity']].values.astype(float)
        server_caps = (server_caps - server_caps.mean(axis=0)) / (server_caps.std(axis=0) + 1e-8)
        
        distances = np.linalg.norm(server_caps - model_req, axis=1)
        scores = 1.0 - (distances / (distances.max() + 1e-10))
    
    elif method == 'load_balanced':
        # Load-Balanced baseline: Distribute models evenly across servers
        scores = np.ones(num_servers) + np.random.random(num_servers) * 0.1
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Get top-K
    top_k_indices = np.argsort(-scores)[:top_k]
    
    # Prepare recommendations
    recommendations = []
    for rank, server_idx in enumerate(top_k_indices):
        server_info = servers_df.iloc[server_idx].to_dict()
        recommendations.append({
            'rank': rank + 1,
            'server_id': int(server_idx + 1),
            'score': float(scores[server_idx]),
            'location': (float(server_info['Lo']), float(server_info['La'])),
            'compute_capacity': float(server_info['ComputationCapacity']),
            'storage_capacity': float(server_info['StorageCapacity']),
            'link_bandwidth': float(server_info['LinkBandwidth'])
        })
    
    return recommendations


def save_inference_results(all_results, model_ids, args, output_dir):
    """
    Save comprehensive inference results for research analysis
    
    Args:
        all_results: Dictionary of all inference results
        model_ids: List of model IDs tested
        args: Command line arguments
        output_dir: Output directory path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save detailed recommendations for each model (CSV)
    print("\nSaving detailed results...")
    
    # HGNN recommendations
    hgnn_records = []
    for model_id in model_ids:
        model_info = all_results[model_id]['model_info']
        for rec in all_results[model_id]['hgnn']:
            hgnn_records.append({
                'model_id': model_id + 1,
                'model_type': model_info['model_type'],
                'arena_score': model_info['arena_score'],
                'model_size': model_info['model_size'],
                'resource_requirement': model_info['resource_requirement'],
                'num_users': model_info['num_users'],
                'rank': rec['rank'],
                'server_id': rec['server_id'],
                'score': rec['score'],
                'server_lo': rec['location'][0],
                'server_la': rec['location'][1],
                'compute_capacity': rec['compute_capacity'],
                'storage_capacity': rec['storage_capacity'],
                'link_bandwidth': rec['link_bandwidth']
            })
    
    hgnn_df = pd.DataFrame(hgnn_records)
    hgnn_df.to_csv(output_dir / 'hgnn_recommendations.csv', index=False)
    print(f"  Saved: {output_dir / 'hgnn_recommendations.csv'}")
    
    # Baseline recommendations
    baseline_methods = ['random', 'popular', 'user_aware', 'resource_matching']
    baseline_names = ['Random', 'Popular', 'User-Aware', 'Resource-Matching']
    
    for method, name in zip(baseline_methods, baseline_names):
        baseline_records = []
        for model_id in model_ids:
            if method in all_results[model_id]:
                for rec in all_results[model_id][method]:
                    baseline_records.append({
                        'model_id': model_id + 1,
                        'rank': rec['rank'],
                        'server_id': rec['server_id'],
                        'score': rec['score'],
                        'server_lo': rec['location'][0],
                        'server_la': rec['location'][1],
                        'compute_capacity': rec['compute_capacity'],
                        'storage_capacity': rec['storage_capacity'],
                        'link_bandwidth': rec['link_bandwidth']
                    })
        
        baseline_df = pd.DataFrame(baseline_records)
        filename = f'{method}_recommendations.csv'
        baseline_df.to_csv(output_dir / filename, index=False)
        print(f"  Saved: {output_dir / filename}")
    
    # 2. Save overlap analysis (CSV)
    overlap_records = []
    for model_id in model_ids:
        hgnn_servers = set([r['server_id'] for r in all_results[model_id]['hgnn']])
        
        row = {
            'model_id': model_id + 1,
            'hgnn_servers': ','.join(map(str, sorted(hgnn_servers)))
        }
        
        for method, name in zip(baseline_methods, baseline_names):
            if method in all_results[model_id]:
                baseline_servers = set([r['server_id'] for r in all_results[model_id][method]])
                overlap = hgnn_servers & baseline_servers
                row[f'{method}_overlap_count'] = len(overlap)
                row[f'{method}_overlap_ratio'] = len(overlap) / args.top_k
                row[f'{method}_servers'] = ','.join(map(str, sorted(baseline_servers)))
                row[f'{method}_common'] = ','.join(map(str, sorted(overlap))) if overlap else ''
        
        overlap_records.append(row)
    
    overlap_df = pd.DataFrame(overlap_records)
    overlap_df.to_csv(output_dir / 'overlap_analysis.csv', index=False)
    print(f"  Saved: {output_dir / 'overlap_analysis.csv'}")
    
    # 3. Save summary statistics (CSV)
    summary_records = []
    for method, name in zip(baseline_methods, baseline_names):
        total_overlap = 0
        for model_id in model_ids:
            if method in all_results[model_id]:
                hgnn_servers = set([r['server_id'] for r in all_results[model_id]['hgnn']])
                baseline_servers = set([r['server_id'] for r in all_results[model_id][method]])
                total_overlap += len(hgnn_servers & baseline_servers)
        
        avg_overlap = total_overlap / len(model_ids)
        avg_overlap_ratio = avg_overlap / args.top_k
        
        summary_records.append({
            'baseline': name,
            'avg_overlap_count': avg_overlap,
            'avg_overlap_ratio': avg_overlap_ratio,
            'total_models_tested': len(model_ids),
            'top_k': args.top_k
        })
    
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(output_dir / 'summary_statistics.csv', index=False)
    print(f"  Saved: {output_dir / 'summary_statistics.csv'}")
    
    # 4. Save complete JSON
    json_output = {
        'metadata': {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'checkpoint': args.checkpoint,
            'model_ids': model_ids,
            'top_k': args.top_k,
            'compare_baselines': args.compare_baselines,
            'device': args.device
        },
        'results': all_results,
        'summary': summary_records
    }
    
    with open(output_dir / 'complete_results.json', 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"  Saved: {output_dir / 'complete_results.json'}")
    
    # 5. Save model info (CSV)
    model_info_records = []
    for model_id in model_ids:
        model_info = all_results[model_id]['model_info']
        model_info_records.append({
            'model_id': model_info['model_id'],
            'model_type': model_info['model_type'],
            'arena_score': model_info['arena_score'],
            'model_size': model_info['model_size'],
            'resource_requirement': model_info['resource_requirement'],
            'num_users': model_info['num_users']
        })
    
    model_info_df = pd.DataFrame(model_info_records)
    model_info_df.to_csv(output_dir / 'model_info.csv', index=False)
    print(f"  Saved: {output_dir / 'model_info.csv'}")
    
    print(f"\nAll results saved to: {output_dir.absolute()}")


def main():
    parser = argparse.ArgumentParser(description='HGNN Model Placement Inference with Baseline Comparison')
    parser.add_argument('--checkpoint', type=str, required=False,
                        default='results/hgnn_k10_h128_e500_20251104_155422/checkpoints/best_model.pth',
                        help='Path to model checkpoint (Ours)')
    parser.add_argument('--checkpoint_sota', type=str, required=False,
                        default='results/hgnn_k10_h128_e500_20251031_191447/checkpoints/best_model.pth',
                        help='Path to SOTA model checkpoint')
    parser.add_argument('--data_root', type=str, default='datasets/server',
                        help='Path to dataset directory')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of servers to recommend')
    parser.add_argument('--model_ids', type=str, default='0,1,2,5,10',
                        help='Comma-separated model IDs to test (0-indexed)')
    parser.add_argument('--compare_baselines', action='store_true', default=True,
                        help='Compare with 4 baselines')
    parser.add_argument('--output_dir', type=str, default='results/inference',
                        help='Directory to save inference results')
    
    args = parser.parse_args()
    
    # Initialize inference engine for main model (Ours)
    print("\n" + "=" * 80)
    print("LOADING OURS MODEL")
    print("=" * 80)
    engine = ModelPlacementInference(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        device=args.device
    )
    
    # Initialize inference engine for SOTA model
    print("\n" + "=" * 80)
    print("LOADING SOTA MODEL")
    print("=" * 80)
    engine_sota = ModelPlacementInference(
        checkpoint_path=args.checkpoint_sota,
        data_root=args.data_root,
        device=args.device
    )
    
    # Parse model IDs
    model_ids = [int(x.strip()) for x in args.model_ids.split(',')]
    
    # Run inference
    print("\n" + "=" * 80)
    print("INFERENCE RESULTS WITH BASELINE COMPARISON")
    print("=" * 80)
    
    all_results = {}
    baseline_methods = ['random', 'popular', 'user_aware', 'resource_matching']
    baseline_names = ['Random', 'Popular', 'User-Aware', 'Resource-Matching']
    
    for model_id in model_ids:
        print(f"\n{'=' * 80}")
        print(f"MODEL {model_id + 1}")
        print('=' * 80)
        
        # Get model info
        model_info = engine.get_model_info(model_id)
        print(f"\nModel Information:")
        print(f"  Type: {model_info['model_type']}")
        print(f"  Arena Score: {model_info['arena_score']:.2f}")
        print(f"  Size: {model_info['model_size']:.2f}")
        print(f"  Resource Requirement: {model_info['resource_requirement']:.2f}")
        print(f"  Associated Users: {model_info['num_users']}")
        
        # HGNN recommendations
        hgnn_recommendations = engine.recommend_servers_for_model(model_id, top_k=args.top_k)
        
        print(f"\n{'HGNN (Ours)':<25} Top-{args.top_k} Recommended Servers:")
        print(f"{'Rank':<6} {'ServerID':<10} {'Score':<12} {'Location':<20} {'Compute':<10}")
        print("-" * 68)
        for rec in hgnn_recommendations:
            print(f"{rec['rank']:<6} {rec['server_id']:<10} {rec['score']:<12.6f} "
                  f"({rec['location'][0]:>6.2f},{rec['location'][1]:>6.2f})   "
                  f"{rec['compute_capacity']:<10.2f}")
        
        # Store results
        model_results = {
            'model_info': model_info,
            'hgnn': hgnn_recommendations
        }
        
        # Baseline comparisons
        if args.compare_baselines:
            print(f"\n{'-' * 80}")
            print("BASELINE COMPARISONS")
            print('-' * 80)
            
            for method, name in zip(baseline_methods, baseline_names):
                baseline_recs = compute_baseline_recommendations(
                    engine.dataset, model_id, top_k=args.top_k, method=method
                )
                model_results[method] = baseline_recs
                
                print(f"\n{name:<25} Top-{args.top_k}:")
                print(f"{'Rank':<6} {'ServerID':<10} {'Score':<12}")
                print("-" * 30)
                for rec in baseline_recs[:args.top_k]:
                    print(f"{rec['rank']:<6} {rec['server_id']:<10} {rec['score']:<12.6f}")
            
            # Overlap analysis
            print(f"\n{'-' * 80}")
            print("OVERLAP ANALYSIS (Common Servers)")
            print('-' * 80)
            hgnn_servers = set([r['server_id'] for r in hgnn_recommendations])
            
            for method, name in zip(baseline_methods, baseline_names):
                baseline_servers = set([r['server_id'] for r in model_results[method]])
                overlap = hgnn_servers & baseline_servers
                overlap_ratio = len(overlap) / args.top_k * 100
                print(f"{name:<25}: {len(overlap)}/{args.top_k} servers ({overlap_ratio:.1f}% overlap)")
                if overlap:
                    print(f"                           Common: {sorted(overlap)}")
        
        all_results[model_id] = model_results
    
    # Summary statistics
    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS")
    print('=' * 80)
    
    if args.compare_baselines:
        print(f"\nAverage Overlap with HGNN (across {len(model_ids)} models):")
        print("-" * 60)
        
        for method, name in zip(baseline_methods, baseline_names):
            total_overlap = 0
            for model_id in model_ids:
                hgnn_servers = set([r['server_id'] for r in all_results[model_id]['hgnn']])
                baseline_servers = set([r['server_id'] for r in all_results[model_id][method]])
                total_overlap += len(hgnn_servers & baseline_servers)
            
            avg_overlap = total_overlap / len(model_ids)
            avg_overlap_ratio = avg_overlap / args.top_k * 100
            print(f"{name:<25}: {avg_overlap:.2f}/{args.top_k} ({avg_overlap_ratio:.1f}%)")
    
    # Save all results to structured files
    print(f"\n{'=' * 80}")
    print("SAVING RESULTS FOR RESEARCH ANALYSIS")
    print('=' * 80)
    save_inference_results(all_results, model_ids, args, args.output_dir)
    
    # Unified metric evaluation against Ground Truth
    print(f"\n{'=' * 80}")
    print("UNIFIED METRIC EVALUATION (Against Ground Truth)")
    print('=' * 80)
    
    # Prepare predictions for all methods
    hgnn_predictions = {}
    hgnn_sota_predictions = {}
    baseline_predictions = {method: {} for method in baseline_methods}
    
    for model_id in range(engine.dataset.num_models):
        # HGNN (Ours) predictions
        hgnn_recs = engine.recommend_servers_for_model(model_id, top_k=args.top_k)
        hgnn_predictions[model_id] = np.array([r['server_id'] - 1 for r in hgnn_recs])  # Convert to 0-indexed
        
        # HGNN (SOTA) predictions
        hgnn_sota_recs = engine_sota.recommend_servers_for_model(model_id, top_k=args.top_k)
        hgnn_sota_predictions[model_id] = np.array([r['server_id'] - 1 for r in hgnn_sota_recs])
        
        # Baseline predictions
        for method in baseline_methods:
            baseline_recs = compute_baseline_recommendations(engine.dataset, model_id, top_k=args.top_k, method=method)
            baseline_predictions[method][model_id] = np.array([r['server_id'] - 1 for r in baseline_recs])
    
    # Add load-balanced baseline
    baseline_predictions['load_balanced'] = {}
    for model_id in range(engine.dataset.num_models):
        load_balanced_recs = compute_baseline_recommendations(engine.dataset, model_id, top_k=args.top_k, method='load_balanced')
        baseline_predictions['load_balanced'][model_id] = np.array([r['server_id'] - 1 for r in load_balanced_recs])
    
    # Compute metrics
    all_metrics = {}
    
    print("\nEvaluating HGNN (Ours)...")
    all_metrics['HGNN'] = compute_ranking_metrics(hgnn_predictions, engine.dataset.model_positive_servers, args.top_k)
    
    print("Evaluating HGNN (SOTA)...")
    all_metrics['HGNN_SOTA'] = compute_ranking_metrics(hgnn_sota_predictions, engine.dataset.model_positive_servers, args.top_k)
    
    for method in baseline_methods + ['load_balanced']:
        method_name = method.replace('_', '-').title()
        print(f"Evaluating {method_name}...")
        all_metrics[method_name] = compute_ranking_metrics(baseline_predictions[method], engine.dataset.model_positive_servers, args.top_k)
    
    # Print comparison table
    print(f"\n{'-' * 88}")
    print(f"{'Method':<20} {'P@{args.top_k}':<12} {'R@{args.top_k}':<12} {'F1@{args.top_k}':<12} {'NDCG@{args.top_k}':<12} {'Hit@{args.top_k}':<12}")
    print('-' * 88)
    
    for method, metrics in all_metrics.items():
        print(f"{method:<20} {metrics[f'precision@{args.top_k}']:.4f}       "
              f"{metrics[f'recall@{args.top_k}']:.4f}       "
              f"{metrics[f'f1@{args.top_k}']:.4f}       "
              f"{metrics[f'ndcg@{args.top_k}']:.4f}       "
              f"{metrics[f'hit_rate@{args.top_k}']:.4f}")
    
    # Compute improvements
    print(f"\n{'-' * 60}")
    print("HGNN RELATIVE IMPROVEMENTS (NDCG@{args.top_k})")
    print('-' * 60)
    
    hgnn_ndcg = all_metrics['HGNN'][f'ndcg@{args.top_k}']
    for method, metrics in all_metrics.items():
        if method == 'HGNN':
            continue
        baseline_ndcg = metrics[f'ndcg@{args.top_k}']
        improvement = (hgnn_ndcg - baseline_ndcg) / baseline_ndcg * 100 if baseline_ndcg > 0 else 0
        print(f"  vs {method:<20}: {improvement:+7.2f}%  (HGNN: {hgnn_ndcg:.4f}, {method}: {baseline_ndcg:.4f})")
    
    # Save unified metrics
    output_dir = Path(args.output_dir)
    metrics_records = []
    for method, metrics in all_metrics.items():
        record = {'method': method}
        record.update(metrics)
        metrics_records.append(record)
    
    metrics_df = pd.DataFrame(metrics_records)
    metrics_csv_path = output_dir / 'unified_metrics_evaluation.csv'
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\n  Saved: {metrics_csv_path}")
    
    metrics_json_path = output_dir / 'unified_metrics_evaluation.json'
    with open(metrics_json_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Saved: {metrics_json_path}")
    
    print(f"\n{'=' * 80}")
    print("INFERENCE COMPLETED")
    print('=' * 80)
    print(f"Results directory: {Path(args.output_dir).absolute()}")
    print(f"Files generated:")
    print(f"  - hgnn_recommendations.csv")
    print(f"  - random_recommendations.csv")
    print(f"  - popular_recommendations.csv")
    print(f"  - user_aware_recommendations.csv")
    print(f"  - resource_matching_recommendations.csv")
    print(f"  - overlap_analysis.csv")
    print(f"  - summary_statistics.csv")
    print(f"  - model_info.csv")
    print(f"  - complete_results.json")
    print(f"  - unified_metrics_evaluation.csv")
    print(f"  - unified_metrics_evaluation.json")
    print('=' * 80)


if __name__ == '__main__':
    main()

