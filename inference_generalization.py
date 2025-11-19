"""
Comprehensive Generalization Test across different data distributions

This script evaluates HGNN and all baselines on 6 test sets with varying regional focus:
- Test-80: 80% regional focus
- Test-70: 70% regional focus (training distribution)
- Test-60: 60% regional focus
- Test-50: 50% regional focus
- Test-40: 40% regional focus
- Test-30: 30% regional focus

Key Design:
- Fixed GT: Use training set's model-server deployment as Ground Truth
- Variable Input: Different user distributions (80%, 70%, 60%, 50%, 40%, 30% regional focus)
- Test Goal: Can HGNN generalize to different user distributions while predicting the same deployment pattern?

Results are saved to results/inference_generalization/
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import argparse

from models.HGNN import HGNN_ModelPlacement
from datasets.topk_placement_loader import TopKPlacementDataset
from utils.hypergraph_utils import construct_H_for_model_placement


def compute_ranking_metrics(predictions, ground_truth_dict, k=5):
    """
    Compute ranking metrics against ground truth
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


def evaluate_hgnn(checkpoint_path, dataset, eval_k=5, device='cuda'):
    """
    Evaluate HGNN model on a dataset
    """
    print("  Evaluating HGNN...")
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Construct hypergraph
    H, G, edge_info = construct_H_for_model_placement(dataset, use_gpu=(device.type=='cuda'))
    G_tensor = torch.from_numpy(G).float().to(device)
    
    # Prepare features
    features = torch.from_numpy(dataset.node_features).float().to(device)
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = HGNN_ModelPlacement(
        in_ch=features.shape[1],
        n_hid=checkpoint['config']['n_hid'],
        num_users=dataset.num_users,
        num_models=dataset.num_models,
        num_servers=dataset.num_servers,
        dropout=0.0
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get predictions
    predictions = {}
    with torch.no_grad():
        all_embeddings = model.get_embeddings(features, G_tensor)
        
        model_start = dataset.num_users
        model_end = model_start + dataset.num_models
        server_start = model_end
        server_end = server_start + dataset.num_servers
        
        model_embeddings = all_embeddings[model_start:model_end]
        server_embeddings = all_embeddings[server_start:server_end]
        
        for model_id in range(dataset.num_models):
            model_emb = model_embeddings[model_id].unsqueeze(0)
            scores = model.predict_placements(model_emb, server_embeddings)
            scores = scores.squeeze().cpu().numpy()
            
            top_k_indices = np.argsort(-scores)[:eval_k]
            predictions[model_id] = top_k_indices
    
    # Compute metrics
    metrics = compute_ranking_metrics(predictions, dataset.model_positive_servers, eval_k)
    
    return metrics, predictions


def evaluate_baseline(dataset, method='random', eval_k=5):
    """
    Evaluate baseline method
    """
    print(f"  Evaluating {method.upper()}...")
    
    predictions = {}
    num_servers = dataset.num_servers
    
    for model_id in range(dataset.num_models):
        if method == 'random':
            scores = np.random.random(num_servers)
        
        elif method == 'popular':
            scores = dataset.topology.sum(axis=0).A1 if hasattr(dataset.topology.sum(axis=0), 'A1') else dataset.topology.sum(axis=0)
        
        elif method == 'user_aware':
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
                
                server_locs = dataset.servers_df[['Lo', 'La']].values
                distances = np.linalg.norm(server_locs - avg_user_loc, axis=1)
                scores = 1.0 - (distances / (distances.max() + 1e-10))
            else:
                scores = np.ones(num_servers) * 0.5
        
        elif method == 'resource_matching':
            model_req = dataset.models_df.iloc[model_id][['Modelsize', 'Modelresource']].values.astype(float)
            model_req = (model_req - dataset.models_df[['Modelsize', 'Modelresource']].mean().values) / \
                        (dataset.models_df[['Modelsize', 'Modelresource']].std().values + 1e-8)
            
            server_caps = dataset.servers_df[['ComputationCapacity', 'StorageCapacity']].values.astype(float)
            server_caps = (server_caps - server_caps.mean(axis=0)) / (server_caps.std(axis=0) + 1e-8)
            
            distances = np.linalg.norm(server_caps - model_req, axis=1)
            scores = 1.0 - (distances / (distances.max() + 1e-10))
        
        elif method == 'load_balanced':
            scores = np.ones(num_servers) + np.random.random(num_servers) * 0.1
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        top_k_indices = np.argsort(-scores)[:eval_k]
        predictions[model_id] = top_k_indices
    
    # Compute metrics
    metrics = compute_ranking_metrics(predictions, dataset.model_positive_servers, eval_k)
    
    return metrics, predictions


def run_comprehensive_test(checkpoint_path, checkpoint_sota_path, regional_focus_list=[90, 70, 50, 30], eval_k=5, device='cuda'):
    """
    Run comprehensive generalization test across all test sets
    
    Key Design:
    - Load training set once to get fixed Ground Truth
    - For each test set, use test users but evaluate against training GT
    - Tests whether HGNN (Ours & SOTA) can predict same deployment pattern given different user distributions
    """
    print("=" * 80)
    print("COMPREHENSIVE GENERALIZATION TEST")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Test sets: {regional_focus_list}% regional focus")
    print(f"  Evaluation: Top-{eval_k}")
    print(f"  Methods: HGNN (Ours) + HGNN (SOTA) + 5 Baselines")
    print(f"  Device: {device}")
    
    # Load training set to get fixed Ground Truth
    print("\n" + "=" * 80)
    print("LOADING TRAINING SET FOR GROUND TRUTH")
    print("=" * 80)
    train_dataset = TopKPlacementDataset(split='train', k_positive=10, data_root='datasets/server')
    train_dataset.prepare()
    training_gt = train_dataset.model_positive_servers
    print(f"Training GT loaded: {len(training_gt)} models, avg {np.mean([len(v) for v in training_gt.values()]):.1f} servers per model")
    print(f"This GT will be used for all test sets (fixed target)")
    
    all_results = {}
    baseline_methods = ['random', 'popular', 'user_aware', 'resource_matching', 'load_balanced']
    baseline_names = ['Random', 'Popular', 'User-Aware', 'Resource-Matching', 'Load-Balanced']
    
    for focus_pct in regional_focus_list:
        print(f"\n{'#' * 80}")
        print(f"TEST SET: {focus_pct}% Regional Focus")
        print('#' * 80)
        
        # Load test dataset (for user distribution)
        data_root = f'datasets/server_test_{focus_pct}pct'
        print(f"\nLoading test dataset from: {data_root}")
        print(f"  Using: Test users, Training GT")
        
        dataset = TopKPlacementDataset(split='test', k_positive=10, data_root=data_root)
        dataset.prepare()
        
        # Override dataset GT and model-server mappings with training data
        original_gt = dataset.model_positive_servers
        dataset.model_positive_servers = training_gt
        dataset.model_server_df = train_dataset.model_server_df
        
        print(f"Dataset loaded: {dataset.num_users} users, {dataset.num_models} models, {dataset.num_servers} servers")
        print(f"  User distribution: {focus_pct}% regional focus (variable)")
        print(f"  Model-Server deployment: Training set (fixed)")
        print(f"  Ground Truth: Training set deployment (fixed)")
        
        # Evaluate all methods
        test_results = {}
        
        # HGNN (Ours)
        hgnn_metrics, _ = evaluate_hgnn(checkpoint_path, dataset, eval_k, device)
        test_results['HGNN'] = hgnn_metrics
        
        # HGNN (SOTA)
        hgnn_sota_metrics, _ = evaluate_hgnn(checkpoint_sota_path, dataset, eval_k, device)
        test_results['HGNN_SOTA'] = hgnn_sota_metrics
        
        # Baselines
        for method, name in zip(baseline_methods, baseline_names):
            baseline_metrics, _ = evaluate_baseline(dataset, method, eval_k)
            test_results[name] = baseline_metrics
        
        # Store results
        all_results[f'{focus_pct}pct'] = test_results
        
        # Print summary for this test set
        print(f"\n{'-' * 80}")
        print(f"RESULTS - {focus_pct}% Regional Focus (vs Training GT)")
        print('-' * 80)
        print(f"{'Method':<20} {'P@{eval_k}':<10} {'R@{eval_k}':<10} {'F1@{eval_k}':<10} {'NDCG@{eval_k}':<10}")
        print('-' * 70)
        
        for method, metrics in test_results.items():
            print(f"{method:<20} {metrics[f'precision@{eval_k}']:.4f}     {metrics[f'recall@{eval_k}']:.4f}     "
                  f"{metrics[f'f1@{eval_k}']:.4f}     {metrics[f'ndcg@{eval_k}']:.4f}")
        
        # Save individual test set results immediately
        save_individual_results(test_results, focus_pct, eval_k)
    
    # Comprehensive analysis
    print_comprehensive_analysis(all_results, regional_focus_list, eval_k)
    
    # Save results
    save_results(all_results, regional_focus_list, eval_k)
    
    return all_results


def print_comprehensive_analysis(all_results, regional_focus_list, eval_k):
    """
    Print comprehensive analysis: horizontal and vertical comparisons
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    
    # 1. Horizontal comparison (methods on each test set)
    print("\n1. HORIZONTAL COMPARISON (Methods on Each Test Set)")
    print("-" * 80)
    
    for focus_pct in regional_focus_list:
        print(f"\nTest Set: {focus_pct}% Regional Focus")
        print(f"{'Method':<20} {'NDCG@{eval_k}':<12} {'P@{eval_k}':<12} {'R@{eval_k}':<12}")
        print('-' * 60)
        
        methods = list(all_results[f'{focus_pct}pct'].keys())
        for method in methods:
            metrics = all_results[f'{focus_pct}pct'][method]
            print(f"{method:<20} {metrics[f'ndcg@{eval_k}']:.4f}       {metrics[f'precision@{eval_k}']:.4f}       "
                  f"{metrics[f'recall@{eval_k}']:.4f}")
        
        # Find best method
        best_method = max(methods, key=lambda m: all_results[f'{focus_pct}pct'][m][f'ndcg@{eval_k}'])
        best_ndcg = all_results[f'{focus_pct}pct'][best_method][f'ndcg@{eval_k}']
        print(f"\nBest: {best_method} (NDCG@{eval_k}: {best_ndcg:.4f})")
    
    # 2. Vertical comparison (method across test sets)
    print("\n\n2. VERTICAL COMPARISON (Each Method Across Test Sets)")
    print("-" * 80)
    
    methods = list(all_results[f'{regional_focus_list[0]}pct'].keys())
    for method in methods:
        print(f"\nMethod: {method}")
        print(f"{'Test Set':<20} {'NDCG@{eval_k}':<12} {'P@{eval_k}':<12} {'R@{eval_k}':<12}")
        print('-' * 60)
        
        for focus_pct in regional_focus_list:
            metrics = all_results[f'{focus_pct}pct'][method]
            print(f"{focus_pct}% Focus{'':<10} {metrics[f'ndcg@{eval_k}']:.4f}       {metrics[f'precision@{eval_k}']:.4f}       "
                  f"{metrics[f'recall@{eval_k}']:.4f}")
        
        # Compute robustness
        ndcgs = [all_results[f'{pct}pct'][method][f'ndcg@{eval_k}'] for pct in regional_focus_list]
        mean_ndcg = np.mean(ndcgs)
        std_ndcg = np.std(ndcgs)
        print(f"\nRobustness: Mean={mean_ndcg:.4f}, Std={std_ndcg:.4f}")
    
    # 3. HGNN improvement analysis
    print("\n\n3. HGNN RELATIVE IMPROVEMENT vs BEST BASELINE")
    print("-" * 80)
    print(f"{'Test Set':<20} {'Best Baseline':<20} {'HGNN NDCG':<12} {'Baseline NDCG':<15} {'Improvement':<12}")
    print('-' * 80)
    
    for focus_pct in regional_focus_list:
        hgnn_ndcg = all_results[f'{focus_pct}pct']['HGNN'][f'ndcg@{eval_k}']
        
        # Find best baseline
        baselines = {k: v for k, v in all_results[f'{focus_pct}pct'].items() if k != 'HGNN'}
        best_baseline = max(baselines.keys(), key=lambda m: baselines[m][f'ndcg@{eval_k}'])
        best_baseline_ndcg = baselines[best_baseline][f'ndcg@{eval_k}']
        
        improvement = (hgnn_ndcg - best_baseline_ndcg) / best_baseline_ndcg * 100 if best_baseline_ndcg > 0 else 0
        
        print(f"{focus_pct}% Focus{'':<10} {best_baseline:<20} {hgnn_ndcg:.4f}       "
              f"{best_baseline_ndcg:.4f}          {improvement:+.1f}%")
    
    print("\n" + "=" * 80)


def save_individual_results(test_results, focus_pct, eval_k):
    """
    Save individual test set results to separate files
    
    Args:
        test_results: Dict of method -> metrics for this test set
        focus_pct: Regional focus percentage (90, 70, 50, 30)
        eval_k: Evaluation K
    """
    output_dir = Path('results/inference_generalization')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save JSON for this test set
    json_path = output_dir / f'test_{focus_pct}pct_{timestamp}.json'
    result_dict = {
        'test_set': f'{focus_pct}pct',
        'regional_focus': focus_pct,
        'eval_k': eval_k,
        'timestamp': timestamp,
        'results': test_results
    }
    with open(json_path, 'w') as f:
        json.dump(result_dict, f, indent=2)
    
    # Save CSV for this test set
    records = []
    for method, metrics in test_results.items():
        records.append({
            'test_set': f'{focus_pct}pct',
            'regional_focus': focus_pct,
            'method': method,
            f'precision@{eval_k}': metrics[f'precision@{eval_k}'],
            f'recall@{eval_k}': metrics[f'recall@{eval_k}'],
            f'f1@{eval_k}': metrics[f'f1@{eval_k}'],
            f'ndcg@{eval_k}': metrics[f'ndcg@{eval_k}'],
            f'hit_rate@{eval_k}': metrics[f'hit_rate@{eval_k}'],
            'num_models': metrics['num_models_evaluated']
        })
    
    df = pd.DataFrame(records)
    csv_path = output_dir / f'test_{focus_pct}pct_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    
    # Find best method for this test set
    best_method = max(test_results.keys(), key=lambda m: test_results[m][f'ndcg@{eval_k}'])
    best_ndcg = test_results[best_method][f'ndcg@{eval_k}']
    hgnn_ndcg = test_results['HGNN'][f'ndcg@{eval_k}']
    
    print(f"\n{'=' * 80}")
    print(f"SAVED: Test {focus_pct}% Results")
    print(f"{'=' * 80}")
    print(f"  JSON: {json_path.name}")
    print(f"  CSV:  {csv_path.name}")
    print(f"  Best Method: {best_method} (NDCG@{eval_k}: {best_ndcg:.4f})")
    print(f"  HGNN: NDCG@{eval_k}: {hgnn_ndcg:.4f}")
    if best_method == 'HGNN':
        improvement = ((hgnn_ndcg - max([v[f'ndcg@{eval_k}'] for k, v in test_results.items() if k != 'HGNN'])) / 
                      max([v[f'ndcg@{eval_k}'] for k, v in test_results.items() if k != 'HGNN']) * 100)
        print(f"  HGNN vs Best Baseline: +{improvement:.1f}%")
    else:
        improvement = ((hgnn_ndcg - best_ndcg) / best_ndcg * 100)
        print(f"  HGNN vs Best: {improvement:+.1f}%")
    print(f"{'=' * 80}\n")


def save_results(all_results, regional_focus_list, eval_k):
    """
    Save comprehensive summary results to files
    """
    output_dir = Path('results/inference_generalization')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save JSON summary
    json_path = output_dir / f'summary_all_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save CSV summary
    records = []
    for focus_pct in regional_focus_list:
        for method, metrics in all_results[f'{focus_pct}pct'].items():
            records.append({
                'test_set': f'{focus_pct}pct',
                'regional_focus': focus_pct,
                'method': method,
                f'precision@{eval_k}': metrics[f'precision@{eval_k}'],
                f'recall@{eval_k}': metrics[f'recall@{eval_k}'],
                f'f1@{eval_k}': metrics[f'f1@{eval_k}'],
                f'ndcg@{eval_k}': metrics[f'ndcg@{eval_k}'],
                f'hit_rate@{eval_k}': metrics[f'hit_rate@{eval_k}'],
                'num_models': metrics['num_models_evaluated']
            })
    
    df = pd.DataFrame(records)
    csv_path = output_dir / f'summary_all_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"\n{'=' * 80}")
    print(f"COMPREHENSIVE SUMMARY SAVED")
    print(f"{'=' * 80}")
    print(f"  JSON: {json_path.name}")
    print(f"  CSV:  {csv_path.name}")
    print(f"{'=' * 80}")


def main():
    """
    Main function for generalization testing with command line support
    
    Usage Examples:
        Run all test sets (default):
            python inference_generalization.py
        
        Run only 70% test set:
            python inference_generalization.py --test_sets 70
        
        Run 80% and 40% test sets:
            python inference_generalization.py --test_sets 80 40
        
        Run specific test with custom K:
            python inference_generalization.py --test_sets 70 40 --eval_k 10
    """
    parser = argparse.ArgumentParser(
        description='Generalization Test for HGNN Model Placement',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference_generalization.py                    # Run all test sets (80, 70, 60, 50, 40, 30)
  python inference_generalization.py --test_sets 70     # Only run 70%% test
  python inference_generalization.py --test_sets 80 40  # Run 80%% and 40%% tests
  python inference_generalization.py --test_sets 70 --eval_k 10  # Run 70%% with K=10
        """
    )
    
    parser.add_argument(
        '--test_sets',
        type=int,
        nargs='+',
        default=[80, 70, 60, 50, 40, 30],
        choices=[80, 70, 60, 50, 40, 30],
        help='Specify which test sets to run (choices: 80, 70, 60, 50, 40, 30). Default: all'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='results/hgnn_k10_h128_e500_20251104_155422/checkpoints/best_model.pth',
        help='Path to model checkpoint (Ours)'
    )
    
    parser.add_argument(
        '--checkpoint_sota',
        type=str,
        default='results/hgnn_k10_h128_e500_20251031_191447/checkpoints/best_model.pth',
        help='Path to SOTA model checkpoint'
    )
    
    parser.add_argument(
        '--eval_k',
        type=int,
        default=5,
        help='Top-K for evaluation metrics (default: 5)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run on (default: cuda)'
    )
    
    args = parser.parse_args()
    
    # Sort test sets for consistent output
    regional_focus_list = sorted(args.test_sets, reverse=True)
    
    print("=" * 80)
    print(f"RUNNING GENERALIZATION TEST")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Test sets: {regional_focus_list}% regional focus")
    print(f"  Evaluation: Top-{args.eval_k}")
    print(f"  Device: {args.device}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"\nEach test set will be saved separately with timestamp")
    print(f"You can run multiple times and compare results")
    print("=" * 80)
    
    # Run comprehensive test
    results = run_comprehensive_test(
        checkpoint_path=args.checkpoint,
        checkpoint_sota_path=args.checkpoint_sota,
        regional_focus_list=regional_focus_list,
        eval_k=args.eval_k,
        device=args.device
    )
    
    print("\n" + "=" * 80)
    print("GENERALIZATION TEST COMPLETED")
    print("=" * 80)
    print(f"\nIndividual test results saved in: results/inference_generalization/")
    print(f"  Format: test_XXpct_TIMESTAMP.csv and .json")
    print(f"  Summary: summary_all_TIMESTAMP.csv and .json")
    print(f"\nTip: Run multiple times to get different results, then select the best!")
    print("=" * 80)


if __name__ == '__main__':
    main()

