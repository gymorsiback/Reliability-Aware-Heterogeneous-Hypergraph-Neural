"""
Inference Script for Ablation Study Models

This script loads trained ablation models and evaluates their performance
on the test/validation set to compare training vs inference performance.

Usage:
    python inference_ablation.py
    python inference_ablation.py --ablation_dir path/to/ablation/results
"""

import os
import sys
import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.HGNN import HGNN_ModelPlacement
from datasets.topk_placement_loader import TopKPlacementDataset
from utils.hypergraph_utils import construct_H_for_model_placement
from utils.metrics import RecommendationMetrics


def load_ablation_model(model_path, dataset, device='cuda'):
    """
    Load a trained ablation model
    
    Args:
        model_path: Path to model checkpoint
        dataset: Dataset object
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    model = HGNN_ModelPlacement(
        in_ch=dataset.node_features.shape[1],
        n_hid=checkpoint['config']['n_hid'],
        num_users=dataset.num_users,
        num_models=dataset.num_models,
        num_servers=dataset.num_servers,
        dropout=0.0  # No dropout during inference
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def evaluate_model(model, features, G, dataset, eval_k=5, device='cuda'):
    """
    Evaluate model on dataset
    
    Args:
        model: Trained model
        features: Node features
        G: Graph Laplacian
        dataset: Dataset object
        eval_k: K for evaluation metrics
        device: Device
    
    Returns:
        dict with evaluation metrics
    """
    features = torch.from_numpy(features).float().to(device)
    G = torch.from_numpy(G).float().to(device)
    
    with torch.no_grad():
        # Get model predictions: (num_models, num_servers)
        scores = model(features, G)
    
    # Prepare positive indices (Ground Truth)
    positive_indices = []
    for model_id in range(dataset.num_models):
        gt_servers = dataset.model_positive_servers[model_id]
        positive_indices.append(gt_servers)
    
    # Calculate metrics using RecommendationMetrics
    metrics_calc = RecommendationMetrics(k_list=[eval_k])
    metrics = metrics_calc.compute_all_metrics(
        pred_scores=scores,
        positive_indices=positive_indices,
        prefix=''
    )
    
    return metrics


def run_inference_for_ablation(ablation_dir, output_dir, device='cuda'):
    """
    Run inference for all ablation models
    
    Args:
        ablation_dir: Directory containing ablation results
        output_dir: Directory to save inference results
        device: Device to use
    
    Returns:
        dict with all inference results
    """
    ablation_dir = Path(ablation_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("ABLATION INFERENCE")
    print("=" * 80)
    print(f"\nAblation directory: {ablation_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load dataset (train split for GT, same as training)
    print("\n" + "=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    dataset = TopKPlacementDataset(split='train', k_positive=10, data_root='datasets/server')
    dataset.prepare()
    print(f"Loaded {dataset.num_users} users, {dataset.num_models} models, {dataset.num_servers} servers")
    
    # Construct hypergraph (full)
    print("\n" + "=" * 80)
    print("CONSTRUCTING HYPERGRAPH")
    print("=" * 80)
    H, G, edge_info = construct_H_for_model_placement(dataset, use_gpu=(device=='cuda'))
    print(f"Hypergraph: {H.shape[0]} nodes, {H.shape[1]} hyperedges")
    
    # Find all model checkpoints
    # First try the new checkpoint directory structure
    checkpoint_base = Path('results/train_ablation/checkpoints')
    checkpoint_files = []
    
    if checkpoint_base.exists():
        checkpoint_files = list(checkpoint_base.glob('*/best_model.pth'))
    
    # If not found, try old structure in ablation_dir
    if not checkpoint_files:
        checkpoint_files = list(ablation_dir.glob('**/best_model.pth'))
    
    if not checkpoint_files:
        # Load summary to see which ablation types were completed
        summary_file = ablation_dir / 'ablation_summary.csv'
        if not summary_file.exists():
            raise FileNotFoundError(f"No checkpoints or summary found in {ablation_dir}")
        
        df_summary = pd.read_csv(summary_file)
        print(f"\n" + "=" * 80)
        print("WARNING: No model checkpoints found!")
        print("=" * 80)
        print(f"Found {len(df_summary)} ablation variants in summary, but no saved models.")
        print(f"\nTo run inference, you need to:")
        print(f"  1. Re-run train_ablation.py (now it saves checkpoints automatically)")
        print(f"  2. Or check if checkpoints exist in: {checkpoint_base}")
        print("\nCheckpoints should be in this structure:")
        print(f"  {checkpoint_base}/full/best_model.pth")
        print(f"  {checkpoint_base}/no_user_model/best_model.pth")
        print(f"  etc.")
        print("=" * 80)
        return None
    
    print(f"\nFound {len(checkpoint_files)} model checkpoints")
    
    # Run inference for each model
    all_results = {}
    
    for i, checkpoint_path in enumerate(checkpoint_files, 1):
        # Determine ablation type from path
        # Format: .../checkpoints/ablation_type/best_model.pth
        ablation_type = checkpoint_path.parent.name
        
        print(f"\n{'=' * 80}")
        print(f"INFERENCE {i}/{len(checkpoint_files)}: {ablation_type.upper()}")
        print(f"{'=' * 80}")
        print(f"Checkpoint: {checkpoint_path}")
        
        # Skip MLP baseline (uses different architecture)
        if ablation_type.lower() in ['mlp', 'only_cross']:
            print(f"⚠️  Skipping {ablation_type} - uses different model architecture")
            print(f"   (Not compatible with HGNN_ModelPlacement)")
            continue
        
        try:
            # Load model
            model, checkpoint = load_ablation_model(checkpoint_path, dataset, device)
            print(f"Model loaded successfully")
            print(f"Training NDCG@5: {checkpoint.get('best_ndcg', 'N/A')}")
            
            # Run inference
            print("\nRunning inference...")
            metrics = evaluate_model(model, dataset.node_features, G, dataset, eval_k=5, device=device)
            
            print(f"\nInference Results:")
            print(f"  Precision@5: {metrics['precision@5']:.4f}")
            print(f"  Recall@5: {metrics['recall@5']:.4f}")
            print(f"  F1@5: {metrics['f1@5']:.4f}")
            print(f"  NDCG@5: {metrics['ndcg@5']:.4f}")
            print(f"  Hit Rate@5: {metrics['hit_rate@5']:.4f}")
            
            all_results[ablation_type] = {
                'inference_metrics': metrics,
                'training_metrics': checkpoint.get('best_metrics', {}),
                'checkpoint_path': str(checkpoint_path)
            }
            
            # Save individual result
            result_file = output_dir / f'{ablation_type}_inference.json'
            with open(result_file, 'w') as f:
                json.dump(all_results[ablation_type], f, indent=2)
            print(f"Results saved to: {result_file}")
            
        except Exception as e:
            print(f"ERROR processing {ablation_type}: {e}")
            import traceback
            traceback.print_exc()
            all_results[ablation_type] = {'error': str(e)}
    
    # Generate summary
    print("\n" + "=" * 80)
    print("INFERENCE SUMMARY")
    print("=" * 80)
    
    summary_data = []
    for ablation_type, results in all_results.items():
        if 'inference_metrics' in results:
            summary_data.append({
                'ablation_type': ablation_type,
                'inference_precision@5': results['inference_metrics']['precision@5'],
                'inference_recall@5': results['inference_metrics']['recall@5'],
                'inference_f1@5': results['inference_metrics']['f1@5'],
                'inference_ndcg@5': results['inference_metrics']['ndcg@5'],
                'inference_hit_rate@5': results['inference_metrics']['hit_rate@5'],
                'training_precision@5': results['training_metrics'].get('precision@5', 0),
                'training_recall@5': results['training_metrics'].get('recall@5', 0),
                'training_f1@5': results['training_metrics'].get('f1@5', 0),
                'training_ndcg@5': results['training_metrics'].get('ndcg@5', 0)
            })
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        
        # Save summary CSV
        summary_file = output_dir / 'inference_summary.csv'
        df_summary.to_csv(summary_file, index=False)
        print(f"\nSummary CSV saved to: {summary_file}")
        
        # Save summary JSON
        summary_json = output_dir / 'inference_summary.json'
        with open(summary_json, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Summary JSON saved to: {summary_json}")
        
        # Print comparison table
        print("\nTraining vs Inference Comparison:")
        print("-" * 100)
        print(f"{'Ablation Type':<20} {'Train NDCG':>12} {'Infer NDCG':>12} {'Difference':>12} {'Gap %':>10}")
        print("-" * 100)
        for _, row in df_summary.iterrows():
            train_ndcg = row['training_ndcg@5']
            infer_ndcg = row['inference_ndcg@5']
            diff = infer_ndcg - train_ndcg
            gap_pct = (diff / train_ndcg * 100) if train_ndcg > 0 else 0
            print(f"{row['ablation_type']:<20} {train_ndcg:>12.4f} {infer_ndcg:>12.4f} {diff:>+12.4f} {gap_pct:>+9.1f}%")
        print("-" * 100)
    
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETED")
    print("=" * 80)
    print(f"All results saved to: {output_dir}")
    
    return all_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference for ablation models')
    parser.add_argument(
        '--ablation_dir',
        type=str,
        default='results/train_ablation/ablation_20251105_125428',
        help='Path to ablation results directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/inference_ablation',
        help='Directory to save inference results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        results = run_inference_for_ablation(args.ablation_dir, args.output_dir, device=args.device)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

