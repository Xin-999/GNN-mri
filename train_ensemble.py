#!/usr/bin/env python
"""
Ensemble Training Script
========================
Combine predictions from multiple trained models for maximum accuracy.

Usage:
    # Step 1: Train individual models first
    python train_advanced_models.py --model braingt --epochs 100
    python train_advanced_models.py --model braingnn --epochs 100
    python train_advanced_models.py --model fbnetgen --epochs 100

    # Step 2: Create ensemble
    python train_ensemble.py --ensemble_type weighted

Ensemble types:
- mean: Simple averaging
- weighted: Learned weights (optimized on validation)
- stacking: Meta-learner MLP
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.brain_gt import BrainGT
from models.brain_gnn import SimpleBrainGNN
from models.fbnetgen import FBNetGenFromGraph
from models.ensemble import load_ensemble_from_checkpoints, evaluate_ensemble

from utils.data_utils import load_graphs_with_normalization, create_dataloaders


def main():
    parser = argparse.ArgumentParser(description="Train ensemble of models")

    # Model checkpoints
    parser.add_argument('--braingt_dir', type=str, default='results_braingt_advanced',
                        help='Directory with BrainGT checkpoints')
    parser.add_argument('--braingnn_dir', type=str, default='results_braingnn_advanced',
                        help='Directory with BrainGNN checkpoints')
    parser.add_argument('--fbnetgen_dir', type=str, default='results_fbnetgen_advanced',
                        help='Directory with FBNetGen checkpoints')

    # Ensemble configuration
    parser.add_argument('--ensemble_type', type=str, default='weighted',
                        choices=['mean', 'weighted', 'stacking'],
                        help='Type of ensemble')
    parser.add_argument('--optimize_epochs', type=int, default=100,
                        help='Epochs to optimize ensemble weights')

    # Data
    parser.add_argument('--fold_dir', type=str, default='data/folds_data',
                        help='Directory with fold data')
    parser.add_argument('--fold_name', type=str, default=None,
                        help='Specific fold (default: all folds)')

    # Output
    parser.add_argument('--output_dir', type=str, default='results_ensemble',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])

    args = parser.parse_args()

    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Get fold files
    fold_dir = Path(args.fold_dir)
    if args.fold_name:
        fold_files = [fold_dir / f"{args.fold_name}.pkl"]
    else:
        fold_files = sorted(fold_dir.glob("graphs_outer*.pkl"))

    print(f"Found {len(fold_files)} folds to process\n")

    # Process each fold
    all_ensemble_results = []

    for fold_path in fold_files:
        fold_name = fold_path.stem
        print(f"\n{'='*60}")
        print(f"Creating ensemble for {fold_name}")
        print(f"{'='*60}\n")

        # Load data
        train_graphs, val_graphs, test_graphs, info = load_graphs_with_normalization(
            str(fold_path),
            normalize_method='standard',
        )

        train_loader, val_loader, test_loader = create_dataloaders(
            train_graphs, val_graphs, test_graphs,
            batch_size=32,
            num_workers=0,
            pin_memory=(args.device == 'cuda'),
        )

        # Define model checkpoint paths
        checkpoint_paths = {
            'braingt': Path(args.braingt_dir) / fold_name / "braingt_best.pt",
            'braingnn': Path(args.braingnn_dir) / fold_name / "braingnn_best.pt",
            'fbnetgen': Path(args.fbnetgen_dir) / fold_name / "fbnetgen_best.pt",
        }

        # Filter out models that don't exist
        available_models = {
            name: path for name, path in checkpoint_paths.items()
            if path.exists()
        }

        if len(available_models) < 2:
            print(f"Warning: Only {len(available_models)} models found for {fold_name}")
            print(f"Need at least 2 models for ensemble. Skipping fold.")
            continue

        print(f"Using models: {list(available_models.keys())}\n")

        # Model classes and configs
        in_dim = train_graphs[0].x.size(-1)

        model_classes = {
            'braingt': BrainGT,
            'braingnn': SimpleBrainGNN,
            'fbnetgen': FBNetGenFromGraph,
        }

        model_configs = {
            'braingt': {
                'in_dim': in_dim,
                'hidden_dim': 128,
                'n_rois': 268,
                'n_transformer_layers': 4,
                'n_gnn_layers': 2,
                'n_heads': 8,
                'dropout': 0.2,
            },
            'braingnn': {
                'in_dim': in_dim,
                'hidden_dim': 128,
                'n_rois': 268,
                'n_communities': 7,
                'n_layers': 3,
                'dropout': 0.3,
            },
            'fbnetgen': {
                'in_dim': in_dim,
                'hidden_dim': 128,
                'n_layers': 3,
                'n_heads': 4,
                'dropout': 0.3,
                'refine_graph': True,
            },
        }

        # Filter configs to only available models
        model_classes_filtered = {k: v for k, v in model_classes.items() if k in available_models}
        model_configs_filtered = {k: v for k, v in model_configs.items() if k in available_models}

        # Load ensemble
        ensemble = load_ensemble_from_checkpoints(
            checkpoint_paths=available_models,
            model_classes=model_classes_filtered,
            model_configs=model_configs_filtered,
            ensemble_type=args.ensemble_type,
            device=args.device,
        )

        # Optimize ensemble weights on validation set
        if args.ensemble_type in ['weighted', 'stacking']:
            print(f"\nOptimizing {args.ensemble_type} ensemble weights...")
            ensemble.optimize_weights(
                val_loader,
                n_epochs=args.optimize_epochs,
                lr=0.01,
                verbose=True,
            )

        # Evaluate on test set
        print(f"\nEvaluating ensemble on test set...")
        fold_output_dir = Path(args.output_dir) / fold_name
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        metrics = evaluate_ensemble(
            ensemble,
            test_loader,
            save_results=fold_output_dir / "ensemble_predictions.csv",
        )

        # Save ensemble weights/model
        if args.ensemble_type in ['weighted', 'stacking']:
            ensemble.save(fold_output_dir / "ensemble_weights.pt")

        # Save metrics
        with open(fold_output_dir / "ensemble_metrics.json", 'w') as f:
            json.dump({
                'fold': fold_name,
                'ensemble_type': args.ensemble_type,
                'models_used': list(available_models.keys()),
                'metrics': metrics,
            }, f, indent=2)

        all_ensemble_results.append(metrics)

    # Aggregate results
    if all_ensemble_results:
        print(f"\n{'='*60}")
        print("Aggregate Ensemble Results Across All Folds")
        print(f"{'='*60}\n")

        mean_mse = np.mean([r['mse'] for r in all_ensemble_results])
        std_mse = np.std([r['mse'] for r in all_ensemble_results])
        mean_r = np.mean([r['r'] for r in all_ensemble_results])
        std_r = np.std([r['r'] for r in all_ensemble_results])

        print(f"MSE:         {mean_mse:.4f} ± {std_mse:.4f}")
        print(f"Correlation: {mean_r:.4f} ± {std_r:.4f}")

        # Save aggregate results
        with open(Path(args.output_dir) / "aggregate_ensemble_results.json", 'w') as f:
            json.dump({
                'ensemble_type': args.ensemble_type,
                'mean_mse': float(mean_mse),
                'std_mse': float(std_mse),
                'mean_r': float(mean_r),
                'std_r': float(std_r),
                'per_fold_results': all_ensemble_results,
            }, f, indent=2)

        print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
