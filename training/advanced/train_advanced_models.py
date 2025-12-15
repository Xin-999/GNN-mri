#!/usr/bin/env python
"""
Advanced GNN Training Script for Brain Connectivity Prediction
==============================================================
Train state-of-the-art models: BrainGT, BrainGNN, or FBNetGen

Usage:
    python train_advanced_models.py --model braingt --epochs 100
    python train_advanced_models.py --model braingnn --hidden_dim 256
    python train_advanced_models.py --model fbnetgen --use_timeseries

Key features:
- Target normalization (CRITICAL FIX!)
- Early stopping with patience
- Learning rate scheduling
- Model checkpointing
- Subject-level evaluation
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

# Add project root to path so we can import modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Import models
from models.brain_gt import BrainGT
from models.brain_gnn import BrainGNN, SimpleBrainGNN
from models.fbnetgen import FBNetGenFromGraph

# Import utilities
from utils.data_utils import (
    load_graphs_with_normalization,
    create_dataloaders,
    compute_metrics,
    aggregate_window_predictions,
    save_predictions,
)


def get_model(model_name: str, in_dim: int, config: dict):
    """Initialize model based on name and config."""
    if model_name == 'braingt':
        return BrainGT(
            in_dim=in_dim,
            hidden_dim=config['hidden_dim'],
            n_rois=config.get('n_rois', 268),
            n_transformer_layers=config.get('n_transformer_layers', 4),
            n_gnn_layers=config.get('n_gnn_layers', 2),
            n_heads=config.get('n_heads', 8),
            dropout=config.get('dropout', 0.2),
            pool_type=config.get('pool_type', 'attention'),
        )
    elif model_name == 'braingnn':
        return SimpleBrainGNN(
            in_dim=in_dim,
            hidden_dim=config['hidden_dim'],
            n_rois=config.get('n_rois', 268),
            n_communities=config.get('n_communities', 7),
            n_layers=config.get('n_layers', 3),
            dropout=config.get('dropout', 0.3),
        )
    elif model_name == 'fbnetgen':
        return FBNetGenFromGraph(
            in_dim=in_dim,
            hidden_dim=config['hidden_dim'],
            n_layers=config.get('n_layers', 3),
            n_heads=config.get('n_heads', 4),
            dropout=config.get('dropout', 0.3),
            refine_graph=config.get('refine_graph', True),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_one_epoch(model, loader, optimizer, criterion, device, model_name=''):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        preds = model(batch)
        target = batch.y.float()

        # Compute loss
        loss = criterion(preds, target)

        # Add regularization for BrainGNN
        if model_name == 'braingnn' and hasattr(model, 'compute_unit_loss'):
            reg_loss = (
                0.01 * model.compute_unit_loss() +
                0.01 * model.compute_glc_loss() +
                0.01 * model.compute_tpk_loss()
            )
            loss = loss + reg_loss

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        num_samples += batch.num_graphs

    return total_loss / max(1, num_samples)


@torch.no_grad()
def evaluate(model, loader, device, subject_ids=None):
    """
    Evaluate model and compute both window-level and subject-level metrics.

    Args:
        model: PyTorch model
        loader: DataLoader
        device: Device
        subject_ids: Subject IDs for each sample (for aggregation)

    Returns:
        metrics: Dict of metrics
        predictions: Array of predictions
        targets: Array of targets
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_subj_ids = []

    for batch in loader:
        batch = batch.to(device)
        preds = model(batch)

        all_preds.append(preds.cpu())
        all_targets.append(batch.y.cpu())

        if hasattr(batch, 'subject_id'):
            all_subj_ids.append(batch.subject_id.cpu())

    predictions = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    # Window-level metrics
    win_metrics = compute_metrics(predictions, targets, prefix='win_')

    # Subject-level metrics (if subject IDs available)
    if all_subj_ids:
        subject_ids_tensor = torch.cat(all_subj_ids).numpy().flatten()
        subj_preds, _ = aggregate_window_predictions(predictions, subject_ids_tensor)
        subj_targets, _ = aggregate_window_predictions(targets, subject_ids_tensor)
        subj_metrics = compute_metrics(subj_preds, subj_targets, prefix='subj_')
    else:
        subj_metrics = {}

    # Combine metrics
    metrics = {**win_metrics, **subj_metrics}

    return metrics, predictions, targets


def train_model(
    model_name: str,
    fold_path: str,
    config: dict,
    output_dir: str,
    device: str = 'cuda',
):
    """
    Train a single model on one fold.

    Args:
        model_name: 'braingt', 'braingnn', or 'fbnetgen'
        fold_path: Path to fold data
        config: Model and training configuration
        output_dir: Where to save results
        device: 'cuda' or 'cpu'
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seed for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Load data with normalization
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} on {Path(fold_path).name}")
    print(f"{'='*60}\n")

    train_graphs, val_graphs, test_graphs, info = load_graphs_with_normalization(
        fold_path,
        normalize_method=config.get('normalize_method', 'standard'),
    )

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_graphs, val_graphs, test_graphs,
        batch_size=config['batch_size'],
        num_workers=0,
        pin_memory=(device == 'cuda'),
    )

    # Initialize model
    in_dim = train_graphs[0].x.size(-1)
    model = get_model(model_name, in_dim, config)
    model = model.to(device)

    print(f"Model: {model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )

    if config.get('scheduler', 'cosine') == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    criterion = nn.MSELoss()

    # Training loop
    best_val_metric = float('inf')  # Using MSE
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, config['epochs'] + 1):
        t0 = time.perf_counter()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, model_name
        )

        # Evaluate
        val_metrics, _, _ = evaluate(model, val_loader, device, info['val_subject_ids'])
        epoch_time = time.perf_counter() - t0

        # Learning rate scheduling
        if config.get('scheduler', 'cosine') == 'cosine':
            scheduler.step()
        else:
            scheduler.step(val_metrics['win_mse'])

        # Log progress
        print(
            f"Epoch {epoch:03d}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Win MSE: {val_metrics['win_mse']:.4f} | "
            f"Val Win r: {val_metrics['win_r']:.3f} | "
            f"Val Subj MSE: {val_metrics.get('subj_mse', float('nan')):.4f} | "
            f"Val Subj r: {val_metrics.get('subj_r', float('nan')):.3f} | "
            f"Time: {epoch_time:.2f}s"
        )

        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            **val_metrics,
            'lr': optimizer.param_groups[0]['lr'],
        })

        # Early stopping (use subject-level MSE if available, else window-level)
        val_metric = val_metrics.get('subj_mse', val_metrics['win_mse'])

        if val_metric < best_val_metric - 1e-5:
            best_val_metric = val_metric
            best_state = model.state_dict().copy()
            patience_counter = 0

            # Save best model
            checkpoint_path = output_dir / f"{model_name}_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config,
            }, checkpoint_path)
        else:
            patience_counter += 1

        if patience_counter >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch} (patience={config['patience']})")
            break

    # Load best model and evaluate on test set
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\n{'='*60}")
    print("Final Evaluation on Test Set")
    print(f"{'='*60}\n")

    test_metrics, test_preds, test_targets = evaluate(
        model, test_loader, device, info['test_subject_ids']
    )

    print("Window-level metrics:")
    print(f"  MSE: {test_metrics['win_mse']:.4f}")
    print(f"  MAE: {test_metrics['win_mae']:.4f}")
    print(f"  r:   {test_metrics['win_r']:.4f} (p={test_metrics['win_p_value']:.2e})")

    if 'subj_mse' in test_metrics:
        print("\nSubject-level metrics:")
        print(f"  MSE: {test_metrics['subj_mse']:.4f}")
        print(f"  MAE: {test_metrics['subj_mae']:.4f}")
        print(f"  r:   {test_metrics['subj_r']:.4f} (p={test_metrics['subj_p_value']:.2e})")

    # Save comprehensive results with tracking info
    import datetime
    import socket

    results = {
        # Model & Training Info
        'model_name': model_name,
        'fold': Path(fold_path).name,
        'timestamp': datetime.datetime.now().isoformat(),
        'hostname': socket.gethostname(),
        'device': device,

        # Configuration
        'config': config,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),

        # Training Summary
        'training_summary': {
            'total_epochs': len(history),
            'best_epoch': len(history) - patience_counter,
            'early_stopped': patience_counter >= config['patience'],
            'final_lr': history[-1]['lr'] if history else config['lr'],
        },

        # Best Validation Metrics
        'best_validation': {
            'epoch': len(history) - patience_counter,
            'win_mse': min(h.get('win_mse', float('inf')) for h in history),
            'win_r': max(h.get('win_r', float('-inf')) for h in history),
            'subj_mse': min(h.get('subj_mse', float('inf')) for h in history if 'subj_mse' in h) if any('subj_mse' in h for h in history) else None,
            'subj_r': max(h.get('subj_r', float('-inf')) for h in history if 'subj_r' in h) if any('subj_r' in h for h in history) else None,
        },

        # Test Metrics (Final Evaluation)
        'test_metrics': {
            'window_level': {
                'mse': test_metrics.get('win_mse'),
                'mae': test_metrics.get('win_mae'),
                'r': test_metrics.get('win_r'),
                'p_value': test_metrics.get('win_p_value'),
                'r2': test_metrics.get('win_r2'),
            },
            'subject_level': {
                'mse': test_metrics.get('subj_mse'),
                'mae': test_metrics.get('subj_mae'),
                'r': test_metrics.get('subj_r'),
                'p_value': test_metrics.get('subj_p_value'),
                'r2': test_metrics.get('subj_r2'),
            } if 'subj_mse' in test_metrics else None,
        },

        # Data Statistics
        'data_info': {
            'n_train_samples': len(train_graphs),
            'n_val_samples': len(val_graphs),
            'n_test_samples': len(test_graphs),
            'n_train_subjects': len(np.unique(info['train_subject_ids'])),
            'n_val_subjects': len(np.unique(info['val_subject_ids'])),
            'n_test_subjects': len(np.unique(info['test_subject_ids'])),
            'input_dim': in_dim,
            'target_range_original': {
                'train': [float(info['train_targets_original'].min()),
                         float(info['train_targets_original'].max())],
                'test': [float(info['test_targets_original'].min()),
                        float(info['test_targets_original'].max())],
            },
        },

        # Full Training History
        'history': history,
    }

    # Save detailed results
    with open(output_dir / f"{model_name}_summary.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Save predictions
    save_predictions(
        test_preds, test_targets,
        info['test_subject_ids'],
        output_dir / f"{model_name}_predictions.json",
        scaler=info['scaler'],
    )

    print(f"\nResults saved to {output_dir}")
    print(f"  - Summary: {output_dir / f'{model_name}_summary.json'}")
    print(f"  - Predictions: {output_dir / f'{model_name}_predictions.json'}")
    print(f"  - Checkpoint: {output_dir / f'{model_name}_best.pt'}")

    return results, model


def main():
    parser = argparse.ArgumentParser(description="Train advanced GNN models")

    # Model selection
    parser.add_argument('--model', type=str, required=True,
                        choices=['braingt', 'braingnn', 'fbnetgen'],
                        help='Model architecture')

    # Data
    parser.add_argument('--fold_dir', type=str, default='../../data/folds_data',
                        help='Directory with fold data')
    parser.add_argument('--fold_name', type=str, default=None,
                        help='Specific fold to train (default: all folds)')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Model hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='Number of layers')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')

    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: results_{model})')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to train on')

    args = parser.parse_args()

    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Output directory
    if args.output_dir is None:
        args.output_dir = f"../../results/advanced/{args.model}"

    # Configuration
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'seed': args.seed,
        'hidden_dim': args.hidden_dim,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'dropout': args.dropout,
        'normalize_method': 'standard',
        'scheduler': 'cosine',
        'n_rois': 268,
    }

    # Model-specific defaults
    if args.model == 'braingt':
        config['n_transformer_layers'] = 4
        config['n_gnn_layers'] = 2
        config['pool_type'] = 'attention'
    elif args.model == 'braingnn':
        config['n_communities'] = 7
    elif args.model == 'fbnetgen':
        config['refine_graph'] = True

    # Get fold files
    fold_dir = Path(args.fold_dir)
    if not fold_dir.exists():
        # Fallback: try from project root
        fold_dir = Path("data/folds_data")

    if not fold_dir.exists():
        print(f"Error: Data directory not found at {args.fold_dir} or data/folds_data")
        return

    if args.fold_name:
        fold_files = [fold_dir / f"{args.fold_name}.pkl"]
    else:
        fold_files = sorted(fold_dir.glob("graphs_outer*.pkl"))

    print(f"Found {len(fold_files)} folds to process\n")

    # Train on each fold
    all_results = []
    for fold_path in fold_files:
        fold_name = fold_path.stem
        fold_output_dir = Path(args.output_dir) / fold_name

        results, model = train_model(
            args.model,
            str(fold_path),
            config,
            str(fold_output_dir),
            args.device,
        )

        all_results.append(results)

    # Aggregate results across folds
    if len(all_results) > 1:
        import datetime

        print(f"\n{'='*60}")
        print("Aggregate Results Across All Folds")
        print(f"{'='*60}\n")

        # Extract metrics
        test_win_rs = [r['test_metrics']['window_level']['r'] for r in all_results]
        test_win_mses = [r['test_metrics']['window_level']['mse'] for r in all_results]

        # Subject-level if available
        has_subject_level = all(r['test_metrics']['subject_level'] is not None for r in all_results)
        if has_subject_level:
            test_subj_rs = [r['test_metrics']['subject_level']['r'] for r in all_results]
            test_subj_mses = [r['test_metrics']['subject_level']['mse'] for r in all_results]

        print("Window-level metrics:")
        print(f"  Mean test r:   {np.mean(test_win_rs):.4f} ± {np.std(test_win_rs):.4f}")
        print(f"  Mean test MSE: {np.mean(test_win_mses):.4f} ± {np.std(test_win_mses):.4f}")

        if has_subject_level:
            print("\nSubject-level metrics:")
            print(f"  Mean test r:   {np.mean(test_subj_rs):.4f} ± {np.std(test_subj_rs):.4f}")
            print(f"  Mean test MSE: {np.mean(test_subj_mses):.4f} ± {np.std(test_subj_mses):.4f}")

        # Comprehensive aggregate summary
        aggregate_summary = {
            # Model Info
            'model_name': args.model,
            'timestamp': datetime.datetime.now().isoformat(),
            'n_folds': len(all_results),

            # Configuration
            'config': config,

            # Aggregate Statistics - Window Level
            'window_level_aggregate': {
                'mse': {
                    'mean': float(np.mean(test_win_mses)),
                    'std': float(np.std(test_win_mses)),
                    'min': float(np.min(test_win_mses)),
                    'max': float(np.max(test_win_mses)),
                    'median': float(np.median(test_win_mses)),
                },
                'r': {
                    'mean': float(np.mean(test_win_rs)),
                    'std': float(np.std(test_win_rs)),
                    'min': float(np.min(test_win_rs)),
                    'max': float(np.max(test_win_rs)),
                    'median': float(np.median(test_win_rs)),
                },
            },

            # Aggregate Statistics - Subject Level (if available)
            'subject_level_aggregate': {
                'mse': {
                    'mean': float(np.mean(test_subj_mses)),
                    'std': float(np.std(test_subj_mses)),
                    'min': float(np.min(test_subj_mses)),
                    'max': float(np.max(test_subj_mses)),
                    'median': float(np.median(test_subj_mses)),
                },
                'r': {
                    'mean': float(np.mean(test_subj_rs)),
                    'std': float(np.std(test_subj_rs)),
                    'min': float(np.min(test_subj_rs)),
                    'max': float(np.max(test_subj_rs)),
                    'median': float(np.median(test_subj_rs)),
                },
            } if has_subject_level else None,

            # Training Statistics
            'training_statistics': {
                'mean_best_epoch': float(np.mean([r['training_summary']['best_epoch'] for r in all_results])),
                'mean_total_epochs': float(np.mean([r['training_summary']['total_epochs'] for r in all_results])),
                'n_early_stopped': sum(1 for r in all_results if r['training_summary']['early_stopped']),
            },

            # Model Size
            'model_info': {
                'mean_parameters': float(np.mean([r['model_parameters'] for r in all_results])),
                'architecture': args.model,
            },

            # Per-Fold Summary (compact)
            'per_fold_summary': [
                {
                    'fold': r['fold'],
                    'test_subj_r': r['test_metrics']['subject_level']['r'] if r['test_metrics']['subject_level'] else r['test_metrics']['window_level']['r'],
                    'test_subj_mse': r['test_metrics']['subject_level']['mse'] if r['test_metrics']['subject_level'] else r['test_metrics']['window_level']['mse'],
                    'best_epoch': r['training_summary']['best_epoch'],
                }
                for r in all_results
            ],

            # Full per-fold results
            'per_fold_detailed': all_results,
        }

        # Save aggregate results
        summary_path = Path(args.output_dir) / f"{args.model}_aggregate_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(aggregate_summary, f, indent=2)

        print(f"\nAggregate summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
