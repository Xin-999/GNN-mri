#!/usr/bin/env python
"""
Enhanced GNN Training Script - With Advanced Improvements
=========================================================
Train enhanced versions of models with additional techniques:
- Virtual Nodes (BrainGT)
- Biased DropEdge (BrainGNN)
- GraphNorm/PairNorm
- Additional Regularization Losses

Usage:
    python train_enhanced_models.py --model braingt --epochs 100
    python train_enhanced_models.py --model braingnn --hidden_dim 256
    python train_enhanced_models.py --model fbnetgen

Key features:
- All original features from train_advanced_models.py
- Additional regularization losses from enhanced models
- Comprehensive JSON tracking for comparison
"""

import os
import argparse
import json
import time
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Import enhanced models
from models_enhanced.brain_gt_enhanced import BrainGTEnhanced
from models_enhanced.brain_gnn_enhanced import BrainGNNEnhanced
from models_enhanced.fbnetgen_enhanced import FBNetGenFromGraphEnhanced

# Import utilities
from utils.data_utils import (
    load_graphs_with_normalization,
    create_dataloaders,
    compute_metrics,
    aggregate_window_predictions,
    save_predictions,
)


def get_model(model_name: str, in_dim: int, config: dict):
    """Initialize enhanced model based on name and config."""
    if model_name == 'braingt':
        return BrainGTEnhanced(
            in_dim=in_dim,
            hidden_dim=config['hidden_dim'],
            n_gnn_layers=config.get('n_gnn_layers', 2),
            n_transformer_layers=config.get('n_transformer_layers', 4),
            n_heads=config.get('n_heads', 8),
            dropout=config.get('dropout', 0.2),
            attention_dropout=config.get('attention_dropout', 0.1),
            edge_dropout=config.get('edge_dropout', 0.1),
            use_virtual_nodes=config.get('use_virtual_nodes', True),
        )
    elif model_name == 'braingnn':
        return BrainGNNEnhanced(
            in_dim=in_dim,
            hidden_dim=config['hidden_dim'],
            n_layers=config.get('n_layers', 3),
            n_rois=config.get('n_rois', 268),
            n_communities=config.get('n_communities', 7),
            dropout=config.get('dropout', 0.3),
            pool_ratio=config.get('pool_ratio', 0.5),
            edge_dropout=config.get('edge_dropout', 0.1),
            biased_drop_strength=config.get('biased_drop_strength', 2.0),
            topk_loss_weight=config.get('topk_loss_weight', 0.2),
            consistency_loss_weight=config.get('consistency_loss_weight', 0.1),
        )
    elif model_name == 'fbnetgen':
        return FBNetGenFromGraphEnhanced(
            in_dim=in_dim,
            hidden_dim=config['hidden_dim'],
            n_gnn_layers=config.get('n_gnn_layers', 3),
            n_heads=config.get('n_heads', 4),
            dropout=config.get('dropout', 0.3),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_one_epoch(model, loader, optimizer, criterion, device, model_name=''):
    """Train for one epoch with additional regularization losses."""
    model.train()
    total_loss = 0.0
    total_pred_loss = 0.0
    total_reg_losses = {}
    num_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        # Enhanced models may return losses
        if model_name in ['braingnn', 'fbnetgen']:
            preds, losses = model(batch, return_losses=True)
            target = batch.y.float()

            # Prediction loss
            pred_loss = criterion(preds, target)

            # Total loss includes regularization
            loss = pred_loss
            for loss_name, loss_value in losses.items():
                loss = loss + loss_value

                # Track individual losses
                if loss_name not in total_reg_losses:
                    total_reg_losses[loss_name] = 0.0
                total_reg_losses[loss_name] += loss_value.item() * batch.num_graphs
        else:
            # BrainGT Enhanced doesn't have additional losses returned
            preds = model(batch)
            target = batch.y.float()
            pred_loss = criterion(preds, target)
            loss = pred_loss

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Track losses
        total_loss += loss.item() * batch.num_graphs
        total_pred_loss += pred_loss.item() * batch.num_graphs
        num_samples += batch.num_graphs

    # Average losses
    avg_loss = total_loss / num_samples
    avg_pred_loss = total_pred_loss / num_samples
    avg_reg_losses = {k: v / num_samples for k, v in total_reg_losses.items()}

    result = {
        'total_loss': avg_loss,
        'pred_loss': avg_pred_loss,
    }
    result.update(avg_reg_losses)

    return result


def evaluate(model, loader, criterion, device, scaler, model_name=''):
    """Evaluate model on validation/test set."""
    model.eval()

    all_preds = []
    all_targets = []
    all_subject_ids = []
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Forward pass (evaluation mode - no losses returned)
            if model_name in ['braingnn', 'fbnetgen']:
                preds = model(batch, return_losses=False)
            else:
                preds = model(batch)

            target = batch.y.float()

            # Loss
            loss = criterion(preds, target)
            total_loss += loss.item() * batch.num_graphs
            num_samples += batch.num_graphs

            # Store predictions
            all_preds.append(preds.cpu().numpy())
            all_targets.append(target.cpu().numpy())

            # Get subject IDs if available
            if hasattr(batch, 'subject_id'):
                all_subject_ids.append(batch.subject_id.cpu().numpy())

    # Concatenate
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    if all_subject_ids:
        all_subject_ids = np.concatenate(all_subject_ids)
    else:
        all_subject_ids = None

    # Denormalize for proper metrics
    all_preds_original = scaler.inverse_transform(all_preds.reshape(-1, 1)).flatten()
    all_targets_original = scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()

    # Window-level metrics
    win_metrics = compute_metrics(all_preds_original, all_targets_original)

    # Subject-level metrics
    subj_metrics = None
    if all_subject_ids is not None:
        subj_preds, subj_targets = aggregate_window_predictions(
            all_preds_original, all_targets_original, all_subject_ids
        )
        subj_metrics = compute_metrics(subj_preds, subj_targets)

    return {
        'loss': total_loss / num_samples,
        'window_metrics': win_metrics,
        'subject_metrics': subj_metrics,
        'predictions': all_preds_original,
        'targets': all_targets_original,
        'subject_ids': all_subject_ids,
    }


def train_fold(model_name, fold_path, config, device):
    """Train model on a single fold."""
    print(f"\n{'='*70}")
    print(f"Training {model_name.upper()} Enhanced on fold: {fold_path.name}")
    print(f"{'='*70}\n")

    # Load data with normalization
    data_dict = load_graphs_with_normalization(
        fold_path,
        normalize_method=config.get('normalize_method', 'standard'),
        verbose=True
    )

    train_list = data_dict['train_list']
    val_list = data_dict['val_list']
    test_list = data_dict['test_list']
    scaler = data_dict['scaler']

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_list, val_list, test_list,
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 0)
    )

    # Initialize model
    in_dim = train_list[0].x.size(1)
    model = get_model(model_name, in_dim, config).to(device)

    print(f"Model: {model_name.upper()} Enhanced")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 1e-5)
    )

    # Scheduler
    if config.get('scheduler') == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    criterion = nn.MSELoss()

    # Training loop
    best_val_metric = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, config['epochs'] + 1):
        epoch_start = time.time()

        # Train
        train_result = train_one_epoch(model, train_loader, optimizer, criterion, device, model_name)

        # Validate
        val_result = evaluate(model, val_loader, criterion, device, scaler, model_name)

        # Update scheduler
        if config.get('scheduler') == 'cosine':
            scheduler.step()
        else:
            scheduler.step(val_result['loss'])

        # Track history
        history.append({
            'epoch': epoch,
            'train_loss': train_result['total_loss'],
            'train_pred_loss': train_result['pred_loss'],
            'val_loss': val_result['loss'],
            'win_mse': val_result['window_metrics']['mse'],
            'win_r': val_result['window_metrics']['r'],
            'subj_mse': val_result['subject_metrics']['mse'] if val_result['subject_metrics'] else None,
            'subj_r': val_result['subject_metrics']['r'] if val_result['subject_metrics'] else None,
            'lr': optimizer.param_groups[0]['lr'],
        })

        # Save regularization losses if present
        for loss_name in ['topk_loss', 'consistency_loss', 'sparsity_loss']:
            if loss_name in train_result:
                history[-1][loss_name] = train_result[loss_name]

        # Print progress
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:03d}/{config['epochs']:03d} | "
              f"Train Loss: {train_result['total_loss']:.4f} | "
              f"Val Loss: {val_result['loss']:.4f} | "
              f"Win r: {val_result['window_metrics']['r']:.4f} | "
              f"Subj r: {val_result['subject_metrics']['r']:.4f if val_result['subject_metrics'] else 0:.4f} | "
              f"Time: {epoch_time:.1f}s")

        # Early stopping based on subject-level correlation
        if val_result['subject_metrics']:
            val_metric = -val_result['subject_metrics']['r']  # Negative because we want to maximize
        else:
            val_metric = val_result['loss']

        if val_metric < best_val_metric:
            best_val_metric = val_metric
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            output_dir = Path(f"results_{model_name}_enhanced") / fold_path.name
            output_dir.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'scaler': scaler,
            }, output_dir / f"{model_name}_best.pt")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\nBest epoch: {best_epoch}")

    # Load best model for testing
    checkpoint = torch.load(output_dir / f"{model_name}_best.pt")
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test evaluation
    test_result = evaluate(model, test_loader, criterion, device, scaler, model_name)

    print(f"\nTest Results:")
    print(f"  Window-level MSE: {test_result['window_metrics']['mse']:.4f}")
    print(f"  Window-level r: {test_result['window_metrics']['r']:.4f}")
    if test_result['subject_metrics']:
        print(f"  Subject-level MSE: {test_result['subject_metrics']['mse']:.4f}")
        print(f"  Subject-level r: {test_result['subject_metrics']['r']:.4f}")

    # Save predictions
    save_predictions(
        test_result['predictions'],
        test_result['targets'],
        test_result['subject_ids'],
        output_dir / f"{model_name}_predictions.csv"
    )

    # Save comprehensive JSON summary
    results = {
        'model_name': f"{model_name}_enhanced",
        'fold': fold_path.name,
        'timestamp': datetime.datetime.now().isoformat(),
        'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown',
        'device': str(device),

        'config': config,

        'model_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),

        'training_summary': {
            'total_epochs': len(history),
            'best_epoch': best_epoch,
            'early_stopped': patience_counter >= config['patience'],
            'final_lr': optimizer.param_groups[0]['lr'],
        },

        'best_validation': {
            'epoch': best_epoch,
            'win_mse': history[best_epoch-1]['win_mse'],
            'win_r': history[best_epoch-1]['win_r'],
            'subj_mse': history[best_epoch-1]['subj_mse'],
            'subj_r': history[best_epoch-1]['subj_r'],
        },

        'test_metrics': {
            'window_level': test_result['window_metrics'],
            'subject_level': test_result['subject_metrics'],
        },

        'data_info': data_dict['data_info'],

        'history': history,
    }

    # Save per-fold summary
    with open(output_dir / f"{model_name}_summary.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


def aggregate_results(model_name, all_results):
    """Aggregate results across all folds."""
    print(f"\n{'='*70}")
    print(f"Aggregating results for {model_name.upper()} Enhanced")
    print(f"{'='*70}\n")

    # Extract test metrics
    window_metrics = []
    subject_metrics = []

    for result in all_results:
        if result['test_metrics']['window_level']:
            window_metrics.append(result['test_metrics']['window_level'])
        if result['test_metrics']['subject_level']:
            subject_metrics.append(result['test_metrics']['subject_level'])

    # Compute aggregates
    def aggregate_metric_list(metrics_list, metric_name):
        values = [m[metric_name] for m in metrics_list]
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
        }

    window_agg = {}
    for metric in ['mse', 'mae', 'r', 'p_value', 'r2']:
        if metric in window_metrics[0]:
            window_agg[metric] = aggregate_metric_list(window_metrics, metric)

    subject_agg = None
    if subject_metrics:
        subject_agg = {}
        for metric in ['mse', 'mae', 'r', 'p_value', 'r2']:
            if metric in subject_metrics[0]:
                subject_agg[metric] = aggregate_metric_list(subject_metrics, metric)

    # Training statistics
    training_stats = {
        'mean_best_epoch': np.mean([r['training_summary']['best_epoch'] for r in all_results]),
        'mean_total_epochs': np.mean([r['training_summary']['total_epochs'] for r in all_results]),
        'n_early_stopped': sum(r['training_summary']['early_stopped'] for r in all_results),
    }

    # Model info
    model_info = {
        'mean_parameters': np.mean([r['model_parameters'] for r in all_results]),
        'architecture': f"{model_name}_enhanced",
    }

    # Per-fold summary
    per_fold_summary = []
    for result in all_results:
        per_fold_summary.append({
            'fold': result['fold'],
            'test_subj_r': result['test_metrics']['subject_level']['r'] if result['test_metrics']['subject_level'] else None,
            'test_subj_mse': result['test_metrics']['subject_level']['mse'] if result['test_metrics']['subject_level'] else None,
            'best_epoch': result['training_summary']['best_epoch'],
        })

    # Create aggregate summary
    aggregate_summary = {
        'model_name': f"{model_name}_enhanced",
        'timestamp': datetime.datetime.now().isoformat(),
        'n_folds': len(all_results),

        'config': all_results[0]['config'],

        'window_level_aggregate': window_agg,
        'subject_level_aggregate': subject_agg,

        'training_statistics': training_stats,
        'model_info': model_info,

        'per_fold_summary': per_fold_summary,
        'per_fold_detailed': all_results,
    }

    # Save aggregate summary
    output_dir = Path(f"results_{model_name}_enhanced")
    with open(output_dir / f"{model_name}_aggregate_summary.json", 'w') as f:
        json.dump(aggregate_summary, f, indent=2)

    # Print summary
    print(f"Aggregate Results:")
    print(f"  Window-level:")
    for metric, values in window_agg.items():
        print(f"    {metric}: {values['mean']:.4f} ± {values['std']:.4f}")

    if subject_agg:
        print(f"  Subject-level:")
        for metric, values in subject_agg.items():
            print(f"    {metric}: {values['mean']:.4f} ± {values['std']:.4f}")

    print(f"\nResults saved to: {output_dir / f'{model_name}_aggregate_summary.json'}")

    return aggregate_summary


def main():
    parser = argparse.ArgumentParser(description="Train Enhanced GNN Models")

    # Model selection
    parser.add_argument('--model', type=str, required=True,
                        choices=['braingt', 'braingnn', 'fbnetgen'],
                        help='Model to train')

    # Data
    parser.add_argument('--data_dir', type=str, default='graphs',
                        help='Directory containing graph data')
    parser.add_argument('--fold_name', type=str, default=None,
                        help='Specific fold to train (default: all folds)')

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine'])

    # Model architecture
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--n_gnn_layers', type=int, default=2)
    parser.add_argument('--n_transformer_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.2)

    # Enhanced model specific
    parser.add_argument('--edge_dropout', type=float, default=0.1,
                        help='DropEdge probability')
    parser.add_argument('--attention_dropout', type=float, default=0.1,
                        help='Attention dropout probability')
    parser.add_argument('--use_virtual_nodes', action='store_true', default=True,
                        help='Use virtual nodes (BrainGT)')
    parser.add_argument('--biased_drop_strength', type=float, default=2.0,
                        help='Biased DropEdge strength (BrainGNN)')

    # System
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()

    # Device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Configuration
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'scheduler': args.scheduler,
        'hidden_dim': args.hidden_dim,
        'n_layers': args.n_layers,
        'n_gnn_layers': args.n_gnn_layers,
        'n_transformer_layers': args.n_transformer_layers,
        'n_heads': args.n_heads,
        'dropout': args.dropout,
        'edge_dropout': args.edge_dropout,
        'attention_dropout': args.attention_dropout,
        'use_virtual_nodes': args.use_virtual_nodes,
        'biased_drop_strength': args.biased_drop_strength,
        'num_workers': args.num_workers,
    }

    # Get fold directories
    data_dir = Path(args.data_dir)
    if args.fold_name:
        fold_dirs = [data_dir / args.fold_name]
    else:
        fold_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    print(f"\nTraining {args.model.upper()} Enhanced on {len(fold_dirs)} folds")

    # Train on all folds
    all_results = []
    for fold_dir in fold_dirs:
        try:
            result = train_fold(args.model, fold_dir, config, device)
            all_results.append(result)
        except Exception as e:
            print(f"Error training on {fold_dir}: {e}")
            continue

    # Aggregate results
    if all_results:
        aggregate_results(args.model, all_results)
    else:
        print("No successful training runs!")


if __name__ == "__main__":
    main()
