#!/usr/bin/env python
"""
Compact Hyperparameter Search using Optuna
==========================================
Smaller, memory-friendly Optuna search to reduce CUDA OOM risk.

Usage:
    # Search for BrainGNN (compact space)
    python training/hyperparameter_search_small.py --model braingnn --n_trials 30

    # Search for FBNetGen (compact space, enhanced)
    python training/hyperparameter_search_small.py --model fbnetgen --use_enhanced --n_trials 30
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path so we can import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import optuna
except ImportError:
    print("Optuna not installed. Install with: pip install optuna")
    exit(1)

from models.brain_gt import BrainGT
from models.brain_gnn import SimpleBrainGNN
from models.fbnetgen import FBNetGenFromGraph

from models_enhanced.brain_gt_enhanced import BrainGTEnhanced
from models_enhanced.brain_gnn_enhanced import BrainGNNEnhanced
from models_enhanced.fbnetgen_enhanced import FBNetGenFromGraphEnhanced

from utils.data_utils import (
    load_graphs_with_normalization,
    create_dataloaders,
    compute_metrics,
)


def objective(trial, model_name, fold_paths, device, n_epochs=15, use_enhanced=False):
    """
    Objective function for compact Optuna optimization.
    Evaluates hyperparameters across ALL folds.

    Args:
        trial: Optuna trial object
        model_name: Model architecture
        fold_paths: List of paths to all fold data files
        device: 'cuda' or 'cpu'
        n_epochs: Number of epochs to train
        use_enhanced: Use enhanced models (default: False)

    Returns:
        validation_metric: Average Pearson correlation (r) across all folds to maximize
    """
    # Clear GPU cache between trials to reduce fragmentation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Set seed for this trial (based on trial number for reproducibility)
    import random
    trial_seed = 42 + trial.number
    random.seed(trial_seed)
    torch.manual_seed(trial_seed)
    np.random.seed(trial_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(trial_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Smaller, memory-friendly search space
    hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128,256])
    lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.3)
    batch_size = trial.suggest_categorical('batch_size', [8, 16,32])
    n_layers = trial.suggest_int('n_layers', 2, 4)

    # Model-specific hyperparameters (compact)
    if model_name == 'braingt':
        n_heads = trial.suggest_categorical('n_heads', [2, 4, 8])
        n_transformer_layers = trial.suggest_int('n_transformer_layers', 2, 3)
        n_gnn_layers = trial.suggest_int('n_gnn_layers', 1, 2)
        pool_type = trial.suggest_categorical('pool_type', ['attention', 'mean'])
    elif model_name == 'braingnn':
        n_communities = trial.suggest_int('n_communities', 5, 8)
    elif model_name == 'fbnetgen':
        n_heads = trial.suggest_categorical('n_heads', [2, 4])
        refine_graph = trial.suggest_categorical('refine_graph', [True, False])

    # Evaluate across all folds
    fold_val_r_scores = []

    for fold_idx, fold_path in enumerate(fold_paths):
        # Load data for this fold
        train_graphs, val_graphs, test_graphs, info = load_graphs_with_normalization(
            fold_path,
            normalize_method='standard',
        )

        train_loader, val_loader, _ = create_dataloaders(
            train_graphs, val_graphs, test_graphs,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=(device == 'cuda'),
        )

        # Initialize model for this fold
        in_dim = train_graphs[0].x.size(-1)

        if use_enhanced:
            if model_name == 'braingt':
                model = BrainGTEnhanced(
                    in_dim=in_dim,
                    hidden_dim=hidden_dim,
                    n_rois=268,
                    n_transformer_layers=n_transformer_layers,
                    n_gnn_layers=n_gnn_layers,
                    n_heads=n_heads,
                    dropout=dropout,
                )
            elif model_name == 'braingnn':
                model = BrainGNNEnhanced(
                    in_dim=in_dim,
                    hidden_dim=hidden_dim,
                    n_rois=268,
                    n_layers=n_layers,
                    dropout=dropout,
                )
            elif model_name == 'fbnetgen':
                model = FBNetGenFromGraphEnhanced(
                    in_dim=in_dim,
                    hidden_dim=hidden_dim,
                    n_layers=n_layers,
                    n_heads=n_heads,
                    dropout=dropout,
                )
            else:
                raise ValueError(f"Unknown model: {model_name}")
        else:
            if model_name == 'braingt':
                model = BrainGT(
                    in_dim=in_dim,
                    hidden_dim=hidden_dim,
                    n_rois=268,
                    n_transformer_layers=n_transformer_layers,
                    n_gnn_layers=n_gnn_layers,
                    n_heads=n_heads,
                    dropout=dropout,
                    pool_type=pool_type,
                )
            elif model_name == 'braingnn':
                model = SimpleBrainGNN(
                    in_dim=in_dim,
                    hidden_dim=hidden_dim,
                    n_rois=268,
                    n_communities=n_communities,
                    n_layers=n_layers,
                    dropout=dropout,
                )
            elif model_name == 'fbnetgen':
                model = FBNetGenFromGraph(
                    in_dim=in_dim,
                    hidden_dim=hidden_dim,
                    n_layers=n_layers,
                    n_heads=n_heads,
                    dropout=dropout,
                    refine_graph=refine_graph,
                )
            else:
                raise ValueError(f"Unknown model: {model_name}")

        model = model.to(device)

        # Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        best_val_r = -float('inf')
        best_val_metrics = {}
        patience_counter = 0
        patience = 5

        for epoch in range(n_epochs):
            try:
                model.train()
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    preds = model(batch)
                    loss = criterion(preds, batch.y.float())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                # Validate
                model.eval()
                all_preds = []
                all_targets = []

                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        preds = model(batch)
                        all_preds.append(preds.cpu())
                        all_targets.append(batch.y.cpu())
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n⚠️  Trial {trial.number} Fold {fold_idx+1}/{len(fold_paths)} failed: CUDA OOM")
                    print(f"   Config: hidden_dim={hidden_dim}, batch_size={batch_size}, n_layers={n_layers}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise optuna.TrialPruned()
                raise

            predictions = torch.cat(all_preds).numpy()
            targets = torch.cat(all_targets).numpy()

            val_metrics = compute_metrics(predictions, targets, prefix='')
            val_r = val_metrics['r']

            if val_r > best_val_r + 1e-5:
                best_val_r = val_r
                best_val_metrics = val_metrics
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

            # Report intermediate value (per fold, per epoch)
            trial.report(best_val_r, fold_idx * n_epochs + epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Store best validation r for this fold
        fold_val_r_scores.append(best_val_r)
        print(f"  Fold {fold_idx+1}/{len(fold_paths)}: val_r = {best_val_r:.4f}")

        # Clear GPU cache after each fold
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Average validation r across all folds
    avg_val_r = np.mean(fold_val_r_scores)
    std_val_r = np.std(fold_val_r_scores)

    trial.set_user_attr('avg_val_r', avg_val_r)
    trial.set_user_attr('std_val_r', std_val_r)
    trial.set_user_attr('fold_val_r_scores', fold_val_r_scores)
    trial.set_user_attr('n_folds', len(fold_paths))

    print(f"Trial {trial.number}: Avg val_r = {avg_val_r:.4f} ± {std_val_r:.4f} across {len(fold_paths)} folds")

    return avg_val_r


def main():
    parser = argparse.ArgumentParser(description="Compact hyperparameter search")

    # Model selection
    parser.add_argument('--model', type=str, required=True,
                        choices=['braingt', 'braingnn', 'fbnetgen'],
                        help='Model architecture')
    parser.add_argument('--use_enhanced', action='store_true',
                        help='Use enhanced models instead of base models')

    # Search configuration
    parser.add_argument('--n_trials', type=int, default=30,
                        help='Number of trials')
    parser.add_argument('--n_epochs', type=int, default=15,
                        help='Epochs per trial')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Parallel jobs (-1 for all CPUs)')

    # Data
    parser.add_argument('--fold_dir', type=str, default='data/folds_data',
                        help='Directory with fold data')
    parser.add_argument('--fold_name', type=str, default=None,
                        help='Specific fold (default: first fold)')

    # Output
    parser.add_argument('--output_dir', type=str, default='hyperparameter_search_results_small',
                        help='Output directory')
    parser.add_argument('--study_name', type=str, default=None,
                        help='Optuna study name (default: model_name_small)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    fold_dir = Path(args.fold_dir)
    if args.fold_name:
        # Single fold mode (for quick testing)
        fold_paths = [fold_dir / f"{args.fold_name}.pkl"]
        print(f"Using single fold: {fold_paths[0].name}\n")
    else:
        # All folds mode (default - for proper CV)
        fold_paths = sorted(fold_dir.glob("graphs_outer*.pkl"))
        if not fold_paths:
            raise FileNotFoundError(f"No fold files found in {fold_dir}")
        print(f"Using all {len(fold_paths)} folds for cross-validation\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    study_name = args.study_name or f"{args.model}_search_small"
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    model_type = "Enhanced" if args.use_enhanced else "Base"
    print(f"Starting compact hyperparameter search for {args.model.upper()} ({model_type})")
    print(f"Number of trials: {args.n_trials}")
    print(f"Epochs per trial: {args.n_epochs}")
    print(f"Number of folds: {len(fold_paths)}")
    print(f"Optimizing for: Average Pearson correlation (r) across all folds\n")

    study.optimize(
        lambda trial: objective(trial, args.model, [str(p) for p in fold_paths], args.device, args.n_epochs, args.use_enhanced),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
    )

    best_trial = study.best_trial
    print(f"\n{'='*60}")
    print("Search Results")
    print(f"{'='*60}\n")
    print(f"Best trial #{best_trial.number}")
    print(f"\nValidation Metrics (averaged across {best_trial.user_attrs.get('n_folds', len(fold_paths))} folds):")
    print(f"  Avg Pearson r:  {best_trial.user_attrs.get('avg_val_r', best_trial.value):.4f} ± {best_trial.user_attrs.get('std_val_r', 0):.4f}")
    print(f"  Per-fold r:     {best_trial.user_attrs.get('fold_val_r_scores', [])}")

    print("\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    results = {
        'model': args.model,
        'model_type': 'enhanced' if args.use_enhanced else 'base',
        'n_folds': len(fold_paths),
        'folds': [p.name for p in fold_paths],
        'n_trials': args.n_trials,
        'best_value': best_trial.value,
        'best_metrics': {
            'avg_r': best_trial.user_attrs.get('avg_val_r', best_trial.value),
            'std_r': best_trial.user_attrs.get('std_val_r', 0),
            'fold_r_scores': best_trial.user_attrs.get('fold_val_r_scores', []),
        },
        'best_params': best_trial.params,
        'all_trials': [
            {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': str(trial.state),
                'metrics': {
                    'avg_r': trial.user_attrs.get('avg_val_r', trial.value),
                    'std_r': trial.user_attrs.get('std_val_r', 0),
                    'n_folds': trial.user_attrs.get('n_folds', len(fold_paths)),
                }
            }
            for trial in study.trials
        ],
    }

    model_suffix = '_enhanced' if args.use_enhanced else '_base'
    results_path = output_dir / f"{args.model}{model_suffix}_search_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")

    try:
        import plotly

        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(output_dir / f"{args.model}_optimization_history.html")

        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(output_dir / f"{args.model}_param_importances.html")

        print(f"Visualizations saved to {output_dir}")
    except ImportError:
        print("Plotly not installed - skipping visualizations")

    print("\nTop 5 trials (by Pearson r):")
    print("="*60)
    trials_df = study.trials_dataframe().sort_values('value', ascending=False).head(5)
    display_df = trials_df[['number', 'value', 'params_hidden_dim', 'params_lr', 'params_dropout']].rename(
        columns={'value': 'pearson_r'}
    )
    print(display_df)


if __name__ == "__main__":
    main()
