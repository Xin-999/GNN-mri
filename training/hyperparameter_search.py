#!/usr/bin/env python
"""
Hyperparameter Search using Optuna
===================================
Automated hyperparameter optimization for brain GNN models.

Usage:
    # Search for BrainGT
    python hyperparameter_search.py --model braingt --n_trials 50

    # Search for BrainGNN
    python hyperparameter_search.py --model braingnn --n_trials 30

Searches over:
- Learning rate
- Weight decay
- Hidden dimension
- Number of layers
- Dropout
- Batch size
"""

import argparse
import json
from pathlib import Path

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

from utils.data_utils import (
    load_graphs_with_normalization,
    create_dataloaders,
    compute_metrics,
)


def objective(trial, model_name, fold_path, device, n_epochs=30):
    """
    Objective function for Optuna optimization.

    Args:
        trial: Optuna trial object
        model_name: Model architecture
        fold_path: Path to fold data
        device: 'cuda' or 'cpu'
        n_epochs: Number of epochs to train

    Returns:
        validation_metric: Metric to minimize (validation MSE)
    """
    # Suggest hyperparameters
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    n_layers = trial.suggest_int('n_layers', 2, 5)

    # Model-specific hyperparameters
    if model_name == 'braingt':
        n_heads = trial.suggest_categorical('n_heads', [4, 8, 16])
        n_transformer_layers = trial.suggest_int('n_transformer_layers', 2, 6)
        n_gnn_layers = trial.suggest_int('n_gnn_layers', 1, 3)
        pool_type = trial.suggest_categorical('pool_type', ['attention', 'mean'])
    elif model_name == 'braingnn':
        n_communities = trial.suggest_int('n_communities', 5, 10)
    elif model_name == 'fbnetgen':
        n_heads = trial.suggest_categorical('n_heads', [2, 4, 8])
        refine_graph = trial.suggest_categorical('refine_graph', [True, False])

    # Load data
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

    # Initialize model
    in_dim = train_graphs[0].x.size(-1)

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

    # Training loop
    best_val_mse = float('inf')
    patience_counter = 0
    patience = 5

    for epoch in range(n_epochs):
        # Train
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

        predictions = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()

        val_metrics = compute_metrics(predictions, targets, prefix='')
        val_mse = val_metrics['mse']

        # Early stopping
        if val_mse < best_val_mse - 1e-5:
            best_val_mse = val_mse
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

        # Report intermediate value for pruning
        trial.report(val_mse, epoch)

        # Handle pruning
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_mse


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search")

    # Model selection
    parser.add_argument('--model', type=str, required=True,
                        choices=['braingt', 'braingnn', 'fbnetgen'],
                        help='Model architecture')

    # Search configuration
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of trials')
    parser.add_argument('--n_epochs', type=int, default=30,
                        help='Epochs per trial')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Parallel jobs (-1 for all CPUs)')

    # Data
    parser.add_argument('--fold_dir', type=str, default='data/folds_data',
                        help='Directory with fold data')
    parser.add_argument('--fold_name', type=str, default=None,
                        help='Specific fold (default: first fold)')

    # Output
    parser.add_argument('--output_dir', type=str, default='hyperparameter_search_results',
                        help='Output directory')
    parser.add_argument('--study_name', type=str, default=None,
                        help='Optuna study name (default: model_name)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])

    args = parser.parse_args()

    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Get fold file
    fold_dir = Path(args.fold_dir)
    if args.fold_name:
        fold_path = fold_dir / f"{args.fold_name}.pkl"
    else:
        fold_files = sorted(fold_dir.glob("graphs_outer*.pkl"))
        if not fold_files:
            raise FileNotFoundError(f"No fold files found in {fold_dir}")
        fold_path = fold_files[0]  # Use first fold for search

    print(f"Using fold: {fold_path.name}\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create Optuna study
    study_name = args.study_name or f"{args.model}_search"

    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    print(f"Starting hyperparameter search for {args.model}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Epochs per trial: {args.n_epochs}\n")

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args.model, str(fold_path), args.device, args.n_epochs),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
    )

    # Results
    print(f"\n{'='*60}")
    print("Search Results")
    print(f"{'='*60}\n")

    print(f"Best trial value (validation MSE): {study.best_trial.value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    # Save results
    results = {
        'model': args.model,
        'fold': fold_path.name,
        'n_trials': args.n_trials,
        'best_value': study.best_trial.value,
        'best_params': study.best_trial.params,
        'all_trials': [
            {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': str(trial.state),
            }
            for trial in study.trials
        ],
    }

    results_path = output_dir / f"{args.model}_search_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")

    # Plot optimization history (if plotly available)
    try:
        import plotly

        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(output_dir / f"{args.model}_optimization_history.html")

        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(output_dir / f"{args.model}_param_importances.html")

        print(f"Visualizations saved to {output_dir}")
    except ImportError:
        print("Plotly not installed - skipping visualizations")

    # Print top 5 trials
    print("\n Top 5 trials:")
    print("="*60)
    trials_df = study.trials_dataframe().sort_values('value').head(5)
    print(trials_df[['number', 'value', 'params_hidden_dim', 'params_lr', 'params_dropout']])


if __name__ == "__main__":
    main()
