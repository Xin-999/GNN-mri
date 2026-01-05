#!/usr/bin/env python
"""
Grid Search Pipeline for Enhanced Models
=========================================

Runs grid search over hyperparameters for BrainGT, BrainGNN, and FBNetGen enhanced models.

Usage:
    # Quick test with small grid
    python run_grid_search_pipeline.py --quick_test --models braingnn fbnetgen

    # Full grid search for all models
    python run_grid_search_pipeline.py \
        --hidden_dims 64 128 256 \
        --dropouts 0.2 0.3 0.4 \
        --learning_rates 1e-4 5e-4 \
        --epochs 100

    # Search specific model only
    python run_grid_search_pipeline.py --models fbnetgen --quick_test

Features:
- Tests all hyperparameter combinations (grid search)
- Trains each model with each configuration
- Saves results in organized folders
- Generates comparison report
- Finds best configuration for each model
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from itertools import product
from datetime import datetime
import pandas as pd


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


def run_training(model_name, config, fold_name, device, project_root):
    """
    Run training for a single configuration.

    Args:
        model_name: Model to train (braingt, braingnn, fbnetgen)
        config: Dict of hyperparameters
        fold_name: Fold to train on (or None for all folds)
        device: cuda or cpu
        project_root: Project root path

    Returns:
        bool: Success status
    """
    config_str = f"h{config['hidden_dim']}_d{config['dropout']}_lr{config['lr']}"
    if model_name == 'braingt':
        config_str += f"_nh{config['n_heads']}"
    elif model_name == 'braingnn':
        config_str += f"_nl{config['n_layers']}"
    elif model_name == 'fbnetgen':
        config_str += f"_nl{config['n_layers']}_nh{config['n_heads']}"

    print(f"\n{'='*70}")
    print(f"Training {model_name.upper()} with config: {config_str}")
    print(f"{'='*70}\n")

    # Build command
    cmd = [
        sys.executable,
        'training/advanced/train_enhanced_models.py',
        '--model', model_name,
        '--hidden_dim', str(config['hidden_dim']),
        '--dropout', str(config['dropout']),
        '--lr', str(config['lr']),
        '--epochs', str(config['epochs']),
        '--device', device,
    ]

    # Model-specific parameters
    if model_name == 'braingt':
        cmd.extend(['--n_heads', str(config['n_heads'])])
    elif model_name in ['braingnn', 'fbnetgen']:
        cmd.extend(['--n_layers', str(config['n_layers'])])

    if model_name == 'fbnetgen':
        cmd.extend(['--n_heads', str(config['n_heads'])])

    # Add fold name if specified
    if fold_name:
        cmd.extend(['--fold_name', fold_name])

    # Run command
    try:
        print(f"Command: {' '.join(cmd)}\n")
        print(f"Working directory: {project_root}\n")

        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=False,
            text=True,
            env={**subprocess.os.environ, 'PYTHONPATH': str(project_root)}
        )

        success = result.returncode == 0

        if success:
            # Move results to grid search folder with config name
            results_base = project_root / 'results' / 'enhanced' / f'{model_name}_enhanced'
            grid_base = project_root / 'results' / 'grid_search' / model_name / config_str
            grid_base.mkdir(parents=True, exist_ok=True)

            # Copy aggregate summary
            aggregate_file = results_base / f'{model_name}_aggregate_summary.json'
            if aggregate_file.exists():
                import shutil
                shutil.copy(aggregate_file, grid_base / 'aggregate_summary.json')

                # Save config info
                with open(grid_base / 'config.json', 'w') as f:
                    json.dump(config, f, indent=2)

            print(f"\n✓ Training {model_name.upper()} with {config_str} completed successfully")
        else:
            print(f"\n✗ Training {model_name.upper()} with {config_str} failed")

        return success

    except Exception as e:
        print(f"\n✗ Training {model_name.upper()} with {config_str} failed with error:")
        print(f"   {str(e)}")
        return False


def generate_grid(args):
    """
    Generate hyperparameter grid.

    Args:
        args: Command-line arguments

    Returns:
        dict: Grid for each model
    """
    grids = {}

    # Common parameters
    base_params = {
        'hidden_dim': args.hidden_dims,
        'dropout': args.dropouts,
        'lr': args.learning_rates,
        'epochs': [args.epochs],
    }

    # BrainGT specific
    if 'braingt' in args.models:
        grids['braingt'] = {
            **base_params,
            'n_heads': args.n_heads_braingt,
            'n_layers': [3],  # Not used but needed for consistency
        }

    # BrainGNN specific
    if 'braingnn' in args.models:
        grids['braingnn'] = {
            **base_params,
            'n_layers': args.n_layers,
            'n_heads': [4],  # Not used but needed for consistency
        }

    # FBNetGen specific
    if 'fbnetgen' in args.models:
        grids['fbnetgen'] = {
            **base_params,
            'n_layers': args.n_layers,
            'n_heads': args.n_heads_fbnetgen,
        }

    return grids


def generate_comparison_report(project_root, models):
    """
    Generate comparison report from all grid search results.

    Args:
        project_root: Project root path
        models: List of models that were searched
    """
    print("\n" + "="*70)
    print("GENERATING GRID SEARCH COMPARISON REPORT")
    print("="*70 + "\n")

    all_results = []

    for model_name in models:
        grid_dir = project_root / 'results' / 'grid_search' / model_name
        if not grid_dir.exists():
            continue

        # Load all configurations for this model
        for config_dir in sorted(grid_dir.iterdir()):
            if not config_dir.is_dir():
                continue

            summary_file = config_dir / 'aggregate_summary.json'
            config_file = config_dir / 'config.json'

            if summary_file.exists() and config_file.exists():
                with open(summary_file) as f:
                    summary = json.load(f)
                with open(config_file) as f:
                    config = json.load(f)

                # Extract metrics
                subj_metrics = summary.get('subject_level', {})
                win_metrics = summary.get('window_level', {})

                result = {
                    'model': model_name.upper(),
                    'config': config_dir.name,
                    'hidden_dim': config.get('hidden_dim'),
                    'dropout': config.get('dropout'),
                    'lr': config.get('lr'),
                    'n_layers': config.get('n_layers'),
                    'n_heads': config.get('n_heads'),
                    'subject_r': subj_metrics.get('r', 0),
                    'subject_mse': subj_metrics.get('mse', float('inf')),
                    'window_r': win_metrics.get('r', 0),
                    'window_mse': win_metrics.get('mse', float('inf')),
                }

                all_results.append(result)

    if not all_results:
        print("⚠️  No results found to compare.")
        return

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Sort by subject-level correlation (descending)
    df = df.sort_values('subject_r', ascending=False)

    # Save to CSV and Excel
    output_dir = project_root / 'results' / 'grid_search'
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / 'comparison_report.csv'
    excel_path = output_dir / 'comparison_report.xlsx'

    df.to_csv(csv_path, index=False)
    df.to_excel(excel_path, index=False)

    # Print summary
    print("Grid Search Results Summary")
    print("-" * 70)
    print(df.to_string(index=False))
    print("\n" + "="*70)

    # Print best configuration for each model
    print("\nBest Configuration per Model (by Subject-Level r):")
    print("-" * 70)

    for model_name in models:
        model_results = df[df['model'] == model_name.upper()]
        if len(model_results) > 0:
            best = model_results.iloc[0]
            print(f"\n{model_name.upper()}:")
            print(f"  Config: {best['config']}")
            print(f"  Subject r: {best['subject_r']:.4f}")
            print(f"  Subject MSE: {best['subject_mse']:.4f}")
            print(f"  Parameters: hidden_dim={best['hidden_dim']}, "
                  f"dropout={best['dropout']:.2f}, lr={best['lr']:.2e}, "
                  f"n_layers={best['n_layers']}, n_heads={best['n_heads']}")

    print("\n" + "="*70)
    print(f"\nResults saved to:")
    print(f"  CSV:   {csv_path}")
    print(f"  Excel: {excel_path}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Grid search pipeline for enhanced GNN models"
    )

    # Quick test mode
    parser.add_argument('--quick_test', action='store_true',
                        help='Fast test run (2 epochs, 1 fold, small grid)')

    # Model selection
    parser.add_argument('--models', nargs='+',
                        choices=['braingt', 'braingnn', 'fbnetgen'],
                        default=['braingt', 'braingnn', 'fbnetgen'],
                        help='Which models to search (default: all three)')

    # Hyperparameter ranges
    parser.add_argument('--hidden_dims', nargs='+', type=int,
                        default=[128],
                        help='Hidden dimensions to search (default: [128])')
    parser.add_argument('--dropouts', nargs='+', type=float,
                        default=[0.3],
                        help='Dropout rates to search (default: [0.3])')
    parser.add_argument('--learning_rates', nargs='+', type=float,
                        default=[1e-4],
                        help='Learning rates to search (default: [1e-4])')
    parser.add_argument('--n_layers', nargs='+', type=int,
                        default=[3],
                        help='Number of layers for BrainGNN/FBNetGen (default: [3])')
    parser.add_argument('--n_heads_braingt', nargs='+', type=int,
                        default=[8],
                        help='Number of heads for BrainGT (default: [8])')
    parser.add_argument('--n_heads_fbnetgen', nargs='+', type=int,
                        default=[4],
                        help='Number of heads for FBNetGen (default: [4])')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])

    args = parser.parse_args()

    # Override for quick test
    if args.quick_test:
        args.epochs = 2
        args.hidden_dims = [64, 128]
        args.dropouts = [0.2, 0.3]
        args.learning_rates = [1e-4]
        args.n_layers = [2, 3]
        args.n_heads_braingt = [4, 8]
        args.n_heads_fbnetgen = [2, 4]
        fold_name = 'graphs_outer1_inner2'
    else:
        fold_name = None

    project_root = get_project_root()

    print("="*70)
    print("GRID SEARCH PIPELINE FOR ENHANCED MODELS")
    print("="*70)
    print(f"\nModels: {', '.join([m.upper() for m in args.models])}")
    print(f"Hidden dims: {args.hidden_dims}")
    print(f"Dropouts: {args.dropouts}")
    print(f"Learning rates: {args.learning_rates}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    if args.quick_test:
        print(f"Fold: {fold_name} (quick test mode)")
    else:
        print(f"Folds: All folds")
    print("="*70)

    # Generate grid
    grids = generate_grid(args)

    # Calculate total runs
    total_runs = 0
    for model_name, grid in grids.items():
        n_configs = 1
        for param_values in grid.values():
            n_configs *= len(param_values)
        print(f"\n{model_name.upper()}: {n_configs} configurations")
        total_runs += n_configs

    print(f"\nTotal training runs: {total_runs}")
    print("="*70)

    # Confirm
    if not args.quick_test and total_runs > 10:
        response = input(f"\nThis will run {total_runs} training jobs. Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    # Run grid search
    start_time = datetime.now()
    results_log = []

    for model_name in args.models:
        grid = grids[model_name]

        # Generate all combinations
        keys = list(grid.keys())
        values = [grid[k] for k in keys]

        for combo in product(*values):
            config = dict(zip(keys, combo))

            success = run_training(
                model_name=model_name,
                config=config,
                fold_name=fold_name,
                device=args.device,
                project_root=project_root
            )

            results_log.append({
                'model': model_name,
                'config': config,
                'success': success
            })

    # Generate comparison report
    generate_comparison_report(project_root, args.models)

    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "="*70)
    print("GRID SEARCH COMPLETION SUMMARY")
    print("="*70)

    successful = sum(1 for r in results_log if r['success'])
    failed = len(results_log) - successful

    print(f"\nTotal runs: {len(results_log)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Duration: {duration}")

    if failed > 0:
        print("\n⚠️  Some runs failed. Check logs above for details.")
    else:
        print("\n✓ All runs completed successfully!")

    print("="*70)


if __name__ == '__main__':
    main()
