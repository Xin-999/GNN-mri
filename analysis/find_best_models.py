#!/usr/bin/env python
"""
Find Best Model Configurations
================================

Scans all base and enhanced model results to find top performers.

Usage:
    # Find best overall
    python find_best_models.py

    # Show top 5 configs per model
    python find_best_models.py --top 5

    # Include both base and enhanced
    python find_best_models.py --include_base
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np


def load_base_model_results(results_dir):
    """
    Load all base model results from results/advanced/.

    Args:
        results_dir: Path to results directory

    Returns:
        List of result dictionaries
    """
    advanced_dir = results_dir / 'advanced'
    if not advanced_dir.exists():
        print(f"⚠️  Base models directory not found: {advanced_dir}")
        return []

    all_results = []

    # Iterate through model directories (braingnn, braingt, fbnetgen)
    for model_dir in sorted(advanced_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        print(f"\nLoading base model: {model_name.upper()}")

        # Collect metrics from all folds
        fold_metrics = {'r': [], 'mse': [], 'mae': [], 'pearson_r': [], 'spearman_r': []}
        fold_count = 0

        for fold_dir in sorted(model_dir.iterdir()):
            if not fold_dir.is_dir() or not fold_dir.name.startswith('graphs_'):
                continue

            # Look for summary JSON
            summary_file = fold_dir / f'{model_name}_summary.json'
            if not summary_file.exists():
                continue

            try:
                with open(summary_file) as f:
                    fold_summary = json.load(f)

                # Extract subject-level metrics (multiple possible formats)
                subj_metrics = None
                if 'subject_metrics' in fold_summary:
                    subj_metrics = fold_summary['subject_metrics']
                elif 'subject_level' in fold_summary:
                    subj_metrics = fold_summary['subject_level']
                elif 'test_metrics' in fold_summary and isinstance(fold_summary['test_metrics'], dict):
                    if 'subject_level' in fold_summary['test_metrics']:
                        subj_metrics = fold_summary['test_metrics']['subject_level']

                if subj_metrics and 'r' in subj_metrics:
                    fold_metrics['r'].append(subj_metrics.get('r', 0))
                    fold_metrics['mse'].append(subj_metrics.get('mse', float('inf')))
                    fold_metrics['mae'].append(subj_metrics.get('mae', float('inf')))
                    fold_metrics['pearson_r'].append(subj_metrics.get('pearson_r', subj_metrics.get('r', 0)))
                    fold_metrics['spearman_r'].append(subj_metrics.get('spearman_r', 0))
                    fold_count += 1
            except Exception as e:
                print(f"  ⚠️  Error loading {fold_dir.name}: {e}")
                continue

        if fold_metrics['r']:
            result = {
                'model': model_name.upper(),
                'type': 'base',
                'config': 'default',
                'n_folds': fold_count,
                'subject_r_mean': float(np.mean(fold_metrics['r'])),
                'subject_r_std': float(np.std(fold_metrics['r'])),
                'subject_mse_mean': float(np.mean(fold_metrics['mse'])),
                'subject_mse_std': float(np.std(fold_metrics['mse'])),
                'subject_mae_mean': float(np.mean(fold_metrics['mae'])),
                'subject_mae_std': float(np.std(fold_metrics['mae'])),
                'pearson_r_mean': float(np.mean(fold_metrics['pearson_r'])),
                'pearson_r_std': float(np.std(fold_metrics['pearson_r'])),
                'spearman_r_mean': float(np.mean(fold_metrics['spearman_r'])),
                'spearman_r_std': float(np.std(fold_metrics['spearman_r'])),
            }
            all_results.append(result)
            print(f"  ✓ Loaded {fold_count} folds - r: {result['subject_r_mean']:.4f} ± {result['subject_r_std']:.4f}")
        else:
            print(f"  ✗ No valid fold results found")

    return all_results


def load_enhanced_model_results(results_dir):
    """
    Load all enhanced model results from results/enhanced/.

    Args:
        results_dir: Path to results directory

    Returns:
        List of result dictionaries
    """
    enhanced_dir = results_dir / 'enhanced'
    if not enhanced_dir.exists():
        print(f"⚠️  Enhanced models directory not found: {enhanced_dir}")
        return []

    all_results = []

    # Iterate through enhanced model directories
    for model_dir in sorted(enhanced_dir.iterdir()):
        if not model_dir.is_dir() or not model_dir.name.endswith('_enhanced'):
            continue

        model_name = model_dir.name.replace('_enhanced', '')
        print(f"\nLoading enhanced model: {model_name.upper()}")

        # Look for aggregate summary
        aggregate_file = model_dir / f'{model_name}_aggregate_summary.json'
        if not aggregate_file.exists():
            print(f"  ✗ Aggregate summary not found")
            continue

        try:
            with open(aggregate_file) as f:
                aggregate = json.load(f)

            # Extract subject-level aggregate metrics
            subj_agg = aggregate.get('subject_level_aggregate', {})
            if not subj_agg:
                print(f"  ✗ No subject-level aggregate metrics")
                continue

            # Handle nested dict structure (mean/std)
            r_mean = subj_agg.get('r', {}).get('mean', 0) if isinstance(subj_agg.get('r'), dict) else subj_agg.get('r', 0)
            r_std = subj_agg.get('r', {}).get('std', 0) if isinstance(subj_agg.get('r'), dict) else 0
            mse_mean = subj_agg.get('mse', {}).get('mean', float('inf')) if isinstance(subj_agg.get('mse'), dict) else subj_agg.get('mse', float('inf'))
            mse_std = subj_agg.get('mse', {}).get('std', 0) if isinstance(subj_agg.get('mse'), dict) else 0
            mae_mean = subj_agg.get('mae', {}).get('mean', float('inf')) if isinstance(subj_agg.get('mae'), dict) else subj_agg.get('mae', float('inf'))
            mae_std = subj_agg.get('mae', {}).get('std', 0) if isinstance(subj_agg.get('mae'), dict) else 0

            pearson_r_mean = subj_agg.get('pearson_r', {}).get('mean', r_mean) if isinstance(subj_agg.get('pearson_r'), dict) else subj_agg.get('pearson_r', r_mean)
            pearson_r_std = subj_agg.get('pearson_r', {}).get('std', r_std) if isinstance(subj_agg.get('pearson_r'), dict) else r_std
            spearman_r_mean = subj_agg.get('spearman_r', {}).get('mean', 0) if isinstance(subj_agg.get('spearman_r'), dict) else subj_agg.get('spearman_r', 0)
            spearman_r_std = subj_agg.get('spearman_r', {}).get('std', 0) if isinstance(subj_agg.get('spearman_r'), dict) else 0

            # Count folds
            n_folds = len(aggregate.get('per_fold_summary', []))

            result = {
                'model': model_name.upper(),
                'type': 'enhanced',
                'config': 'default',
                'n_folds': n_folds,
                'subject_r_mean': float(r_mean),
                'subject_r_std': float(r_std),
                'subject_mse_mean': float(mse_mean),
                'subject_mse_std': float(mse_std),
                'subject_mae_mean': float(mae_mean),
                'subject_mae_std': float(mae_std),
                'pearson_r_mean': float(pearson_r_mean),
                'pearson_r_std': float(pearson_r_std),
                'spearman_r_mean': float(spearman_r_mean),
                'spearman_r_std': float(spearman_r_std),
            }
            all_results.append(result)
            print(f"  ✓ Loaded {n_folds} folds - r: {result['subject_r_mean']:.4f} ± {result['subject_r_std']:.4f}")

        except Exception as e:
            print(f"  ⚠️  Error loading enhanced model: {e}")
            continue

    return all_results


def load_grid_search_results(results_dir):
    """
    Load all grid search results from results/grid_search/.

    Args:
        results_dir: Path to results directory

    Returns:
        List of result dictionaries
    """
    grid_dir = results_dir / 'grid_search'
    if not grid_dir.exists():
        print(f"\n⚠️  Grid search directory not found: {grid_dir}")
        return []

    all_results = []
    print(f"\nLoading grid search results...")

    # Iterate through model directories
    for model_dir in sorted(grid_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        # Iterate through config directories
        for config_dir in sorted(model_dir.iterdir()):
            if not config_dir.is_dir():
                continue

            config_str = config_dir.name
            aggregate_file = config_dir / 'aggregate_summary.json'
            config_file = config_dir / 'config.json'

            if not aggregate_file.exists() or not config_file.exists():
                continue

            try:
                with open(aggregate_file) as f:
                    aggregate = json.load(f)
                with open(config_file) as f:
                    config = json.load(f)

                # Extract metrics
                subj_agg = aggregate.get('subject_level_aggregate', {})

                r_mean = subj_agg.get('r', {}).get('mean', 0) if isinstance(subj_agg.get('r'), dict) else subj_agg.get('r', 0)
                r_std = subj_agg.get('r', {}).get('std', 0) if isinstance(subj_agg.get('r'), dict) else 0
                mse_mean = subj_agg.get('mse', {}).get('mean', float('inf')) if isinstance(subj_agg.get('mse'), dict) else subj_agg.get('mse', float('inf'))
                mse_std = subj_agg.get('mse', {}).get('std', 0) if isinstance(subj_agg.get('mse'), dict) else 0
                mae_mean = subj_agg.get('mae', {}).get('mean', float('inf')) if isinstance(subj_agg.get('mae'), dict) else subj_agg.get('mae', float('inf'))
                mae_std = subj_agg.get('mae', {}).get('std', 0) if isinstance(subj_agg.get('mae'), dict) else 0

                pearson_r_mean = subj_agg.get('pearson_r', {}).get('mean', r_mean) if isinstance(subj_agg.get('pearson_r'), dict) else subj_agg.get('pearson_r', r_mean)
                pearson_r_std = subj_agg.get('pearson_r', {}).get('std', r_std) if isinstance(subj_agg.get('pearson_r'), dict) else r_std
                spearman_r_mean = subj_agg.get('spearman_r', {}).get('mean', 0) if isinstance(subj_agg.get('spearman_r'), dict) else subj_agg.get('spearman_r', 0)
                spearman_r_std = subj_agg.get('spearman_r', {}).get('std', 0) if isinstance(subj_agg.get('spearman_r'), dict) else 0

                n_folds = len(aggregate.get('per_fold_summary', []))

                result = {
                    'model': model_name.upper(),
                    'type': 'grid_search',
                    'config': config_str,
                    'hidden_dim': config.get('hidden_dim'),
                    'dropout': config.get('dropout'),
                    'lr': config.get('lr'),
                    'n_layers': config.get('n_layers'),
                    'n_heads': config.get('n_heads'),
                    'n_folds': n_folds,
                    'subject_r_mean': float(r_mean),
                    'subject_r_std': float(r_std),
                    'subject_mse_mean': float(mse_mean),
                    'subject_mse_std': float(mse_std),
                    'subject_mae_mean': float(mae_mean),
                    'subject_mae_std': float(mae_std),
                    'pearson_r_mean': float(pearson_r_mean),
                    'pearson_r_std': float(pearson_r_std),
                    'spearman_r_mean': float(spearman_r_mean),
                    'spearman_r_std': float(spearman_r_std),
                }
                all_results.append(result)

            except Exception as e:
                print(f"  ⚠️  Error loading {model_name}/{config_str}: {e}")
                continue

    if all_results:
        print(f"  ✓ Loaded {len(all_results)} grid search configurations")

    return all_results


def print_results(df, top_n=None, include_base=False):
    """
    Print results in a formatted table.

    Args:
        df: DataFrame with results
        top_n: Show top N results per model
        include_base: Include base model results
    """
    if df.empty:
        print("\n❌ No results found!")
        return

    # Filter by type if needed
    if not include_base:
        df = df[df['type'] != 'base'].copy()

    if df.empty:
        print("\n❌ No enhanced/grid search results found!")
        return

    print("\n" + "="*100)
    print("BEST MODEL CONFIGURATIONS")
    print("="*100)

    # Overall best
    best_overall = df.loc[df['subject_r_mean'].idxmax()]
    print(f"\n🏆 BEST OVERALL:")
    print(f"   Model: {best_overall['model']}")
    print(f"   Type: {best_overall['type']}")
    print(f"   Config: {best_overall['config']}")
    print(f"   Subject Pearson r: {best_overall['pearson_r_mean']:.4f} ± {best_overall['pearson_r_std']:.4f}")
    print(f"   Subject Spearman r: {best_overall['spearman_r_mean']:.4f} ± {best_overall['spearman_r_std']:.4f}")
    print(f"   Subject MSE: {best_overall['subject_mse_mean']:.4f} ± {best_overall['subject_mse_std']:.4f}")
    print(f"   Folds: {best_overall['n_folds']}")

    if best_overall['type'] == 'grid_search':
        print(f"\n   Hyperparameters:")
        print(f"     - hidden_dim: {best_overall['hidden_dim']}")
        print(f"     - dropout: {best_overall['dropout']:.2f}")
        print(f"     - lr: {best_overall['lr']:.2e}")
        print(f"     - n_layers: {best_overall['n_layers']}")
        print(f"     - n_heads: {best_overall['n_heads']}")

    # Best per model
    print(f"\n{'='*100}")
    print("TOP CONFIGURATIONS PER MODEL")
    print("="*100)

    for model in sorted(df['model'].unique()):
        model_df = df[df['model'] == model].copy()
        model_df = model_df.sort_values('subject_r_mean', ascending=False)

        print(f"\n{model}:")
        print("-" * 100)

        display_df = model_df.head(top_n) if top_n else model_df

        for idx, (_, row) in enumerate(display_df.iterrows(), 1):
            print(f"\n  Rank {idx}: [{row['type'].upper()}] {row['config']}")
            print(f"    Pearson r:  {row['pearson_r_mean']:.4f} ± {row['pearson_r_std']:.4f}")
            print(f"    Spearman r: {row['spearman_r_mean']:.4f} ± {row['spearman_r_std']:.4f}")
            print(f"    MSE: {row['subject_mse_mean']:.4f} ± {row['subject_mse_std']:.4f}")
            print(f"    MAE: {row['subject_mae_mean']:.4f} ± {row['subject_mae_std']:.4f}")
            print(f"    Folds: {row['n_folds']}")

            if row['type'] == 'grid_search':
                print(f"    Config: h={row['hidden_dim']}, d={row['dropout']:.2f}, lr={row['lr']:.2e}, "
                      f"layers={row['n_layers']}, heads={row['n_heads']}")

    print("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser(
        description="Find best model configurations across all results"
    )

    parser.add_argument('--results_dir', type=str, default=None,
                        help='Path to results directory (default: auto-detect)')
    parser.add_argument('--top', type=int, default=None,
                        help='Show top N configs per model (default: all)')
    parser.add_argument('--include_base', action='store_true',
                        help='Include base models in comparison')
    parser.add_argument('--save_csv', action='store_true',
                        help='Save results to CSV file')

    args = parser.parse_args()

    # Determine results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        # Try to auto-detect
        project_root = Path(__file__).parent.parent
        results_dir = project_root / 'results'
        if not results_dir.exists():
            results_dir = Path('results')

    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        print("Please specify --results_dir")
        return

    print(f"Scanning results directory: {results_dir}")
    print("="*100)

    # Load all results
    all_results = []

    # Load base models
    if args.include_base:
        base_results = load_base_model_results(results_dir)
        all_results.extend(base_results)

    # Load enhanced models
    enhanced_results = load_enhanced_model_results(results_dir)
    all_results.extend(enhanced_results)

    # Load grid search results
    grid_results = load_grid_search_results(results_dir)
    all_results.extend(grid_results)

    if not all_results:
        print("\n❌ No results found in any directory!")
        return

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Print results
    print_results(df, top_n=args.top, include_base=args.include_base)

    # Save to CSV if requested
    if args.save_csv:
        output_file = results_dir / 'best_models_summary.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to: {output_file}")


if __name__ == '__main__':
    main()
