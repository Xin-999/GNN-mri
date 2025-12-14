#!/usr/bin/env python
"""
GATv2 Configuration Comparison Script
======================================

Compare performance across different GATv2 hyperparameter configurations.

Usage:
    # Compare all trained models in results/gatv2/
    python compare_gatv2_configs.py

    # Compare specific result directories
    python compare_gatv2_configs.py --results_dirs results/gatv2/improved results/gatv2/grid_*

    # Save to custom location
    python compare_gatv2_configs.py --output my_comparison.xlsx
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Any


def load_fold_summary(fold_dir: Path) -> Dict[str, Any]:
    """Load summary JSON from a single fold directory."""
    summary_path = fold_dir / "gatv2_summary.json"

    if not summary_path.exists():
        return None

    try:
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        return summary
    except Exception as e:
        print(f"  Warning: Error loading {summary_path}: {e}")
        return None


def extract_config_performance(results_dir: Path) -> Dict[str, Any]:
    """
    Extract configuration and performance metrics from a results directory.

    Args:
        results_dir: Path to results directory (e.g., results/gatv2/improved)

    Returns:
        Dictionary with config and aggregated metrics
    """
    results_dir = Path(results_dir)

    if not results_dir.exists():
        return None

    # Find all fold subdirectories (graphs_outer*_inner*)
    fold_dirs = sorted([d for d in results_dir.iterdir()
                       if d.is_dir() and d.name.startswith('graphs_')])

    if not fold_dirs:
        print(f"  No fold directories found in {results_dir}")
        return None

    # Collect metrics from all folds
    all_summaries = []
    configs = []

    for fold_dir in fold_dirs:
        summary = load_fold_summary(fold_dir)
        if summary:
            all_summaries.append(summary)
            if 'config' in summary:
                configs.append(summary['config'])

    if not all_summaries:
        return None

    # Use first config (should be same across all folds)
    config = configs[0] if configs else {}

    # Extract test metrics
    subject_r_list = []
    subject_mse_list = []
    subject_mae_list = []
    subject_rmse_list = []

    window_r_list = []
    window_mse_list = []
    window_mae_list = []

    epochs_list = []

    for summary in all_summaries:
        if 'test_metrics' not in summary:
            continue

        test_metrics = summary['test_metrics']

        # Subject-level metrics (most important)
        if 'subject_level' in test_metrics:
            subj = test_metrics['subject_level']
            if 'r' in subj:
                subject_r_list.append(subj['r'])
            if 'mse' in subj:
                subject_mse_list.append(subj['mse'])
            if 'mae' in subj:
                subject_mae_list.append(subj['mae'])
            if 'rmse' in subj:
                subject_rmse_list.append(subj['rmse'])

        # Window-level metrics
        if 'window_level' in test_metrics:
            win = test_metrics['window_level']
            if 'r' in win:
                window_r_list.append(win['r'])
            if 'mse' in win:
                window_mse_list.append(win['mse'])
            if 'mae' in win:
                window_mae_list.append(win['mae'])

        # Training info
        if 'training_summary' in summary:
            if 'best_epoch' in summary['training_summary']:
                epochs_list.append(summary['training_summary']['best_epoch'])

    return {
        # Identification
        'Config_Name': results_dir.name,
        'Path': str(results_dir),
        'N_Folds': len(all_summaries),

        # Hyperparameters
        'Hidden_Dim': config.get('hidden_dim', 'N/A'),
        'N_Layers': config.get('n_layers', 'N/A'),
        'N_Heads': config.get('n_heads', 'N/A'),
        'Dropout': config.get('dropout', 'N/A'),
        'Edge_Dropout': config.get('edge_dropout', 'N/A'),
        'Learning_Rate': config.get('lr', 'N/A'),
        'Batch_Size': config.get('batch_size', 'N/A'),
        'Patience': config.get('patience', 'N/A'),

        # Subject-level Performance (PRIMARY METRIC)
        'Subject_R_Mean': np.mean(subject_r_list) if subject_r_list else np.nan,
        'Subject_R_Std': np.std(subject_r_list) if subject_r_list else np.nan,
        'Subject_R_Min': np.min(subject_r_list) if subject_r_list else np.nan,
        'Subject_R_Max': np.max(subject_r_list) if subject_r_list else np.nan,
        'Subject_MSE_Mean': np.mean(subject_mse_list) if subject_mse_list else np.nan,
        'Subject_MSE_Std': np.std(subject_mse_list) if subject_mse_list else np.nan,
        'Subject_MAE_Mean': np.mean(subject_mae_list) if subject_mae_list else np.nan,
        'Subject_MAE_Std': np.std(subject_mae_list) if subject_mae_list else np.nan,
        'Subject_RMSE_Mean': np.mean(subject_rmse_list) if subject_rmse_list else np.nan,

        # Window-level Performance
        'Window_R_Mean': np.mean(window_r_list) if window_r_list else np.nan,
        'Window_R_Std': np.std(window_r_list) if window_r_list else np.nan,
        'Window_MSE_Mean': np.mean(window_mse_list) if window_mse_list else np.nan,
        'Window_MAE_Mean': np.mean(window_mae_list) if window_mae_list else np.nan,

        # Training Info
        'Avg_Best_Epoch': np.mean(epochs_list) if epochs_list else np.nan,
    }


def compare_configurations(results_dirs: List[str] = None,
                          output_path: str = 'gatv2_config_comparison.xlsx'):
    """
    Compare GATv2 configurations and save results to Excel.

    Args:
        results_dirs: List of paths to results directories. If None, auto-discover from results/gatv2/
        output_path: Path to save comparison Excel file
    """
    # Auto-discover if not specified
    if results_dirs is None:
        base_dir = Path('results/gatv2')
        if base_dir.exists():
            results_dirs = [str(d) for d in base_dir.iterdir() if d.is_dir()]
        else:
            print("❌ No results/gatv2/ directory found!")
            return None

    if not results_dirs:
        print("❌ No results directories specified!")
        return None

    print("="*80)
    print("GATv2 CONFIGURATION COMPARISON")
    print("="*80)

    # Extract performance from each configuration
    all_configs = []
    for results_dir in results_dirs:
        print(f"\nProcessing: {results_dir}")
        config_info = extract_config_performance(Path(results_dir))
        if config_info:
            all_configs.append(config_info)
            print(f"  ✓ Loaded {config_info['N_Folds']} folds")
            if not np.isnan(config_info['Subject_R_Mean']):
                print(f"  Subject R: {config_info['Subject_R_Mean']:.4f} ± {config_info['Subject_R_Std']:.4f}")
            else:
                print(f"  Window R: {config_info['Window_R_Mean']:.4f} ± {config_info['Window_R_Std']:.4f}")

    if not all_configs:
        print("\n❌ No valid configurations found!")
        return None

    # Create DataFrame
    df = pd.DataFrame(all_configs)

    # Sort by Subject R (descending - higher is better)
    # If Subject R not available, sort by Window R
    if 'Subject_R_Mean' in df.columns and not df['Subject_R_Mean'].isna().all():
        df = df.sort_values('Subject_R_Mean', ascending=False)
        sort_metric = 'Subject_R_Mean'
    else:
        df = df.sort_values('Window_R_Mean', ascending=False)
        sort_metric = 'Window_R_Mean'

    # Add rank
    df.insert(0, 'Rank', range(1, len(df) + 1))

    # Save to Excel with multiple sheets
    output_path = Path(output_path)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Ranking Summary (key metrics only)
        summary_cols = [
            'Rank', 'Config_Name', 'N_Folds',
            'Subject_R_Mean', 'Subject_R_Std',
            'Subject_MSE_Mean', 'Subject_MAE_Mean',
            'Hidden_Dim', 'N_Layers', 'Dropout', 'Learning_Rate'
        ]
        # Only include columns that exist
        summary_cols = [c for c in summary_cols if c in df.columns]
        df[summary_cols].to_excel(writer, sheet_name='Ranking', index=False)

        # Sheet 2: Hyperparameters
        hyperparam_cols = [
            'Rank', 'Config_Name',
            'Hidden_Dim', 'N_Layers', 'N_Heads',
            'Dropout', 'Edge_Dropout',
            'Learning_Rate', 'Batch_Size', 'Patience'
        ]
        hyperparam_cols = [c for c in hyperparam_cols if c in df.columns]
        df[hyperparam_cols].to_excel(writer, sheet_name='Hyperparameters', index=False)

        # Sheet 3: Subject-level Metrics (detailed)
        subject_cols = [
            'Rank', 'Config_Name',
            'Subject_R_Mean', 'Subject_R_Std', 'Subject_R_Min', 'Subject_R_Max',
            'Subject_MSE_Mean', 'Subject_MSE_Std',
            'Subject_MAE_Mean', 'Subject_MAE_Std',
            'Subject_RMSE_Mean'
        ]
        subject_cols = [c for c in subject_cols if c in df.columns]
        if len(subject_cols) > 2:  # Has actual metrics
            df[subject_cols].to_excel(writer, sheet_name='Subject_Metrics', index=False)

        # Sheet 4: Window-level Metrics
        window_cols = [
            'Rank', 'Config_Name',
            'Window_R_Mean', 'Window_R_Std',
            'Window_MSE_Mean', 'Window_MAE_Mean'
        ]
        window_cols = [c for c in window_cols if c in df.columns]
        df[window_cols].to_excel(writer, sheet_name='Window_Metrics', index=False)

        # Sheet 5: Full Data (all columns)
        df.to_excel(writer, sheet_name='Full_Data', index=False)

    print(f"\n{'='*80}")
    print(f"✓ Comparison saved to: {output_path}")
    print(f"{'='*80}\n")

    # Print top 5 configurations
    print(f"Top 5 Configurations (ranked by {sort_metric}):")
    print("-" * 80)

    for i, row in df.head(5).iterrows():
        print(f"\n{row['Rank']}. {row['Config_Name']}")

        if not np.isnan(row.get('Subject_R_Mean', np.nan)):
            print(f"   Subject R:   {row['Subject_R_Mean']:.4f} ± {row['Subject_R_Std']:.4f}")
            print(f"   Subject MSE: {row['Subject_MSE_Mean']:.2f} ± {row['Subject_MSE_Std']:.2f}")
        else:
            print(f"   Window R:    {row['Window_R_Mean']:.4f} ± {row['Window_R_Std']:.4f}")
            print(f"   Window MSE:  {row['Window_MSE_Mean']:.2f}")

        print(f"   Config: hidden_dim={row['Hidden_Dim']}, n_layers={row['N_Layers']}, "
              f"n_heads={row['N_Heads']}")
        print(f"           dropout={row['Dropout']}, edge_dropout={row['Edge_Dropout']}, "
              f"lr={row['Learning_Rate']}")

    print()

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compare GATv2 hyperparameter configurations"
    )
    parser.add_argument(
        '--results_dirs',
        nargs='+',
        default=None,
        help='Paths to results directories (auto-discovers from results/gatv2/ if not specified)'
    )
    parser.add_argument(
        '--output',
        default='gatv2_config_comparison.xlsx',
        help='Output Excel file path'
    )

    args = parser.parse_args()

    compare_configurations(args.results_dirs, args.output)


if __name__ == "__main__":
    main()
