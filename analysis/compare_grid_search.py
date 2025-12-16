#!/usr/bin/env python
"""
Grid Search Results Comparison
================================

Compare GATv2 grid search configurations by reading aggregate summaries.

Usage:
    # Compare all grid search results
    python compare_grid_search.py

    # Show top N configs
    python compare_grid_search.py --top 10

    # Save to custom location
    python compare_grid_search.py --output my_grid_comparison.xlsx
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Any


def load_grid_search_results(grid_dir: Path = None) -> List[Dict[str, Any]]:
    """
    Load all aggregate summaries from grid search.

    Args:
        grid_dir: Path to grid search results directory

    Returns:
        List of config summaries
    """
    if grid_dir is None:
        # Try both possible locations
        grid_dir = Path("results/gatv2/grid")
        if not grid_dir.exists():
            grid_dir = Path("../../results/gatv2/grid")
        if not grid_dir.exists():
            print("❌ Grid search results directory not found!")
            return []

    print(f"Loading results from: {grid_dir}")

    # Find all config directories
    config_dirs = sorted([d for d in grid_dir.iterdir() if d.is_dir()])

    print(f"Found {len(config_dirs)} config directories")

    # Load aggregate summaries
    results = []
    for config_dir in config_dirs:
        aggregate_path = config_dir / "gatv2_aggregate_summary.json"

        if not aggregate_path.exists():
            print(f"  ⚠️  Skipping {config_dir.name} - no aggregate summary")
            continue

        try:
            with open(aggregate_path, 'r') as f:
                summary = json.load(f)
            results.append(summary)
            print(f"  ✓ Loaded {config_dir.name}")
        except Exception as e:
            print(f"  ❌ Error loading {config_dir.name}: {e}")

    return results


def create_comparison_dataframe(results: List[Dict]) -> pd.DataFrame:
    """
    Create comparison DataFrame from grid search results.

    Args:
        results: List of aggregate summaries

    Returns:
        DataFrame with all configs and metrics
    """
    rows = []

    for summary in results:
        config = summary['config']
        subj_r = summary['subject_level_r']
        subj_mse = summary['subject_level_mse']

        row = {
            'Config_Name': summary['config_name'],
            'Config_Index': summary.get('config_index', 0),
            'N_Folds': summary['n_folds'],

            # Subject-level metrics (most important)
            'Subject_R_Mean': subj_r['mean'],
            'Subject_R_Std': subj_r['std'],
            'Subject_R_Min': subj_r['min'],
            'Subject_R_Max': subj_r['max'],
            'Subject_MSE_Mean': subj_mse['mean'],
            'Subject_MSE_Std': subj_mse['std'],

            # Hyperparameters
            'Hidden_Dim': config['hidden_dim'],
            'N_Layers': config['n_layers'],
            'N_Heads': config['n_heads'],
            'Dropout': config['dropout'],
            'Edge_Dropout': config['edge_dropout'],
            'Learning_Rate': config['lr'],
            'Batch_Size': config['batch_size'],
            'Epochs': config['epochs'],
            'Patience': config['patience'],
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by Subject R (descending - higher is better)
    df = df.sort_values('Subject_R_Mean', ascending=False)

    # Add rank
    df.insert(0, 'Rank', range(1, len(df) + 1))

    return df


def print_summary(df: pd.DataFrame, top_n: int = 10):
    """Print comparison summary to console."""
    print("\n" + "="*100)
    print(f"GATV2 GRID SEARCH COMPARISON - TOP {min(top_n, len(df))} CONFIGURATIONS")
    print("="*100 + "\n")

    # Show top N configs
    display_df = df.head(top_n)

    # Print ranking table
    print("Subject-Level Performance (Higher R = Better):")
    print("-" * 100)
    cols = ['Rank', 'Subject_R_Mean', 'Subject_R_Std', 'Subject_MSE_Mean',
            'Hidden_Dim', 'N_Layers', 'N_Heads', 'Dropout', 'Learning_Rate']
    print(display_df[cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    # Best config details
    print("\n" + "="*100)
    best = df.iloc[0]
    print(f"🏆 BEST CONFIGURATION: {best['Config_Name']}")
    print("="*100)
    print(f"  Subject R:     {best['Subject_R_Mean']:.4f} ± {best['Subject_R_Std']:.4f}")
    print(f"  Subject MSE:   {best['Subject_MSE_Mean']:.2f} ± {best['Subject_MSE_Std']:.2f}")
    print(f"  R Range:       [{best['Subject_R_Min']:.4f}, {best['Subject_R_Max']:.4f}]")
    print(f"\n  Hyperparameters:")
    print(f"    Hidden Dim:    {best['Hidden_Dim']}")
    print(f"    Layers:        {best['N_Layers']}")
    print(f"    Heads:         {best['N_Heads']}")
    print(f"    Dropout:       {best['Dropout']}")
    print(f"    Edge Dropout:  {best['Edge_Dropout']}")
    print(f"    Learning Rate: {best['Learning_Rate']}")
    print(f"    Batch Size:    {best['Batch_Size']}")
    print("="*100 + "\n")


def save_excel(df: pd.DataFrame, output_path: str = 'grid_search_comparison.xlsx'):
    """Save comparison to Excel with multiple sheets."""
    output_path = Path(output_path)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Full Ranking
        rank_cols = [
            'Rank', 'Config_Name', 'N_Folds',
            'Subject_R_Mean', 'Subject_R_Std', 'Subject_R_Min', 'Subject_R_Max',
            'Subject_MSE_Mean', 'Subject_MSE_Std',
            'Hidden_Dim', 'N_Layers', 'N_Heads',
            'Dropout', 'Edge_Dropout', 'Learning_Rate'
        ]
        df[rank_cols].to_excel(writer, sheet_name='Ranking', index=False)

        # Sheet 2: Hyperparameters
        hyperparam_cols = [
            'Rank', 'Config_Name',
            'Hidden_Dim', 'N_Layers', 'N_Heads',
            'Dropout', 'Edge_Dropout',
            'Learning_Rate', 'Batch_Size', 'Epochs', 'Patience'
        ]
        df[hyperparam_cols].to_excel(writer, sheet_name='Hyperparameters', index=False)

        # Sheet 3: Performance Metrics
        metrics_cols = [
            'Rank', 'Config_Name', 'N_Folds',
            'Subject_R_Mean', 'Subject_R_Std', 'Subject_R_Min', 'Subject_R_Max',
            'Subject_MSE_Mean', 'Subject_MSE_Std'
        ]
        df[metrics_cols].to_excel(writer, sheet_name='Performance', index=False)

        # Sheet 4: All Data
        df.to_excel(writer, sheet_name='All_Data', index=False)

    print(f"✓ Saved comparison to: {output_path}")
    print(f"  - 4 sheets: Ranking, Hyperparameters, Performance, All_Data")


def main():
    parser = argparse.ArgumentParser(description="Compare GATv2 grid search results")

    parser.add_argument('--grid_dir', type=str, default=None,
                       help='Path to grid search results directory')
    parser.add_argument('--top', type=int, default=10,
                       help='Number of top configs to display (default: 10)')
    parser.add_argument('--output', type=str, default='grid_search_comparison.xlsx',
                       help='Output Excel file path')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save to Excel (print only)')

    args = parser.parse_args()

    # Load results
    print("="*100)
    print("GATV2 GRID SEARCH COMPARISON")
    print("="*100 + "\n")

    results = load_grid_search_results(
        Path(args.grid_dir) if args.grid_dir else None
    )

    if not results:
        print("\n❌ No valid results found!")
        print("\nMake sure you have run grid search and have aggregate summaries in:")
        print("  results/gatv2/grid/*/gatv2_aggregate_summary.json")
        return

    print(f"\nLoaded {len(results)} configurations")

    # Create comparison DataFrame
    df = create_comparison_dataframe(results)

    # Print summary
    print_summary(df, args.top)

    # Save to Excel
    if not args.no_save:
        save_excel(df, args.output)

    print("\n✓ Comparison complete!")


if __name__ == "__main__":
    main()
