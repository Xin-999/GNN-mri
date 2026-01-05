#!/usr/bin/env python
"""
Enhanced Models Grid Search Comparison
========================================

Analyze and visualize grid search results for enhanced models.

Usage:
    # Show all results
    python compare_enhanced_grid_search.py

    # Show top 5 for each model
    python compare_enhanced_grid_search.py --top 5

    # Visualize hyperparameter impact
    python compare_enhanced_grid_search.py --plot
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_grid_search_results(grid_dir=None):
    """
    Load all grid search results.

    Args:
        grid_dir: Path to grid search directory

    Returns:
        DataFrame with all results
    """
    if grid_dir is None:
        grid_dir = Path("results/grid_search")
        if not grid_dir.exists():
            grid_dir = Path("../../results/grid_search")
        if not grid_dir.exists():
            print("❌ Grid search results directory not found!")
            return pd.DataFrame()

    all_results = []

    # Load results for each model
    for model_dir in sorted(grid_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name == 'comparison_report.csv':
            continue

        model_name = model_dir.name

        # Load each configuration
        for config_dir in sorted(model_dir.iterdir()):
            if not config_dir.is_dir():
                continue

            summary_file = config_dir / 'aggregate_summary.json'
            config_file = config_dir / 'config.json'

            if summary_file.exists() and config_file.exists():
                with open(summary_file) as f:
                    summary = json.load(f)
                with open(config_file) as f:
                    config = json.load(f)

                # Extract metrics - enhanced models use 'subject_level_aggregate' structure
                subj_agg = summary.get('subject_level_aggregate', {})
                win_agg = summary.get('window_level_aggregate', {})

                # Extract mean and std from nested structure
                subject_r = subj_agg.get('r', {}).get('mean', 0) if isinstance(subj_agg.get('r'), dict) else subj_agg.get('r', 0)
                subject_r_std = subj_agg.get('r', {}).get('std', 0) if isinstance(subj_agg.get('r'), dict) else 0
                subject_mse = subj_agg.get('mse', {}).get('mean', float('inf')) if isinstance(subj_agg.get('mse'), dict) else subj_agg.get('mse', float('inf'))
                subject_mae = subj_agg.get('mae', {}).get('mean', float('inf')) if isinstance(subj_agg.get('mae'), dict) else subj_agg.get('mae', float('inf'))

                window_r = win_agg.get('r', {}).get('mean', 0) if isinstance(win_agg.get('r'), dict) else win_agg.get('r', 0)
                window_mse = win_agg.get('mse', {}).get('mean', float('inf')) if isinstance(win_agg.get('mse'), dict) else win_agg.get('mse', float('inf'))

                result = {
                    'model': model_name.upper(),
                    'config_name': config_dir.name,
                    'hidden_dim': config.get('hidden_dim'),
                    'dropout': config.get('dropout'),
                    'lr': config.get('lr'),
                    'n_layers': config.get('n_layers'),
                    'n_heads': config.get('n_heads'),
                    'epochs': config.get('epochs'),
                    'subject_r': subject_r,
                    'subject_r_std': subject_r_std,
                    'subject_mse': subject_mse,
                    'subject_mae': subject_mae,
                    'window_r': window_r,
                    'window_mse': window_mse,
                }

                all_results.append(result)

    return pd.DataFrame(all_results)


def print_summary(df, top_n=None):
    """Print summary of grid search results."""
    if df.empty:
        print("No results to display.")
        return

    print("\n" + "="*80)
    print("GRID SEARCH RESULTS SUMMARY")
    print("="*80)

    # Overall best
    best_overall = df.loc[df['subject_r'].idxmax()]
    print(f"\nBest Overall Configuration:")
    print(f"  Model: {best_overall['model']}")
    print(f"  Config: {best_overall['config_name']}")
    print(f"  Subject r: {best_overall['subject_r']:.4f}")
    print(f"  Subject MSE: {best_overall['subject_mse']:.4f}")

    # Best per model
    print(f"\n{'='*80}")
    print("BEST CONFIGURATION PER MODEL")
    print("="*80)

    for model in df['model'].unique():
        model_df = df[df['model'] == model].copy()
        model_df = model_df.sort_values('subject_r', ascending=False)

        print(f"\n{model}:")
        print("-" * 80)

        # Show top N or all
        display_df = model_df.head(top_n) if top_n else model_df

        for idx, row in display_df.iterrows():
            print(f"\n  Rank {list(display_df.index).index(idx) + 1}:")
            print(f"    Config: {row['config_name']}")
            print(f"    Subject r: {row['subject_r']:.4f}")
            print(f"    Subject MSE: {row['subject_mse']:.4f}")
            print(f"    Parameters:")
            print(f"      - hidden_dim: {row['hidden_dim']}")
            print(f"      - dropout: {row['dropout']:.2f}")
            print(f"      - lr: {row['lr']:.2e}")
            print(f"      - n_layers: {row['n_layers']}")
            print(f"      - n_heads: {row['n_heads']}")

    print("\n" + "="*80)


def plot_hyperparameter_impact(df, output_dir=None):
    """
    Plot impact of each hyperparameter on performance.

    Args:
        df: DataFrame with grid search results
        output_dir: Directory to save plots
    """
    if df.empty:
        print("No data to plot.")
        return

    if output_dir is None:
        output_dir = Path("results/grid_search/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_style('whitegrid')
    models = df['model'].unique()

    # Plot for each model
    for model in models:
        model_df = df[df['model'] == model]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{model} - Hyperparameter Impact on Subject-Level r', fontsize=16)

        # Hidden dimension
        if len(model_df['hidden_dim'].unique()) > 1:
            ax = axes[0, 0]
            hidden_grouped = model_df.groupby('hidden_dim')['subject_r'].agg(['mean', 'std'])
            ax.errorbar(hidden_grouped.index, hidden_grouped['mean'],
                       yerr=hidden_grouped['std'], marker='o', capsize=5)
            ax.set_xlabel('Hidden Dimension')
            ax.set_ylabel('Subject r')
            ax.set_title('Hidden Dimension Impact')
            ax.grid(True, alpha=0.3)

        # Dropout
        if len(model_df['dropout'].unique()) > 1:
            ax = axes[0, 1]
            dropout_grouped = model_df.groupby('dropout')['subject_r'].agg(['mean', 'std'])
            ax.errorbar(dropout_grouped.index, dropout_grouped['mean'],
                       yerr=dropout_grouped['std'], marker='o', capsize=5)
            ax.set_xlabel('Dropout Rate')
            ax.set_ylabel('Subject r')
            ax.set_title('Dropout Impact')
            ax.grid(True, alpha=0.3)

        # Learning rate
        if len(model_df['lr'].unique()) > 1:
            ax = axes[0, 2]
            lr_grouped = model_df.groupby('lr')['subject_r'].agg(['mean', 'std'])
            ax.errorbar(lr_grouped.index, lr_grouped['mean'],
                       yerr=lr_grouped['std'], marker='o', capsize=5)
            ax.set_xlabel('Learning Rate')
            ax.set_ylabel('Subject r')
            ax.set_title('Learning Rate Impact')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)

        # Number of layers
        if len(model_df['n_layers'].unique()) > 1:
            ax = axes[1, 0]
            layers_grouped = model_df.groupby('n_layers')['subject_r'].agg(['mean', 'std'])
            ax.errorbar(layers_grouped.index, layers_grouped['mean'],
                       yerr=layers_grouped['std'], marker='o', capsize=5)
            ax.set_xlabel('Number of Layers')
            ax.set_ylabel('Subject r')
            ax.set_title('Layers Impact')
            ax.grid(True, alpha=0.3)

        # Number of heads
        if len(model_df['n_heads'].unique()) > 1:
            ax = axes[1, 1]
            heads_grouped = model_df.groupby('n_heads')['subject_r'].agg(['mean', 'std'])
            ax.errorbar(heads_grouped.index, heads_grouped['mean'],
                       yerr=heads_grouped['std'], marker='o', capsize=5)
            ax.set_xlabel('Number of Heads')
            ax.set_ylabel('Subject r')
            ax.set_title('Heads Impact')
            ax.grid(True, alpha=0.3)

        # All configs comparison
        ax = axes[1, 2]
        configs = model_df['config_name'].values
        subject_r = model_df['subject_r'].values
        colors = plt.cm.viridis(np.linspace(0, 1, len(configs)))

        bars = ax.barh(range(len(configs)), subject_r, color=colors)
        ax.set_yticks(range(len(configs)))
        ax.set_yticklabels([c[:20] + '...' if len(c) > 20 else c for c in configs], fontsize=8)
        ax.set_xlabel('Subject r')
        ax.set_title('All Configurations')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plot_path = output_dir / f'{model.lower()}_hyperparameter_impact.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {plot_path}")
        plt.close()

    # Cross-model comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    model_best = df.groupby('model')['subject_r'].max().sort_values(ascending=False)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax.bar(range(len(model_best)), model_best.values, color=colors[:len(model_best)])
    ax.set_xticks(range(len(model_best)))
    ax.set_xticklabels(model_best.index, fontsize=12)
    ax.set_ylabel('Best Subject r', fontsize=12)
    ax.set_title('Best Performance per Model', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, model_best.values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plot_path = output_dir / 'model_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    plt.close()

    print(f"\nAll plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare grid search results for enhanced models"
    )

    parser.add_argument('--top', type=int, default=None,
                        help='Show top N configs per model (default: all)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plots')

    args = parser.parse_args()

    # Load results
    print("Loading grid search results...")
    df = load_grid_search_results()

    if df.empty:
        print("No grid search results found.")
        return

    print(f"Loaded {len(df)} configurations across {df['model'].nunique()} models")

    # Print summary
    print_summary(df, top_n=args.top)

    # Generate plots
    if args.plot:
        print("\nGenerating plots...")
        output_dir = Path(args.output_dir) if args.output_dir else None
        plot_hyperparameter_impact(df, output_dir)

    # Save detailed CSV
    output_dir = Path("results/grid_search")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'detailed_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to: {csv_path}")


if __name__ == '__main__':
    main()
