#!/usr/bin/env python
"""
Model Comparison Tool
=====================
Compare performance of all trained models from JSON summaries.

Usage:
    # Compare all models
    python compare_models.py

    # Compare specific models
    python compare_models.py --models braingt braingnn

    # Generate HTML report
    python compare_models.py --html

    # Compare specific folds
    python compare_models.py --fold graphs_outer1_inner1
"""

import argparse
import json
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional


def load_model_summary(model_name: str, results_dir: Optional[str] = None) -> Optional[Dict]:
    """Load aggregate summary for a model."""
    if results_dir is None:
        results_dir = f"results_{model_name}_advanced"

    summary_path = Path(results_dir) / f"{model_name}_aggregate_summary.json"

    if not summary_path.exists():
        print(f"Warning: Summary not found at {summary_path}")
        return None

    with open(summary_path) as f:
        return json.load(f)


def load_fold_summary(model_name: str, fold_name: str, results_dir: Optional[str] = None) -> Optional[Dict]:
    """Load summary for a specific fold."""
    if results_dir is None:
        results_dir = f"results_{model_name}_advanced"

    summary_path = Path(results_dir) / fold_name / f"{model_name}_summary.json"

    if not summary_path.exists():
        return None

    with open(summary_path) as f:
        return json.load(f)


def create_comparison_table(models: List[str]) -> pd.DataFrame:
    """Create comparison table from model summaries."""

    data = []

    for model_name in models:
        summary = load_model_summary(model_name)

        if summary is None:
            continue

        # Extract key metrics
        subj_agg = summary.get('subject_level_aggregate')
        win_agg = summary['window_level_aggregate']

        row = {
            'Model': model_name.upper(),
            'Parameters (M)': summary['model_info']['mean_parameters'] / 1e6,
            'Window MSE': win_agg['mse']['mean'],
            'Window MSE (std)': win_agg['mse']['std'],
            'Window r': win_agg['r']['mean'],
            'Window r (std)': win_agg['r']['std'],
        }

        if subj_agg:
            row.update({
                'Subject MSE': subj_agg['mse']['mean'],
                'Subject MSE (std)': subj_agg['mse']['std'],
                'Subject r': subj_agg['r']['mean'],
                'Subject r (std)': subj_agg['r']['std'],
            })

        # Training info
        row.update({
            'Mean Epochs': summary['training_statistics']['mean_best_epoch'],
            'Early Stopped': summary['training_statistics']['n_early_stopped'],
            'N Folds': summary['n_folds'],
        })

        data.append(row)

    return pd.DataFrame(data)


def print_comparison(df: pd.DataFrame):
    """Print formatted comparison table."""
    print("\n" + "="*100)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*100 + "\n")

    # Format for display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)

    print("Subject-Level Performance (Most Important):")
    print("-" * 100)
    if 'Subject r' in df.columns:
        cols = ['Model', 'Subject r', 'Subject r (std)', 'Subject MSE', 'Subject MSE (std)', 'Parameters (M)']
        print(df[cols].to_string(index=False))
    else:
        print("Subject-level metrics not available")

    print("\n\nWindow-Level Performance:")
    print("-" * 100)
    cols = ['Model', 'Window r', 'Window r (std)', 'Window MSE', 'Window MSE (std)']
    print(df[cols].to_string(index=False))

    print("\n\nTraining Summary:")
    print("-" * 100)
    cols = ['Model', 'Mean Epochs', 'Early Stopped', 'N Folds']
    print(df[cols].to_string(index=False))

    # Highlight best model
    print("\n" + "="*100)
    if 'Subject r' in df.columns:
        best_idx = df['Subject r'].idxmax()
        best_model = df.loc[best_idx, 'Model']
        best_r = df.loc[best_idx, 'Subject r']
        print(f"🏆 BEST MODEL: {best_model} (Subject r = {best_r:.4f})")
    else:
        best_idx = df['Window r'].idxmax()
        best_model = df.loc[best_idx, 'Model']
        best_r = df.loc[best_idx, 'Window r']
        print(f"🏆 BEST MODEL: {best_model} (Window r = {best_r:.4f})")
    print("="*100 + "\n")


def create_fold_comparison(models: List[str], fold_name: str) -> pd.DataFrame:
    """Compare models on a specific fold."""
    data = []

    for model_name in models:
        summary = load_fold_summary(model_name, fold_name)

        if summary is None:
            continue

        test_metrics = summary['test_metrics']

        row = {
            'Model': model_name.upper(),
            'Fold': fold_name,
            'Window MSE': test_metrics['window_level']['mse'],
            'Window r': test_metrics['window_level']['r'],
        }

        if test_metrics['subject_level']:
            row.update({
                'Subject MSE': test_metrics['subject_level']['mse'],
                'Subject r': test_metrics['subject_level']['r'],
            })

        row.update({
            'Best Epoch': summary['training_summary']['best_epoch'],
            'Total Epochs': summary['training_summary']['total_epochs'],
        })

        data.append(row)

    return pd.DataFrame(data)


def generate_html_report(models: List[str], output_path: str = "model_comparison.html"):
    """Generate HTML report with comparison."""
    df = create_comparison_table(models)

    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Comparison Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }}
            h1 {{
                color: #333;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #555;
                margin-top: 30px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                background-color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin: 20px 0;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
                padding: 12px;
                text-align: left;
            }}
            td {{
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .best {{
                background-color: #c8e6c9;
                font-weight: bold;
            }}
            .metric {{
                font-family: 'Courier New', monospace;
            }}
            .timestamp {{
                color: #888;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <h1>🧠 Brain GNN Model Comparison Report</h1>
        <p class="timestamp">Generated: {timestamp}</p>

        <h2>📊 Subject-Level Performance</h2>
        {subject_table}

        <h2>📈 Window-Level Performance</h2>
        {window_table}

        <h2>⚙️ Training Summary</h2>
        {training_table}

        <h2>🏆 Recommendation</h2>
        <p style="font-size: 1.2em; padding: 20px; background-color: #e8f5e9; border-left: 4px solid #4CAF50;">
            <strong>Best Model:</strong> {best_model} (Subject r = {best_r:.4f})
        </p>
    </body>
    </html>
    """

    import datetime

    # Create HTML tables
    if 'Subject r' in df.columns:
        subject_cols = ['Model', 'Subject r', 'Subject r (std)', 'Subject MSE', 'Subject MSE (std)']
        subject_table = df[subject_cols].to_html(index=False, classes='dataframe')
        best_idx = df['Subject r'].idxmax()
    else:
        subject_table = "<p>Subject-level metrics not available</p>"
        best_idx = df['Window r'].idxmax()

    window_cols = ['Model', 'Window r', 'Window r (std)', 'Window MSE', 'Window MSE (std)']
    window_table = df[window_cols].to_html(index=False, classes='dataframe')

    training_cols = ['Model', 'Mean Epochs', 'Early Stopped', 'N Folds', 'Parameters (M)']
    training_table = df[training_cols].to_html(index=False, classes='dataframe')

    best_model = df.loc[best_idx, 'Model']
    best_r = df.loc[best_idx, 'Subject r'] if 'Subject r' in df.columns else df.loc[best_idx, 'Window r']

    html = html_template.format(
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        subject_table=subject_table,
        window_table=window_table,
        training_table=training_table,
        best_model=best_model,
        best_r=best_r,
    )

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"HTML report saved to: {output_path}")


def save_comparison_json(models: List[str], output_path: str = "model_comparison.json"):
    """Save comparison as JSON."""
    comparison = {}

    for model_name in models:
        summary = load_model_summary(model_name)
        if summary:
            comparison[model_name] = {
                'subject_level': summary.get('subject_level_aggregate'),
                'window_level': summary['window_level_aggregate'],
                'training_stats': summary['training_statistics'],
                'model_info': summary['model_info'],
            }

    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"Comparison JSON saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare trained models")

    parser.add_argument('--models', nargs='+', default=['braingt', 'braingnn', 'fbnetgen'],
                        help='Models to compare')
    parser.add_argument('--fold', type=str, default=None,
                        help='Compare specific fold')
    parser.add_argument('--html', action='store_true',
                        help='Generate HTML report')
    parser.add_argument('--csv', type=str, default=None,
                        help='Save comparison as CSV')
    parser.add_argument('--json', type=str, default=None,
                        help='Save comparison as JSON')

    args = parser.parse_args()

    # Compare specific fold
    if args.fold:
        df = create_fold_comparison(args.models, args.fold)
        print(f"\n=== Fold Comparison: {args.fold} ===\n")
        print(df.to_string(index=False))

    # Compare aggregate results
    else:
        df = create_comparison_table(args.models)

        if df.empty:
            print("No model summaries found. Train models first:")
            print("  python train_advanced_models.py --model braingt")
            return

        # Print comparison
        print_comparison(df)

        # Save CSV
        if args.csv:
            df.to_csv(args.csv, index=False)
            print(f"Comparison saved to: {args.csv}")

        # Generate HTML
        if args.html:
            generate_html_report(args.models)

        # Save JSON
        if args.json:
            save_comparison_json(args.models, args.json)


if __name__ == "__main__":
    main()
