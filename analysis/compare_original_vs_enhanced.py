#!/usr/bin/env python
"""
Original vs Enhanced Model Comparison Tool
=========================================
Compare performance between original and enhanced model versions.

Usage:
    # Compare all models
    python compare_original_vs_enhanced.py

    # Compare specific model
    python compare_original_vs_enhanced.py --model braingt

    # Generate HTML report
    python compare_original_vs_enhanced.py --html

    # Export to CSV
    python compare_original_vs_enhanced.py --csv comparison.csv
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def load_summary(model_name: str, enhanced: bool = False) -> Optional[Dict]:
    """Load aggregate summary for a model."""
    suffix = "_enhanced" if enhanced else "_advanced"
    results_dir = Path(f"results_{model_name}{suffix}")
    summary_path = results_dir / f"{model_name}_aggregate_summary.json"

    if not summary_path.exists():
        print(f"Warning: Summary not found at {summary_path}")
        return None

    with open(summary_path) as f:
        return json.load(f)


def create_comparison_dataframe(models: List[str]) -> pd.DataFrame:
    """Create comparison DataFrame."""
    data = []

    for model_name in models:
        # Load original
        original_summary = load_summary(model_name, enhanced=False)

        # Load enhanced
        enhanced_summary = load_summary(model_name, enhanced=True)

        if original_summary is None and enhanced_summary is None:
            continue

        # Extract metrics
        for version, summary in [('Original', original_summary), ('Enhanced', enhanced_summary)]:
            if summary is None:
                continue

            subj_agg = summary.get('subject_level_aggregate')
            win_agg = summary['window_level_aggregate']

            row = {
                'Model': model_name.upper(),
                'Version': version,
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

            row.update({
                'Mean Epochs': summary['training_statistics']['mean_best_epoch'],
                'N Folds': summary['n_folds'],
            })

            data.append(row)

    return pd.DataFrame(data)


def compute_improvements(df: pd.DataFrame) -> pd.DataFrame:
    """Compute improvement percentages."""
    improvements = []

    for model in df['Model'].unique():
        model_df = df[df['Model'] == model]

        if len(model_df) != 2:
            continue

        original = model_df[model_df['Version'] == 'Original']
        enhanced = model_df[model_df['Version'] == 'Enhanced']

        if original.empty or enhanced.empty:
            continue

        # Compute improvements
        improvement_row = {
            'Model': model,
        }

        # For MSE, lower is better (negative improvement = better)
        if 'Subject MSE' in original.columns:
            orig_mse = original['Subject MSE'].values[0]
            enh_mse = enhanced['Subject MSE'].values[0]
            improvement_row['Subject MSE Δ%'] = ((enh_mse - orig_mse) / orig_mse) * 100

        # For r, higher is better (positive improvement = better)
        if 'Subject r' in original.columns:
            orig_r = original['Subject r'].values[0]
            enh_r = enhanced['Subject r'].values[0]
            improvement_row['Subject r Δ%'] = ((enh_r - orig_r) / orig_r) * 100
            improvement_row['Subject r Abs Δ'] = enh_r - orig_r

        # Window level
        orig_win_r = original['Window r'].values[0]
        enh_win_r = enhanced['Window r'].values[0]
        improvement_row['Window r Δ%'] = ((enh_win_r - orig_win_r) / orig_win_r) * 100

        improvements.append(improvement_row)

    return pd.DataFrame(improvements)


def print_comparison(df: pd.DataFrame):
    """Print formatted comparison."""
    print("\n" + "="*100)
    print("ORIGINAL vs ENHANCED MODEL COMPARISON")
    print("="*100 + "\n")

    # Format options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)

    # Subject-level comparison
    if 'Subject r' in df.columns:
        print("Subject-Level Performance (Most Important):")
        print("-" * 100)
        cols = ['Model', 'Version', 'Subject r', 'Subject r (std)', 'Subject MSE', 'Subject MSE (std)']
        print(df[cols].to_string(index=False))
    else:
        print("Subject-level metrics not available")

    # Window-level comparison
    print("\n\nWindow-Level Performance:")
    print("-" * 100)
    cols = ['Model', 'Version', 'Window r', 'Window r (std)', 'Window MSE', 'Window MSE (std)']
    print(df[cols].to_string(index=False))

    # Training info
    print("\n\nTraining Info:")
    print("-" * 100)
    cols = ['Model', 'Version', 'Parameters (M)', 'Mean Epochs', 'N Folds']
    print(df[cols].to_string(index=False))

    # Improvements
    print("\n\n" + "="*100)
    print("IMPROVEMENT SUMMARY")
    print("="*100 + "\n")

    improvements_df = compute_improvements(df)

    if not improvements_df.empty:
        print(improvements_df.to_string(index=False))

        # Highlight best improvements
        if 'Subject r Δ%' in improvements_df.columns:
            best_model = improvements_df.loc[improvements_df['Subject r Δ%'].idxmax(), 'Model']
            best_improvement = improvements_df.loc[improvements_df['Subject r Δ%'].idxmax(), 'Subject r Δ%']
            print(f"\n🏆 BEST IMPROVEMENT: {best_model} (+{best_improvement:.2f}% on Subject r)")
    else:
        print("Could not compute improvements - missing data")

    print("\n" + "="*100 + "\n")


def generate_html_report(df: pd.DataFrame, output_path: str = "original_vs_enhanced_comparison.html"):
    """Generate HTML comparison report."""
    improvements_df = compute_improvements(df)

    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Original vs Enhanced Model Comparison</title>
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
            .enhanced {{
                background-color: #e8f5e9;
                font-weight: bold;
            }}
            .original {{
                background-color: #fff3e0;
            }}
            .improvement {{
                background-color: #c8e6c9;
            }}
            .degradation {{
                background-color: #ffccbc;
            }}
            .timestamp {{
                color: #888;
                font-size: 0.9em;
            }}
            .highlight {{
                font-size: 1.2em;
                padding: 20px;
                background-color: #e8f5e9;
                border-left: 4px solid #4CAF50;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <h1>🧠 Original vs Enhanced Model Comparison</h1>
        <p class="timestamp">Generated: {timestamp}</p>

        <h2>📊 Subject-Level Performance</h2>
        {subject_table}

        <h2>📈 Window-Level Performance</h2>
        {window_table}

        <h2>⚙️ Training Info</h2>
        {training_table}

        <h2>🚀 Improvement Summary</h2>
        {improvements_table}

        <div class="highlight">
            <strong>Key Findings:</strong><br>
            {key_findings}
        </div>
    </body>
    </html>
    """

    import datetime

    # Create tables
    if 'Subject r' in df.columns:
        subject_cols = ['Model', 'Version', 'Subject r', 'Subject r (std)', 'Subject MSE', 'Subject MSE (std)']
        subject_table = df[subject_cols].to_html(index=False, classes='dataframe')
    else:
        subject_table = "<p>Subject-level metrics not available</p>"

    window_cols = ['Model', 'Version', 'Window r', 'Window r (std)', 'Window MSE', 'Window MSE (std)']
    window_table = df[window_cols].to_html(index=False, classes='dataframe')

    training_cols = ['Model', 'Version', 'Parameters (M)', 'Mean Epochs', 'N Folds']
    training_table = df[training_cols].to_html(index=False, classes='dataframe')

    improvements_table = improvements_df.to_html(index=False, classes='dataframe') if not improvements_df.empty else "<p>No improvements data</p>"

    # Key findings
    key_findings = []
    if not improvements_df.empty and 'Subject r Δ%' in improvements_df.columns:
        best_model = improvements_df.loc[improvements_df['Subject r Δ%'].idxmax(), 'Model']
        best_improvement = improvements_df.loc[improvements_df['Subject r Δ%'].idxmax(), 'Subject r Δ%']
        key_findings.append(f"🏆 Best improvement: {best_model} (+{best_improvement:.2f}% on Subject r)")

        avg_improvement = improvements_df['Subject r Δ%'].mean()
        key_findings.append(f"📊 Average improvement across models: {avg_improvement:+.2f}%")

    key_findings_html = "<br>".join(key_findings) if key_findings else "No improvements detected"

    html = html_template.format(
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        subject_table=subject_table,
        window_table=window_table,
        training_table=training_table,
        improvements_table=improvements_table,
        key_findings=key_findings_html,
    )

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"HTML report saved to: {output_path}")


def print_detailed_comparison(model_name: str):
    """Print detailed comparison for a specific model."""
    original = load_summary(model_name, enhanced=False)
    enhanced = load_summary(model_name, enhanced=True)

    if original is None or enhanced is None:
        print(f"Missing data for {model_name}")
        return

    print(f"\n{'='*70}")
    print(f"DETAILED COMPARISON: {model_name.upper()}")
    print(f"{'='*70}\n")

    # Subject-level metrics
    if original.get('subject_level_aggregate') and enhanced.get('subject_level_aggregate'):
        orig_subj = original['subject_level_aggregate']
        enh_subj = enhanced['subject_level_aggregate']

        print("Subject-Level Metrics:")
        print("-" * 70)

        for metric in ['mse', 'r', 'r2']:
            if metric in orig_subj:
                orig_val = orig_subj[metric]['mean']
                enh_val = enh_subj[metric]['mean']
                diff = enh_val - orig_val
                pct = (diff / orig_val) * 100 if orig_val != 0 else 0

                symbol = "📈" if (metric == 'r' and diff > 0) or (metric == 'mse' and diff < 0) else "📉"

                print(f"  {metric.upper()}:")
                print(f"    Original: {orig_val:.4f} ± {orig_subj[metric]['std']:.4f}")
                print(f"    Enhanced: {enh_val:.4f} ± {enh_subj[metric]['std']:.4f}")
                print(f"    Change: {diff:+.4f} ({pct:+.2f}%) {symbol}")

    # Per-fold comparison
    print("\n\nPer-Fold Comparison:")
    print("-" * 70)

    orig_folds = {f['fold']: f for f in original['per_fold_summary']}
    enh_folds = {f['fold']: f for f in enhanced['per_fold_summary']}

    common_folds = set(orig_folds.keys()) & set(enh_folds.keys())

    improvements = []
    for fold in sorted(common_folds):
        orig_r = orig_folds[fold]['test_subj_r']
        enh_r = enh_folds[fold]['test_subj_r']

        if orig_r is not None and enh_r is not None:
            improvement = enh_r - orig_r
            improvements.append(improvement)
            symbol = "✓" if improvement > 0 else "✗"
            print(f"  {fold}: {orig_r:.4f} → {enh_r:.4f} ({improvement:+.4f}) {symbol}")

    if improvements:
        print(f"\n  Average fold improvement: {np.mean(improvements):+.4f}")
        print(f"  Improved folds: {sum(1 for i in improvements if i > 0)}/{len(improvements)}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Compare Original vs Enhanced Models")

    parser.add_argument('--models', nargs='+', default=['braingt', 'braingnn', 'fbnetgen'],
                        help='Models to compare')
    parser.add_argument('--model', type=str, default=None,
                        help='Detailed comparison for specific model')
    parser.add_argument('--html', action='store_true',
                        help='Generate HTML report')
    parser.add_argument('--csv', type=str, default=None,
                        help='Save comparison as CSV')

    args = parser.parse_args()

    # Detailed comparison for single model
    if args.model:
        print_detailed_comparison(args.model)
        return

    # Overall comparison
    df = create_comparison_dataframe(args.models)

    if df.empty:
        print("No data found. Please train models first:")
        print("  Original: python train_advanced_models.py --model braingt")
        print("  Enhanced: python train_enhanced_models.py --model braingt")
        return

    # Print comparison
    print_comparison(df)

    # Save CSV
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Comparison saved to: {args.csv}")

    # Generate HTML
    if args.html:
        generate_html_report(df)


if __name__ == "__main__":
    main()
