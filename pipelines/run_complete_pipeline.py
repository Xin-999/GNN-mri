#!/usr/bin/env python
"""
Complete Training Pipeline - One Click Solution
===============================================
Runs the entire advanced GNN training pipeline automatically.

This script:
1. Trains BrainGT (Graph Transformer) (Parameters: 1,216,770)
2. Trains BrainGNN (ROI-aware) (Parameters: 418,302)
3. Trains FBNetGen (Task-aware) (Parameters: 176,514)
4. Creates weighted ensemble
5. Generates comparison report

Usage:
    python run_complete_pipeline.py --epochs 100
    python run_complete_pipeline.py --quick_test  # Fast test run

    # Train specific models only
    python run_complete_pipeline.py --models braingnn fbnetgen --epochs 100
    python run_complete_pipeline.py --models fbnetgen --quick_test

    # BASE vs ENHANCED
    python run_complete_pipeline.py --use_base --epochs 100 --device cuda
    python run_complete_pipeline.py --use_enhanced --epochs 100 --device cuda

    # COMPARE BASE and ENHANCED (runs both automatically!)
    python run_complete_pipeline.py --compare_base_enhanced --quick_test
    python run_complete_pipeline.py --compare_base_enhanced --models braingnn fbnetgen --epochs 100

"""

import argparse
import subprocess
import sys
import os
import json
from pathlib import Path
import time


def run_command(cmd, description, cwd=None):
    """Run a shell command and handle errors."""
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print(f"{'='*70}\n")
    print(f"Command: {' '.join(cmd)}\n")
    if cwd:
        print(f"Working directory: {cwd}\n")

    start_time = time.time()

    # Set up environment with PYTHONPATH
    env = os.environ.copy()
    if cwd:
        # Add project root to PYTHONPATH so Python can find modules
        env['PYTHONPATH'] = str(cwd)
        print(f"PYTHONPATH: {cwd}\n")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True,
            cwd=cwd,
            env=env,
        )
        elapsed = time.time() - start_time
        print(f"\n✓ {description} completed in {elapsed/60:.1f} minutes")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error:")
        print(e)
        return False


def generate_comparison_report(output_dir, project_root, selected_models=None):
    """Generate a comparison report of selected models."""
    print(f"\n{'='*70}")
    print("GENERATING COMPARISON REPORT")
    print(f"{'='*70}\n")

    models = selected_models if selected_models else ['braingt', 'braingnn', 'fbnetgen']
    results = {}

    for model in models:
        # Check enhanced path (primary)
        result_file = project_root / f"results/enhanced/{model}_enhanced/{model}_aggregate_summary.json"
        if not result_file.exists():
            # Fallback to advanced path
            result_file = project_root / f"results/advanced/{model}/{model}_aggregate_summary.json"
        if not result_file.exists():
            # Fallback to old path
            result_file = project_root / f"results_{model}_advanced/aggregate_results.json"

        if result_file.exists():
            with open(result_file) as f:
                results[model] = json.load(f)

    # Load ensemble results
    ensemble_file = project_root / "results/ensemble/aggregate_ensemble_results.json"
    if not ensemble_file.exists():
        # Fallback to old path
        ensemble_file = project_root / "results_ensemble/aggregate_ensemble_results.json"

    if ensemble_file.exists():
        with open(ensemble_file) as f:
            results['ensemble'] = json.load(f)

    # Generate report
    report_path = Path(output_dir) / "comparison_report.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("MODEL COMPARISON REPORT\n")
        f.write("="*70 + "\n\n")

        f.write("Performance Summary (Subject-Level):\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Model':<20} {'Mean r':<15} {'Std r':<15} {'Mean MSE':<15}\n")
        f.write("-" * 70 + "\n")

        model_scores = []
        for name, data in results.items():
            # Handle new JSON structure
            if 'subject_level_aggregate' in data and data['subject_level_aggregate']:
                r_mean = data['subject_level_aggregate']['r']['mean']
                r_std = data['subject_level_aggregate']['r']['std']
                mse_mean = data['subject_level_aggregate']['mse']['mean']
                f.write(f"{name.upper():<20} {r_mean:<15.4f} {r_std:<15.4f} {mse_mean:<15.4f}\n")
                model_scores.append((name, r_mean))
            # Fallback to old structure
            elif 'mean_test_r' in data:
                r_mean = data['mean_test_r']
                r_std = data.get('std_test_r', 0)
                mse_mean = data['mean_test_mse']
                f.write(f"{name.upper():<20} {r_mean:<15.4f} {r_std:<15.4f} {mse_mean:<15.4f}\n")
                model_scores.append((name, r_mean))

        f.write("\n" + "="*70 + "\n")
        f.write("\nRecommendation:\n")

        if 'ensemble' in results:
            f.write("✓ Use ENSEMBLE for best accuracy\n")
        elif model_scores:
            # Find best single model
            best_model = max(model_scores, key=lambda x: x[1])
            f.write(f"✓ Best single model: {best_model[0].upper()} (r={best_model[1]:.4f})\n")
        else:
            f.write("⚠️  No model results found\n")

    print(f"Report saved to: {report_path}")

    # Print to console
    with open(report_path, encoding='utf-8') as f:
        print(f.read())


def run_pipeline_version(args, project_root, epochs, fold_arg, use_enhanced, models):
    """
    Run complete pipeline for either base or enhanced models.

    Args:
        args: Command-line arguments
        project_root: Project root path
        epochs: Number of epochs
        fold_arg: Fold argument list
        use_enhanced: True for enhanced, False for base
        models: List of models to train

    Returns:
        success_log: List of (model_name, success) tuples
    """
    success_log = []

    # Determine which training script and labels to use
    if use_enhanced:
        train_script = 'training/advanced/train_enhanced_models.py'
        model_suffix = ' Enhanced'
        results_dir = 'enhanced'
    else:
        train_script = 'training/advanced/train_advanced_models.py'
        model_suffix = ''
        results_dir = 'advanced'

    # Train selected models
    if not args.skip_training:
        # Step 1: Train BrainGT (if selected)
        if 'braingt' in models:
            cmd = [
                sys.executable, train_script,
                '--model', 'braingt',
                '--epochs', str(epochs),
                '--hidden_dim', '128',
                '--n_heads', '8',
                '--lr', '1e-4',
                '--dropout', '0.2',
                '--device', args.device,
            ] + fold_arg

            success = run_command(cmd, f"Training BrainGT{model_suffix} (Graph Transformer)", cwd=project_root)
            success_log.append((f'BrainGT{model_suffix}', success))

        # Step 2: Train BrainGNN (if selected)
        if 'braingnn' in models:
            cmd = [
                sys.executable, train_script,
                '--model', 'braingnn',
                '--epochs', str(epochs),
                '--hidden_dim', '128',
                '--n_layers', '3',
                '--dropout', '0.3',
                '--device', args.device,
            ] + fold_arg

            success = run_command(cmd, f"Training BrainGNN{model_suffix} (ROI-Aware)", cwd=project_root)
            success_log.append((f'BrainGNN{model_suffix}', success))

        # Step 3: Train FBNetGen (if selected)
        if 'fbnetgen' in models:
            cmd = [
                sys.executable, train_script,
                '--model', 'fbnetgen',
                '--epochs', str(epochs),
                '--hidden_dim', '128',
                '--n_layers', '3',
                '--n_heads', '4',
                '--dropout', '0.3',
                '--device', args.device,
            ] + fold_arg

            success = run_command(cmd, f"Training FBNetGen{model_suffix} (Task-Aware)", cwd=project_root)
            success_log.append((f'FBNetGen{model_suffix}', success))

    # Step 4: Create Ensemble (skip in compare mode, will do separately)
    if not args.skip_ensemble and not args.compare_base_enhanced:
        if use_enhanced:
            ensemble_script = 'training/advanced/train_ensemble.py'
        else:
            ensemble_script = 'training/advanced/train_ensemble.py'  # Same script

        cmd = [
            sys.executable, ensemble_script,
            '--ensemble_type', 'weighted',
            '--optimize_epochs', '100',
            '--device', args.device,
        ] + fold_arg

        success = run_command(cmd, f"Creating Weighted Ensemble ({model_suffix.strip() or 'Base'})", cwd=project_root)
        success_log.append((f'Ensemble{model_suffix}', success))

    return success_log


def generate_base_vs_enhanced_comparison(project_root, models, output_path, fold_name=None):
    """
    Generate comparison report between base and enhanced models.

    Args:
        project_root: Project root path
        models: List of models that were trained
        output_path: Output directory path
        fold_name: Specific fold name for quick test mode
    """
    print("\n" + "="*70)
    print("GENERATING BASE vs ENHANCED COMPARISON REPORT")
    print("="*70 + "\n")

    comparison_data = []
    per_fold_data = {}  # Store per-fold results for full CV

    for model_name in models:
        model_upper = model_name.upper()

        base_metrics = {}
        enhanced_metrics = {}
        base_metrics_std = {}
        enhanced_metrics_std = {}

        # === Load Base Results ===
        if fold_name:
            # Quick test mode - single fold
            base_paths = [
                project_root / f'results/advanced/{model_name}/{fold_name}/{model_name}_summary.json',
                project_root / f'results/advanced/{model_name}/{fold_name}.pkl/{model_name}_summary.json',
                Path(f'results/advanced/{model_name}/{fold_name}/{model_name}_summary.json'),
                Path(f'../../results/advanced/{model_name}/{fold_name}/{model_name}_summary.json'),
            ]

            base_file = None
            for path in base_paths:
                print(f"Checking base path: {path}")
                if path.exists():
                    base_file = path
                    print(f"  ✓ Found!")
                    break

            if base_file:
                with open(base_file) as f:
                    base_summary = json.load(f)

                    # Extract metrics from various possible structures
                    if 'subject_metrics' in base_summary:
                        base_metrics = base_summary['subject_metrics']
                    elif 'subject_level' in base_summary:
                        base_metrics = base_summary['subject_level']
                    elif 'test_metrics' in base_summary and isinstance(base_summary['test_metrics'], dict):
                        if 'subject_level' in base_summary['test_metrics']:
                            base_metrics = base_summary['test_metrics']['subject_level']

                print(f"  Base metrics: r={base_metrics.get('r', 'N/A')}, mse={base_metrics.get('mse', 'N/A')}")
            else:
                print(f"  ✗ Base file not found!")

        else:
            # Full CV mode - aggregate across all folds
            print(f"Loading all folds for {model_name} base model...")
            base_model_dir = project_root / f'results/advanced/{model_name}'

            if base_model_dir.exists():
                fold_metrics = {'r': [], 'mse': [], 'mae': []}
                fold_details = []

                # Find all fold directories
                for fold_dir in sorted(base_model_dir.iterdir()):
                    if fold_dir.is_dir():
                        summary_paths = [
                            fold_dir / f'{model_name}_summary.json',
                            fold_dir / f'{model_name}_test_summary.json',
                        ]

                        for summary_path in summary_paths:
                            if summary_path.exists():
                                with open(summary_path) as f:
                                    fold_summary = json.load(f)

                                    # Extract metrics
                                    fold_base_metrics = None
                                    if 'subject_metrics' in fold_summary:
                                        fold_base_metrics = fold_summary['subject_metrics']
                                    elif 'subject_level' in fold_summary:
                                        fold_base_metrics = fold_summary['subject_level']
                                    elif 'test_metrics' in fold_summary and isinstance(fold_summary['test_metrics'], dict):
                                        if 'subject_level' in fold_summary['test_metrics']:
                                            fold_base_metrics = fold_summary['test_metrics']['subject_level']

                                    if fold_base_metrics and 'r' in fold_base_metrics:
                                        fold_metrics['r'].append(fold_base_metrics.get('r', 0))
                                        fold_metrics['mse'].append(fold_base_metrics.get('mse', float('inf')))
                                        fold_metrics['mae'].append(fold_base_metrics.get('mae', float('inf')))
                                        fold_details.append({
                                            'fold': fold_dir.name,
                                            'r': fold_base_metrics.get('r', 0),
                                            'mse': fold_base_metrics.get('mse', float('inf')),
                                        })
                                break

                # Compute aggregates
                if fold_metrics['r']:
                    import numpy as np
                    base_metrics = {
                        'r': np.mean(fold_metrics['r']),
                        'mse': np.mean(fold_metrics['mse']),
                        'mae': np.mean(fold_metrics['mae']),
                    }
                    base_metrics_std = {
                        'r': np.std(fold_metrics['r']),
                        'mse': np.std(fold_metrics['mse']),
                        'mae': np.std(fold_metrics['mae']),
                    }
                    per_fold_data[f'{model_name}_base'] = fold_details
                    print(f"  Loaded {len(fold_metrics['r'])} folds")
                    print(f"  Base mean r: {base_metrics['r']:.4f} ± {base_metrics_std['r']:.4f}")
                else:
                    print(f"  ✗ No valid fold results found!")
            else:
                print(f"  ✗ Base model directory not found: {base_model_dir}")

        # === Load Enhanced Results ===
        enhanced_file = project_root / f'results/enhanced/{model_name}_enhanced/{model_name}_aggregate_summary.json'

        if enhanced_file.exists():
            with open(enhanced_file) as f:
                enhanced_summary = json.load(f)

                # Enhanced models have aggregate statistics built-in
                subject_agg = enhanced_summary.get('subject_level_aggregate', {})
                if subject_agg:
                    enhanced_metrics = {
                        'r': subject_agg.get('r', {}).get('mean', 0),
                        'mse': subject_agg.get('mse', {}).get('mean', float('inf')),
                        'mae': subject_agg.get('mae', {}).get('mean', float('inf')),
                    }
                    enhanced_metrics_std = {
                        'r': subject_agg.get('r', {}).get('std', 0),
                        'mse': subject_agg.get('mse', {}).get('std', 0),
                        'mae': subject_agg.get('mae', {}).get('std', 0),
                    }

                    # Get per-fold details if available
                    if 'per_fold_summary' in enhanced_summary:
                        fold_details = []
                        for fold_info in enhanced_summary['per_fold_summary']:
                            if 'subject_level' in fold_info:
                                fold_details.append({
                                    'fold': fold_info.get('fold', 'unknown'),
                                    'r': fold_info['subject_level'].get('r', 0),
                                    'mse': fold_info['subject_level'].get('mse', float('inf')),
                                })
                        per_fold_data[f'{model_name}_enhanced'] = fold_details

                    print(f"  Enhanced mean r: {enhanced_metrics['r']:.4f} ± {enhanced_metrics_std['r']:.4f}")
        else:
            print(f"  Enhanced file not found: {enhanced_file}")

        # === Store Comparison Data ===
        if base_metrics or enhanced_metrics:
            base_r = base_metrics.get('r', 0) if base_metrics else 0
            base_mse = base_metrics.get('mse', float('inf')) if base_metrics else float('inf')
            enhanced_r = enhanced_metrics.get('r', 0) if enhanced_metrics else 0
            enhanced_mse = enhanced_metrics.get('mse', float('inf')) if enhanced_metrics else float('inf')

            base_r_std = base_metrics_std.get('r', 0) if base_metrics_std else 0
            base_mse_std = base_metrics_std.get('mse', 0) if base_metrics_std else 0
            enhanced_r_std = enhanced_metrics_std.get('r', 0) if enhanced_metrics_std else 0
            enhanced_mse_std = enhanced_metrics_std.get('mse', 0) if enhanced_metrics_std else 0

            comparison_data.append({
                'model': model_upper,
                'base_r': base_r,
                'base_r_std': base_r_std,
                'enhanced_r': enhanced_r,
                'enhanced_r_std': enhanced_r_std,
                'base_mse': base_mse,
                'base_mse_std': base_mse_std,
                'enhanced_mse': enhanced_mse,
                'enhanced_mse_std': enhanced_mse_std,
                'improvement_r': enhanced_r - base_r,
                'improvement_mse': base_mse - enhanced_mse,
            })

    # Save comparison report
    report_path = output_path / 'base_vs_enhanced_comparison.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*90 + "\n")
        f.write("BASE vs ENHANCED MODELS COMPARISON\n")
        f.write("="*90 + "\n\n")

        # Add fold information
        if fold_name:
            f.write(f"Evaluation Mode: Quick Test (Single Fold)\n")
            f.write(f"Fold: {fold_name}\n")
            f.write(f"Note: Single fold evaluation - no standard deviation available\n")
        else:
            f.write(f"Evaluation Mode: Full Cross-Validation\n")
            f.write(f"Folds: 25 folds (5×5 nested cross-validation)\n")
            f.write(f"Format: Mean ± Std across all folds\n")
        f.write("\n" + "="*90 + "\n\n")

        # === Subject-Level Correlation Comparison ===
        f.write("Subject-Level Correlation (r) - Higher is Better:\n")
        f.write("-"*90 + "\n")

        if fold_name:
            # Single fold format
            f.write(f"{'Model':<15} {'Base r':<20} {'Enhanced r':<20} {'Improvement':<15}\n")
            f.write("-"*90 + "\n")
            for data in comparison_data:
                f.write(f"{data['model']:<15} "
                       f"{data['base_r']:<20.4f} "
                       f"{data['enhanced_r']:<20.4f} "
                       f"{data['improvement_r']:+.4f}\n")
        else:
            # Multi-fold format with Mean ± Std
            f.write(f"{'Model':<15} {'Base r':<25} {'Enhanced r':<25} {'Improvement':<15}\n")
            f.write("-"*90 + "\n")
            for data in comparison_data:
                base_str = f"{data['base_r']:.4f} ± {data['base_r_std']:.4f}"
                enhanced_str = f"{data['enhanced_r']:.4f} ± {data['enhanced_r_std']:.4f}"
                f.write(f"{data['model']:<15} "
                       f"{base_str:<25} "
                       f"{enhanced_str:<25} "
                       f"{data['improvement_r']:+.4f}\n")

        # === MSE Comparison ===
        f.write("\n" + "="*90 + "\n")
        f.write("\nSubject-Level MSE - Lower is Better:\n")
        f.write("-"*90 + "\n")

        if fold_name:
            # Single fold format
            f.write(f"{'Model':<15} {'Base MSE':<20} {'Enhanced MSE':<20} {'Reduction':<15}\n")
            f.write("-"*90 + "\n")
            for data in comparison_data:
                f.write(f"{data['model']:<15} "
                       f"{data['base_mse']:<20.4f} "
                       f"{data['enhanced_mse']:<20.4f} "
                       f"{data['improvement_mse']:+.4f}\n")
        else:
            # Multi-fold format with Mean ± Std
            f.write(f"{'Model':<15} {'Base MSE':<25} {'Enhanced MSE':<25} {'Reduction':<15}\n")
            f.write("-"*90 + "\n")
            for data in comparison_data:
                base_str = f"{data['base_mse']:.4f} ± {data['base_mse_std']:.4f}"
                enhanced_str = f"{data['enhanced_mse']:.4f} ± {data['enhanced_mse_std']:.4f}"
                f.write(f"{data['model']:<15} "
                       f"{base_str:<25} "
                       f"{enhanced_str:<25} "
                       f"{data['improvement_mse']:+.4f}\n")

        # === Per-Fold Breakdown ===
        if not fold_name and per_fold_data:
            f.write("\n" + "="*90 + "\n")
            f.write("\nPER-FOLD BREAKDOWN:\n")
            f.write("="*90 + "\n")

            for model_name in models:
                model_upper = model_name.upper()
                base_key = f'{model_name}_base'
                enhanced_key = f'{model_name}_enhanced'

                if base_key in per_fold_data or enhanced_key in per_fold_data:
                    f.write(f"\n{model_upper}:\n")
                    f.write("-"*90 + "\n")
                    f.write(f"{'Fold':<30} {'Base r':<15} {'Enhanced r':<15} {'Improvement':<15}\n")
                    f.write("-"*90 + "\n")

                    base_folds = {fold['fold']: fold for fold in per_fold_data.get(base_key, [])}
                    enhanced_folds = {fold['fold']: fold for fold in per_fold_data.get(enhanced_key, [])}

                    all_fold_names = sorted(set(base_folds.keys()) | set(enhanced_folds.keys()))

                    for fold in all_fold_names:
                        base_r = base_folds[fold]['r'] if fold in base_folds else 0.0
                        enhanced_r = enhanced_folds[fold]['r'] if fold in enhanced_folds else 0.0
                        improvement = enhanced_r - base_r

                        f.write(f"{fold:<30} "
                               f"{base_r:<15.4f} "
                               f"{enhanced_r:<15.4f} "
                               f"{improvement:+.4f}\n")

        # === Summary ===
        f.write("\n" + "="*90 + "\n")
        f.write("\nSUMMARY:\n")
        f.write("-"*90 + "\n")

        avg_improvement_r = sum(d['improvement_r'] for d in comparison_data) / len(comparison_data) if comparison_data else 0
        if avg_improvement_r > 0:
            f.write(f"✓ Enhanced models show average improvement of {avg_improvement_r:.4f} in correlation\n")
        else:
            f.write(f"✗ Enhanced models show average decrease of {abs(avg_improvement_r):.4f} in correlation\n")

        best_model = max(comparison_data, key=lambda x: x['enhanced_r']) if comparison_data else None
        if best_model:
            f.write(f"✓ Best enhanced model: {best_model['model']} (r={best_model['enhanced_r']:.4f})\n")

    print(f"Comparison report saved to: {report_path}\n")

    # Print to console
    with open(report_path, encoding='utf-8') as f:
        print(f.read())


def print_comparison_summary(all_success_logs):
    """
    Print final summary for comparison mode.

    Args:
        all_success_logs: Dict with 'base' and 'enhanced' success logs
    """
    print("\n" + "="*70)
    print("COMPARISON MODE COMPLETION SUMMARY")
    print("="*70 + "\n")

    for version, success_log in all_success_logs.items():
        print(f"{version.upper()} Models:")
        for model_name, success in success_log:
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"  {model_name:<30} {status}")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Run complete training pipeline")

    # Quick test mode
    parser.add_argument('--quick_test', action='store_true',
                        help='Fast test run (10 epochs, 1 fold)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])

    # Model selection
    parser.add_argument('--use_enhanced', action='store_true', default=True,
                        help='Use enhanced models (default: True)')
    parser.add_argument('--use_base', dest='use_enhanced', action='store_false',
                        help='Use base models instead of enhanced')
    parser.add_argument('--compare_base_enhanced', action='store_true',
                        help='Run BOTH base and enhanced models, then compare results')
    parser.add_argument('--models', nargs='+',
                        choices=['braingt', 'braingnn', 'fbnetgen'],
                        default=['braingt', 'braingnn', 'fbnetgen'],
                        help='Which models to train (default: all three)')

    # What to run
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip model training (only ensemble)')
    parser.add_argument('--skip_ensemble', action='store_true',
                        help='Skip ensemble creation')

    # Output
    parser.add_argument('--output_dir', type=str, default='results/complete_pipeline',
                        help='Output directory for final report')

    args = parser.parse_args()

    # Get project root for proper path resolution
    project_root = Path(__file__).parent.parent

    # Configuration
    if args.quick_test:
        print("\n*** QUICK TEST MODE - Using reduced settings ***\n")
        epochs = 2
        # Use a valid fold (not the corrupted graphs_outer1_inner1)
        fold_arg = ['--fold_name', 'graphs_outer1_inner2']
    else:
        epochs = args.epochs
        fold_arg = []

    # Create output directory (relative to project root)
    output_path = project_root / args.output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    # Handle compare_base_enhanced mode
    if args.compare_base_enhanced:
        print("\n" + "="*70)
        print("COMPARE MODE: Running BOTH Base and Enhanced Models")
        print("="*70)

        all_success_logs = {}

        # Run base models first
        print("\n" + "="*70)
        print("PHASE 1: Training BASE Models")
        print("="*70)
        success_log_base = run_pipeline_version(
            args, project_root, epochs, fold_arg,
            use_enhanced=False, models=args.models
        )
        all_success_logs['base'] = success_log_base

        # Run enhanced models
        print("\n" + "="*70)
        print("PHASE 2: Training ENHANCED Models")
        print("="*70)
        success_log_enhanced = run_pipeline_version(
            args, project_root, epochs, fold_arg,
            use_enhanced=True, models=args.models
        )
        all_success_logs['enhanced'] = success_log_enhanced

        # Generate comparison report
        fold_name = 'graphs_outer1_inner2' if args.quick_test else None
        generate_base_vs_enhanced_comparison(project_root, args.models, output_path, fold_name=fold_name)

        # Print final summary
        print_comparison_summary(all_success_logs)
        return

    success_log = []

    # Determine which training script and labels to use
    if args.use_enhanced:
        train_script = 'training/advanced/train_enhanced_models.py'
        model_suffix = ' Enhanced'
        results_dir = 'enhanced'
    else:
        train_script = 'training/advanced/train_advanced_models.py'
        model_suffix = ''
        results_dir = 'advanced'

    # Train selected models
    if not args.skip_training:
        # Step 1: Train BrainGT (if selected)
        if 'braingt' in args.models:
            cmd = [
                sys.executable, train_script,
                '--model', 'braingt',
                '--epochs', str(epochs),
                '--hidden_dim', '128',
                '--n_heads', '8',
                '--lr', '1e-4',
                '--dropout', '0.2',
                '--device', args.device,
            ] + fold_arg

            success = run_command(cmd, f"Training BrainGT{model_suffix} (Graph Transformer)", cwd=project_root)
            success_log.append((f'BrainGT{model_suffix}', success))

        # Step 2: Train BrainGNN (if selected)
        if 'braingnn' in args.models:
            cmd = [
                sys.executable, train_script,
                '--model', 'braingnn',
                '--epochs', str(epochs),
                '--hidden_dim', '128',
                '--n_layers', '3',
                '--dropout', '0.3',
                '--device', args.device,
            ] + fold_arg

            success = run_command(cmd, f"Training BrainGNN{model_suffix} (ROI-Aware)", cwd=project_root)
            success_log.append((f'BrainGNN{model_suffix}', success))

        # Step 3: Train FBNetGen (if selected)
        if 'fbnetgen' in args.models:
            cmd = [
                sys.executable, train_script,
                '--model', 'fbnetgen',
                '--epochs', str(epochs),
                '--hidden_dim', '128',
                '--n_layers', '3',
                '--n_heads', '4',
                '--dropout', '0.3',
                '--device', args.device,
            ] + fold_arg

            success = run_command(cmd, f"Training FBNetGen{model_suffix} (Task-Aware)", cwd=project_root)
            success_log.append((f'FBNetGen{model_suffix}', success))

    # Step 4: Create Ensemble
    if not args.skip_ensemble:
        # Set correct model directories based on base vs enhanced
        if args.use_enhanced:
            model_dirs = {
                'braingt': 'results/enhanced/braingt_enhanced',
                'braingnn': 'results/enhanced/braingnn_enhanced',
                'fbnetgen': 'results/enhanced/fbnetgen_enhanced',
            }
        else:
            model_dirs = {
                'braingt': 'results/advanced/braingt',
                'braingnn': 'results/advanced/braingnn',
                'fbnetgen': 'results/advanced/fbnetgen',
            }

        cmd = [
            sys.executable, 'training/advanced/train_ensemble.py',
            '--ensemble_type', 'weighted',
            '--optimize_epochs', '100',
            '--device', args.device,
            '--braingt_dir', model_dirs['braingt'],
            '--braingnn_dir', model_dirs['braingnn'],
            '--fbnetgen_dir', model_dirs['fbnetgen'],
        ] + fold_arg

        success = run_command(cmd, "Creating Weighted Ensemble", cwd=project_root)
        success_log.append(('Ensemble', success))

    # Generate comparison report
    generate_comparison_report(output_path, project_root, selected_models=args.models)

    # Final summary
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETION SUMMARY")
    print(f"{'='*70}\n")

    for step, success in success_log:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{step:<20} {status}")

    all_success = all(s for _, s in success_log)

    if all_success:
        print(f"\n🎉 All steps completed successfully!")
        print(f"\nResults saved in:")
        if args.use_enhanced:
            print(f"  - results/enhanced/braingt_enhanced/")
            print(f"  - results/enhanced/braingnn_enhanced/")
            print(f"  - results/enhanced/fbnetgen_enhanced/")
        else:
            print(f"  - results/advanced/braingt/")
            print(f"  - results/advanced/braingnn/")
            print(f"  - results/advanced/fbnetgen/")
        print(f"  - results/ensemble/")
        print(f"  - {args.output_dir}/comparison_report.txt")
    else:
        print(f"\n⚠️  Some steps failed. Check logs above.")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
