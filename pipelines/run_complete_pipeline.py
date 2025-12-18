#!/usr/bin/env python
"""
Complete Training Pipeline - One Click Solution
===============================================
Runs the entire advanced GNN training pipeline automatically.

This script:
1. Trains BrainGT (Graph Transformer)
2. Trains BrainGNN (ROI-aware)
3. Trains FBNetGen (Task-aware)
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


def generate_comparison_report(output_dir, project_root):
    """Generate a comparison report of all models."""
    print(f"\n{'='*70}")
    print("GENERATING COMPARISON REPORT")
    print(f"{'='*70}\n")

    models = ['braingt', 'braingnn', 'fbnetgen']
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
        cmd = [
            sys.executable, 'training/advanced/train_ensemble.py',
            '--ensemble_type', 'weighted',
            '--optimize_epochs', '100',
            '--device', args.device,
        ] + fold_arg

        success = run_command(cmd, "Creating Weighted Ensemble", cwd=project_root)
        success_log.append(('Ensemble', success))

    # Generate comparison report
    generate_comparison_report(output_path, project_root)

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
