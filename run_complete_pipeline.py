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
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
import time


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print(f"{'='*70}\n")
    print(f"Command: {' '.join(cmd)}\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True,
        )
        elapsed = time.time() - start_time
        print(f"\n✓ {description} completed in {elapsed/60:.1f} minutes")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error:")
        print(e)
        return False


def generate_comparison_report(output_dir):
    """Generate a comparison report of all models."""
    print(f"\n{'='*70}")
    print("GENERATING COMPARISON REPORT")
    print(f"{'='*70}\n")

    models = ['braingt', 'braingnn', 'fbnetgen']
    results = {}

    for model in models:
        result_file = Path(f"results_{model}_advanced/aggregate_results.json")
        if result_file.exists():
            with open(result_file) as f:
                results[model] = json.load(f)

    # Load ensemble results
    ensemble_file = Path("results_ensemble/aggregate_ensemble_results.json")
    if ensemble_file.exists():
        with open(ensemble_file) as f:
            results['ensemble'] = json.load(f)

    # Generate report
    report_path = Path(output_dir) / "comparison_report.txt"

    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MODEL COMPARISON REPORT\n")
        f.write("="*70 + "\n\n")

        f.write("Performance Summary (Subject-Level):\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Model':<20} {'Mean r':<15} {'Std r':<15} {'Mean MSE':<15}\n")
        f.write("-" * 70 + "\n")

        for name, data in results.items():
            if 'mean_test_r' in data:
                r_mean = data['mean_test_r']
                r_std = data.get('std_test_r', 0)
                mse_mean = data['mean_test_mse']
                f.write(f"{name.upper():<20} {r_mean:<15.4f} {r_std:<15.4f} {mse_mean:<15.4f}\n")

        f.write("\n" + "="*70 + "\n")
        f.write("\nRecommendation:\n")

        if 'ensemble' in results:
            f.write("✓ Use ENSEMBLE for best accuracy\n")
        else:
            # Find best single model
            best_model = max(
                [(name, data['mean_test_r']) for name, data in results.items() if 'mean_test_r' in data],
                key=lambda x: x[1]
            )
            f.write(f"✓ Best single model: {best_model[0].upper()} (r={best_model[1]:.4f})\n")

    print(f"Report saved to: {report_path}")

    # Print to console
    with open(report_path) as f:
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

    # What to run
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip model training (only ensemble)')
    parser.add_argument('--skip_ensemble', action='store_true',
                        help='Skip ensemble creation')

    # Output
    parser.add_argument('--output_dir', type=str, default='complete_pipeline_results',
                        help='Output directory for final report')

    args = parser.parse_args()

    # Configuration
    if args.quick_test:
        print("\n*** QUICK TEST MODE - Using reduced settings ***\n")
        epochs = 10
        fold_arg = ['--fold_name', 'graphs_outer1_inner1']
    else:
        epochs = args.epochs
        fold_arg = []

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    success_log = []

    # Step 1: Train BrainGT
    if not args.skip_training:
        cmd = [
            sys.executable, 'train_advanced_models.py',
            '--model', 'braingt',
            '--epochs', str(epochs),
            '--hidden_dim', '128',
            '--n_heads', '8',
            '--lr', '1e-4',
            '--dropout', '0.2',
            '--device', args.device,
        ] + fold_arg

        success = run_command(cmd, "Training BrainGT (Graph Transformer)")
        success_log.append(('BrainGT', success))

        # Step 2: Train BrainGNN
        cmd = [
            sys.executable, 'train_advanced_models.py',
            '--model', 'braingnn',
            '--epochs', str(epochs),
            '--hidden_dim', '128',
            '--n_layers', '3',
            '--dropout', '0.3',
            '--device', args.device,
        ] + fold_arg

        success = run_command(cmd, "Training BrainGNN (ROI-Aware)")
        success_log.append(('BrainGNN', success))

        # Step 3: Train FBNetGen
        cmd = [
            sys.executable, 'train_advanced_models.py',
            '--model', 'fbnetgen',
            '--epochs', str(epochs),
            '--hidden_dim', '128',
            '--n_layers', '3',
            '--n_heads', '4',
            '--dropout', '0.3',
            '--device', args.device,
        ] + fold_arg

        success = run_command(cmd, "Training FBNetGen (Task-Aware)")
        success_log.append(('FBNetGen', success))

    # Step 4: Create Ensemble
    if not args.skip_ensemble:
        cmd = [
            sys.executable, 'train_ensemble.py',
            '--ensemble_type', 'weighted',
            '--optimize_epochs', '100',
            '--device', args.device,
        ] + fold_arg

        success = run_command(cmd, "Creating Weighted Ensemble")
        success_log.append(('Ensemble', success))

    # Generate comparison report
    generate_comparison_report(args.output_dir)

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
        print(f"  - results_braingt_advanced/")
        print(f"  - results_braingnn_advanced/")
        print(f"  - results_fbnetgen_advanced/")
        print(f"  - results_ensemble/")
        print(f"  - {args.output_dir}/comparison_report.txt")
    else:
        print(f"\n⚠️  Some steps failed. Check logs above.")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
