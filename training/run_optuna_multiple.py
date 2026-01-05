#!/usr/bin/env python
"""
Run Optuna Hyperparameter Search for Multiple Models
=====================================================

Automatically runs Optuna search for multiple models consecutively.

Usage:
    # Search for 2 models
    python run_optuna_multiple.py --models braingnn fbnetgen --n_trials 50

    # Search for all 3 models
    python run_optuna_multiple.py --models braingt braingnn fbnetgen --n_trials 100

    # Quick test
    python run_optuna_multiple.py --models braingnn fbnetgen --n_trials 20 --n_epochs 20
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_optuna_for_model(model_name, args):
    """
    Run Optuna search for a single model.

    Args:
        model_name: Model to search
        args: Command-line arguments

    Returns:
        bool: Success status
    """
    print("\n" + "="*70)
    print(f"OPTUNA SEARCH FOR {model_name.upper()}")
    print("="*70 + "\n")

    cmd = [
        sys.executable,
        'training/hyperparameter_search.py',
        '--model', model_name,
        '--n_trials', str(args.n_trials),
        '--n_epochs', str(args.n_epochs),
        '--n_jobs', str(args.n_jobs),
        '--device', args.device,
    ]

    if args.fold_name:
        cmd.extend(['--fold_name', args.fold_name])

    if args.output_dir:
        cmd.extend(['--output_dir', args.output_dir])

    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ {model_name.upper()} search completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {model_name.upper()} search failed with error:")
        print(f"   {str(e)}")
        return False


def print_summary(results, start_time):
    """Print final summary of all searches."""
    duration = datetime.now() - start_time

    print("\n" + "="*70)
    print("OPTUNA SEARCH COMPLETION SUMMARY")
    print("="*70 + "\n")

    print("Results:")
    for model_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {model_name.upper():<20} {status}")

    successful = sum(1 for s in results.values() if s)
    total = len(results)

    print(f"\nTotal: {successful}/{total} successful")
    print(f"Duration: {duration}")

    print("\n" + "="*70)
    print("RESULTS LOCATION")
    print("="*70 + "\n")

    for model_name, success in results.items():
        if success:
            print(f"{model_name.upper()}:")
            print(f"  Best params: hyperparameter_search_results/{model_name}_best_params.json")
            print(f"  Study data:  hyperparameter_search_results/{model_name}_study.csv")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter search for multiple models consecutively"
    )

    # Model selection
    parser.add_argument('--models', nargs='+', required=True,
                        choices=['braingt', 'braingnn', 'fbnetgen'],
                        help='Models to search (space-separated)')

    # Search configuration
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of trials per model (default: 50)')
    parser.add_argument('--n_epochs', type=int, default=30,
                        help='Epochs per trial (default: 30)')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Parallel jobs per model (default: 1)')

    # Data
    parser.add_argument('--fold_name', type=str, default=None,
                        help='Specific fold (default: first fold)')

    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: hyperparameter_search_results)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])

    # Execution
    parser.add_argument('--stop_on_error', action='store_true',
                        help='Stop if any model search fails (default: continue)')

    args = parser.parse_args()

    print("="*70)
    print("OPTUNA MULTI-MODEL HYPERPARAMETER SEARCH")
    print("="*70)
    print(f"\nModels to search: {', '.join([m.upper() for m in args.models])}")
    print(f"Trials per model: {args.n_trials}")
    print(f"Epochs per trial: {args.n_epochs}")
    print(f"Parallel jobs: {args.n_jobs}")
    print(f"Device: {args.device}")
    print(f"Total searches: {len(args.models)}")

    total_trials = len(args.models) * args.n_trials
    print(f"Total trials: {total_trials}")
    print("="*70)

    # Confirm if large search
    if total_trials > 100:
        response = input(f"\nThis will run {total_trials} total trials. Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    # Run searches
    start_time = datetime.now()
    results = {}

    for i, model_name in enumerate(args.models, 1):
        print(f"\n{'='*70}")
        print(f"SEARCH {i}/{len(args.models)}: {model_name.upper()}")
        print(f"{'='*70}")

        success = run_optuna_for_model(model_name, args)
        results[model_name] = success

        if not success and args.stop_on_error:
            print(f"\n⚠️  Stopping due to error in {model_name.upper()}")
            break

    # Print summary
    print_summary(results, start_time)

    # Exit code
    if all(results.values()):
        print("\n✓ All searches completed successfully!")
        sys.exit(0)
    else:
        print("\n⚠️  Some searches failed. Check logs above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
