#!/usr/bin/env python
"""
Simple GATv2 Prediction Script
===============================

Makes predictions using trained GATv2 models and outputs results to Excel.

Usage:
    python predict_gatv2_simple.py --model_dir results_gatv2_interpretable --device cuda
    python predict_gatv2_simple.py --model_dir results_gatv2_interpretable --device cpu --split test
    python predict_gatv2_simple.py --model_dir results_gatv2_interpretable --fold graphs_outer1_inner2
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add training directory to path for model imports
training_dir = Path(__file__).parent.parent / "training" / "gatv2"
sys.path.insert(0, str(training_dir))

try:
    from train_gatv2_improved import ImprovedGATv2Regressor
except Exception as e:
    print(f"Warning: Could not import ImprovedGATv2Regressor: {e}")
    ImprovedGATv2Regressor = None

try:
    from train_gatv2_basic import GATv2Regressor
except Exception as e:
    print(f"Warning: Could not import GATv2Regressor: {e}")
    GATv2Regressor = None


def load_model_and_data(model_dir, fold_name, device='cpu'):
    """
    Load trained model and data for a specific fold.

    Args:
        model_dir: Directory containing trained models
        fold_name: Name of fold (e.g., 'graphs_outer1_inner1')
        device: Device to load model on

    Returns:
        model, data_dict with train/val/test graphs, scaler, config
    """
    model_path = Path(model_dir) / fold_name / "gatv2_best.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"\nLoading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle both old and new checkpoint formats
    if 'model_state_dict' in checkpoint:
        config = checkpoint.get('config', {})
        scaler = checkpoint.get('scaler', None)
        state_dict = checkpoint['model_state_dict']
    else:
        config = {}
        scaler = checkpoint.get('scaler', None)
        state_dict = checkpoint
        print("Warning: Old checkpoint format detected")

    # Load fold data
    fold_data_path = Path(f"../../data/folds_data/{fold_name}.pkl")
    if not fold_data_path.exists():
        fold_data_path = Path(f"data/folds_data/{fold_name}.pkl")

    if not fold_data_path.exists():
        raise FileNotFoundError(f"Data file not found: {fold_name}.pkl")

    print(f"Loading data from {fold_data_path}")
    graphs_dict = torch.load(fold_data_path, map_location='cpu', weights_only=False)

    # Extract data
    train_2d = graphs_dict.get('train_graphs', [])
    val_2d = graphs_dict.get('val_graphs', [])
    test_2d = graphs_dict.get('test_graphs', [])

    # Get input dimension from first non-padded graph
    sample_graph = None
    for row in train_2d + val_2d + test_2d:
        for g in row:
            if not (hasattr(g, "pad") and bool(g.pad)):
                sample_graph = g
                break
        if sample_graph:
            break

    if sample_graph is None:
        raise ValueError("No valid graphs found in data")

    in_dim = sample_graph.x.size(-1)

    # Try to create and load model
    model = None

    # Try ImprovedGATv2Regressor first
    if ImprovedGATv2Regressor is not None:
        try:
            temp_model = ImprovedGATv2Regressor(
                in_dim=in_dim,
                hidden_dim=config.get('hidden_dim', 64),
                n_layers=config.get('n_layers', 3),
                n_heads=config.get('n_heads', 4),
                dropout=config.get('dropout', 0.2),
                edge_dropout=config.get('edge_dropout', 0.1),
            )
            temp_model.load_state_dict(state_dict)
            model = temp_model
            print(f"Loaded model: ImprovedGATv2Regressor")
        except Exception as e:
            print(f"Could not load ImprovedGATv2Regressor: {type(e).__name__}")

    # Try GATv2Regressor (basic) if improved didn't work
    if model is None and GATv2Regressor is not None:
        try:
            temp_model = GATv2Regressor(
                in_dim=in_dim,
                hidden_dim=config.get('hidden_dim', 32),
            )
            temp_model.load_state_dict(state_dict)
            model = temp_model
            print(f"Loaded model: GATv2Regressor")
        except Exception as e:
            print(f"Could not load GATv2Regressor: {type(e).__name__}")

    if model is None:
        raise RuntimeError("Could not load model - no compatible architecture found")

    model = model.to(device)
    model.eval()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, {
        'train_2d': train_2d,
        'val_2d': val_2d,
        'test_2d': test_2d,
    }, scaler, config


def flatten_graphs(graphs_2d, split_name='split'):
    """
    Flatten 2D graph list and extract subject IDs and targets.

    Args:
        graphs_2d: 2D list of graphs [subject][window]
        split_name: Name of this split

    Returns:
        List of graphs, list of subject IDs, list of targets
    """
    graphs = []
    subject_ids = []
    targets = []

    for subj_idx, row in enumerate(graphs_2d):
        for window_idx, g in enumerate(row):
            # Skip padded graphs
            if hasattr(g, "pad") and bool(g.pad):
                continue

            # Add subject_id if not present
            if not hasattr(g, 'subject_id'):
                g.subject_id = torch.tensor([subj_idx], dtype=torch.long)

            graphs.append(g)
            subject_ids.append(g.subject_id.item() if torch.is_tensor(g.subject_id) else g.subject_id)
            targets.append(g.y.item() if torch.is_tensor(g.y) else g.y)

    return graphs, subject_ids, targets


def predict_split(model, graphs, scaler, device, batch_size=32):
    """
    Make predictions on a list of graphs.

    Args:
        model: Trained model
        graphs: List of graphs
        scaler: Target scaler (optional)
        device: Device to run on
        batch_size: Batch size for prediction

    Returns:
        numpy array of predictions
    """
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    predictions = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            predictions.append(pred.cpu().numpy())

    predictions = np.concatenate(predictions)

    # Inverse transform if scaler available
    if scaler is not None:
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    return predictions


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    r, p_value = pearsonr(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return {
        'r': r,
        'p_value': p_value,
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mae,
    }


def predict_fold(model_dir, fold_name, device, split='all'):
    """
    Make predictions for a single fold.

    Args:
        model_dir: Directory containing trained models
        fold_name: Name of fold
        device: Device to run on
        split: Which split to predict ('train', 'val', 'test', or 'all')

    Returns:
        Dictionary with predictions and metrics
    """
    print(f"\n{'='*70}")
    print(f"Processing fold: {fold_name}")
    print(f"{'='*70}")

    # Load model and data
    model, data_dict, scaler, config = load_model_and_data(model_dir, fold_name, device)

    results = {}

    # Determine which splits to process
    splits_to_process = []
    if split == 'all':
        splits_to_process = [('train', data_dict['train_2d']),
                            ('val', data_dict['val_2d']),
                            ('test', data_dict['test_2d'])]
    else:
        key = f"{split}_2d" if split != 'test' else 'test_2d'
        if split == 'train':
            splits_to_process = [('train', data_dict['train_2d'])]
        elif split == 'val':
            splits_to_process = [('val', data_dict['val_2d'])]
        elif split == 'test':
            splits_to_process = [('test', data_dict['test_2d'])]

    # Process each split
    for split_name, graphs_2d in splits_to_process:
        if not graphs_2d:
            print(f"No {split_name} data available")
            continue

        print(f"\nPredicting {split_name} split...")

        # Flatten graphs
        graphs, subject_ids, targets = flatten_graphs(graphs_2d, split_name)

        if not graphs:
            print(f"No valid graphs in {split_name} split")
            continue

        print(f"  {len(graphs)} windows from {len(set(subject_ids))} subjects")

        # Make predictions
        predictions = predict_split(model, graphs, scaler, device)

        # Calculate metrics
        metrics = calculate_metrics(targets, predictions)

        print(f"  Metrics: r={metrics['r']:.4f}, MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}")

        # Store results
        results[split_name] = {
            'subject_ids': subject_ids,
            'targets': targets,
            'predictions': predictions,
            'errors': np.array(predictions) - np.array(targets),
            'absolute_errors': np.abs(np.array(predictions) - np.array(targets)),
            'metrics': metrics,
        }

    return results


def save_to_excel(all_results, output_dir, model_dir_name):
    """
    Save prediction results to Excel files.

    Args:
        all_results: Dictionary of results by fold
        output_dir: Output directory
        model_dir_name: Name of model directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-fold results
    for fold_name, fold_results in all_results.items():
        if fold_results is None:
            continue

        excel_path = output_dir / f"{fold_name}_predictions.xlsx"

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Write each split to a separate sheet
            for split_name, split_data in fold_results.items():
                df = pd.DataFrame({
                    'Subject_ID': split_data['subject_ids'],
                    'True_Score': split_data['targets'],
                    'Predicted_Score': split_data['predictions'],
                    'Error': split_data['errors'],
                    'Absolute_Error': split_data['absolute_errors'],
                })

                df.to_excel(writer, sheet_name=split_name.capitalize(), index=False)

                # Add metrics summary
                metrics_df = pd.DataFrame([split_data['metrics']])
                metrics_df.to_excel(writer, sheet_name=f"{split_name.capitalize()}_Metrics", index=False)

        print(f"Saved: {excel_path}")

    # Create aggregate summary
    summary_rows = []
    for fold_name, fold_results in all_results.items():
        if fold_results is None:
            continue

        for split_name, split_data in fold_results.items():
            summary_rows.append({
                'Fold': fold_name,
                'Split': split_name,
                'N_Windows': len(split_data['subject_ids']),
                'N_Subjects': len(set(split_data['subject_ids'])),
                'Pearson_r': split_data['metrics']['r'],
                'P_value': split_data['metrics']['p_value'],
                'MSE': split_data['metrics']['mse'],
                'RMSE': split_data['metrics']['rmse'],
                'MAE': split_data['metrics']['mae'],
            })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = output_dir / f"{model_dir_name}_summary.xlsx"
        summary_df.to_excel(summary_path, index=False)
        print(f"\nSaved aggregate summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Make predictions with trained GATv2 models")

    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing trained models (e.g., results_gatv2_interpretable)')
    parser.add_argument('--output_dir', type=str, default='results/predictions',
                        help='Output directory for Excel files')
    parser.add_argument('--split', type=str, default='all',
                        choices=['train', 'val', 'test', 'all'],
                        help='Which split to predict on')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'])
    parser.add_argument('--fold', type=str, default=None,
                        help='Predict specific fold only (e.g., graphs_outer1_inner1)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for prediction')

    args = parser.parse_args()

    # Device setup
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Get fold directories
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        return

    if args.fold:
        fold_dirs = [args.fold]
    else:
        # Find all fold subdirectories
        fold_dirs = sorted([d.name for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('graphs_')])

    if not fold_dirs:
        print(f"No fold directories found in {model_dir}")
        return

    print(f"\nFound {len(fold_dirs)} folds to predict:")
    for fold in fold_dirs:
        print(f"  - {fold}")

    # Predict all folds
    all_results = {}

    for fold_name in fold_dirs:
        try:
            results = predict_fold(args.model_dir, fold_name, device, split=args.split)
            all_results[fold_name] = results
        except Exception as e:
            print(f"\nError predicting fold {fold_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[fold_name] = None
            continue

    # Save results to Excel
    if all_results:
        model_dir_name = Path(args.model_dir).name
        save_to_excel(all_results, args.output_dir, model_dir_name)

        print(f"\n{'='*70}")
        print(f"Predictions complete!")
        print(f"Results saved to: {args.output_dir}/")
        print(f"{'='*70}")
    else:
        print("\nNo successful predictions!")


if __name__ == "__main__":
    main()
