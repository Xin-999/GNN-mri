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


def load_subject_ids_and_scores(csv_path='data/ListSort_AgeAdj.csv'):
    """
    Load real subject IDs and cognitive scores from CSV.

    Returns:
        subject_ids: List of subject IDs in order
        scores: List of cognitive scores in order
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"Warning: {csv_path} not found. Using array indices as subject IDs.")
        return None, None

    df = pd.read_csv(csv_file)
    subject_ids = df['Subject'].tolist()
    scores = df['ListSort_AgeAdj'].tolist()

    print(f"\nLoaded {len(subject_ids)} subjects from {csv_path}")
    print(f"Original score range: [{min(scores):.2f}, {max(scores):.2f}]")

    return subject_ids, scores


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

    # Verify scaler is loaded
    if scaler is not None:
        print(f"Scaler loaded - Mean: {scaler.mean_[0]:.2f}, Std: {scaler.scale_[0]:.2f}")
    else:
        print("WARNING: No scaler found in checkpoint! Predictions will be in normalized scale.")

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

    # Extract real subject indices
    train_indices = graphs_dict.get('train_indices', [])
    val_indices = graphs_dict.get('val_indices', [])
    test_indices = graphs_dict.get('test_indices', [])

    # Get input dimension from first non-padded graph
    sample_graph = None
    # Convert numpy arrays to lists for iteration
    all_graphs = list(train_2d) + list(val_2d) + list(test_2d)
    for row in all_graphs:
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
            print(f"Could not load ImprovedGATv2Regressor: {type(e).__name__}: {str(e)[:200]}")
    else:
        print("ImprovedGATv2Regressor class is None (import failed)")

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
            print(f"Could not load GATv2Regressor: {type(e).__name__}: {str(e)[:200]}")
    elif model is None:
        print("GATv2Regressor class is None (import failed)")

    if model is None:
        raise RuntimeError("Could not load model - no compatible architecture found")

    model = model.to(device)
    model.eval()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, {
        'train_2d': train_2d,
        'val_2d': val_2d,
        'test_2d': test_2d,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
    }, scaler, config


def flatten_graphs(graphs_2d, fold_subject_indices=None, all_subject_ids=None, split_name='split'):
    """
    Flatten 2D graph list and extract subject IDs and targets.

    Args:
        graphs_2d: 2D list of graphs [subject][window]
        fold_subject_indices: Array indices for this split (e.g., [0, 1, 5, 7...])
        all_subject_ids: Full list of real subject IDs to lookup from
        split_name: Name of this split

    Returns:
        List of graphs, list of real subject IDs, list of targets
    """
    graphs = []
    subject_ids = []
    targets = []

    for subj_idx, row in enumerate(graphs_2d):
        # Get the real subject ID by looking up the fold index
        if fold_subject_indices is not None and all_subject_ids is not None:
            if subj_idx < len(fold_subject_indices):
                array_index = int(fold_subject_indices[subj_idx])
                if array_index < len(all_subject_ids):
                    real_subject_id = all_subject_ids[array_index]
                else:
                    real_subject_id = array_index
            else:
                real_subject_id = subj_idx
        else:
            real_subject_id = subj_idx  # Fallback to index if no mapping available

        for window_idx, g in enumerate(row):
            # Skip padded graphs
            if hasattr(g, "pad") and bool(g.pad):
                continue

            graphs.append(g)
            subject_ids.append(real_subject_id)
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
        numpy array of predictions (denormalized if scaler exists)
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
    predictions_normalized = predictions.copy()

    # Inverse transform if scaler available
    if scaler is not None:
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        print(f"    Denormalized predictions: {predictions_normalized[:3]} -> {predictions[:3]}")
    else:
        print(f"    WARNING: No scaler - predictions remain normalized!")

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

    # Load real subject IDs and scores from CSV
    all_subject_ids, all_scores = load_subject_ids_and_scores()

    # Load model and data
    model, data_dict, scaler, config = load_model_and_data(model_dir, fold_name, device)

    results = {}

    # Determine which splits to process
    splits_to_process = []
    if split == 'all':
        splits_to_process = [('train', data_dict['train_2d'], data_dict['train_indices']),
                            ('val', data_dict['val_2d'], data_dict['val_indices']),
                            ('test', data_dict['test_2d'], data_dict['test_indices'])]
    else:
        if split == 'train':
            splits_to_process = [('train', data_dict['train_2d'], data_dict['train_indices'])]
        elif split == 'val':
            splits_to_process = [('val', data_dict['val_2d'], data_dict['val_indices'])]
        elif split == 'test':
            splits_to_process = [('test', data_dict['test_2d'], data_dict['test_indices'])]

    # Process each split
    for split_name, graphs_2d, real_indices in splits_to_process:
        if len(graphs_2d) == 0:
            print(f"No {split_name} data available")
            continue

        print(f"\nPredicting {split_name} split...")

        # Flatten graphs (with real subject IDs from CSV)
        graphs, subject_ids, targets_normalized = flatten_graphs(
            graphs_2d,
            fold_subject_indices=real_indices,
            all_subject_ids=all_subject_ids,
            split_name=split_name
        )

        if not graphs:
            print(f"No valid graphs in {split_name} split")
            continue

        # Debug: Check subject_ids
        print(f"  Subject IDs: {subject_ids[:5]}... (first 5)")
        print(f"  Unique subjects: {sorted(set(subject_ids))[:10]}...")

        # Get true scores from CSV for each window's subject
        if all_subject_ids is not None and all_scores is not None:
            # Map subject ID to score
            score_map = {sid: score for sid, score in zip(all_subject_ids, all_scores)}
            # Get score for each window based on its subject ID
            targets = np.array([score_map.get(sid, np.nan) for sid in subject_ids])
            print(f"  {len(graphs)} windows from {len(set(subject_ids))} subjects")
            print(f"  Target range (from CSV): [{np.nanmin(targets):.2f}, {np.nanmax(targets):.2f}]")
        else:
            # Fallback: Use scaler to denormalize
            targets_array = np.array(targets_normalized)
            if scaler is not None:
                targets = scaler.inverse_transform(targets_array.reshape(-1, 1)).flatten()
                print(f"  {len(graphs)} windows from {len(set(subject_ids))} subjects")
                print(f"  Target range (from scaler): [{np.min(targets):.2f}, {np.max(targets):.2f}]")
            else:
                targets = targets_array
                print(f"  WARNING: No scaler or CSV - targets remain normalized!")

        # Make predictions (predictions are denormalized in predict_split if scaler exists)
        predictions = predict_split(model, graphs, scaler, device)
        print(f"  Prediction range: [{np.min(predictions):.2f}, {np.max(predictions):.2f}]")

        # Calculate metrics (both targets and predictions are in original scale)
        metrics = calculate_metrics(targets, predictions)

        print(f"  Metrics: r={metrics['r']:.4f}, MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}")

        # Store results (convert to lists for Excel compatibility)
        results[split_name] = {
            'subject_ids': [int(sid) for sid in subject_ids],  # Ensure integers
            'targets': targets.tolist() if isinstance(targets, np.ndarray) else list(targets),
            'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else list(predictions),
            'errors': (np.array(predictions) - np.array(targets)).tolist(),
            'absolute_errors': np.abs(np.array(predictions) - np.array(targets)).tolist(),
            'metrics': metrics,
        }

    return results


def save_fold_to_excel(fold_name, fold_results, output_dir):
    """
    Save a single fold's prediction results to Excel file.

    Args:
        fold_name: Name of the fold
        fold_results: Results dictionary for this fold
        output_dir: Output directory
    """
    if fold_results is None:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"Saved: {excel_path}\n")


def save_aggregate_summary(all_results, output_dir, model_dir_name):
    """
    Save aggregate summary across all folds.

    Args:
        all_results: Dictionary of results by fold
        output_dir: Output directory
        model_dir_name: Name of model directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

            # Save this fold's results immediately
            save_fold_to_excel(fold_name, results, args.output_dir)

        except Exception as e:
            print(f"\nError predicting fold {fold_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[fold_name] = None
            continue

    # Save aggregate summary
    if all_results:
        model_dir_name = Path(args.model_dir).name
        save_aggregate_summary(all_results, args.output_dir, model_dir_name)

        print(f"\n{'='*70}")
        print(f"Predictions complete!")
        print(f"Results saved to: {args.output_dir}/")
        print(f"{'='*70}")
    else:
        print("\nNo successful predictions!")


if __name__ == "__main__":
    main()
