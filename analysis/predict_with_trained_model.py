#!/usr/bin/env python
"""
Model Prediction and Evaluation Script
======================================
Load trained models and make predictions with comprehensive Excel output.

Outputs:
- Subject ID
- True values (normalized and original)
- Predicted values (normalized and original)
- MSE (both scales)
- Per-fold and aggregate results

Usage:
    python predict_with_trained_model.py --model_dir results_gatv2_interpretable
    python predict_with_trained_model.py --model_dir results_braingt_enhanced --split test
    python predict_with_trained_model.py --model_dir results_braingnn_enhanced --split all
"""

import os
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


def load_model_checkpoint(checkpoint_path, device='cpu'):
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: Device to load model on

    Returns:
        checkpoint dict with model_state_dict, config, scaler, etc.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint


def load_graphs_and_indices(fold_path):
    """
    Load graphs and indices from fold directory.

    Args:
        fold_path: Path to fold directory (e.g., graphs/graphs_outer1_inner1/)

    Returns:
        dict with train_2d, val_2d, test_2d graphs and indices
    """
    fold_path = Path(fold_path)

    # Load graphs
    train_2d = torch.load(fold_path / 'train_2d.pt')
    val_2d = torch.load(fold_path / 'val_2d.pt')
    test_2d = torch.load(fold_path / 'test_2d.pt')

    # Load indices if available
    indices_path = fold_path / 'split_indices.pt'
    if indices_path.exists():
        indices = torch.load(indices_path)
    else:
        # Fallback: Try to load from older format
        indices = None
        print(f"Warning: split_indices.pt not found in {fold_path}")

    return {
        'train_2d': train_2d,
        'val_2d': val_2d,
        'test_2d': test_2d,
        'indices': indices,
    }


def get_subject_ids_from_graphs(graphs):
    """
    Extract subject IDs from graphs if available.

    Args:
        graphs: List of PyG Data objects

    Returns:
        numpy array of subject IDs or None
    """
    subject_ids = []
    for g in graphs:
        if hasattr(g, 'subject_id'):
            subject_ids.append(g.subject_id.item() if torch.is_tensor(g.subject_id) else g.subject_id)
        elif hasattr(g, 'idx'):
            subject_ids.append(g.idx.item() if torch.is_tensor(g.idx) else g.idx)

    if subject_ids:
        return np.array(subject_ids)
    else:
        return None


def predict_with_model(model, data_loader, device, model_type=''):
    """
    Make predictions using trained model.

    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for data
        device: Device to run on
        model_type: Type of model (for handling different return types)

    Returns:
        predictions, targets, subject_ids (all as numpy arrays)
    """
    model.eval()

    all_preds = []
    all_targets = []
    all_subject_ids = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)

            # Forward pass
            # Some models may return additional outputs, handle both cases
            try:
                output = model(batch)
                if isinstance(output, tuple):
                    preds = output[0]  # First element is predictions
                else:
                    preds = output
            except Exception as e:
                print(f"Error during prediction: {e}")
                # Try alternative forward pass (for enhanced models)
                try:
                    if hasattr(model, 'forward') and 'return_losses' in model.forward.__code__.co_varnames:
                        preds = model(batch, return_losses=False)
                    else:
                        preds = model(batch)
                except:
                    raise e

            target = batch.y.float()

            # Store predictions
            all_preds.append(preds.cpu().numpy())
            all_targets.append(target.cpu().numpy())

            # Get subject IDs if available
            if hasattr(batch, 'subject_id'):
                all_subject_ids.append(batch.subject_id.cpu().numpy())
            elif hasattr(batch, 'idx'):
                all_subject_ids.append(batch.idx.cpu().numpy())

    # Concatenate
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    if all_subject_ids:
        all_subject_ids = np.concatenate(all_subject_ids)
    else:
        # Generate sequential IDs if not available
        all_subject_ids = np.arange(len(all_preds))

    return all_preds, all_targets, all_subject_ids


def denormalize_predictions(normalized_values, scaler):
    """
    Denormalize predictions using saved scaler.

    Args:
        normalized_values: Normalized predictions/targets
        scaler: Fitted StandardScaler or similar

    Returns:
        Denormalized values
    """
    if scaler is None:
        return normalized_values

    normalized_values = np.asarray(normalized_values).reshape(-1, 1)
    original_values = scaler.inverse_transform(normalized_values).flatten()
    return original_values


def compute_metrics_dict(y_true, y_pred):
    """
    Compute comprehensive metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        dict with metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    r, p_value = pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R': r,
        'R2': r2,
        'P_value': p_value,
        'N_samples': len(y_true),
    }


def aggregate_by_subject(predictions, targets, subject_ids):
    """
    Aggregate window-level predictions to subject-level.

    Args:
        predictions: Window-level predictions
        targets: Window-level targets
        subject_ids: Subject identifiers

    Returns:
        subject_predictions, subject_targets, unique_subject_ids
    """
    unique_subjects = np.unique(subject_ids)

    subject_preds = []
    subject_targets = []

    for subj_id in unique_subjects:
        mask = (subject_ids == subj_id)
        subject_preds.append(predictions[mask].mean())
        subject_targets.append(targets[mask].mean())

    return np.array(subject_preds), np.array(subject_targets), unique_subjects


def create_prediction_dataframe(
    predictions_norm,
    targets_norm,
    subject_ids,
    predictions_orig=None,
    targets_orig=None,
    window_ids=None
):
    """
    Create DataFrame with all prediction information.

    Args:
        predictions_norm: Normalized predictions
        targets_norm: Normalized targets
        subject_ids: Subject IDs
        predictions_orig: Original scale predictions
        targets_orig: Original scale targets
        window_ids: Window identifiers (optional)

    Returns:
        pandas DataFrame
    """
    data = {
        'Subject_ID': subject_ids,
        'True_Normalized': targets_norm,
        'Predicted_Normalized': predictions_norm,
        'Error_Normalized': targets_norm - predictions_norm,
    }

    if window_ids is not None:
        data['Window_ID'] = window_ids

    if predictions_orig is not None and targets_orig is not None:
        data['True_Original'] = targets_orig
        data['Predicted_Original'] = predictions_orig
        data['Error_Original'] = targets_orig - predictions_orig
        data['Absolute_Error_Original'] = np.abs(targets_orig - predictions_orig)

    df = pd.DataFrame(data)

    # Reorder columns
    cols = ['Subject_ID']
    if window_ids is not None:
        cols.append('Window_ID')

    cols.extend([
        'True_Original', 'Predicted_Original', 'Error_Original', 'Absolute_Error_Original',
        'True_Normalized', 'Predicted_Normalized', 'Error_Normalized'
    ])

    # Only include columns that exist
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    return df


def predict_fold(model_dir, fold_name, graphs_dir, device, split='test'):
    """
    Make predictions for a single fold.

    Args:
        model_dir: Directory containing model checkpoint
        fold_name: Name of fold (e.g., 'graphs_outer1_inner1')
        graphs_dir: Directory containing graph data
        device: Device to run on
        split: Which split to predict on ('train', 'val', 'test', or 'all')

    Returns:
        dict with results
    """
    print(f"\n{'='*70}")
    print(f"Predicting fold: {fold_name}")
    print(f"{'='*70}\n")

    # Find model checkpoint
    model_dir = Path(model_dir) / fold_name
    checkpoint_files = list(model_dir.glob('*_best.pt'))

    if not checkpoint_files:
        print(f"No model checkpoint found in {model_dir}")
        return None

    checkpoint_path = checkpoint_files[0]
    print(f"Loading checkpoint: {checkpoint_path.name}")

    # Load checkpoint
    checkpoint = load_model_checkpoint(checkpoint_path, device)

    # Load corresponding data
    fold_data_dir = Path(graphs_dir) / fold_name
    data_dict = load_graphs_and_indices(fold_data_dir)

    # Get scaler from checkpoint
    scaler = checkpoint.get('scaler', None)

    # Reconstruct model
    # Try to automatically detect model type
    config = checkpoint.get('config', {})
    model_state = checkpoint['model_state_dict']

    # Import models based on what's available
    try:
        from models.brain_gt import BrainGT
        from models.brain_gnn import BrainGNN, SimpleBrainGNN
        from models.fbnetgen import FBNetGenFromGraph
    except:
        pass

    try:
        from models_enhanced.brain_gt_enhanced import BrainGTEnhanced
        from models_enhanced.brain_gnn_enhanced import BrainGNNEnhanced
        from models_enhanced.fbnetgen_enhanced import FBNetGenFromGraphEnhanced
    except:
        pass

    # Detect model architecture from state dict
    model = None
    model_name = checkpoint_path.stem.replace('_best', '')

    # Get input dimension from data
    sample_graph = data_dict['train_2d'][0] if data_dict['train_2d'] else data_dict['test_2d'][0]
    in_dim = sample_graph.x.size(1)

    # Try to create model based on name and config
    if 'braingt' in model_name.lower():
        try:
            if 'enhanced' in str(model_dir):
                model = BrainGTEnhanced(in_dim=in_dim, **config)
            else:
                model = BrainGT(in_dim=in_dim, **config)
        except:
            print("Could not create BrainGT model, trying generic approach...")
    elif 'braingnn' in model_name.lower():
        try:
            if 'enhanced' in str(model_dir):
                model = BrainGNNEnhanced(in_dim=in_dim, **config)
            else:
                model = SimpleBrainGNN(in_dim=in_dim, **config)
        except:
            print("Could not create BrainGNN model, trying generic approach...")
    elif 'fbnetgen' in model_name.lower():
        try:
            if 'enhanced' in str(model_dir):
                model = FBNetGenFromGraphEnhanced(in_dim=in_dim, **config)
            else:
                model = FBNetGenFromGraph(in_dim=in_dim, **config)
        except:
            print("Could not create FBNetGen model, trying generic approach...")

    # If model still not created, try loading from 1modgatv2.py
    if model is None:
        try:
            from models.gatv2 import GATv2Model
            model = GATv2Model(in_dim=in_dim, **config)
        except:
            try:
                # Import from 1modgatv2.py if exists
                import sys
                sys.path.insert(0, str(Path(__file__).parent))
                from modgatv2 import GATv2Model
                model = GATv2Model(in_dim=in_dim, **config)
            except:
                print("ERROR: Could not create model. Please check model architecture.")
                return None

    # Load model weights
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Determine which splits to predict
    splits_to_predict = []
    if split == 'all':
        splits_to_predict = ['train', 'val', 'test']
    else:
        splits_to_predict = [split]

    results = {}

    for split_name in splits_to_predict:
        print(f"\n--- Predicting on {split_name} set ---")

        # Get data for this split
        if split_name == 'train':
            graphs = data_dict['train_2d']
        elif split_name == 'val':
            graphs = data_dict['val_2d']
        else:  # test
            graphs = data_dict['test_2d']

        if not graphs:
            print(f"No {split_name} data found, skipping...")
            continue

        # Create dataloader
        loader = DataLoader(graphs, batch_size=32, shuffle=False)

        # Make predictions
        predictions_norm, targets_norm, subject_ids = predict_with_model(
            model, loader, device, model_type=model_name
        )

        # Denormalize if scaler available
        if scaler is not None:
            predictions_orig = denormalize_predictions(predictions_norm, scaler)
            targets_orig = denormalize_predictions(targets_norm, scaler)
            print(f"Denormalized predictions using saved scaler")
        else:
            predictions_orig = predictions_norm
            targets_orig = targets_norm
            print(f"Warning: No scaler found, using normalized values")

        # Compute window-level metrics
        window_metrics_norm = compute_metrics_dict(targets_norm, predictions_norm)
        window_metrics_orig = compute_metrics_dict(targets_orig, predictions_orig)

        print(f"\nWindow-level metrics (Original scale):")
        print(f"  MSE:  {window_metrics_orig['MSE']:.4f}")
        print(f"  MAE:  {window_metrics_orig['MAE']:.4f}")
        print(f"  RMSE: {window_metrics_orig['RMSE']:.4f}")
        print(f"  R:    {window_metrics_orig['R']:.4f}")
        print(f"  R2:   {window_metrics_orig['R2']:.4f}")
        print(f"  N:    {window_metrics_orig['N_samples']}")

        # Aggregate to subject-level
        subj_preds_norm, subj_targets_norm, unique_subjects = aggregate_by_subject(
            predictions_norm, targets_norm, subject_ids
        )

        if scaler is not None:
            subj_preds_orig = denormalize_predictions(subj_preds_norm, scaler)
            subj_targets_orig = denormalize_predictions(subj_targets_norm, scaler)
        else:
            subj_preds_orig = subj_preds_norm
            subj_targets_orig = subj_targets_norm

        subject_metrics_norm = compute_metrics_dict(subj_targets_norm, subj_preds_norm)
        subject_metrics_orig = compute_metrics_dict(subj_targets_orig, subj_preds_orig)

        print(f"\nSubject-level metrics (Original scale):")
        print(f"  MSE:  {subject_metrics_orig['MSE']:.4f}")
        print(f"  MAE:  {subject_metrics_orig['MAE']:.4f}")
        print(f"  RMSE: {subject_metrics_orig['RMSE']:.4f}")
        print(f"  R:    {subject_metrics_orig['R']:.4f}")
        print(f"  R2:   {subject_metrics_orig['R2']:.4f}")
        print(f"  N:    {subject_metrics_orig['N_samples']}")

        # Create DataFrames
        window_df = create_prediction_dataframe(
            predictions_norm, targets_norm, subject_ids,
            predictions_orig, targets_orig,
            window_ids=np.arange(len(predictions_norm))
        )

        subject_df = create_prediction_dataframe(
            subj_preds_norm, subj_targets_norm, unique_subjects,
            subj_preds_orig, subj_targets_orig
        )

        results[split_name] = {
            'window_df': window_df,
            'subject_df': subject_df,
            'window_metrics_norm': window_metrics_norm,
            'window_metrics_orig': window_metrics_orig,
            'subject_metrics_norm': subject_metrics_norm,
            'subject_metrics_orig': subject_metrics_orig,
        }

    return results


def save_results_to_excel(all_results, output_dir, model_dir_name):
    """
    Save all results to Excel files.

    Args:
        all_results: dict mapping fold_name -> results
        output_dir: Output directory
        model_dir_name: Name of model directory (for naming)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-fold results
    for fold_name, results in all_results.items():
        if results is None:
            continue

        fold_output = output_dir / f"{model_dir_name}_{fold_name}.xlsx"

        with pd.ExcelWriter(fold_output, engine='openpyxl') as writer:
            for split_name, split_results in results.items():
                # Window-level predictions
                sheet_name = f"{split_name}_windows"
                split_results['window_df'].to_excel(writer, sheet_name=sheet_name, index=False)

                # Subject-level predictions
                sheet_name = f"{split_name}_subjects"
                split_results['subject_df'].to_excel(writer, sheet_name=sheet_name, index=False)

                # Metrics
                sheet_name = f"{split_name}_metrics"
                metrics_data = {
                    'Metric': [],
                    'Window_Level_Normalized': [],
                    'Window_Level_Original': [],
                    'Subject_Level_Normalized': [],
                    'Subject_Level_Original': [],
                }

                for metric_name in ['MSE', 'MAE', 'RMSE', 'R', 'R2', 'P_value', 'N_samples']:
                    metrics_data['Metric'].append(metric_name)
                    metrics_data['Window_Level_Normalized'].append(
                        split_results['window_metrics_norm'][metric_name]
                    )
                    metrics_data['Window_Level_Original'].append(
                        split_results['window_metrics_orig'][metric_name]
                    )
                    metrics_data['Subject_Level_Normalized'].append(
                        split_results['subject_metrics_norm'][metric_name]
                    )
                    metrics_data['Subject_Level_Original'].append(
                        split_results['subject_metrics_orig'][metric_name]
                    )

                metrics_df = pd.DataFrame(metrics_data)
                metrics_df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"Saved: {fold_output}")

    # Create aggregate summary
    aggregate_file = output_dir / f"{model_dir_name}_aggregate_summary.xlsx"

    with pd.ExcelWriter(aggregate_file, engine='openpyxl') as writer:
        # Aggregate metrics across folds
        for split_name in ['train', 'val', 'test']:
            aggregate_data = {
                'Fold': [],
                'Window_MSE': [],
                'Window_R': [],
                'Subject_MSE': [],
                'Subject_R': [],
            }

            for fold_name, results in all_results.items():
                if results is None or split_name not in results:
                    continue

                split_results = results[split_name]
                aggregate_data['Fold'].append(fold_name)
                aggregate_data['Window_MSE'].append(split_results['window_metrics_orig']['MSE'])
                aggregate_data['Window_R'].append(split_results['window_metrics_orig']['R'])
                aggregate_data['Subject_MSE'].append(split_results['subject_metrics_orig']['MSE'])
                aggregate_data['Subject_R'].append(split_results['subject_metrics_orig']['R'])

            if aggregate_data['Fold']:
                agg_df = pd.DataFrame(aggregate_data)

                # Add summary statistics
                summary_row = {
                    'Fold': 'MEAN',
                    'Window_MSE': agg_df['Window_MSE'].mean(),
                    'Window_R': agg_df['Window_R'].mean(),
                    'Subject_MSE': agg_df['Subject_MSE'].mean(),
                    'Subject_R': agg_df['Subject_R'].mean(),
                }
                agg_df = pd.concat([agg_df, pd.DataFrame([summary_row])], ignore_index=True)

                summary_row = {
                    'Fold': 'STD',
                    'Window_MSE': agg_df['Window_MSE'][:-1].std(),
                    'Window_R': agg_df['Window_R'][:-1].std(),
                    'Subject_MSE': agg_df['Subject_MSE'][:-1].std(),
                    'Subject_R': agg_df['Subject_R'][:-1].std(),
                }
                agg_df = pd.concat([agg_df, pd.DataFrame([summary_row])], ignore_index=True)

                sheet_name = f"{split_name}_aggregate"
                agg_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\nSaved aggregate summary: {aggregate_file}")


def main():
    parser = argparse.ArgumentParser(description="Make predictions with trained models")

    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing trained models (e.g., results_gatv2_interpretable)')
    parser.add_argument('--graphs_dir', type=str, default='../../data',
                        help='Directory containing graph data folders')
    parser.add_argument('--output_dir', type=str, default='../../results/predictions',
                        help='Output directory for Excel files')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test', 'all'],
                        help='Which split to predict on')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--fold', type=str, default=None,
                        help='Predict specific fold only (e.g., graphs_outer1_inner1)')

    args = parser.parse_args()

    # Device
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
        fold_dirs = sorted([d.name for d in model_dir.iterdir() if d.is_dir()])

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
            results = predict_fold(
                args.model_dir,
                fold_name,
                args.graphs_dir,
                device,
                split=args.split
            )
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
        save_results_to_excel(all_results, args.output_dir, model_dir_name)

        print(f"\n{'='*70}")
        print(f"Predictions complete!")
        print(f"Results saved to: {args.output_dir}/")
        print(f"{'='*70}")
    else:
        print("\nNo successful predictions!")


if __name__ == "__main__":
    main()
