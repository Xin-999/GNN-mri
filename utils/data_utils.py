"""
Data loading and preprocessing utilities with target normalization
==================================================================
Critical fix: Normalize cognitive scores for stable training
"""

import os
import numpy as np
import torch
import dill
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import LeaveOneOut
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import warnings


def normalize_targets(
    scores: np.ndarray,
    method: str = 'standard',  # 'standard', 'minmax', or 'robust'
    scaler: Optional[object] = None,
    fit: bool = True,
) -> Tuple[np.ndarray, object]:
    """
    Normalize cognitive scores for stable training.

    CRITICAL: Your scores range ~94-131 without normalization,
    causing training instability. This fixes it.

    Args:
        scores: (n_subjects,) array of cognitive scores
        method: Normalization method
            - 'standard': zero mean, unit variance (recommended)
            - 'minmax': scale to [0, 1] or [-1, 1]
            - 'robust': robust to outliers (median, IQR)
        scaler: Existing scaler object (for test set)
        fit: Whether to fit scaler (True for train, False for test)

    Returns:
        normalized_scores: (n_subjects,) normalized scores
        scaler: Fitted scaler object (save for inverse transform)

    Example:
        >>> # Training
        >>> train_scores_norm, scaler = normalize_targets(train_scores, fit=True)
        >>> # Testing (use same scaler!)
        >>> test_scores_norm, _ = normalize_targets(test_scores, scaler=scaler, fit=False)
        >>> # Inverse transform predictions
        >>> predictions_original = scaler.inverse_transform(predictions_normalized.reshape(-1, 1)).flatten()
    """
    scores = np.asarray(scores, dtype=float).reshape(-1, 1)

    if scaler is None:
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler(feature_range=(-1, 1))
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    if fit:
        normalized = scaler.fit_transform(scores)
        print(f"Target normalization ({method}):")
        print(f"  Original range: [{scores.min():.2f}, {scores.max():.2f}]")
        print(f"  Normalized range: [{normalized.min():.2f}, {normalized.max():.2f}]")
        print(f"  Mean: {normalized.mean():.4f}, Std: {normalized.std():.4f}")
    else:
        normalized = scaler.transform(scores)

    return normalized.flatten(), scaler


def load_graphs_with_normalization(
    fold_path: str,
    normalize_method: str = 'standard',
    weights_only: bool = False,
) -> Tuple[List, List, List, Dict]:
    """
    Load graphs from a fold with NORMALIZED targets.

    This is the key fix for your overfitting issue!

    Args:
        fold_path: Path to fold pickle file
        normalize_method: Method for target normalization
        weights_only: PyTorch load weights_only flag

    Returns:
        train_graphs: List of training PyG Data objects (normalized targets)
        val_graphs: List of validation PyG Data objects (normalized targets)
        test_graphs: List of test PyG Data objects (normalized targets)
        info: Dict with scaler and metadata
    """
    print(f"Loading fold from {fold_path}")
    graphs = torch.load(fold_path, map_location="cpu", weights_only=weights_only)

    train_2d = graphs["train_graphs"]
    val_2d   = graphs["val_graphs"]
    test_2d  = graphs["test_graphs"]

    # Flatten graphs and collect targets
    def flatten_graphs(arr2d):
        flat = []
        targets = []
        subject_ids = []
        for subj_idx, row in enumerate(arr2d):
            for g in row:
                if hasattr(g, "pad") and bool(g.pad):
                    continue
                g.subject_id = torch.tensor([subj_idx], dtype=torch.long)
                flat.append(g)
                targets.append(float(g.y.item()))
                subject_ids.append(subj_idx)
        return flat, np.array(targets), np.array(subject_ids)

    train_list, train_targets, train_subj = flatten_graphs(train_2d)
    val_list, val_targets, val_subj = flatten_graphs(val_2d)
    test_list, test_targets, test_subj = flatten_graphs(test_2d)

    # CRITICAL: Normalize targets
    # Fit scaler on training data only!
    train_targets_norm, scaler = normalize_targets(
        train_targets, method=normalize_method, fit=True
    )
    val_targets_norm, _ = normalize_targets(
        val_targets, method=normalize_method, scaler=scaler, fit=False
    )
    test_targets_norm, _ = normalize_targets(
        test_targets, method=normalize_method, scaler=scaler, fit=False
    )

    # Update graph targets with normalized values
    for i, g in enumerate(train_list):
        g.y = torch.tensor(train_targets_norm[i], dtype=torch.float32)
    for i, g in enumerate(val_list):
        g.y = torch.tensor(val_targets_norm[i], dtype=torch.float32)
    for i, g in enumerate(test_list):
        g.y = torch.tensor(test_targets_norm[i], dtype=torch.float32)

    print(f"#train graphs (windows): {len(train_list)}")
    print(f"#val   graphs (windows): {len(val_list)}")
    print(f"#test  graphs (windows): {len(test_list)}")

    info = {
        'scaler': scaler,
        'train_targets_original': train_targets,
        'val_targets_original': val_targets,
        'test_targets_original': test_targets,
        'train_subject_ids': train_subj,
        'val_subject_ids': val_subj,
        'test_subject_ids': test_subj,
    }

    return train_list, val_list, test_list, info


def create_loocv_splits(
    all_graphs: np.ndarray,
    n_subjects: int,
) -> List[Dict]:
    """
    Create Leave-One-Subject-Out CV splits.

    Better than nested k-fold for N=184 subjects.

    Args:
        all_graphs: (n_subjects, n_windows) array of PyG Data
        n_subjects: Number of subjects

    Returns:
        splits: List of dicts with 'train', 'val', 'test' indices
    """
    loo = LeaveOneOut()
    splits = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(loo.split(range(n_subjects))):
        # Further split train_val into train and val (90/10)
        n_train = int(len(train_val_idx) * 0.9)
        train_idx = train_val_idx[:n_train]
        val_idx = train_val_idx[n_train:]

        splits.append({
            'train': train_idx,
            'val': val_idx,
            'test': test_idx,
            'fold': fold_idx,
        })

    print(f"Created {len(splits)} LOOCV splits")
    return splits


def get_timeseries_data(
    data_path: str = './data/all_shen_roi_ts',
    score_csv: str = './data/ListSort_AgeAdj.csv',
    score_column: str = 'ListSort_AgeAdj',
    normalize_timeseries: bool = True,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Load raw time-series data for FBNetGen (end-to-end training).

    Args:
        data_path: Directory with ROI time-series .txt files
        score_csv: Path to cognitive scores CSV
        score_column: Column name for target scores
        normalize_timeseries: Whether to z-score time-series

    Returns:
        timeseries_list: List of (time, n_rois) arrays per subject
        cognitive_scores: (n_subjects,) array of scores
        subject_ids: (n_subjects,) array of subject identifiers
    """
    from step1_compute_ldw import load_data

    print("Loading raw fMRI time-series...")
    data, cognitive_scores = load_data(
        data_dir=data_path,
        score_csv=score_csv,
        score_column=score_column,
        score_key='subject',
    )

    # Convert to numpy arrays
    timeseries_list = [np.array(ts) for ts in data]

    # Optional: Normalize time-series (recommended)
    if normalize_timeseries:
        print("Normalizing time-series (z-score per ROI)...")
        timeseries_list = [
            (ts - ts.mean(axis=0, keepdims=True)) / (ts.std(axis=0, keepdims=True) + 1e-8)
            for ts in timeseries_list
        ]

    # Extract subject IDs (assuming filename format: {subject}_{run}.txt)
    data_path = Path(data_path)
    txt_files = sorted(data_path.glob('*.txt'))
    subject_ids = np.array([f.stem.split('_')[0] for f in txt_files])

    print(f"Loaded {len(timeseries_list)} subjects")
    print(f"ROI count: {timeseries_list[0].shape[1]}")
    print(f"Time points range: {min(ts.shape[0] for ts in timeseries_list)} - {max(ts.shape[0] for ts in timeseries_list)}")

    return timeseries_list, cognitive_scores, subject_ids


def create_dataloaders(
    train_graphs: List,
    val_graphs: List,
    test_graphs: List,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch Geometric DataLoaders with proper settings.

    Args:
        train_graphs, val_graphs, test_graphs: Lists of PyG Data objects
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory (faster GPU transfer)

    Returns:
        train_loader, val_loader, test_loader
    """
    train_loader = DataLoader(
        train_graphs,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_graphs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_graphs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def aggregate_window_predictions(
    predictions: np.ndarray,
    subject_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate window-level predictions to subject-level.

    Critical for proper evaluation - windows from same subject
    are NOT independent samples!

    Args:
        predictions: (n_windows,) window-level predictions
        subject_ids: (n_windows,) subject ID for each window

    Returns:
        subject_predictions: (n_subjects,) averaged predictions
        unique_subjects: (n_subjects,) subject IDs
    """
    unique_subjects = np.unique(subject_ids)
    subject_predictions = []

    for subj in unique_subjects:
        subj_mask = (subject_ids == subj)
        subj_pred = predictions[subj_mask].mean()
        subject_predictions.append(subj_pred)

    return np.array(subject_predictions), unique_subjects


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    prefix: str = '',
) -> Dict[str, float]:
    """
    Compute evaluation metrics for regression.

    Args:
        predictions: (n_samples,) predicted values
        targets: (n_samples,) ground truth values
        prefix: Prefix for metric names (e.g., 'test_')

    Returns:
        metrics: Dictionary of metric name -> value
    """
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    if len(predictions) < 2:
        warnings.warn(f"Too few samples ({len(predictions)}) to compute meaningful metrics")
        return {f'{prefix}mse': float('nan'), f'{prefix}pearson_r': float('nan'), f'{prefix}spearman_r': float('nan')}

    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)

    # Pearson correlation (linear)
    pearson_r, pearson_p = pearsonr(predictions, targets)

    # Spearman correlation (rank-based)
    spearman_r, spearman_p = spearmanr(predictions, targets)

    r2 = r2_score(targets, predictions)

    metrics = {
        f'{prefix}mse': float(mse),
        f'{prefix}mae': float(mae),
        f'{prefix}pearson_r': float(pearson_r),
        f'{prefix}pearson_p': float(pearson_p),
        f'{prefix}spearman_r': float(spearman_r),
        f'{prefix}spearman_p': float(spearman_p),
        f'{prefix}r2': float(r2),
        # Keep 'r' as alias for pearson_r for backward compatibility
        f'{prefix}r': float(pearson_r),
        f'{prefix}p_value': float(pearson_p),
    }

    return metrics


def permutation_test(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_permutations: int = 1000,
    metric: str = 'pearson',
    random_state: int = 42
) -> Dict[str, float]:
    """
    Permutation test for correlation significance.

    Creates null distribution by shuffling labels (without retraining).

    Args:
        predictions: Model predictions (fixed)
        targets: True labels (will be shuffled)
        n_permutations: Number of random shuffles
        metric: 'pearson' or 'spearman'
        random_state: Random seed for reproducibility

    Returns:
        results: Dict with real_r, perm_p_value, null_distribution
    """
    from scipy.stats import pearsonr, spearmanr

    np.random.seed(random_state)

    # Compute real correlation
    if metric == 'pearson':
        real_r, _ = pearsonr(predictions, targets)
    elif metric == 'spearman':
        real_r, _ = spearmanr(predictions, targets)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Build null distribution by shuffling targets
    null_r = []
    for _ in range(n_permutations):
        # Shuffle targets (break true relationship)
        shuffled_targets = np.random.permutation(targets)

        # Compute correlation with shuffled targets
        if metric == 'pearson':
            null_corr, _ = pearsonr(predictions, shuffled_targets)
        else:
            null_corr, _ = spearmanr(predictions, shuffled_targets)

        null_r.append(null_corr)

    null_r = np.array(null_r)

    # Compute permutation p-value (two-tailed)
    # How many null correlations are as extreme as real correlation?
    p_value = np.mean(np.abs(null_r) >= np.abs(real_r))

    results = {
        f'{metric}_r': float(real_r),
        f'{metric}_perm_p': float(p_value),
        f'{metric}_null_mean': float(np.mean(null_r)),
        f'{metric}_null_std': float(np.std(null_r)),
        f'{metric}_null_distribution': null_r.tolist(),
    }

    return results


def save_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    subject_ids: np.ndarray,
    save_path: Path,
    scaler: Optional[object] = None,
):
    """
    Save predictions to JSON with optional inverse normalization.

    Args:
        predictions: Predicted values (possibly normalized)
        targets: Ground truth values (possibly normalized)
        subject_ids: Subject identifiers
        save_path: Path to save JSON
        scaler: Scaler object for inverse transform
    """
    import json

    # Inverse transform if scaler provided
    if scaler is not None:
        pred_original = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        target_original = scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
    else:
        pred_original = predictions
        target_original = targets

    # Create list of prediction entries
    predictions_list = []
    for i in range(len(subject_ids)):
        predictions_list.append({
            'subject_id': int(subject_ids[i]),
            'prediction': float(pred_original[i]),
            'target': float(target_original[i]),
            'prediction_normalized': float(predictions[i]),
            'target_normalized': float(targets[i]),
            'error': float(pred_original[i] - target_original[i]),
        })

    # Save as JSON
    with open(save_path, 'w') as f:
        json.dump(predictions_list, f, indent=2)

    print(f"Saved predictions to {save_path}")


def load_all_folds_with_normalization(
    fold_dir: str = 'data/folds_data',
    normalize_method: str = 'standard',
) -> Dict[str, Tuple]:
    """
    Load all fold files with normalized targets.

    Args:
        fold_dir: Directory containing fold pickle files
        normalize_method: Normalization method

    Returns:
        folds_dict: Dict mapping fold_name -> (train, val, test, info)
    """
    fold_dir = Path(fold_dir)
    fold_files = sorted(
        f for f in fold_dir.glob("graphs_outer*.pkl")
    )

    print(f"Found {len(fold_files)} fold files")

    folds_dict = {}
    for fold_path in fold_files:
        fold_name = fold_path.stem
        train, val, test, info = load_graphs_with_normalization(
            str(fold_path),
            normalize_method=normalize_method,
        )
        folds_dict[fold_name] = (train, val, test, info)

    return folds_dict
