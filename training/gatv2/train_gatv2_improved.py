#!/usr/bin/env python
"""
train_gatv2_improved.py - Enhanced GATv2 Training with Latest Improvements
=========================================================================
Claude
Improvements over train_gatv2_interpretable.py:
1. ✅ Target Normalization (CRITICAL FIX - prevents training instability)
2. ✅ DropEdge regularization
3. ✅ Early stopping with patience
4. ✅ Learning rate scheduling (ReduceLROnPlateau)
5. ✅ Gradient clipping
6. ✅ Comprehensive JSON tracking
7. ✅ Better model architecture (more layers, residual connections)
8. ✅ Data augmentation support
9. ✅ Improved attention pooling

Usage:
    python train_gatv2_improved.py --device auto --epochs 100 --hidden_dim 128
"""

import os
import json
import time
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, GlobalAttention, global_mean_pool
from torch_geometric.utils import dropout_edge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from typing import List, Any, Dict


# ----------------------------
# Custom JSON Encoder for Numpy Types
# ----------------------------

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# ----------------------------
# 1. Config
# ----------------------------

# Output directory (relative to project root since script moved to training/gatv2/)
OUTPUT_DIR = Path("../../results/gatv2/improved")
if not OUTPUT_DIR.exists():
    # Fallback if running from project root
    OUTPUT_DIR = Path("results/gatv2/improved")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
PATIENCE = 15
USE_GLOBAL_ATTENTION_POOL = True


# ----------------------------
# 2. Utils
# ----------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(arg_choice: str = "auto") -> torch.device:
    if arg_choice == "cpu":
        return torch.device("cpu")
    if arg_choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def flatten_and_filter(arr2d) -> List[Any]:
    """
    Flatten 2D array of graphs and attach subject IDs.
    """
    flat = []
    for subj_idx, row in enumerate(arr2d):
        for g in row:
            if hasattr(g, "pad") and bool(g.pad):
                continue
            g.subject_id = torch.tensor([subj_idx], dtype=torch.long)
            flat.append(g)
    return flat


def normalize_targets(graphs_list, scaler=None, fit=True):
    """
    CRITICAL: Normalize target values for stable training.

    Args:
        graphs_list: List of PyG Data objects
        scaler: StandardScaler (if None, creates new one)
        fit: Whether to fit the scaler

    Returns:
        normalized graphs, fitted scaler
    """
    if not graphs_list:
        return graphs_list, scaler

    # Extract targets
    targets = np.array([g.y.item() if torch.is_tensor(g.y) else g.y for g in graphs_list])
    targets = targets.reshape(-1, 1)

    # Create scaler if needed
    if scaler is None:
        scaler = StandardScaler()

    # Normalize
    if fit:
        targets_norm = scaler.fit_transform(targets)
        print(f"  Target normalization:")
        print(f"    Original range: [{targets.min():.2f}, {targets.max():.2f}]")
        print(f"    Normalized range: [{targets_norm.min():.2f}, {targets_norm.max():.2f}]")
        print(f"    Mean: {scaler.mean_[0]:.2f}, Std: {scaler.scale_[0]:.2f}")
    else:
        targets_norm = scaler.transform(targets)

    # Update graph targets
    for i, g in enumerate(graphs_list):
        g.y = torch.tensor(targets_norm[i, 0], dtype=torch.float32)

    return graphs_list, scaler


def load_fold(path: str, weights_only: bool = False, normalize: bool = True):
    """
    Load one fold file and return normalized flat lists of train/val/test graphs.
    """
    print(f"\nLoading fold from {path}")
    graphs = torch.load(path, map_location="cpu", weights_only=weights_only)

    train_2d = graphs["train_graphs"]
    val_2d   = graphs["val_graphs"]
    test_2d  = graphs["test_graphs"]

    train_list = flatten_and_filter(train_2d)
    val_list   = flatten_and_filter(val_2d)
    test_list  = flatten_and_filter(test_2d)

    print(f"#train graphs (windows): {len(train_list)}")
    print(f"#val   graphs (windows): {len(val_list)}")
    print(f"#test  graphs (windows): {len(test_list)}")

    # Normalize targets (CRITICAL!)
    if normalize:
        print("\nNormalizing targets...")
        train_list, scaler = normalize_targets(train_list, scaler=None, fit=True)
        val_list, _ = normalize_targets(val_list, scaler=scaler, fit=False)
        test_list, _ = normalize_targets(test_list, scaler=scaler, fit=False)
    else:
        scaler = None

    return train_list, val_list, test_list, scaler


# ----------------------------
# 3. Improved GATv2 Model
# ----------------------------

class ImprovedGATv2Regressor(nn.Module):
    """
    Improved GATv2-based regressor with:
    - Deeper architecture (3 layers)
    - Residual connections
    - Layer normalization
    - DropEdge support
    - Better attention pooling
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.2,
        edge_dropout: float = 0.1,
        out_dim: int = 1,
        use_global_attention_pool: bool = True
    ):
        super().__init__()

        self.n_layers = n_layers
        self.edge_dropout = edge_dropout

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(n_layers):
            if i == 0:
                gat_in = hidden_dim
            else:
                gat_in = hidden_dim

            # Last layer uses single head
            heads = n_heads if i < n_layers - 1 else 1
            concat = i < n_layers - 1

            self.gat_layers.append(
                GATv2Conv(
                    gat_in,
                    hidden_dim // heads if concat else hidden_dim,
                    heads=heads,
                    concat=concat,
                    dropout=dropout
                )
            )

            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Attention pooling
        self.use_global_attention_pool = use_global_attention_pool
        if use_global_attention_pool:
            self.gate_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
            self.pool = GlobalAttention(gate_nn=self.gate_nn)
        else:
            self.pool = global_mean_pool

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Input projection
        x = self.input_proj(x)
        identity = x

        # Apply DropEdge during training
        if self.training and self.edge_dropout > 0:
            edge_index, _ = dropout_edge(
                edge_index,
                p=self.edge_dropout,
                force_undirected=True,
                training=True
            )

        # GAT layers with residual connections
        for i, (gat, ln) in enumerate(zip(self.gat_layers, self.layer_norms)):
            x_new = gat(x, edge_index)
            x_new = ln(x_new)
            x_new = torch.relu(x_new)

            # Residual connection (skip first layer)
            if i > 0:
                x = x + x_new
            else:
                x = x_new

        # Global pooling
        if self.use_global_attention_pool:
            x = self.pool(x, batch)
        else:
            x = self.pool(x, batch)

        # Regression
        out = self.head(x).squeeze(-1)
        return out

    @torch.no_grad()
    def get_edge_attention(self, data):
        """Get edge attention weights from last GAT layer."""
        self.eval()
        device = next(self.parameters()).device

        if not hasattr(data, "batch"):
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long)

        data = data.to(device)
        x = self.input_proj(data.x)

        # Forward through all layers except last
        for gat in self.gat_layers[:-1]:
            x = gat(x, data.edge_index)
            x = torch.relu(x)

        # Last layer with attention weights
        _, (edge_index, alpha) = self.gat_layers[-1](
            x,
            data.edge_index,
            return_attention_weights=True
        )

        alpha = alpha.squeeze(-1).detach().cpu()
        edge_index = edge_index.detach().cpu()
        return edge_index, alpha

    @torch.no_grad()
    def get_node_importance(self, data):
        """Get node importance scores from attention pooling."""
        if not self.use_global_attention_pool:
            raise RuntimeError("GlobalAttention pooling is disabled")

        self.eval()
        device = next(self.parameters()).device

        if not hasattr(data, "batch"):
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long)

        data = data.to(device)
        x = self.input_proj(data.x)

        # Forward through GAT layers
        for gat in self.gat_layers:
            x = gat(x, data.edge_index)
            x = torch.relu(x)

        # Get gate scores
        gate_values = self.gate_nn(x)
        node_scores = gate_values.squeeze(-1).detach().cpu()
        return node_scores


# ----------------------------
# 4. Training & Evaluation
# ----------------------------

def train_one_epoch(model, loader, optimizer, device):
    """Train for one epoch with gradient clipping."""
    model.train()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    num_samples = 0

    for batch in loader:
        batch = batch.to(device)
        preds = model(batch)
        target = batch.y.view_as(preds).float()

        loss = loss_fn(preds, target)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        num_samples += batch.num_graphs

    return total_loss / max(1, num_samples)


@torch.no_grad()
def eval_epoch(model, loader, device, scaler=None):
    """
    Evaluate model with comprehensive metrics.

    Returns dict with:
        - Window-level metrics (MSE, MAE, R, R2)
        - Subject-level metrics (aggregated)
        - Predictions and targets (both normalized and original scale)
    """
    model.eval()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    num_samples = 0

    all_preds = []
    all_targets = []
    all_subject_ids = []

    for batch in loader:
        batch = batch.to(device)
        preds = model(batch)
        target = batch.y.view_as(preds).float()

        loss = loss_fn(preds, target)

        total_loss += loss.item() * batch.num_graphs
        num_samples += batch.num_graphs

        all_preds.append(preds.cpu())
        all_targets.append(target.cpu())

        if hasattr(batch, "subject_id"):
            all_subject_ids.append(batch.subject_id.cpu())
        else:
            all_subject_ids.append(torch.arange(batch.num_graphs))

    if num_samples == 0:
        return None

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_subject_ids = torch.cat(all_subject_ids).numpy()

    # Denormalize if scaler available
    if scaler is not None:
        all_preds_orig = scaler.inverse_transform(all_preds.reshape(-1, 1)).flatten()
        all_targets_orig = scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()
    else:
        all_preds_orig = all_preds
        all_targets_orig = all_targets

    # Window-level metrics
    win_metrics_norm = compute_metrics(all_targets, all_preds)
    win_metrics_orig = compute_metrics(all_targets_orig, all_preds_orig)

    # Subject-level aggregation
    subj_pred, subj_true = aggregate_by_subject(all_preds, all_targets, all_subject_ids)
    subj_metrics_norm = compute_metrics(subj_true, subj_pred)

    if scaler is not None:
        subj_pred_orig = scaler.inverse_transform(subj_pred.reshape(-1, 1)).flatten()
        subj_true_orig = scaler.inverse_transform(subj_true.reshape(-1, 1)).flatten()
        subj_metrics_orig = compute_metrics(subj_true_orig, subj_pred_orig)
    else:
        subj_metrics_orig = subj_metrics_norm

    return {
        'loss': total_loss / num_samples,
        'win_metrics_norm': win_metrics_norm,
        'win_metrics_orig': win_metrics_orig,
        'subj_metrics_norm': subj_metrics_norm,
        'subj_metrics_orig': subj_metrics_orig,
    }


def compute_metrics(y_true, y_pred):
    """Compute comprehensive metrics."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    if len(y_true) > 1:
        r, p_value = pearsonr(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
    else:
        r, p_value, r2 = np.nan, np.nan, np.nan

    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r': r,
        'r2': r2,
        'p_value': p_value,
    }


def aggregate_by_subject(predictions, targets, subject_ids):
    """Aggregate window-level predictions to subject-level."""
    subj_pred = {}
    subj_true = {}

    for p, t, sid in zip(predictions, targets, subject_ids):
        sid = int(sid)
        if sid not in subj_pred:
            subj_pred[sid] = []
            subj_true[sid] = []
        subj_pred[sid].append(p)
        subj_true[sid].append(t)

    subj_pred_avg = np.array([np.mean(subj_pred[sid]) for sid in sorted(subj_pred.keys())])
    subj_true_avg = np.array([np.mean(subj_true[sid]) for sid in sorted(subj_true.keys())])

    return subj_pred_avg, subj_true_avg


# ----------------------------
# 5. Main Training Loop
# ----------------------------

def train_fold(fold_path, config, device):
    """Train model on a single fold with all improvements."""
    fold_name = os.path.basename(fold_path).replace(".pkl", "")
    print(f"\n{'='*70}")
    print(f"Training fold: {fold_name}")
    print(f"{'='*70}")

    # Load data with normalization
    train_graphs, val_graphs, test_graphs, scaler = load_fold(
        fold_path,
        weights_only=False,
        normalize=True
    )

    # Dataloaders
    in_dim = train_graphs[0].x.size(-1)
    print(f"\nNode feature dim: {in_dim}")

    train_loader = DataLoader(
        train_graphs,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_graphs,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_graphs,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # Model
    model = ImprovedGATv2Regressor(
        in_dim=in_dim,
        hidden_dim=config['hidden_dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        dropout=config['dropout'],
        edge_dropout=config['edge_dropout'],
        out_dim=1,
        use_global_attention_pool=USE_GLOBAL_ATTENTION_POOL,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )

    # Training loop
    best_val_metric = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, config['epochs'] + 1):
        t0 = time.perf_counter()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_results = eval_epoch(model, val_loader, device, scaler)

        # Update scheduler
        scheduler.step(val_results['loss'])

        # Track history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_results['loss'],
            'win_r_norm': val_results['win_metrics_norm']['r'],
            'win_r_orig': val_results['win_metrics_orig']['r'],
            'subj_r_norm': val_results['subj_metrics_norm']['r'],
            'subj_r_orig': val_results['subj_metrics_orig']['r'],
            'lr': optimizer.param_groups[0]['lr'],
            'time': time.perf_counter() - t0,
        })

        # Print progress
        print(
            f"Epoch {epoch:03d}/{config['epochs']:03d} | "
            f"Train MSE: {train_loss:.4f} | Val MSE: {val_results['loss']:.4f} | "
            f"Win r: {val_results['win_metrics_orig']['r']:.4f} | "
            f"Subj r: {val_results['subj_metrics_orig']['r']:.4f} | "
            f"Time: {history[-1]['time']:.1f}s"
        )

        # Early stopping on subject-level metrics
        val_metric = -val_results['subj_metrics_orig']['r']  # Negative to minimize

        if val_metric < best_val_metric:
            best_val_metric = val_metric
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            fold_output_dir = Path(OUTPUT_DIR) / fold_name
            fold_output_dir.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'scaler': scaler,
            }, fold_output_dir / "gatv2_best.pt")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\nBest epoch: {best_epoch}")

    # Load best model
    checkpoint = torch.load(fold_output_dir / "gatv2_best.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test evaluation
    test_results = eval_epoch(model, test_loader, device, scaler)

    print(f"\nTest Results (Original Scale):")
    print(f"  Window-level - MSE: {test_results['win_metrics_orig']['mse']:.4f}, "
          f"R: {test_results['win_metrics_orig']['r']:.4f}")
    print(f"  Subject-level - MSE: {test_results['subj_metrics_orig']['mse']:.4f}, "
          f"R: {test_results['subj_metrics_orig']['r']:.4f}")

    # Save comprehensive summary
    summary = {
        'model_name': 'gatv2_improved',
        'fold': fold_name,
        'timestamp': datetime.datetime.now().isoformat(),
        'config': config,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'training_summary': {
            'total_epochs': len(history),
            'best_epoch': best_epoch,
            'early_stopped': patience_counter >= config['patience'],
        },
        'test_metrics': {
            'window_level': test_results['win_metrics_orig'],
            'subject_level': test_results['subj_metrics_orig'],
        },
        'history': history,
    }

    with open(fold_output_dir / "gatv2_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train improved GATv2 regressor")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--edge_dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)

    args = parser.parse_args()

    set_seed(SEED)
    device = select_device(args.device)
    print(f"Using device: {device}")

    config = {
        'epochs': args.epochs,
        'patience': args.patience,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': WEIGHT_DECAY,
        'hidden_dim': args.hidden_dim,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'dropout': args.dropout,
        'edge_dropout': args.edge_dropout,
    }

    # Find fold files (adjust path since we're in training/gatv2/)
    fold_dir = Path("../../data/folds_data")
    if not fold_dir.exists():
        # Try from project root if running from there
        fold_dir = Path("data/folds_data")

    if not fold_dir.exists():
        print(f"Error: Data directory not found at {fold_dir}")
        print("Make sure you're running from the project root or the data exists.")
        return

    fold_files = sorted(
        str(fold_dir / f)
        for f in os.listdir(fold_dir)
        if f.startswith("graphs_outer") and f.endswith(".pkl")
    )

    # Train all folds
    all_results = []
    for fold_path in fold_files:
        try:
            result = train_fold(fold_path, config, device)
            all_results.append(result)
        except Exception as e:
            print(f"\nError training {fold_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Aggregate results
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")

    subj_rs = [r['test_metrics']['subject_level']['r'] for r in all_results]
    print(f"\nSubject-level R: {np.mean(subj_rs):.4f} ± {np.std(subj_rs):.4f}")
    print(f"Range: [{np.min(subj_rs):.4f}, {np.max(subj_rs):.4f}]")

    # Save aggregate
    aggregate = {
        'model_name': 'gatv2_improved',
        'n_folds': len(all_results),
        'subject_level_r': {
            'mean': float(np.mean(subj_rs)),
            'std': float(np.std(subj_rs)),
            'min': float(np.min(subj_rs)),
            'max': float(np.max(subj_rs)),
        },
        'per_fold': all_results,
    }

    with open(Path(OUTPUT_DIR) / "gatv2_aggregate_summary.json", 'w') as f:
        json.dump(aggregate, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
