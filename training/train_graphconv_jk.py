#!/usr/bin/env python
"""
Train a Jumping-Knowledge GraphConv regressor across CV folds.
Uses GraphConv layers with residual connections and JK aggregation.

Examples:
  # Full argument example
  python training/train_graphconv_jk.py \
    --fold_dir data/folds_data \
    --output_dir results/graphconv_jk \
    --run_name graphconv_jk_full \
    --device cuda \
    --seed 42 \
    --normalize_method standard \
    --epochs 30 \
    --patience 6 \
    --batch_size 16 \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --early_stop_metric subj_mse \
    --hidden_dim 128 \
    --num_layers 4 \
    --dropout 0.3 \
    --jk_mode cat \
    --pool attention

  # Single-fold quick run
  python training/train_graphconv_jk.py --fold_name graphs_outer1_inner1

Arguments:
  --fold_dir: directory with fold .pkl files
  --fold_name: specific fold name (mutually exclusive with --fold_indices)
  --fold_indices: list of fold indices to use
  --output_dir: results directory
  --run_name: subfolder name for this run
  --device: cuda | cpu | auto
  --seed: random seed
  --normalize_method: standard | minmax | robust
  --epochs: training epochs per fold
  --patience: early-stop patience
  --batch_size: batch size
  --lr: learning rate
  --weight_decay: weight decay
  --early_stop_metric: subj_mse | subj_r | win_mse | win_r
  --hidden_dim: hidden dimension size
  --num_layers: number of GraphConv layers
  --dropout: dropout for hidden layers/head
  --jk_mode: last | sum | cat
  --pool: mean | attention
  --no_edge_weight: disable edge weights (default uses edge weights)
  --no_residual: disable residual connections
"""

import argparse
import datetime
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GraphConv, GlobalAttention, global_mean_pool

# Add project root to path so we can import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.data_utils import (
    aggregate_window_predictions,
    compute_metrics,
    create_dataloaders,
    load_graphs_with_normalization,
)


class JKGraphConvRegressor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.3,
        jk_mode: str = "cat",
        pool: str = "attention",
        use_edge_weight: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if jk_mode not in {"last", "sum", "cat"}:
            raise ValueError(f"Unknown jk_mode: {jk_mode}")
        if pool not in {"mean", "attention"}:
            raise ValueError(f"Unknown pool: {pool}")

        self.use_edge_weight = use_edge_weight
        self.residual = residual
        self.jk_mode = jk_mode
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_ch = in_dim if layer_idx == 0 else hidden_dim
            self.convs.append(GraphConv(in_ch, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        jk_dim = hidden_dim * num_layers if jk_mode == "cat" else hidden_dim
        if pool == "attention":
            gate_nn = nn.Sequential(
                nn.Linear(jk_dim, jk_dim),
                nn.ReLU(),
                nn.Linear(jk_dim, 1),
            )
            self.pool = GlobalAttention(gate_nn=gate_nn)
        else:
            self.pool = None

        self.head = nn.Sequential(
            nn.Linear(jk_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None

        edge_weight = None
        if self.use_edge_weight and edge_attr is not None:
            edge_weight = edge_attr.squeeze(-1) if edge_attr.dim() > 1 else edge_attr

        layer_outputs = []
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h_in = h
            h = conv(h, edge_index, edge_weight=edge_weight)
            h = F.relu(h)
            h = norm(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.residual and h_in.shape == h.shape:
                h = h + h_in
            layer_outputs.append(h)

        if self.jk_mode == "cat":
            h = torch.cat(layer_outputs, dim=-1)
        elif self.jk_mode == "sum":
            h = torch.stack(layer_outputs, dim=0).sum(dim=0)
        else:
            h = layer_outputs[-1]

        if self.pool is None:
            h = global_mean_pool(h, batch)
        else:
            h = self.pool(h, batch)

        out = self.head(h).squeeze(-1)
        return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    num_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        preds = model(batch)
        loss = criterion(preds, batch.y.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        num_samples += batch.num_graphs

    return total_loss / max(1, num_samples)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    scaler=None,
) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_targets = []
    all_subj_ids = []

    for batch in loader:
        batch = batch.to(device)
        preds = model(batch)
        all_preds.append(preds.cpu())
        all_targets.append(batch.y.cpu())
        if hasattr(batch, "subject_id"):
            all_subj_ids.append(batch.subject_id.cpu())

    predictions = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    metrics = {}
    metrics.update(compute_metrics(predictions, targets, prefix="win_"))

    if all_subj_ids:
        subject_ids = torch.cat(all_subj_ids).numpy().flatten()
        subj_preds, _ = aggregate_window_predictions(predictions, subject_ids)
        subj_targets, _ = aggregate_window_predictions(targets, subject_ids)
        metrics.update(compute_metrics(subj_preds, subj_targets, prefix="subj_"))

    if scaler is not None:
        preds_raw = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        targets_raw = scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
        metrics.update(compute_metrics(preds_raw, targets_raw, prefix="win_raw_"))
        if all_subj_ids:
            subject_ids = torch.cat(all_subj_ids).numpy().flatten()
            subj_preds_raw, _ = aggregate_window_predictions(preds_raw, subject_ids)
            subj_targets_raw, _ = aggregate_window_predictions(targets_raw, subject_ids)
            metrics.update(compute_metrics(subj_preds_raw, subj_targets_raw, prefix="subj_raw_"))

    return metrics


def metric_mode(metric_name: str) -> str:
    metric = metric_name.lower()
    if "mse" in metric or "mae" in metric or "loss" in metric:
        return "min"
    return "max"


def get_metric(metrics: Dict[str, float], primary: str, fallback: str) -> float:
    value = metrics.get(primary)
    if value is None or np.isnan(value):
        value = metrics.get(fallback, float("nan"))
    return value


def fallback_metric(primary: str) -> str:
    if primary.endswith("_r"):
        return "win_r"
    return "win_mse"


def collect_folds(
    fold_dir: Path,
    fold_name: str,
    fold_indices: List[int],
) -> Tuple[List[Path], str]:
    if fold_name:
        fold_paths = [fold_dir / f"{fold_name}.pkl"]
        return fold_paths, f"single_fold_{fold_name}"

    all_fold_paths = sorted(fold_dir.glob("graphs_outer*.pkl"))
    if not all_fold_paths:
        raise FileNotFoundError(f"No fold files found in {fold_dir}")

    if fold_indices is not None:
        fold_paths = []
        for idx in fold_indices:
            if idx < 0 or idx >= len(all_fold_paths):
                raise ValueError(f"Fold index {idx} out of range (0-{len(all_fold_paths)-1})")
            fold_paths.append(all_fold_paths[idx])
        fold_indices_str = "_".join(map(str, fold_indices))
        return fold_paths, f"folds_{len(fold_paths)}_indices_{fold_indices_str}"

    return all_fold_paths, f"all_{len(all_fold_paths)}_folds"


def summarize_metric(values: List[float]) -> Dict[str, float]:
    return {
        "mean": float(np.nanmean(values)) if values else float("nan"),
        "std": float(np.nanstd(values)) if values else float("nan"),
    }


def train_fold(
    fold_path: Path,
    config: Dict,
    device: torch.device,
) -> Dict:
    train_graphs, val_graphs, test_graphs, info = load_graphs_with_normalization(
        str(fold_path),
        normalize_method=config["normalize_method"],
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        train_graphs,
        val_graphs,
        test_graphs,
        batch_size=config["batch_size"],
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    in_dim = train_graphs[0].x.size(-1)
    model = JKGraphConvRegressor(
        in_dim=in_dim,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        jk_mode=config["jk_mode"],
        pool=config["pool"],
        use_edge_weight=config["use_edge_weight"],
        residual=config["residual"],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.MSELoss()

    best_metric = float("inf") if metric_mode(config["early_stop_metric"]) == "min" else -float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(config["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device, scaler=info.get("scaler"))

        fallback = fallback_metric(config["early_stop_metric"])
        current_metric = get_metric(val_metrics, config["early_stop_metric"], fallback)
        if metric_mode(config["early_stop_metric"]) == "max":
            improved = current_metric > best_metric + 1e-5
        else:
            improved = current_metric < best_metric - 1e-5

        if improved:
            best_metric = current_metric
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = evaluate(model, val_loader, device, scaler=info.get("scaler"))
    test_metrics = evaluate(model, test_loader, device, scaler=info.get("scaler"))

    val_r = get_metric(val_metrics, "subj_r", "win_r")
    test_r = get_metric(test_metrics, "subj_r", "win_r")
    print(f"Fold {fold_path.name}: val_r={val_r:.4f} | test_r={test_r:.4f}")

    return {
        "fold": fold_path.name,
        "train_loss": float(train_loss),
        "best_val_metric": float(best_metric),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train JK-GraphConv across folds")
    parser.add_argument("--fold_dir", type=str, default="data/folds_data")
    parser.add_argument("--fold_name", type=str, default=None)
    parser.add_argument("--fold_indices", nargs="+", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="results/graphconv_jk")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize_method", type=str, default="standard",
                        choices=["standard", "minmax", "robust"])

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--early_stop_metric", type=str, default="subj_mse",
                        choices=["subj_mse", "subj_r", "win_mse", "win_r"])

    # Model hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--jk_mode", type=str, default="cat", choices=["last", "sum", "cat"])
    parser.add_argument("--pool", type=str, default="attention", choices=["mean", "attention"])
    parser.add_argument("--no_edge_weight", action="store_false", dest="use_edge_weight")
    parser.set_defaults(use_edge_weight=True)
    parser.add_argument("--no_residual", action="store_false", dest="residual")
    parser.set_defaults(residual=True)

    args = parser.parse_args()

    device = select_device(args.device)
    set_seed(args.seed)

    fold_dir = Path(args.fold_dir)
    fold_paths, fold_config_str = collect_folds(fold_dir, args.fold_name, args.fold_indices)

    run_name = args.run_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / fold_config_str / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config["output_dir"] = str(output_dir)

    all_results = []
    for fold_path in fold_paths:
        print(f"Training JK-GraphConv on {fold_path.name}")
        fold_result = train_fold(fold_path, config, device)
        all_results.append(fold_result)

    val_subj_r = [r["val_metrics"].get("subj_r") for r in all_results]
    val_subj_mse = [r["val_metrics"].get("subj_mse") for r in all_results]
    test_subj_r = [r["test_metrics"].get("subj_r") for r in all_results]
    test_subj_mse = [r["test_metrics"].get("subj_mse") for r in all_results]

    aggregate = {
        "val_subj_r": summarize_metric(val_subj_r),
        "val_subj_mse": summarize_metric(val_subj_mse),
        "test_subj_r": summarize_metric(test_subj_r),
        "test_subj_mse": summarize_metric(test_subj_mse),
    }

    results = {
        "model": "jk_graphconv",
        "config": config,
        "n_folds": len(fold_paths),
        "folds": [p.name for p in fold_paths],
        "aggregate": aggregate,
        "per_fold": all_results,
    }

    results_path = output_dir / "jk_graphconv_cv_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
