#!/usr/bin/env python
"""
train_gatv2_grid.py

Train an interpretable GATv2-based GNN on Movie-HCP Ledoit-Wolf graphs
with a small hyperparameter grid search.

- Loads all CV folds (graphs_outer*_inner*.pkl).
- Flattens [subject, window] arrays into a list of PyG Data objects.
- Drops padding windows.
- Attaches subject_id to each window graph.
- For each config in HYPERPARAM_GRID:
    - Trains a GATv2 regressor across all folds.
    - Uses subject-level validation MSE for early stopping within each fold.
    - Records subject-level validation r and test r.
- After all configs and folds, reports the best config based on
  mean validation subject-level r across folds.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, GlobalAttention, global_mean_pool
from typing import List, Any

# ----------------------------
# 1. Config & hyperparameter grid
# ----------------------------

from pathlib import Path

OUTPUT_DIR = Path("../../results/gatv2/grid")
if not OUTPUT_DIR.parent.exists():
    # Fallback: try from project root
    OUTPUT_DIR = Path("results/gatv2/grid")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
EPOCHS = 20
USE_GLOBAL_ATTENTION_POOL = True  # if False, use simple mean pooling

# 👉 You can edit this grid however you like
HYPERPARAM_GRID = [
    {"hidden_dim": 32, "heads1": 2, "lr": 1e-3, "weight_decay": 1e-4},
    {"hidden_dim": 64, "heads1": 2, "lr": 1e-3, "weight_decay": 1e-4},
    {"hidden_dim": 32, "heads1": 4, "lr": 5e-4, "weight_decay": 1e-4},
]


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


def flatten_and_filter(arr2d) -> List[Any]:
    """
    arr2d: np.ndarray or list with shape [n_subjects, n_win],
           each entry is a PyG Data object.

    We:
      - drop padding windows (g.pad == True)
      - attach g.subject_id so we can aggregate per subject later

    Returns:
        flat list of non-padding Data objects.
    """
    flat = []
    for subj_idx, row in enumerate(arr2d):
        for g in row:
            # some Graphs have pad=True for padded windows
            if hasattr(g, "pad") and bool(g.pad):
                continue
            # Attach subject index as a tensor so PyG DataLoader can batch it
            g.subject_id = torch.tensor([subj_idx], dtype=torch.long)
            flat.append(g)
    return flat


def load_fold(path: str, weights_only: bool = False):
    """
    Load one fold file and return flat lists of train/val/test graphs.
    """
    print(f"Loading fold from {path}")
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

    return train_list, val_list, test_list


# ----------------------------
# 3. Interpretable GATv2 model
# ----------------------------

class GATv2Regressor(nn.Module):
    """
    Interpretable GATv2-based regressor:
    - Two GATv2Conv layers (edge-level attention).
    - Global attention pooling for node importance (optional).
    - MLP head for regression.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 32, heads1: int = 2,
                 out_dim: int = 1, use_global_attention_pool: bool = True):
        super().__init__()

        # First GATv2 layer
        self.gat1 = GATv2Conv(in_dim, hidden_dim, heads=heads1, concat=True)
        gat1_out_dim = hidden_dim * heads1

        # Second GATv2 layer (single head)
        self.gat2 = GATv2Conv(gat1_out_dim, hidden_dim, heads=1, concat=True)

        self.use_global_attention_pool = use_global_attention_pool
        if use_global_attention_pool:
            # gate_nn takes node features and returns scalar gates → attention over nodes
            self.gate_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            self.pool = GlobalAttention(gate_nn=self.gate_nn)
        else:
            # fallback: simple mean pooling
            self.pool = global_mean_pool

        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.gat1(x, edge_index).relu()
        x = self.gat2(x, edge_index).relu()

        if self.use_global_attention_pool:
            x = self.pool(x, batch)  # [batch_size, hidden_dim]
        else:
            x = self.pool(x, batch)

        x = self.lin1(x).relu()
        out = self.lin2(x).squeeze(-1)  # [batch_size]
        return out


# ----------------------------
# 4. Training & evaluation
# ----------------------------

def train_one_epoch(model, loader, optimizer, device):
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
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        num_samples += batch.num_graphs

    return total_loss / max(1, num_samples)


@torch.no_grad()
def eval_epoch(model, loader, device):
    """
    Returns:
        mse_win  : window-level MSE
        r_win    : window-level Pearson r
        mse_subj : subject-level MSE (after averaging windows per subject)
        r_subj   : subject-level Pearson r
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
        return float("nan"), float("nan"), float("nan"), float("nan")

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_subject_ids = torch.cat(all_subject_ids)

    # ----- window-level metrics -----
    mse_win = total_loss / num_samples

    if all_preds.numel() > 1:
        px = all_preds - all_preds.mean()
        py = all_targets - all_targets.mean()
        r_win = (px * py).sum() / (px.norm() * py.norm() + 1e-8)
        r_win = r_win.item()
    else:
        r_win = float("nan")

    # ----- subject-level aggregation -----
    subj_pred = {}
    subj_true = {}

    for p, t, sid in zip(all_preds, all_targets, all_subject_ids):
        sid = int(sid.item())
        if sid not in subj_pred:
            subj_pred[sid] = []
            subj_true[sid] = []
        subj_pred[sid].append(p.item())
        subj_true[sid].append(t.item())

    subj_pred_avg = []
    subj_true_avg = []
    for sid in subj_pred.keys():
        subj_pred_avg.append(sum(subj_pred[sid]) / len(subj_pred[sid]))
        subj_true_avg.append(sum(subj_true[sid]) / len(subj_true[sid]))

    subj_pred_avg = torch.tensor(subj_pred_avg)
    subj_true_avg = torch.tensor(subj_true_avg)

    mse_subj = nn.functional.mse_loss(subj_pred_avg, subj_true_avg).item()

    if subj_pred_avg.numel() > 1:
        px = subj_pred_avg - subj_pred_avg.mean()
        py = subj_true_avg - subj_true_avg.mean()
        r_subj = (px * py).sum() / (px.norm() * py.norm() + 1e-8)
        r_subj = r_subj.item()
    else:
        r_subj = float("nan")

    return mse_win, r_win, mse_subj, r_subj


# ----------------------------
# 5. Main: grid search over configs
# ----------------------------

def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    fold_dir = Path("../../data/folds_data")
    if not fold_dir.exists():
        # Fallback: try from project root
        fold_dir = Path("data/folds_data")

    if not fold_dir.exists():
        print(f"Error: Data directory not found")
        return

    fold_files = sorted(
        str(fold_dir / f)
        for f in os.listdir(fold_dir)
        if f.startswith("graphs_outer") and f.endswith(".pkl")
    )

    all_config_summaries = []

    for cfg_idx, cfg in enumerate(HYPERPARAM_GRID):
        print("\n====================================")
        print(f"Config {cfg_idx}: {cfg}")
        print("====================================")

        cfg_val_r_subj_list = []
        cfg_test_r_subj_list = []

        for fold_path in fold_files:
            print("\n------------------------------")
            print("Training on fold:", fold_path)
            print("------------------------------")

            train_graphs, val_graphs, test_graphs = load_fold(fold_path)

            in_dim = train_graphs[0].x.size(-1)
            print(f"Node feature dim: {in_dim}")

            train_loader = DataLoader(
                train_graphs, batch_size=BATCH_SIZE, shuffle=True,
                num_workers=0, pin_memory=torch.cuda.is_available()
            )
            val_loader   = DataLoader(
                val_graphs, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=0, pin_memory=torch.cuda.is_available()
            )
            test_loader  = DataLoader(
                test_graphs, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=0, pin_memory=torch.cuda.is_available()
            )

            model = GATv2Regressor(
                in_dim=in_dim,
                hidden_dim=cfg["hidden_dim"],
                heads1=cfg["heads1"],
                out_dim=1,
                use_global_attention_pool=USE_GLOBAL_ATTENTION_POOL,
            ).to(device)

            optimizer = optim.Adam(
                model.parameters(),
                lr=cfg["lr"],
                weight_decay=cfg["weight_decay"]
            )

            best_val_loss = float("inf")
            best_state = None

            for epoch in range(1, EPOCHS + 1):
                t0 = time.perf_counter()    
                train_loss = train_one_epoch(model, train_loader, optimizer, device)
                val_mse_win, val_r_win, val_mse_subj, val_r_subj = eval_epoch(model, val_loader, device)
                epoch_sec = time.perf_counter() - t0
                print(
                    f"[cfg{cfg_idx} | {os.path.basename(fold_path)}] "
                    f"Epoch {epoch:03d} | train MSE={train_loss:.4f} | "
                    f"val_win MSE={val_mse_win:.4f} | val_win r={val_r_win:.3f} | "
                    f"val_subj MSE={val_mse_subj:.4f} | val_subj r={val_r_subj:.3f} |" 
                    f"time={epoch_sec:.2f}s"
                )

                # Early stopping criterion: subject-level validation MSE
                if val_mse_subj < best_val_loss:
                    best_val_loss = val_mse_subj
                    best_state = model.state_dict()

            # Reload best model for this fold
            if best_state is not None:
                model.load_state_dict(best_state)

            # Recompute val metrics at best epoch
            val_mse_win, val_r_win, val_mse_subj, val_r_subj = eval_epoch(model, val_loader, device)
            # And compute test metrics
            test_mse_win, test_r_win, test_mse_subj, test_r_subj = eval_epoch(model, test_loader, device)

            print(
                f"[cfg{cfg_idx} | {os.path.basename(fold_path)}] "
                f"BEST val_subj MSE={val_mse_subj:.4f} | BEST val_subj r={val_r_subj:.3f}"
            )
            print(
                f"[cfg{cfg_idx} | {os.path.basename(fold_path)}] "
                f"Test window  MSE: {test_mse_win:.4f} | Test window  r: {test_r_win:.3f} | "
                f"Test subject MSE: {test_mse_subj:.4f} | Test subject r: {test_r_subj:.3f}"
            )

            # Save best model for this config+fold
            fold_name = os.path.basename(fold_path).replace(".pkl", "")
            model_path = OUTPUT_DIR / f"gatv2_cfg{cfg_idx}_best_{fold_name}.pt"
            if best_state is not None:
                torch.save(best_state, model_path)
                print(f"Saved best model for config {cfg_idx}, fold {fold_name} to {model_path}")

            cfg_val_r_subj_list.append(val_r_subj)
            cfg_test_r_subj_list.append(test_r_subj)

        # Aggregate over folds
        mean_val_r_subj = float(np.nanmean(cfg_val_r_subj_list))
        mean_test_r_subj = float(np.nanmean(cfg_test_r_subj_list))

        cfg_summary = {
            "config_id": cfg_idx,
            "hidden_dim": cfg["hidden_dim"],
            "heads1": cfg["heads1"],
            "lr": cfg["lr"],
            "weight_decay": cfg["weight_decay"],
            "mean_val_r_subj": mean_val_r_subj,
            "mean_test_r_subj": mean_test_r_subj,
            "per_fold_val_r_subj": cfg_val_r_subj_list,
            "per_fold_test_r_subj": cfg_test_r_subj_list,
        }

        all_config_summaries.append(cfg_summary)
        print("\n===== Config", cfg_idx, "summary =====")
        print(cfg_summary)

    # Choose best config by mean validation subject-level r
    mean_val_rs = [c["mean_val_r_subj"] for c in all_config_summaries]
    best_idx = int(np.nanargmax(mean_val_rs))
    best_cfg = all_config_summaries[best_idx]

    print("\n==============================")
    print("BEST CONFIG (by mean val_subj r):")
    print(best_cfg)
    print("==============================")

    # Save all configs summary to JSON
    summary_path = OUTPUT_DIR / "gatv2_grid_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_config_summaries, f, indent=2)
    print(f"\nSaved grid search summary to {summary_path}")


if __name__ == "__main__":
    main()
