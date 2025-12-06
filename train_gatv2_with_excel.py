#!/usr/bin/env python
"""
train_gatv2_interpretable.py

Train an interpretable GATv2-based GNN on Movie-HCP Ledoit-Wolf graphs.

- Loads all CV folds (graphs_outer*_inner*.pkl).
- Flattens [subject, window] arrays into a list of PyG Data objects.
- Drops padding windows.
- Attaches subject_id to each window graph.
- Trains a GATv2 regressor to predict cognitive scores.
- Reports both window-level and subject-level metrics.
- Saves the best model (per fold) based on subject-level validation MSE.
- NEW: saves validation (and test) predictions to Excel files.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd  # <-- NEW

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, GlobalAttention, global_mean_pool
from typing import List, Any

# ----------------------------
# 1. Config
# ----------------------------

OUTPUT_DIR = "results_gatv2_interpretable"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PREDICTION_DIR = "results_gatv2_predictions"   # <-- NEW
os.makedirs(PREDICTION_DIR, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 20
USE_GLOBAL_ATTENTION_POOL = True  # if False, use simple mean pooling


# ----------------------------
# 2. Utils
# ----------------------------

def set_seed(seed: int = 42):
    import numpy as np
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
            if hasattr(g, "pad") and bool(g.pad):
                continue
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
    def __init__(self, in_dim: int, hidden_dim: int = 32, out_dim: int = 1,
                 use_global_attention_pool: bool = True):
        super().__init__()

        heads1 = 2
        self.gat1 = GATv2Conv(in_dim, hidden_dim, heads=heads1, concat=True)
        gat1_out_dim = hidden_dim * heads1

        self.gat2 = GATv2Conv(gat1_out_dim, hidden_dim, heads=1, concat=True)

        self.use_global_attention_pool = use_global_attention_pool
        if use_global_attention_pool:
            self.gate_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            self.pool = GlobalAttention(gate_nn=self.gate_nn)
        else:
            self.pool = global_mean_pool

        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.gat1(x, edge_index).relu()
        x = self.gat2(x, edge_index).relu()

        if self.use_global_attention_pool:
            x = self.pool(x, batch)
        else:
            x = self.pool(x, batch)

        x = self.lin1(x).relu()
        out = self.lin2(x).squeeze(-1)
        return out

    @torch.no_grad()
    def get_edge_attention(self, data):
        self.eval()
        device = next(self.parameters()).device

        if not hasattr(data, "batch"):
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long)

        data = data.to(device)
        x = self.gat1(data.x, data.edge_index).relu()
        _, (edge_index, alpha) = self.gat2(
            x,
            data.edge_index,
            return_attention_weights=True
        )
        alpha = alpha.squeeze(-1).detach().cpu()
        edge_index = edge_index.detach().cpu()
        return edge_index, alpha

    @torch.no_grad()
    def get_node_importance(self, data):
        if not self.use_global_attention_pool:
            raise RuntimeError("GlobalAttention pooling is disabled; cannot compute node importance.")

        self.eval()
        device = next(self.parameters()).device

        if not hasattr(data, "batch"):
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long)

        data = data.to(device)
        x = self.gat1(data.x, data.edge_index).relu()
        x = self.gat2(x, data.edge_index).relu()

        gate_values = self.gate_nn(x)
        node_scores = gate_values.squeeze(-1).detach().cpu()
        return node_scores


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

    mse_win = total_loss / num_samples

    if all_preds.numel() > 1:
        px = all_preds - all_preds.mean()
        py = all_targets - all_targets.mean()
        r_win = (px * py).sum() / (px.norm() * py.norm() + 1e-8)
        r_win = r_win.item()
    else:
        r_win = float("nan")

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


@torch.no_grad()
def save_split_predictions(
    model,
    loader,
    device,
    fold_name: str,
    split_name: str,
    output_dir: str,
):
    """
    Run model on all graphs in loader and save predictions to Excel.

    - Sheet 'windows': one row per window graph (subject_id, y_true, y_pred)
    - Sheet 'subjects': aggregated per subject (mean y_true, mean y_pred, n_windows)
    """
    model.eval()

    all_preds = []
    all_targets = []
    all_subject_ids = []

    for batch in loader:
        batch = batch.to(device)
        preds = model(batch)

        all_preds.append(preds.cpu())
        all_targets.append(batch.y.view_as(preds).cpu())

        if hasattr(batch, "subject_id"):
            all_subject_ids.append(batch.subject_id.cpu())
        else:
            all_subject_ids.append(torch.arange(batch.num_graphs))

    if not all_preds:
        print(f"No data for {split_name} on fold {fold_name}, skipping prediction export.")
        return

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_subject_ids = torch.cat(all_subject_ids).numpy()

    # Window-level dataframe
    df_win = pd.DataFrame({
        "fold": fold_name,
        "split": split_name,
        "subject_id": all_subject_ids,
        "y_true": all_targets,
        "y_pred": all_preds,
    })

    # Subject-level aggregated dataframe
    df_subj = (
        df_win.groupby("subject_id")
              .agg(
                  y_true=("y_true", "mean"),
                  y_pred=("y_pred", "mean"),
                  n_windows=("y_true", "size"),
              )
              .reset_index()
    )
    df_subj.insert(0, "fold", fold_name)
    df_subj.insert(1, "split", split_name)

    out_path = os.path.join(output_dir, f"{split_name}_predictions_{fold_name}.xlsx")
    with pd.ExcelWriter(out_path) as writer:
        df_win.to_excel(writer, sheet_name="windows", index=False)
        df_subj.to_excel(writer, sheet_name="subjects", index=False)

    print(f"Saved {split_name} predictions for {fold_name} to {out_path}")


# ----------------------------
# 5. Main
# ----------------------------

def main():
    set_seed(SEED)

    fold_dir = "data/folds_data"
    fold_files = sorted(
        os.path.join(fold_dir, f)
        for f in os.listdir(fold_dir)
        if f.startswith("graphs_outer") and f.endswith(".pkl")
    )

    all_test_results = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    for fold_path in fold_files:
        print("\n==============================")
        print("Training on fold:", fold_path)
        print("==============================")

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
            hidden_dim=32,
            out_dim=1,
            use_global_attention_pool=USE_GLOBAL_ATTENTION_POOL,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(1, EPOCHS + 1):
            t0 = time.perf_counter()
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_mse_win, val_r_win, val_mse_subj, val_r_subj = eval_epoch(model, val_loader, device)
            epoch_sec = time.perf_counter() - t0
            print(
                f"[{os.path.basename(fold_path)}] "
                f"Epoch {epoch:03d} | train MSE={train_loss:.4f} | "
                f"val_win MSE={val_mse_win:.4f} | val_win r={val_r_win:.3f} | "
                f"val_subj MSE={val_mse_subj:.4f} | val_subj r={val_r_subj:.3f} |"
                f"time={epoch_sec:.2f}s"
            )

            if val_mse_subj < best_val_loss:
                best_val_loss = val_mse_subj
                best_state = model.state_dict()

        if best_state is not None:
            model.load_state_dict(best_state)

        test_mse_win, test_r_win, test_mse_subj, test_r_subj = eval_epoch(model, test_loader, device)
        print(
            f"[{os.path.basename(fold_path)}] "
            f"Test window  MSE: {test_mse_win:.4f} | Test window  r: {test_r_win:.3f} | "
            f"Test subject MSE: {test_mse_subj:.4f} | Test subject r: {test_r_subj:.3f}"
        )

        fold_name = os.path.basename(fold_path).replace(".pkl", "")
        model_path = os.path.join(OUTPUT_DIR, f"gatv2_best_{fold_name}.pt")
        if best_state is not None:
            torch.save(best_state, model_path)
            print(f"Saved best model for {fold_name} to {model_path}")

        # NEW: save predictions for validation and test splits
        save_split_predictions(model, val_loader,  device, fold_name, "val",  PREDICTION_DIR)
        save_split_predictions(model, test_loader, device, fold_name, "test", PREDICTION_DIR)

        all_test_results.append(
            {
                "fold": os.path.basename(fold_path),
                "test_mse_win": test_mse_win,
                "test_r_win": test_r_win,
                "test_mse_subj": test_mse_subj,
                "test_r_subj": test_r_subj,
            }
        )

    print("\n=== Summary over folds ===")
    for res in all_test_results:
        print(res)

    summary_path = os.path.join(OUTPUT_DIR, "gatv2_test_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_test_results, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
