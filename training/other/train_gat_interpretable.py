#!/usr/bin/env python
"""
train_gat_interpretable.py

Train an interpretable GAT-based GNN on Movie-HCP Ledoit-Wolf graphs.

- Loads one CV fold (graphs_outer1_inner1.pkl by default).
- Flattens [run, window] arrays into a list of PyG Data objects.
- Drops padding windows.
- Trains a GAT regressor to predict cognitive scores.
- Exposes helper functions to inspect attention weights and node importance.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GlobalAttention, global_mean_pool
from typing import List, Dict, Any

# ----------------------------
# 1. Config
# ----------------------------

# FOLD_PATH = "data/folds_data/"  # change if you like
# fold_dir = "data/folds_data"
# fold_files = sorted(
#     os.path.join(fold_dir, f)
#     for f in os.listdir(fold_dir)
#     if f.startswith("graphs_outer") and f.endswith(".pkl")
# )

OUTPUT_DIR = "results_gat_interpretable"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    arr2d: np.ndarray or list with shape [n_runs, n_win], each entry is a PyG Data object.

    Returns:
        flat list of non-padding Data objects.
    """
    flat = []
    for row in arr2d:
        for g in row:
            # some Graphs have pad=True for padded windows
            if hasattr(g, "pad") and bool(g.pad):
                continue
            flat.append(g)
    return flat


def load_fold(path: str, weights_only: bool = False):
    """
    Load one fold file and return flat lists of train/val/test graphs.
    """
    print(f"Loading fold from {path}")
    # Torch 2.6 defaulted torch.load to weights_only=True, but these fold files
    # store full Python objects (PyG Data). Force weights_only=False for compatibility.
    graphs = torch.load(path, map_location="cpu", weights_only=weights_only)

    # The keys in your step2 script are 'train_graphs', 'test_graphs', 'val_graphs'
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
# 3. Interpretable GAT model
# ----------------------------

class GATRegressor(nn.Module):
    """
    Interpretable GAT-based regressor:
    - Two GATConv layers (edge-level attention).
    - Global attention pooling for node importance (optional).
    - MLP head for regression.

    We also expose helper methods:
    - get_edge_attention(data)
    - get_node_importance(data)
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 1,
                 use_global_attention_pool: bool = True):
        super().__init__()

        heads1 = 4  # multi-head on first layer
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads1, concat=True)  # -> hidden_dim * heads1
        gat1_out_dim = hidden_dim * heads1

        self.gat2 = GATConv(gat1_out_dim, hidden_dim, heads=1, concat=True)  # -> hidden_dim

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
        self.dropout = nn.Dropout(0.2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.gat1(x, edge_index).relu()
        x = self.gat2(x, edge_index).relu()

        if self.use_global_attention_pool:
            x = self.pool(x, batch)  # [batch_size, hidden_dim]
        else:
            x = self.pool(x, batch)

        x = self.lin1(x).relu()
        x = self.dropout(x)
        out = self.lin2(x).squeeze(-1)  # [batch_size]
        return out

    @torch.no_grad()
    def get_edge_attention(self, data):
        """
        Returns attention weights per edge for a *single* graph.

        data: PyG Data (batch dimension = 1 or no batch)
        Returns:
            edge_index: [2, E]
            alpha:      [E] attention coefficients from second GAT layer
        """
        self.eval()
        device = next(self.parameters()).device

        # Ensure it has batch attr
        if not hasattr(data, "batch"):
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long)

        data = data.to(device)
        # First GAT (we don't need its alphas here)
        x = self.gat1(data.x, data.edge_index).relu()

        # Second GAT with attention weights
        _, (edge_index, alpha) = self.gat2(
            x,
            data.edge_index,
            return_attention_weights=True
        )
        # alpha shape: [E, heads], but we used heads=1, so squeeze
        alpha = alpha.squeeze(-1).detach().cpu()
        edge_index = edge_index.detach().cpu()
        return edge_index, alpha

    @torch.no_grad()
    def get_node_importance(self, data):
        """
        Approximate node importance using the gate values from GlobalAttention.

        Returns:
            node_scores: [num_nodes] tensor with unnormalized attention scores per node.
        """
        if not self.use_global_attention_pool:
            raise RuntimeError("GlobalAttention pooling is disabled; cannot compute node importance.")

        self.eval()
        device = next(self.parameters()).device

        if not hasattr(data, "batch"):
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long)

        data = data.to(device)
        x = self.gat1(data.x, data.edge_index).relu()
        x = self.gat2(x, data.edge_index).relu()

        # gate_nn gives scalar per node → attention logits
        gate_values = self.gate_nn(x)  # [N, 1]
        # separate by batch if you had multiple graphs; here assume single graph
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

    for batch in loader:
        batch = batch.to(device)
        preds = model(batch)
        target = batch.y.view_as(preds).float()

        loss = loss_fn(preds, target)

        total_loss += loss.item() * batch.num_graphs
        num_samples += batch.num_graphs

        all_preds.append(preds.cpu())
        all_targets.append(target.cpu())

    if num_samples == 0:
        return float("nan"), None, None

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    mse = total_loss / num_samples

    # Optional: compute Pearson r
    if all_preds.numel() > 1:
        px = all_preds - all_preds.mean()
        py = all_targets - all_targets.mean()
        r = (px * py).sum() / (px.norm() * py.norm() + 1e-8)
        r = r.item()
    else:
        r = float("nan")

    return mse, all_preds, all_targets, r


def main():
    set_seed(SEED)

    fold_dir = "data/folds_data"
    fold_files = sorted(
        os.path.join(fold_dir, f)
        for f in os.listdir(fold_dir)
        if f.startswith("graphs_outer") and f.endswith(".pkl")
    )

    
    all_test_results = []

    for fold_path in fold_files:
        print("\n==============================")
        print("Training on fold:", fold_path)
        print("==============================")

        # 1) Load data from THIS fold
        train_graphs, val_graphs, test_graphs = load_fold(fold_path)

        # 2) Dataloaders (for this fold)
        in_dim = train_graphs[0].x.size(-1)
        print(f"Node feature dim: {in_dim}")

        train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_graphs,   batch_size=BATCH_SIZE, shuffle=False)
        test_loader  = DataLoader(test_graphs,  batch_size=BATCH_SIZE, shuffle=False)

        # 3) Fresh model + optimizer for THIS fold
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GATRegressor(
            in_dim=in_dim,
            hidden_dim=64,
            out_dim=1,
            use_global_attention_pool=USE_GLOBAL_ATTENTION_POOL,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_loss, _, _, val_r = eval_epoch(model, val_loader, device)
            print(
                f"[{os.path.basename(fold_path)}] "
                f"Epoch {epoch:03d} | train MSE={train_loss:.4f} | "
                f"val MSE={val_loss:.4f} | val r={val_r:.3f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict()

        # Use best model for test
        if best_state is not None:
            model.load_state_dict(best_state)

        test_mse, _, _, test_r = eval_epoch(model, test_loader, device)
        print(f"[{os.path.basename(fold_path)}] Test MSE: {test_mse:.4f} | Test r: {test_r:.3f}")

        all_test_results.append(
            {"fold": os.path.basename(fold_path), "test_mse": test_mse, "test_r": test_r}
        )

    # summarize over folds
    print("\n=== Summary over folds ===")
    for res in all_test_results:
        print(res)

if __name__ == "__main__":
    main()
