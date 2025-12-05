#!/usr/bin/env python
"""
explain_gatv2_gnnexplainer.py

Post-hoc interpretability for your trained GATv2 model on Movie-HCP graphs,
using GNNExplainer (like Amendola et al. did for BraTS, but here for graph
regression on fMRI).

- Loads one CV fold (graphs_outer*_inner*.pkl).
- Loads the corresponding best GATv2 model checkpoint.
- Picks one test graph (one window from one subject).
- Uses torch_geometric.explain.Explainer + GNNExplainer to get
  node and edge importance masks for that graph.

Requirements:
  - Same environment as train_gatv2_interpretable.py
  - torch_geometric >= 2.3 (for torch_geometric.explain)
"""

import os
import torch
import torch.nn as nn

from typing import List, Any

from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, GlobalAttention, global_mean_pool
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer


# ----------------------------
# 1. Config
# ----------------------------

FOLD_DIR = "data/folds_data"
MODEL_DIR = "results_gatv2_interpretable"

# choose which fold to explain
FOLD_BASENAME = "graphs_outer1_inner1.pkl"   # change if you want another fold
CHECKPOINT_NAME = f"gatv2_best_{FOLD_BASENAME.replace('.pkl', '')}.pt"

TOPK_EDGES = 20  # how many edges to print as most important


# ----------------------------
# 2. Utils: flatten + load fold
# ----------------------------

def flatten_and_filter(arr2d) -> List[Any]:
    """
    arr2d: np.ndarray or list with shape [n_subjects, n_win],
           each entry is a PyG Data object.

    Returns:
        flat list of non-padding Data objects.
    """
    flat = []
    for subj_idx, row in enumerate(arr2d):
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
# 3. GATv2 model (same as training script)
# ----------------------------

class GATv2Regressor(nn.Module):
    """
    Same architecture as in train_gatv2_interpretable.py
    """

    def __init__(self, in_dim: int, hidden_dim: int = 32, out_dim: int = 1,
                 use_global_attention_pool: bool = True):
        super().__init__()

        heads1 = 2  # must match what you used in training
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

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.gat1(x, edge_index).relu()
        x = self.gat2(x, edge_index).relu()

        if self.use_global_attention_pool:
            x = self.pool(x, batch)
        else:
            x = self.pool(x, batch)

        x = self.lin1(x).relu()
        out = self.lin2(x).squeeze(-1)  # [batch_size]
        return out


class WrappedGATv2(nn.Module):
    """
    Simple wrapper so that Explainer can call model(x, edge_index, batch).

    Explainer in PyG 2.3+ expects model(x, edge_index, batch=...) for
    graph-level tasks, not a Data object.
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            # single graph, all nodes in one graph
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        data = Data(x=x, edge_index=edge_index, batch=batch)
        return self.base_model(data)


# ----------------------------
# 4. Main: load model, pick graph, run GNNExplainer
# ----------------------------

def main():
    fold_path = os.path.join(FOLD_DIR, FOLD_BASENAME)
    ckpt_path = os.path.join(MODEL_DIR, CHECKPOINT_NAME)

    if not os.path.isfile(fold_path):
        raise FileNotFoundError(f"Fold file not found: {fold_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # 1) Load data
    train_graphs, val_graphs, test_graphs = load_fold(fold_path)

    # just to know feature dim
    in_dim = train_graphs[0].x.size(-1)
    print(f"Node feature dim: {in_dim}")

    # 2) Build and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    base_model = GATv2Regressor(
        in_dim=in_dim,
        hidden_dim=32,
        out_dim=1,
        use_global_attention_pool=True,
    ).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    base_model.load_state_dict(state_dict)
    base_model.eval()

    model = WrappedGATv2(base_model).to(device)

    # 3) Choose a test graph to explain
    #    Example: first test graph
    graph = test_graphs[0]
    # make sure it has batch attribute for consistency
    if not hasattr(graph, "batch"):
        graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)

    x = graph.x.to(device)
    edge_index = graph.edge_index.to(device)
    batch = graph.batch.to(device)
    y_true = float(graph.y.view(-1)[0].item())

    with torch.no_grad():
        y_pred = float(model(x, edge_index, batch=batch).view(-1)[0].item())

    print("\nExplaining graph from test set:")
    if hasattr(graph, "subject_id"):
        print(f"  subject_id: {int(graph.subject_id.item())}")
    print(f"  true score: {y_true:.4f}")
    print(f"  pred score: {y_pred:.4f}")

    # 4) Build GNNExplainer via Explainer interface
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200, lr=0.01),
        explanation_type='model',
        node_mask_type='attributes',  # importance over node features
        edge_mask_type='object',      # importance over edges
        model_config=dict(
            mode='regression',
            task_level='graph',
            return_type='raw',        # model returns raw scores
        ),
        # keep only top-K most important edges in the final explanation
        threshold_config=dict(
            threshold_type='topk',
            value=TOPK_EDGES,
        ),
    )

    # 5) Run explanation
    explanation = explainer(x, edge_index, batch=batch)

    node_mask = explanation.node_mask       # [num_nodes, num_features] or [num_nodes]
    edge_mask = explanation.edge_mask       # [num_edges]

    print("\n=== Explanation summary ===")
    print(f"node_mask shape: {node_mask.shape}")
    print(f"edge_mask shape: {edge_mask.shape}")

    # 6) Print top-K important edges
    # Sort edges by importance
    edge_importance = edge_mask.detach().cpu()
    sorted_idx = torch.argsort(edge_importance, descending=True)

    print(f"\nTop {TOPK_EDGES} edges by importance (u -> v, score):")
    top_k = min(TOPK_EDGES, sorted_idx.numel())
    for i in range(top_k):
        e_idx = int(sorted_idx[i].item())
        u = int(edge_index[0, e_idx].item())
        v = int(edge_index[1, e_idx].item())
        score = float(edge_importance[e_idx].item())
        print(f"  edge {i+1:02d}: {u:3d} -> {v:3d} | importance={score:.4f}")

    # 7) (Optional) Summarize node importance
    # If node_mask is [N, F], you can aggregate across features:
    if node_mask.dim() == 2:
        node_scores = node_mask.mean(dim=1)  # simple average over features
    else:
        node_scores = node_mask

    node_scores = node_scores.detach().cpu()
    node_sorted = torch.argsort(node_scores, descending=True)
    top_nodes = min(20, node_sorted.numel())

    print("\nTop nodes by importance:")
    for i in range(top_nodes):
        nid = int(node_sorted[i].item())
        score = float(node_scores[nid].item())
        print(f"  node {nid:3d} | importance={score:.4f}")

    print("\nDone. You can now map node indices back to Shen ROI labels "
          "to interpret which brain regions and connections drive the prediction.")


if __name__ == "__main__":
    main()
