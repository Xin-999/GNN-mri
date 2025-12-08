#!/usr/bin/env python
"""
predict_with_trained_gatv2.py

Use already-trained GATv2 models to make predictions on val/test splits
and save subject-level predictions to Excel.

Assumptions:
- Fold files live in: data/folds_data/graphs_outer*_inner*.pkl
- Trained models live in: results_gatv2_interpretable/gatv2_best_<fold_name>.pt
  (same naming as in train_gatv2_interpretable.py)
- Model architecture (hidden_dim, heads1, use_global_attention_pool) matches training.
"""

import os
import argparse
from typing import List, Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, GlobalAttention, global_mean_pool

def get_target_mean_std(csv_path: str, score_column: str = "ListSort_AgeAdj"):
    df = pd.read_csv(csv_path)
    scores = df[score_column].astype(float).values
    mu = float(scores.mean())
    sigma = float(scores.std(ddof=0))  # population std (matches usual z-score)
    print(f"Denorm stats: mean={mu:.4f}, std={sigma:.4f}")
    return mu, sigma


# ----------------------------
# 1. GATv2 model (must match training)
# ----------------------------

class GATv2Regressor(nn.Module):
    def __init__(self, in_dim: int,
                 hidden_dim: int = 32,
                 heads1: int = 2,
                 out_dim: int = 1,
                 use_global_attention_pool = True):
        super().__init__()

        self.gat1 = GATv2Conv(in_dim, hidden_dim, heads=heads1, concat=True)
        gat1_out_dim = hidden_dim * heads1

        self.gat2 = GATv2Conv(gat1_out_dim, hidden_dim, heads=1, concat=True)

        self.use_global_attention_pool = use_global_attention_pool
        if use_global_attention_pool:
            self.gate_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
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


# ----------------------------
# 2. Helpers to load folds & flatten graphs
# ----------------------------

def flatten_and_filter(arr2d) -> List[Any]:
    """
    arr2d: [n_subjects, n_windows] of PyG Data.

    - drop padding windows (g.pad == True)
    - attach integer g.subject_id (0..n_subjects-1)
    """
    flat = []
    for subj_idx, row in enumerate(arr2d):
        for g in row:
            if hasattr(g, "pad") and bool(g.pad):
                continue
            g.subject_id = torch.tensor([subj_idx], dtype=torch.long)
            flat.append(g)
    return flat


def load_split_from_fold(
    fold_path: str,
    split: str
) -> Tuple[List[Any], str, Dict[int, Any]]:
    """
    Load graphs for a given split ("train"/"val"/"test") from a fold file.

    Returns:
        graphs_list: flattened list of Data
        fold_name:   e.g. "graphs_outer1_inner1"
        subj_label_map: {subject_index -> subject_label} if available
    """
    print(f"\nLoading fold from {fold_path}")
    graphs = torch.load(fold_path, map_location="cpu", weights_only=False)

    fold_name = os.path.basename(fold_path).replace(".pkl", "")

    if split == "train":
        arr2d = graphs["train_graphs"]
    elif split == "val":
        arr2d = graphs["val_graphs"]
    elif split == "test":
        arr2d = graphs["test_graphs"]
    else:
        raise ValueError(f"Unknown split: {split}")

    graphs_list = flatten_and_filter(arr2d)
    print(f"{fold_name} | split={split}: {len(graphs_list)} graphs (windows)")

    # Try to recover real subject IDs if they were stored
    subj_labels = None
    for key in ["subjects", "subject_ids", "subj_ids", "subject_list"]:
        if key in graphs:
            subj_labels = graphs[key]
            break

    subj_label_map: Dict[int, Any] = {}
    if subj_labels is not None:
        # subj_labels should be aligned with first axis (subjects)
        for idx, lab in enumerate(list(subj_labels)):
            subj_label_map[idx] = lab

    return graphs_list, fold_name, subj_label_map


# ----------------------------
# 3. Run model and aggregate per subject
# ----------------------------

@torch.no_grad()
def predict_subject_level(
    model: nn.Module,
    graphs_list: List[Any],
    device: torch.device,
    fold_name: str,
    split: str,
    subj_label_map: Dict[int, Any],
    mu: float,
    sigma: float,
) -> pd.DataFrame:
    """
    Run model on all graphs and aggregate per subject.

    Returns DataFrame with:
        fold, split, subject_id, subject_label (if known),
        y_true, y_pred, n_windows
    """
    loader = DataLoader(
        graphs_list,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    rows: List[Dict[str, Any]] = []

    for batch in loader:
        batch = batch.to(device)
        preds = model(batch)
        targets = batch.y.view_as(preds).float()

        if hasattr(batch, "subject_id"):
            subj_ids = batch.subject_id.view(-1).cpu().numpy()
        else:
            subj_ids = np.arange(batch.num_graphs)

        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()

        for sid, y_true, y_pred in zip(subj_ids, targets_np, preds_np):
            sid_int = int(sid)
            label = subj_label_map.get(sid_int, sid_int)  # fallback to index
            rows.append(
                {
                    "fold": fold_name,
                    "split": split,
                    "subject_id": sid_int,
                    "subject_label": label,
                    "y_true": float(y_true),
                    "y_pred": float(y_pred),
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # aggregate per subject (average over windows)
    grouped = df.groupby(["fold", "split", "subject_id", "subject_label"])
    df_subj = grouped.agg(
        y_true=("y_true", "mean"),
        y_pred=("y_pred", "mean"),
        n_windows=("y_pred", "count"),
    ).reset_index()

    # 🔹 denormalise back to original ListSort_AgeAdj scale
    df_subj["y_true_raw"] = df_subj["y_true"] * sigma + mu
    df_subj["y_pred_raw"] = df_subj["y_pred"] * sigma + mu

    return df_subj


# ----------------------------
# 4. Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Use trained GATv2 models to predict and save Excel files."
    )
    parser.add_argument(
        "--fold-dir",
        default="data/folds_data",
        help="Directory with graphs_outer*_inner*.pkl",
    )
    parser.add_argument(
        "--model-dir",
        default="results_gatv2_interpretable",
        help="Directory with gatv2_best_<fold_name>.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="results_gatv2_predictions",
        help="Where to save Excel outputs.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Which split to run predictions on.",
    )
    # These must match how you trained the model
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--heads1", type=int, default=2)
    parser.add_argument(
        "--use-global-attention-pool",
        action="store_true",
        help="Set this if training used GlobalAttention pooling.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    beh_csv = "./data/ListSort_AgeAdj.csv"
    beh_col = "ListSort_AgeAdj"
    mu, sigma = get_target_mean_std(beh_csv, beh_col)

    # find all fold files
    fold_files = sorted(
        os.path.join(args.fold_dir, f)
        for f in os.listdir(args.fold_dir)
        if f.startswith("graphs_outer") and f.endswith(".pkl")
    )
    if not fold_files:
        raise FileNotFoundError(f"No fold .pkl files found in {args.fold_dir}")

    all_dfs = []

    for fold_path in fold_files:
        graphs_list, fold_name, subj_label_map = load_split_from_fold(
            fold_path, args.split
        )

        if not graphs_list:
            print(f"{fold_name}: no graphs for split={args.split}, skipping.")
            continue

        in_dim = graphs_list[0].x.size(-1)
        print(f"{fold_name}: node feature dim = {in_dim}")

        # build model and load saved weights
        model = GATv2Regressor(
            in_dim=in_dim,
            hidden_dim=args.hidden_dim,
            heads1=args.heads1,
            out_dim=1,
            use_global_attention_pool=args.use_global_attention_pool,
        ).to(device)

        model_path = os.path.join(
            args.model_dir, f"gatv2_best_{fold_name}.pt"
        )
        if not os.path.exists(model_path):
            print(f"WARNING: model for fold {fold_name} not found at {model_path}")
            continue

        print(f"Loading model weights from {model_path}")
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)

        # run predictions and aggregate per subject
        df_subj = predict_subject_level(
            model=model,
            graphs_list=graphs_list,
            device=device,
            fold_name=fold_name,
            split=args.split,
            subj_label_map=subj_label_map,
            mu=mu,
            sigma=sigma,
        )

        if df_subj.empty:
            print(f"{fold_name}: no predictions produced.")
            continue

        all_dfs.append(df_subj)

        # save one Excel per fold
        out_path = os.path.join(
            args.output_dir,
            f"{args.split}_predictions_{fold_name}.xlsx",
        )
        df_subj.to_excel(out_path, index=False)
        print(f"Saved {len(df_subj)} subjects to {out_path}")

    # also save combined Excel over all folds
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_out = os.path.join(
            args.output_dir,
            f"{args.split}_predictions_all_folds.xlsx",
        )
        combined.to_excel(combined_out, index=False)
        print(f"\nSaved combined predictions to {combined_out}")


if __name__ == "__main__":
    main()
