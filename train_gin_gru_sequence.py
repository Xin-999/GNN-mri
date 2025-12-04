#!/usr/bin/env python
"""
train_gin_gru_sequence.py

Subject-level sequence model for Movie-HCP Ledoit-Wolf graphs.

- Each subject = sequence of dynamic FC graphs (windows).
- Spatial encoder: 2-layer GINConv + global mean pooling.
- Temporal encoder: GRU over graph embeddings.
- Output: one cognitive score per subject.

- Loads CV folds (graphs_outer*_inner*.pkl) created by step2_prepare_data.py.
- Uses subject-level data: each training sample is (list_of_graphs_for_subject, label).
- Reports subject-level MSE and Pearson r on val and test.
- Saves best model per fold based on lowest validation MSE.
"""

import os
import json
import time
from typing import List, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from torch_geometric.data import Batch
from torch_geometric.nn import GINConv, global_mean_pool

# ----------------------------
# 1. Config
# ----------------------------

OUTPUT_DIR = "results_gin_gru_sequence"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
BATCH_SIZE = 8        # smaller, since each sample is a whole subject sequence
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 20


# ----------------------------
# 2. Utilities
# ----------------------------

def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SubjectSequenceDataset(Dataset):
    """
    Wraps a 2D array/list of PyG Data objects [n_subjects, n_win]
    into a subject-level Dataset.

    Each __getitem__ returns:
        (list_of_Data_windows, target_y: float)
    """

    def __init__(self, graphs_2d: Any):
        self.subjects = []

        for subj_row in graphs_2d:
            # subj_row: sequence of windows for this subject
            windows = []
            y_val = None

            for g in subj_row:
                # skip padding windows if present
                if hasattr(g, "pad") and bool(g.pad):
                    continue

                windows.append(g)

                # assume y is stored on each window and identical per subject
                if getattr(g, "y", None) is not None and y_val is None:
                    # g.y could be shape [1] or []
                    y_tensor = g.y.view(-1).float()
                    y_val = float(y_tensor[0].item())

            if len(windows) == 0:
                # no valid windows; skip subject
                continue
            if y_val is None:
                raise ValueError("Subject has windows but no 'y' label found.")

            self.subjects.append((windows, y_val))

        if len(self.subjects) == 0:
            raise ValueError("No non-empty subjects found in graphs_2d.")

        print(f"Built SubjectSequenceDataset with {len(self.subjects)} subjects.")

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int):
        windows, y_val = self.subjects[idx]
        return windows, torch.tensor(y_val, dtype=torch.float32)


def collate_subject_batch(
    batch: List[Tuple[List[Any], torch.Tensor]]
) -> Tuple[Batch, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for subject-level batches.

    Args:
        batch: list of (windows_list, y) for each subject in the batch.

    Returns:
        big_batch: PyG Batch containing all windows of all subjects.
        lengths:  tensor of shape [batch_size], number of windows per subject.
        ys:       tensor of shape [batch_size], subject labels.
    """
    all_graphs = []
    lengths = []
    ys = []

    for windows, y in batch:
        lengths.append(len(windows))
        ys.append(y)
        all_graphs.extend(windows)

    big_batch = Batch.from_data_list(all_graphs)
    lengths = torch.tensor(lengths, dtype=torch.long)
    ys = torch.stack(ys).view(-1)  # [batch_size]

    return big_batch, lengths, ys


def load_fold_subject_level(path: str, weights_only: bool = False):
    """
    Load one fold file and return subject-level Datasets for train/val/test.
    """
    print(f"Loading fold (subject-level) from {path}")
    graphs = torch.load(path, map_location="cpu", weights_only=weights_only)

    train_2d = graphs["train_graphs"]  # [n_subjects, n_win]
    val_2d   = graphs["val_graphs"]
    test_2d  = graphs["test_graphs"]

    train_dataset = SubjectSequenceDataset(train_2d)
    val_dataset   = SubjectSequenceDataset(val_2d)
    test_dataset  = SubjectSequenceDataset(test_2d)

    print(f"Subjects: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset


# ----------------------------
# 3. GIN + GRU model
# ----------------------------

class GINGRURegressor(nn.Module):
    """
    Subject-level model:

    - Spatial encoder: 2-layer GINConv + global_mean_pool → graph embedding per window.
    - Temporal encoder: GRU over window embeddings for each subject.
    - Head: linear → regression score per subject.
    """

    def __init__(
        self,
        in_dim: int,
        gnn_hidden: int = 64,
        gru_hidden: int = 64,
        out_dim: int = 1,
    ):
        super().__init__()

        # GIN MLPs
        mlp1 = nn.Sequential(
            nn.Linear(in_dim, gnn_hidden),
            nn.ReLU(),
            nn.Linear(gnn_hidden, gnn_hidden),
        )
        mlp2 = nn.Sequential(
            nn.Linear(gnn_hidden, gnn_hidden),
            nn.ReLU(),
            nn.Linear(gnn_hidden, gnn_hidden),
        )

        self.gin1 = GINConv(mlp1)
        self.gin2 = GINConv(mlp2)

        self.gru = nn.GRU(
            input_size=gnn_hidden,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.lin_out = nn.Linear(gru_hidden, out_dim)

    def encode_graphs(self, big_batch: Batch) -> torch.Tensor:
        """
        big_batch: PyG Batch containing all windows in the batch.
        Returns:
            graph_emb: [num_windows_in_batch, gnn_hidden]
        """
        x, edge_index, batch_idx = big_batch.x, big_batch.edge_index, big_batch.batch

        h = self.gin1(x, edge_index).relu()
        h = self.gin2(h, edge_index).relu()

        g = global_mean_pool(h, batch_idx)  # [num_windows_in_batch, gnn_hidden]
        return g

    def forward(self, big_batch: Batch, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            big_batch: PyG Batch with all windows in the batch.
            lengths:  tensor [batch_size], number of windows per subject.

        Returns:
            preds: [batch_size] tensor of subject-level predictions.
        """
        device = next(self.parameters()).device
        lengths = lengths.to(device)

        # Spatial encoding
        graph_emb = self.encode_graphs(big_batch.to(device))
        # graph_emb is concatenated by subjects; we need to split by lengths
        seq_list = torch.split(graph_emb, lengths.tolist(), dim=0)  # list of [T_i, gnn_hidden]

        # Pad to [B, T_max, gnn_hidden]
        padded = pad_sequence(seq_list, batch_first=True)  # [B, T_max, gnn_hidden]

        # Pack for GRU
        packed = pack_padded_sequence(
            padded,
            lengths.cpu(),  # lengths must be on CPU for pack_padded_sequence
            batch_first=True,
            enforce_sorted=False,
        )

        _, h_n = self.gru(packed)  # h_n: [num_layers, B, gru_hidden]
        h_last = h_n[-1]           # [B, gru_hidden]

        out = self.lin_out(h_last).squeeze(-1)  # [B]
        return out


# ----------------------------
# 4. Training & evaluation
# ----------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.MSELoss()

    total_loss = 0.0
    num_samples = 0

    for big_batch, lengths, targets in loader:
        big_batch = big_batch.to(device)
        lengths = lengths.to(device)
        targets = targets.to(device)

        preds = model(big_batch, lengths)
        loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        num_samples += targets.size(0)

    return total_loss / max(1, num_samples)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    loss_fn = nn.MSELoss()

    total_loss = 0.0
    num_samples = 0

    all_preds = []
    all_targets = []

    for big_batch, lengths, targets in loader:
        big_batch = big_batch.to(device)
        lengths = lengths.to(device)
        targets = targets.to(device)

        preds = model(big_batch, lengths)
        loss = loss_fn(preds, targets)

        total_loss += loss.item() * targets.size(0)
        num_samples += targets.size(0)

        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

    if num_samples == 0:
        return float("nan"), float("nan")

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    mse = total_loss / num_samples

    # Pearson r across subjects
    if all_preds.numel() > 1:
        px = all_preds - all_preds.mean()
        py = all_targets - all_targets.mean()
        r = (px * py).sum() / (px.norm() * py.norm() + 1e-8)
        r = r.item()
    else:
        r = float("nan")

    return mse, r


# ----------------------------
# 5. Main loop over folds
# ----------------------------

def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    fold_dir = "data/folds_data"
    fold_files = sorted(
        os.path.join(fold_dir, f)
        for f in os.listdir(fold_dir)
        if f.startswith("graphs_outer") and f.endswith(".pkl")
    )

    all_fold_results = []

    for fold_path in fold_files:
        print("\n==============================")
        print("Training (subject-level) on fold:", fold_path)
        print("==============================")

        train_dataset, val_dataset, test_dataset = load_fold_subject_level(fold_path)

        # Infer node feature dimension from first non-empty window of first train subject
        first_windows, _ = train_dataset[0]
        in_dim = first_windows[0].x.size(-1)
        print(f"Node feature dim: {in_dim}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_subject_batch,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_subject_batch,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_subject_batch,
        )

        model = GINGRURegressor(
            in_dim=in_dim,
            gnn_hidden=64,
            gru_hidden=64,
            out_dim=1,
        ).to(device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

        best_val_mse = float("inf")
        best_state = None

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_mse, val_r = eval_epoch(model, val_loader, device)
            dt = time.time() - t0

            print(
                f"[{os.path.basename(fold_path)}] "
                f"Epoch {epoch:03d} | train MSE={train_loss:.4f} | "
                f"val_subj MSE={val_mse:.4f} | val_subj r={val_r:.3f} | time={dt:.2f}s"
            )

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_state = model.state_dict()

        # Use best model for test
        if best_state is not None:
            model.load_state_dict(best_state)

        test_mse, test_r = eval_epoch(model, test_loader, device)
        print(
            f"[{os.path.basename(fold_path)}] "
            f"Test subject MSE: {test_mse:.4f} | Test subject r: {test_r:.3f}"
        )

        # Save best model for this fold
        fold_name = os.path.basename(fold_path).replace(".pkl", "")
        model_path = os.path.join(OUTPUT_DIR, f"gingru_best_{fold_name}.pt")
        if best_state is not None:
            torch.save(best_state, model_path)
            print(f"Saved best model for {fold_name} to {model_path}")

        all_fold_results.append(
            {
                "fold": os.path.basename(fold_path),
                "test_mse_subj": float(test_mse),
                "test_r_subj": float(test_r),
            }
        )

    # summarize over folds
    print("\n=== Summary over folds (subject-level GIN+GRU) ===")
    for res in all_fold_results:
        print(res)

    # Save JSON summary
    summary_path = os.path.join(OUTPUT_DIR, "gingru_test_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_fold_results, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
