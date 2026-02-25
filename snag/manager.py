import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from utils.data_utils import (
    load_graphs_with_normalization,
    aggregate_window_predictions,
    compute_metrics,
)
from .model import GATv2GraphNet


class GATv2RegressionManager(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.loss_fn = nn.MSELoss()

        fold_path = self._resolve_fold_path()
        train_list, val_list, test_list, info = load_graphs_with_normalization(
            str(fold_path),
            normalize_method=args.normalize_method,
        )
        self.scaler = info.get("scaler")

        self.train_loader = DataLoader(
            train_list,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )
        self.val_loader = DataLoader(
            val_list,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )
        self.test_loader = DataLoader(
            test_list,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )

        sample_graph = train_list[0]
        self.in_feats = sample_graph.num_features
        self.num_label = 1
        self.args.in_feats = self.in_feats
        self.args.num_label = self.num_label

    def _resolve_fold_path(self):
        if self.args.fold_path:
            return Path(self.args.fold_path)

        fold_dir = Path(self.args.fold_dir)
        fold_files = sorted(fold_dir.glob("graphs_outer*.pkl"))
        if not fold_files:
            raise FileNotFoundError(f"No fold files found in {fold_dir}")

        if self.args.fold_name:
            for path in fold_files:
                if path.stem == self.args.fold_name:
                    return path
            raise FileNotFoundError(
                f"Fold {self.args.fold_name} not found in {fold_dir}"
            )

        return fold_files[0]

    def build_gnn(self, actions):
        model = GATv2GraphNet(actions, self.in_feats, self.num_label, self.args)
        return model

    def _run_epoch(self, model, loader, optimizer=None):
        is_train = optimizer is not None
        if is_train:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_subject_ids = []

        for batch in loader:
            batch = batch.to(self.device)
            preds = model(batch.x, batch.edge_index, batch.batch)
            targets = batch.y.view(-1).float()
            loss = self.loss_fn(preds, targets)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            all_preds.append(preds.detach().cpu())
            all_targets.append(targets.detach().cpu())
            if hasattr(batch, "subject_id"):
                all_subject_ids.append(batch.subject_id.cpu())

        preds_np = torch.cat(all_preds).numpy()
        targets_np = torch.cat(all_targets).numpy()
        metrics = compute_metrics(preds_np, targets_np, prefix="win_")

        if all_subject_ids:
            subject_ids = torch.cat(all_subject_ids).numpy().flatten()
            if len(subject_ids) == len(preds_np):
                subj_preds, _ = aggregate_window_predictions(preds_np, subject_ids)
                subj_targets, _ = aggregate_window_predictions(targets_np, subject_ids)
                metrics.update(compute_metrics(subj_preds, subj_targets, prefix="subj_"))

        avg_loss = total_loss / max(1, len(loader.dataset))
        return metrics, avg_loss

    def evaluate(self, actions=None, format="two"):
        model = self.build_gnn(actions)
        model.to(self.device)
        metrics, _ = self._run_epoch(model, self.val_loader)
        val_score = metrics.get("subj_r", metrics.get("win_r", 0.0))
        return val_score, metrics

    def train(self, actions=None, format="two", evaluate_test=False):
        model = self.build_gnn(actions)
        model.to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )

        best_val = -float("inf")
        best_state = None
        best_metrics = None
        patience = self.args.patience
        patience_left = patience

        for _ in range(self.args.epochs):
            self._run_epoch(model, self.train_loader, optimizer)
            val_metrics, _ = self._run_epoch(model, self.val_loader)
            val_score = val_metrics.get("subj_r", val_metrics.get("win_r", 0.0))

            if val_score > best_val:
                best_val = val_score
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                best_metrics = val_metrics
                patience_left = patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        result = {
            "val_score": best_val,
            "val_metrics": best_metrics,
        }

        if evaluate_test and best_state is not None:
            model.load_state_dict(best_state)
            test_metrics, _ = self._run_epoch(model, self.test_loader)
            result["test_metrics"] = test_metrics
            result["model_state"] = best_state

        return result

    def test_with_param(self, actions=None, format="two", evaluate_test=False):
        return self.train(actions=actions, format=format, evaluate_test=evaluate_test)
