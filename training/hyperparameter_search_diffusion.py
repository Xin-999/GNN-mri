#!/usr/bin/env python
"""
Diffusion GNN hyperparameter search (APPNP/SGC).

Runs Optuna across folds, tracks subject-level Pearson r and MSE,
and saves full results to JSON.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import optuna
except ImportError:
    print("Optuna not installed. Install with: pip install optuna")
    sys.exit(1)

# Add project root to path so we can import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.gnn_diffusion import AppnpDiffusionRegressor, SgcRegressor
from utils.data_utils import (
    aggregate_window_predictions,
    compute_metrics,
    create_dataloaders,
    load_graphs_with_normalization,
)


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


def build_model(model_name: str, in_dim: int, params: Dict) -> nn.Module:
    if model_name == "appnp":
        return AppnpDiffusionRegressor(
            in_dim=in_dim,
            hidden_dim=params["hidden_dim"],
            mlp_layers=params["mlp_layers"],
            dropout=params["dropout"],
            diffusion_steps=params["diffusion_steps"],
            alpha=params["alpha"],
            diffusion_dropout=params["diffusion_dropout"],
            readout=params["readout"],
            use_edge_weight=params["use_edge_weight"],
        )
    if model_name == "sgc":
        return SgcRegressor(
            in_dim=in_dim,
            hidden_dim=params["hidden_dim"],
            sgc_k=params["sgc_k"],
            dropout=params["dropout"],
            readout=params["readout"],
            use_edge_weight=params["use_edge_weight"],
        )
    raise ValueError(f"Unknown model: {model_name}")


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


def get_metric(metrics: Dict[str, float], primary: str, fallback: str) -> float:
    value = metrics.get(primary)
    if value is None or np.isnan(value):
        value = metrics.get(fallback, float("nan"))
    return value


def metric_mode(metric_name: str) -> str:
    metric = metric_name.lower()
    if "mse" in metric or "mae" in metric or "loss" in metric:
        return "min"
    return "max"


def objective(
    trial,
    model_name: str,
    fold_paths: List[Path],
    device: torch.device,
    n_epochs: int,
    patience: int,
    objective_name: str,
    mse_weight: float,
    normalize_method: str,
) -> float:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    set_seed(42)

    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    mlp_layers = trial.suggest_int("mlp_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    readout = trial.suggest_categorical("readout", ["mean", "add"])
    use_edge_weight = trial.suggest_categorical("use_edge_weight", [True, False])

    params = {
        "hidden_dim": hidden_dim,
        "mlp_layers": mlp_layers,
        "dropout": dropout,
        "readout": readout,
        "use_edge_weight": use_edge_weight,
    }

    if model_name == "appnp":
        params.update({
            "diffusion_steps": trial.suggest_int("diffusion_steps", 3, 15),
            "alpha": trial.suggest_float("alpha", 0.05, 0.2),
            "diffusion_dropout": trial.suggest_float("diffusion_dropout", 0.0, 0.2),
        })
    else:
        params.update({
            "sgc_k": trial.suggest_int("sgc_k", 2, 6),
        })

    fold_r_scores: List[float] = []
    fold_mse_scores: List[float] = []
    fold_best_metrics: List[Dict[str, float]] = []

    for fold_idx, fold_path in enumerate(fold_paths):
        train_graphs, val_graphs, test_graphs, info = load_graphs_with_normalization(
            str(fold_path),
            normalize_method=normalize_method,
        )

        train_loader, val_loader, _ = create_dataloaders(
            train_graphs,
            val_graphs,
            test_graphs,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        in_dim = train_graphs[0].x.size(-1)
        model = build_model(model_name, in_dim, params).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        best_metric = float("inf") if metric_mode("subj_mse") == "min" else -float("inf")
        best_metrics = {}
        patience_counter = 0

        for epoch in range(n_epochs):
            try:
                train_one_epoch(model, train_loader, optimizer, criterion, device)
                val_metrics = evaluate(model, val_loader, device, scaler=info.get("scaler"))
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise optuna.TrialPruned()
                raise

            current_metric = get_metric(val_metrics, "subj_mse", "win_mse")
            if metric_mode("subj_mse") == "max":
                improved = current_metric > best_metric + 1e-5
            else:
                improved = current_metric < best_metric - 1e-5

            if improved:
                best_metric = current_metric
                best_metrics = val_metrics
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        fold_val_r = get_metric(best_metrics, "subj_r", "win_r")
        fold_val_mse = get_metric(best_metrics, "subj_mse", "win_mse")
        fold_r_scores.append(fold_val_r)
        fold_mse_scores.append(fold_val_mse)
        fold_best_metrics.append({
            "fold": fold_path.name,
            "best_val_r": fold_val_r,
            "best_val_mse": fold_val_mse,
        })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        partial_avg_r = float(np.nanmean(fold_r_scores))
        partial_avg_mse = float(np.nanmean(fold_mse_scores))
        if objective_name == "mse":
            trial.report(partial_avg_mse, fold_idx)
        else:
            trial.report(partial_avg_r, fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    avg_val_r = float(np.nanmean(fold_r_scores))
    std_val_r = float(np.nanstd(fold_r_scores))
    avg_val_mse = float(np.nanmean(fold_mse_scores))
    std_val_mse = float(np.nanstd(fold_mse_scores))

    trial.set_user_attr("avg_val_r", avg_val_r)
    trial.set_user_attr("std_val_r", std_val_r)
    trial.set_user_attr("avg_val_mse", avg_val_mse)
    trial.set_user_attr("std_val_mse", std_val_mse)
    trial.set_user_attr("fold_r_scores", fold_r_scores)
    trial.set_user_attr("fold_mse_scores", fold_mse_scores)
    trial.set_user_attr("fold_best_metrics", fold_best_metrics)
    trial.set_user_attr("n_folds", len(fold_paths))

    if objective_name == "pearson_r":
        return avg_val_r
    if objective_name == "mse":
        return avg_val_mse
    return avg_val_r - mse_weight * avg_val_mse


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


def best_trial_by_metric(study, metric_key: str, direction: str):
    best_trial = None
    best_value = None
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        value = trial.user_attrs.get(metric_key)
        if value is None:
            continue
        if best_trial is None:
            best_trial = trial
            best_value = value
            continue
        if direction == "max" and value > best_value:
            best_trial = trial
            best_value = value
        if direction == "min" and value < best_value:
            best_trial = trial
            best_value = value
    return best_trial


def main() -> None:
    parser = argparse.ArgumentParser(description="Diffusion GNN hyperparameter search")
    parser.add_argument("--model", type=str, required=True, choices=["appnp", "sgc"])
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--n_epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--fold_dir", type=str, default="data/folds_data")
    parser.add_argument("--fold_name", type=str, default=None)
    parser.add_argument("--fold_indices", nargs="+", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="hyperparameter_search_results_diffusion")
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--objective", type=str, default="composite",
                        choices=["pearson_r", "mse", "composite"])
    parser.add_argument("--mse_weight", type=float, default=0.5)
    parser.add_argument("--normalize_method", type=str, default="standard",
                        choices=["standard", "minmax", "robust"])

    args = parser.parse_args()

    device = select_device(args.device)
    fold_dir = Path(args.fold_dir)
    fold_paths, fold_config_str = collect_folds(fold_dir, args.fold_name, args.fold_indices)

    base_output_dir = Path(args.output_dir)
    output_dir = base_output_dir / fold_config_str / f"trials_{args.n_trials}"
    output_dir.mkdir(parents=True, exist_ok=True)

    study_name = args.study_name or f"{args.model}_diffusion_search"
    direction = "minimize" if args.objective == "mse" else "maximize"

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        lambda trial: objective(
            trial,
            args.model,
            fold_paths,
            device,
            args.n_epochs,
            args.patience,
            args.objective,
            args.mse_weight,
            args.normalize_method,
        ),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
    )

    best_objective_trial = study.best_trial
    best_r_trial = best_trial_by_metric(study, "avg_val_r", "max")
    best_mse_trial = best_trial_by_metric(study, "avg_val_mse", "min")

    results = {
        "model": args.model,
        "objective": args.objective,
        "objective_mse_weight": args.mse_weight,
        "n_folds": len(fold_paths),
        "folds": [p.name for p in fold_paths],
        "n_trials": args.n_trials,
        "best_objective_trial": {
            "trial_number": best_objective_trial.number,
            "value": best_objective_trial.value,
            "avg_val_r": best_objective_trial.user_attrs.get("avg_val_r"),
            "std_val_r": best_objective_trial.user_attrs.get("std_val_r"),
            "avg_val_mse": best_objective_trial.user_attrs.get("avg_val_mse"),
            "std_val_mse": best_objective_trial.user_attrs.get("std_val_mse"),
            "params": best_objective_trial.params,
        },
        "best_avg_r_trial": {
            "trial_number": best_r_trial.number if best_r_trial else None,
            "avg_val_r": best_r_trial.user_attrs.get("avg_val_r") if best_r_trial else None,
            "std_val_r": best_r_trial.user_attrs.get("std_val_r") if best_r_trial else None,
            "avg_val_mse": best_r_trial.user_attrs.get("avg_val_mse") if best_r_trial else None,
            "params": best_r_trial.params if best_r_trial else {},
        },
        "best_avg_mse_trial": {
            "trial_number": best_mse_trial.number if best_mse_trial else None,
            "avg_val_mse": best_mse_trial.user_attrs.get("avg_val_mse") if best_mse_trial else None,
            "std_val_mse": best_mse_trial.user_attrs.get("std_val_mse") if best_mse_trial else None,
            "avg_val_r": best_mse_trial.user_attrs.get("avg_val_r") if best_mse_trial else None,
            "params": best_mse_trial.params if best_mse_trial else {},
        },
        "all_trials": [
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": str(trial.state),
                "metrics": {
                    "avg_val_r": trial.user_attrs.get("avg_val_r"),
                    "std_val_r": trial.user_attrs.get("std_val_r"),
                    "avg_val_mse": trial.user_attrs.get("avg_val_mse"),
                    "std_val_mse": trial.user_attrs.get("std_val_mse"),
                    "fold_r_scores": trial.user_attrs.get("fold_r_scores", []),
                    "fold_mse_scores": trial.user_attrs.get("fold_mse_scores", []),
                },
            }
            for trial in study.trials
        ],
    }

    results_path = output_dir / f"{args.model}_diffusion_search_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
