#!/usr/bin/env python
"""
Export per-fold *_predictions.json and *_summary.json to Excel.

Creates:
  - One Excel file per fold with sheets: "predictions" and "summary"
  - An aggregate summary Excel with per-fold metrics + mean/std rows

Usage:
  python analysis/export_results_to_excel.py --results_root "C:\\path\\to\\results\\advanced\\fbnetgen_v2"
  python analysis/export_results_to_excel.py --results_root "C:\\path\\to\\results\\advanced\\fbnetgen_v2" --output_dir "C:\\path\\to\\excel"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def find_result_files(folder: Path) -> Tuple[Optional[Path], Optional[Path]]:
    summary = None
    preds = None
    for item in folder.iterdir():
        name = item.name.lower()
        if name.endswith("_summary.json") and summary is None:
            summary = item
        if name.endswith("_predictions.json") and preds is None:
            preds = item
    return summary, preds


def extract_metrics(summary: Dict) -> Dict[str, Optional[float]]:
    test_metrics = summary.get("test_metrics", {}) if isinstance(summary, dict) else {}
    window = test_metrics.get("window_level", {}) or {}
    subject = test_metrics.get("subject_level", {}) or {}

    return {
        "window_mse": window.get("mse"),
        "window_mae": window.get("mae"),
        "window_pearson_r": window.get("pearson_r"),
        "window_pearson_p": window.get("pearson_p"),
        "window_spearman_r": window.get("spearman_r"),
        "window_spearman_p": window.get("spearman_p"),
        "window_r2": window.get("r2"),
        "subject_mse": subject.get("mse"),
        "subject_mae": subject.get("mae"),
        "subject_pearson_r": subject.get("pearson_r"),
        "subject_pearson_p": subject.get("pearson_p"),
        "subject_spearman_r": subject.get("spearman_r"),
        "subject_spearman_p": subject.get("spearman_p"),
        "subject_r2": subject.get("r2"),
    }


def flatten_summary(summary: Dict) -> List[Dict[str, str]]:
    rows = []

    def add_row(key, value):
        rows.append({"key": key, "value": value})

    if not isinstance(summary, dict):
        add_row("error", "summary JSON is not a dict")
        return rows

    for key in ["model_name", "fold", "timestamp", "hostname", "device"]:
        if key in summary:
            add_row(key, summary.get(key))

    config = summary.get("config", {})
    if isinstance(config, dict):
        for k, v in sorted(config.items()):
            add_row(f"config.{k}", v)

    training_summary = summary.get("training_summary", {})
    if isinstance(training_summary, dict):
        for k, v in sorted(training_summary.items()):
            add_row(f"training.{k}", v)

    best_validation = summary.get("best_validation", {})
    if isinstance(best_validation, dict):
        for k, v in sorted(best_validation.items()):
            add_row(f"best_validation.{k}", v)

    data_info = summary.get("data_info", {})
    if isinstance(data_info, dict):
        for k, v in sorted(data_info.items()):
            add_row(f"data.{k}", v)

    metrics = extract_metrics(summary)
    for k, v in metrics.items():
        add_row(f"metrics.{k}", v)

    history = summary.get("history")
    if isinstance(history, list):
        add_row("history_len", len(history))

    return rows


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_sheet_name(name: str) -> str:
    name = name.replace("/", "_").replace("\\", "_")
    return name[:31]


def ensure_unique_dir(base_dir: Path) -> Path:
    base_dir = Path(base_dir)
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    if base_dir.is_dir() and any(base_dir.iterdir()):
        suffix = 1
        while True:
            candidate = base_dir.with_name(f"{base_dir.name}_{suffix}")
            if not candidate.exists():
                candidate.mkdir(parents=True, exist_ok=True)
                return candidate
            suffix += 1

    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Export results JSONs to Excel")
    parser.add_argument("--results_root", type=str, required=True,
                        help="Path containing per-fold results")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save Excel files (default: results_root/excel_exports)")
    parser.add_argument("--single_workbook", action="store_true",
                        help="Write one Excel file with all folds as tabs")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    if not results_root.exists():
        print(f"Error: results_root not found: {results_root}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else results_root / "excel_exports"
    output_dir = ensure_unique_dir(output_dir)

    summary_path, preds_path = find_result_files(results_root)
    if summary_path and preds_path:
        fold_dirs = [results_root]
    else:
        fold_dirs = [d for d in results_root.iterdir() if d.is_dir()]

    if not fold_dirs:
        print(f"No fold directories found under {results_root}")
        sys.exit(1)

    aggregate_rows = []
    combined_results = []

    for fold_dir in fold_dirs:
        summary_path, preds_path = find_result_files(fold_dir)
        if not summary_path or not preds_path:
            print(f"Skipping {fold_dir.name}: missing summary/predictions JSON")
            continue

        summary = load_json(summary_path)
        predictions = load_json(preds_path)

        fold_name = summary.get("fold") if isinstance(summary, dict) else None
        if not fold_name:
            fold_name = fold_dir.name

        preds_df = pd.DataFrame(predictions)
        summary_df = pd.DataFrame(flatten_summary(summary))

        if args.single_workbook:
            combined_results.append((fold_name, preds_df, summary_df))
        else:
            out_path = output_dir / f"{fold_name}_results.xlsx"
            with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
                preds_df.to_excel(writer, sheet_name="predictions", index=False)
                summary_df.to_excel(writer, sheet_name="summary", index=False)

        metrics = extract_metrics(summary)
        metrics["fold"] = fold_name
        aggregate_rows.append(metrics)

        if not args.single_workbook:
            print(f"Saved {out_path}")

    if aggregate_rows:
        agg_df = pd.DataFrame(aggregate_rows)
        numeric_cols = [c for c in agg_df.columns if c != "fold"]

        mean_row = {"fold": "MEAN"}
        std_row = {"fold": "STD"}
        for col in numeric_cols:
            mean_row[col] = agg_df[col].mean()
            std_row[col] = agg_df[col].std()

        agg_df = pd.concat([agg_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

        agg_path = output_dir / "aggregate_summary.xlsx"
        with pd.ExcelWriter(agg_path, engine="openpyxl") as writer:
            agg_df.to_excel(writer, sheet_name="summary", index=False)

        print(f"Saved {agg_path}")

    if args.single_workbook and combined_results:
        combined_path = output_dir / "all_folds_results.xlsx"
        with pd.ExcelWriter(combined_path, engine="openpyxl") as writer:
            for fold_name, preds_df, summary_df in combined_results:
                preds_sheet = safe_sheet_name(f"{fold_name}_preds")
                summary_sheet = safe_sheet_name(f"{fold_name}_summary")
                preds_df.to_excel(writer, sheet_name=preds_sheet, index=False)
                summary_df.to_excel(writer, sheet_name=summary_sheet, index=False)
            if aggregate_rows:
                agg_df.to_excel(writer, sheet_name="aggregate_summary", index=False)

        print(f"Saved {combined_path}")


if __name__ == "__main__":
    main()
