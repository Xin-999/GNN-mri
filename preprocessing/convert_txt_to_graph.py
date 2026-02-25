#!/usr/bin/env python
"""
Convert one ROI time-series .txt file into PyG graph windows.

This script reuses the same core computations used by training:
- step1_compute_ldw.extract_ldw_corr
- step2_prepare_data.pad_graph_seq
- step2_prepare_data.convert2graphs

Usage:
  python preprocessing/convert_txt_to_graph.py \
    --txt_path data/all_shen_roi_ts/100610_MOVIE2_7T_PA_shen268_roi_ts_gsr.txt \
    --output_path data/single_input/100610_graph_windows.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import step1_compute_ldw
import step2_prepare_data


def load_single_timeseries(txt_path: Path, expected_nrois: int) -> np.ndarray:
    ts = np.loadtxt(txt_path)
    if ts.ndim == 1:
        ts = ts[:, np.newaxis]

    # Auto-fix orientation if file is [ROI, T] instead of [T, ROI]
    if ts.shape[0] == expected_nrois and ts.shape[1] != expected_nrois:
        ts = ts.T

    if ts.shape[1] != expected_nrois:
        raise ValueError(
            f"Expected {expected_nrois} ROIs, got shape {ts.shape}. "
            "Please provide a [T, N_ROI] file with matching ROI count."
        )
    return ts


def flatten_non_padding(graphs_2d):
    flat = []
    for row in graphs_2d:
        for g in row:
            if hasattr(g, "pad") and bool(g.pad):
                continue
            flat.append(g)
    return flat


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert one time-series txt into graph windows")
    parser.add_argument("--txt_path", type=str, required=True, help="Path to one ROI time-series .txt")
    parser.add_argument("--output_path", type=str, default="data/single_input/single_graph_windows.pt",
                        help="Path to save converted graphs")
    parser.add_argument("--expected_nrois", type=int, default=268, help="Expected number of ROIs")
    parser.add_argument("--wsize", type=int, default=20, help="Sliding window size")
    parser.add_argument("--shift", type=int, default=10, help="Sliding window shift")
    parser.add_argument("--target", type=float, default=0.0,
                        help="Dummy target value used for Data.y (for inference this is not used)")
    args = parser.parse_args()

    txt_path = Path(args.txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"Input file not found: {txt_path}")

    ts = load_single_timeseries(txt_path, args.expected_nrois)
    print(f"Loaded: {txt_path}")
    print(f"Time-series shape: {ts.shape} (T, ROI)")

    # Reuse step1 logic (single subject list)
    node_feats, adj_mats, nwin = step1_compute_ldw.extract_ldw_corr(
        [ts], wSize=args.wsize, shift=args.shift
    )
    print(f"Extracted windows: {nwin[0]}")

    # Reuse step2 logic to get the exact graph format used in training
    corr_padded, seqlens = step2_prepare_data.pad_graph_seq(node_feats)
    adj_padded, _ = step2_prepare_data.pad_graph_seq(adj_mats)
    scores = np.asarray([args.target], dtype=float)
    graphs_2d = step2_prepare_data.convert2graphs(corr_padded, adj_padded, seqlens, scores)
    graphs_flat = flatten_non_padding(graphs_2d)

    # Attach single-subject id for downstream aggregation if needed
    for g in graphs_flat:
        g.subject_id = torch.tensor([0], dtype=torch.long)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "source_txt": str(txt_path),
        "time_series_shape": list(ts.shape),
        "expected_nrois": args.expected_nrois,
        "wsize": args.wsize,
        "shift": args.shift,
        "n_windows": int(nwin[0]),
        "graphs_2d": graphs_2d,     # [subject, window] Data objects
        "graphs_flat": graphs_flat, # non-padding Data objects
        "dummy_target": args.target,
    }
    torch.save(payload, output_path)
    print(f"Saved converted graphs to: {output_path}")


if __name__ == "__main__":
    main()
