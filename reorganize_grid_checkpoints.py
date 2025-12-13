#!/usr/bin/env python
"""
Reorganize grid search checkpoint files for prediction scripts.

Converts from flat structure:
  gatv2_cfg0_best_graphs_outer1_inner1.pt
To hierarchical structure:
  graphs_outer1_inner1/gatv2_best.pt
"""
import shutil
from pathlib import Path

# Source directory with flat structure
source_dir = Path("results_gatv2_grid")

# Find all checkpoint files
checkpoint_files = list(source_dir.glob("gatv2_cfg*_best_graphs_*.pt"))

if not checkpoint_files:
    print(f"No checkpoint files found in {source_dir}")
    exit(1)

print(f"Found {len(checkpoint_files)} checkpoint files to reorganize\n")

for ckpt_file in checkpoint_files:
    # Extract fold name: gatv2_cfg0_best_graphs_outer1_inner1.pt -> graphs_outer1_inner1
    # Remove the prefix "gatv2_cfg0_best_" or "gatv2_cfg1_best_" etc.
    filename = ckpt_file.stem

    # Find "graphs_" in the filename
    if "graphs_" in filename:
        fold_name = filename[filename.index("graphs_"):]
    else:
        print(f"Warning: Could not extract fold name from {ckpt_file.name}")
        continue

    # Create subdirectory
    fold_dir = source_dir / fold_name
    fold_dir.mkdir(exist_ok=True)

    # Copy checkpoint to subdirectory
    dest_file = fold_dir / "gatv2_best.pt"
    shutil.copy2(ckpt_file, dest_file)

    print(f"Created {fold_dir.relative_to(source_dir)}/gatv2_best.pt")

print(f"\nDone! Reorganized {len(checkpoint_files)} checkpoints")
print(f"\nNow you can run:")
print(f"  python analysis/predict_gatv2_simple.py --model_dir {source_dir} --device cuda --split test")
