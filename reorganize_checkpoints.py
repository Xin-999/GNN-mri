#!/usr/bin/env python
"""
Reorganize checkpoint files for interpretability analysis.

Converts from flat structure:
  gatv2_best_graphs_outer1_inner1.pt
To hierarchical structure:
  graphs_outer1_inner1/gatv2_best.pt
"""
import shutil
from pathlib import Path

# Source directory with flat structure
source_dir = Path("results_gatv2_interpretable")

# Find all checkpoint files
checkpoint_files = list(source_dir.glob("gatv2_best_graphs_*.pt"))

if not checkpoint_files:
    print(f"No checkpoint files found in {source_dir}")
    exit(1)

print(f"Found {len(checkpoint_files)} checkpoint files to reorganize\n")

for ckpt_file in checkpoint_files:
    # Extract fold name: gatv2_best_graphs_outer1_inner1.pt -> graphs_outer1_inner1
    fold_name = ckpt_file.stem.replace("gatv2_best_", "")

    # Create subdirectory
    fold_dir = source_dir / fold_name
    fold_dir.mkdir(exist_ok=True)

    # Copy checkpoint to subdirectory
    dest_file = fold_dir / "gatv2_best.pt"
    shutil.copy2(ckpt_file, dest_file)

    print(f"Created {fold_dir.relative_to(source_dir)}/gatv2_best.pt")

print(f"\nDone! Reorganized {len(checkpoint_files)} checkpoints")
print(f"\nNow you can run:")
print(f"  python analysis/analyze_gatv2_interpretability.py --model_dir {source_dir} --device cuda --top_k 20")
