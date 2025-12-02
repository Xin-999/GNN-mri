# Movie HCP Brain Graphs
> Build Ledoit-Wolf sliding-window brain graphs and PyTorch Geometric folds for the HCP 7T Movie dataset using Shen 268 ROI time-series plus cognitive scores.

Dynamic functional connectivity pipeline for the HCP 7T Movie/Rest dataset.  
Step 1 loads Shen 268 ROI time-series (`data/all_shen_roi_ts/*.txt`), associates each run with a user-provided cognitive regression score, builds sliding‑window Ledoit‑Wolf correlation graphs, and stores them under `data/ldw_data`. Step 2 pads those sequences, creates nested cross‑validation splits, and serializes PyTorch Geometric graph objects for downstream GNN experiments.

## Project Layout
- `step1_compute_ldw.py` – loads raw ROI time‑series, looks up cognitive regression scores (by default `Subject` → `cogn_PC1` in `data/cogn_pc_scores.csv`), computes Ledoit‑Wolf covariance per window, applies proportional thresholding, and writes `LDW_movie-hcp_data.pkl` plus `win_info.pkl`.
- `step2_prepare_data.py` – pads node/adjacency sequences, builds nested K-Fold splits tailored to the small Movie HCP sample, converts each window to `torch_geometric.data.Data`, and saves `graphs_outerX_innerY.pkl` under `data/folds_data`.
- `data/all_shen_roi_ts/` – input ROI time-series (tab-delimited, one file per subject/run).
- `requirements.txt` – Python dependencies.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# install torch-geometric version matching your torch/cu build:
pip install torch-geometric
```
See https://pytorch-geometric.readthedocs.io/ for platform-specific wheels.

## Usage
1. Place the Shen ROI time-series `.txt` files in `data/all_shen_roi_ts/`.
   - You can download the Movie HCP ROI time series directly from https://github.com/esfinn/movie_cpm/tree/master/data/all_shen_roi_ts and copy the files into this folder.
   - Cognitive targets default to the cognitive principal components file (`data/cogn_pc_scores.csv`) via `Subject` → `cogn_PC1`. To use a different column (e.g., `cogn_PC3`) or an entirely different CSV, pass `--score-column cogn_PC3` or `--score-csv /path/to/file.csv`. Set `--score-key run_id --score-template` if you prefer per-run targets; the script will create a template CSV listing every run identifier you need to fill.
2. Generate Ledoit-Wolf graphs (outputs `data/ldw_data/LDW_movie-hcp_data.pkl` + `win_info.pkl`):
   ```bash
   python3 step1_compute_ldw.py \
     --score-column cogn_PC1 \
     --score-csv ./data/cogn_pc_scores.csv \
     --score-key subject
   ```
   Customize `wSize`, `shift`, etc. inside the script if needed.
   > **Note:** Sliding-window parameters (`wSize`, `shift`, and the proportional threshold `p`) live inside `step1_compute_ldw.py`. Tweak them before running if your analysis needs different temporal granularity or sparsity.
3. Create cross-validation folds and PyG graph tensors (requires `torch_geometric`):
   ```bash
   python3 step2_prepare_data.py
   ```
   Outputs `data/folds_data/graphs_outer*_inner*.pkl`.

## Notes
- Cognitive scores are user-defined regression targets; choose any column in `data/cogn_pc_scores.csv` or supply your own CSV via the `--score-*` flags. The processed pickle stores them under `cognitive_scores` for downstream models.
- Fold generation automatically reduces the number of splits when there are too few samples, ensuring the Movie HCP subset remains usable.
- Ledoit-Wolf estimation uses `sklearn.covariance.ledoit_wolf`, which is more stable than the estimator class on macOS.
