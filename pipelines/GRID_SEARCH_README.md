# Grid Search Pipeline for Enhanced Models

## Overview

The grid search pipeline automatically tests multiple hyperparameter combinations for BrainGT Enhanced, BrainGNN Enhanced, and FBNetGen Enhanced models.

## Quick Start

### 1. Quick Test (Recommended First)

Test with a small grid to verify everything works:

```bash
python pipelines/run_grid_search_pipeline.py --quick_test --models braingnn fbnetgen
```

This runs:
- 2 models (BrainGNN, FBNetGen)
- 2 epochs per config
- 1 fold (graphs_outer1_inner2)
- Small hyperparameter grid (8 configs per model)
- **Total: 16 training runs (~10-15 minutes)**

### 2. Single Model Grid Search

Search for best config for one model:

```bash
# FBNetGen only
python pipelines/run_grid_search_pipeline.py \
    --models fbnetgen \
    --hidden_dims 64 128 256 \
    --dropouts 0.2 0.3 0.4 \
    --learning_rates 1e-4 5e-4 \
    --n_layers 2 3 4 \
    --n_heads_fbnetgen 2 4 8 \
    --epochs 100
```

This searches: 3×3×2×3×3 = **162 configurations**

### 3. Full Grid Search (All Models)

**⚠️ WARNING: This can take many hours/days!**

```bash
python pipelines/run_grid_search_pipeline.py \
    --hidden_dims 64 128 256 \
    --dropouts 0.2 0.3 0.4 \
    --learning_rates 1e-4 5e-4 1e-3 \
    --epochs 100
```

## Hyperparameter Ranges

### Common Parameters (All Models)

- `--hidden_dims`: Hidden layer dimensions (default: [128])
- `--dropouts`: Dropout rates (default: [0.3])
- `--learning_rates`: Learning rates (default: [1e-4])
- `--epochs`: Number of training epochs (default: 100)

### Model-Specific Parameters

**BrainGT:**
- `--n_heads_braingt`: Attention heads (default: [8])

**BrainGNN:**
- `--n_layers`: GNN layers (default: [3])

**FBNetGen:**
- `--n_layers`: GNN layers (default: [3])
- `--n_heads_fbnetgen`: Attention heads (default: [4])

## Results Structure

```
results/grid_search/
├── braingt/
│   ├── h64_d0.2_lr1e-04_nh4/
│   │   ├── aggregate_summary.json
│   │   └── config.json
│   ├── h64_d0.2_lr1e-04_nh8/
│   └── ...
├── braingnn/
│   ├── h64_d0.2_lr1e-04_nl2/
│   └── ...
├── fbnetgen/
│   ├── h64_d0.2_lr1e-04_nl2_nh2/
│   └── ...
├── comparison_report.csv
├── comparison_report.xlsx
└── plots/
    ├── braingt_hyperparameter_impact.png
    ├── braingnn_hyperparameter_impact.png
    ├── fbnetgen_hyperparameter_impact.png
    └── model_comparison.png
```

## Analyzing Results

### View Summary in Terminal

```bash
python analysis/compare_enhanced_grid_search.py
```

### Show Top 5 Configs per Model

```bash
python analysis/compare_enhanced_grid_search.py --top 5
```

### Generate Visualization Plots

```bash
python analysis/compare_enhanced_grid_search.py --plot
```

This creates:
- Hyperparameter impact plots (how each parameter affects performance)
- Model comparison plots
- Config-by-config comparison

### Open Results in Excel

```bash
# Open in Excel/LibreOffice
results/grid_search/comparison_report.xlsx
```

Columns:
- `model`: Model name
- `config`: Configuration identifier
- `subject_r`: Subject-level correlation (↑ higher is better)
- `subject_mse`: Subject-level MSE (↓ lower is better)
- `hidden_dim`, `dropout`, `lr`, etc.: Hyperparameters

## Example Workflows

### Workflow 1: Find Best FBNetGen Config

```bash
# 1. Run grid search (quick test first)
python pipelines/run_grid_search_pipeline.py \
    --models fbnetgen \
    --quick_test

# 2. Check results
python analysis/compare_enhanced_grid_search.py --top 3

# 3. Run full grid search with best ranges
python pipelines/run_grid_search_pipeline.py \
    --models fbnetgen \
    --hidden_dims 128 256 \
    --dropouts 0.2 0.3 \
    --learning_rates 1e-4 5e-4 \
    --n_layers 3 4 \
    --n_heads_fbnetgen 4 8 \
    --epochs 100

# 4. Visualize results
python analysis/compare_enhanced_grid_search.py --plot
```

### Workflow 2: Compare All Models

```bash
# 1. Quick test all models
python pipelines/run_grid_search_pipeline.py --quick_test

# 2. View comparison
python analysis/compare_enhanced_grid_search.py --plot

# 3. Pick best model, run full search
python pipelines/run_grid_search_pipeline.py \
    --models fbnetgen \
    --hidden_dims 64 128 256 \
    --epochs 100
```

### Workflow 3: Coarse-to-Fine Search

```bash
# 1. Coarse search (wide range, fewer values)
python pipelines/run_grid_search_pipeline.py \
    --models braingnn \
    --hidden_dims 64 256 \
    --dropouts 0.2 0.4 \
    --learning_rates 1e-5 1e-3 \
    --epochs 50

# 2. Check best region
python analysis/compare_enhanced_grid_search.py --top 1

# Output shows: h128, d0.3, lr1e-4 is best region

# 3. Fine search (narrow range, more values)
python pipelines/run_grid_search_pipeline.py \
    --models braingnn \
    --hidden_dims 96 128 160 \
    --dropouts 0.25 0.3 0.35 \
    --learning_rates 5e-5 1e-4 2e-4 \
    --epochs 100
```

## Interpreting Results

### Subject-Level Correlation (r)

- **r > 0.5**: Excellent (strong brain-behavior relationship)
- **r = 0.3-0.5**: Good (meaningful relationship)
- **r = 0.1-0.3**: Weak (some signal)
- **r < 0.1**: Poor (barely better than random)

### Comparing Configs

**Example output:**
```
BRAINGNN:
  Rank 1:
    Subject r: 0.4523
    Parameters: hidden_dim=128, dropout=0.3, lr=1e-04
  Rank 2:
    Subject r: 0.4201
    Parameters: hidden_dim=256, dropout=0.2, lr=5e-04
```

**Interpretation:**
- Config 1 is best (r=0.4523)
- Config 2 is close (r=0.4201)
- Difference of 0.03 is meaningful

## Tips

### Reduce Search Time

1. **Use --quick_test first** to validate the pipeline
2. **Search one model at a time** instead of all three
3. **Use fewer hyperparameter values** (e.g., 2-3 values per param)
4. **Reduce --epochs** for initial exploration (50 instead of 100)

### Effective Search Strategy

1. **Start coarse**: Wide range, few values
2. **Identify promising region**: Check top configs
3. **Refine**: Narrow range around best config, more values
4. **Validate**: Re-run best config multiple times

### Parallel Execution

To run multiple searches simultaneously:

```bash
# Terminal 1
python pipelines/run_grid_search_pipeline.py --models braingt --quick_test

# Terminal 2 (different GPU)
python pipelines/run_grid_search_pipeline.py --models braingnn --device cuda:1 --quick_test
```

## Troubleshooting

### "No results found to compare"

Check if grid search completed successfully:
```bash
ls results/grid_search/
```

Should see model directories (braingt/, braingnn/, fbnetgen/).

### Out of Memory Errors

Reduce:
- `--hidden_dims` (try 64, 128 instead of 256, 512)
- Batch size (modify in train_enhanced_models.py)

### Taking Too Long

- Use `--quick_test` for testing
- Search one model at a time
- Reduce hyperparameter grid size
- Reduce `--epochs`

## Advanced Usage

### Custom Hyperparameter Grid

Edit the script to add new hyperparameters:

```python
# In run_grid_search_pipeline.py, add new argument:
parser.add_argument('--weight_decays', nargs='+', type=float,
                    default=[1e-5], help='Weight decay values')
```

### Using Best Config for Full Training

After finding best config, train on all folds:

```bash
# From grid search, best config is: h128_d0.3_lr1e-04_nl3

# Train on all folds
python training/advanced/train_enhanced_models.py \
    --model braingnn \
    --hidden_dim 128 \
    --dropout 0.3 \
    --lr 1e-4 \
    --n_layers 3 \
    --epochs 100 \
    --device cuda
```

## Summary

✓ **Quick test**: `--quick_test` for fast validation
✓ **Model selection**: `--models` to choose which models to search
✓ **Hyperparameter ranges**: Multiple arguments for each parameter
✓ **Automatic comparison**: Results saved to CSV/Excel automatically
✓ **Visualization**: Use `--plot` flag to generate plots
✓ **Flexible**: Search one model or all three simultaneously

**Start with quick test, then expand to full grid search!**
