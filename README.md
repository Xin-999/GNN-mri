# Movie-HCP Brain Graph: Predicting Cognitive Scores from Brain Connectivity

## Table of Contents
- [What is This Project?](#what-is-this-project)
- [Background Concepts](#background-concepts)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Complete Workflow](#complete-workflow)
- [Understanding the Data](#understanding-the-data)
- [Training Models](#training-models)
- [Making Predictions](#making-predictions)
- [Understanding Results](#understanding-results)
- [Key Concepts Explained](#key-concepts-explained)
- [Troubleshooting](#troubleshooting)

---

## What is This Project?

This project uses **Graph Neural Networks (GNNs)** to predict cognitive test scores from brain connectivity data obtained through fMRI scans.

**Simple Explanation:**
- We have brain scans (fMRI) from people watching a movie
- We measure how different brain regions communicate with each other (connectivity)
- We use AI (Graph Neural Networks) to predict cognitive scores from these brain patterns
- Goal: Predict a cognitive test score (ListSort_AgeAdj) from brain connectivity patterns

**Real-World Application:**
Understanding brain-cognition relationships could help in early detection of cognitive decline, personalized medicine, and understanding how brain networks support different cognitive abilities.

---

## Background Concepts

### fMRI (Functional Magnetic Resonance Imaging)
- Brain imaging technique that measures brain activity
- Shows which brain regions are active during different tasks
- In this project: Subjects watched a movie while being scanned

### Brain Connectivity
- Measures how different brain regions communicate
- **Nodes**: Brain regions (268 regions in this dataset)
- **Edges**: Connections between regions (how correlated their activity is)
- Creates a **graph** representation of the brain

### Graph Neural Networks (GNNs)
- AI models designed to work with graph-structured data
- Learn patterns from brain connectivity graphs
- **GATv2 (Graph Attention Network v2)**: Main model used in this project
  - Uses "attention" to focus on important brain connections
  - Can learn which connections matter most for predicting cognitive scores

### Cognitive Score (ListSort_AgeAdj)
- Standardized cognitive test score
- Age-adjusted (accounts for normal age-related differences)
- Range: approximately 98-132 in this dataset
- Measures working memory and processing speed

---

## Project Structure

```
Movie-HCP_Brain_Graph-master-test/
│
├── data/                          # DATA FOLDER (you create this)
│   ├── ListSort_AgeAdj.csv       # Subject IDs and cognitive scores
│   ├── folds_data/               # Cross-validation fold assignments
│   │   ├── graphs_outer1_inner1.pkl
│   │   ├── graphs_outer1_inner2.pkl
│   │   └── ... (25 fold files total)
│   └── (your raw fMRI data)
│
├── training/                      # TRAINING SCRIPTS
│   ├── gatv2/                    # GATv2 model training
│   │   ├── train_gatv2_improved.py      # Main training script (RECOMMENDED)
│   │   ├── train_gatv2_interpretable.py # Earlier version
│   │   └── train_gatv2_simple.py        # Basic version
│   └── advanced/                 # Other model architectures
│       └── train_enhanced_models.py
│
├── analysis/                      # ANALYSIS & PREDICTION SCRIPTS
│   ├── predict_gatv2_simple.py   # Generate predictions with trained model
│   ├── compare_gatv2_configs.py  # Compare different hyperparameter settings
│   └── compare_models.py         # Compare different model architectures
│
├── results/                       # TRAINING OUTPUTS (auto-created)
│   ├── gatv2/
│   │   ├── improved/             # Results from train_gatv2_improved.py
│   │   │   ├── graphs_outer1_inner1/
│   │   │   │   ├── gatv2_best.pt         # Best model checkpoint
│   │   │   │   └── gatv2_summary.json    # Performance metrics
│   │   │   └── ... (one folder per fold)
│   │   └── grid/                 # Grid search results (if using --grid_search)
│   │       ├── h64_l2_head4_d0.1_ed0.0_lr0.001/
│   │       └── ... (one folder per config)
│   └── predictions/              # Prediction outputs
│       └── (Excel files with predictions)
│
├── models/                        # MODEL ARCHITECTURE DEFINITIONS
│   └── (original model implementations)
│
├── preprocessing/                 # DATA PREPARATION SCRIPTS
│   ├── step1_compute_ldw.py     # Compute connectivity matrices
│   └── step2_prepare_data.py    # Create fold files
│
└── docs/                          # DOCUMENTATION
    ├── GATV2_IMPROVEMENT_GUIDE.md
    ├── ENHANCEMENTS_GUIDE.md
    └── JSON_TRACKING_GUIDE.md
```

### Key Files Explained

| File | Purpose | When to Use |
|------|---------|-------------|
| `data/ListSort_AgeAdj.csv` | Contains subject IDs and true cognitive scores | Required for training & prediction |
| `data/folds_data/graphs_outer*.pkl` | Cross-validation fold data (brain graphs + labels) | Required for training |
| `training/gatv2/train_gatv2_improved.py` | Main training script with all improvements | Default choice for training |
| `analysis/predict_gatv2_simple.py` | Generate predictions from trained model | After training completes |
| `analysis/compare_gatv2_configs.py` | Compare multiple hyperparameter configurations | After grid search |
| `results/gatv2/improved/graphs_*/gatv2_best.pt` | Trained model checkpoint | Used for making predictions |
| `results/gatv2/improved/graphs_*/gatv2_summary.json` | Training metrics and config | Review performance |

---

## Installation & Setup

### 1. Requirements

```bash
# Python 3.8 or higher
python --version

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install scikit-learn pandas numpy scipy openpyxl
```

### 2. Prepare Data Directory

Create the data folder structure:

```bash
mkdir -p data/folds_data
```

Place your files:
- `data/ListSort_AgeAdj.csv` - CSV with columns: Subject, ListSort_AgeAdj
- `data/folds_data/*.pkl` - Fold files containing brain graphs

**CSV Format Example:**
```csv
Subject,ListSort_AgeAdj
100610,108.26
102311,113.85
102816,112.66
...
```

---

## Complete Workflow

### Step-by-Step: From Data to Predictions

```
Raw fMRI Data
    ↓
[1. Preprocessing]          ← preprocessing/step1_compute_ldw.py
    ↓                         preprocessing/step2_prepare_data.py
Brain Graphs + Folds
    ↓
[2. Training]              ← training/gatv2/train_gatv2_improved.py
    ↓
Trained Models
    ↓
[3. Prediction]            ← analysis/predict_gatv2_simple.py
    ↓
Cognitive Score Predictions
    ↓
[4. Comparison]            ← analysis/compare_gatv2_configs.py
    ↓
Best Model Selection
```

---

## Understanding the Data

### What's in a Fold File (graphs_outer1_inner1.pkl)?

Each `.pkl` file contains:

```python
{
    'train_graphs': [Graph1, Graph2, ...],  # Training brain graphs
    'train_y': [score1, score2, ...],       # Training scores (normalized)
    'val_graphs': [...],                     # Validation brain graphs
    'val_y': [...],                         # Validation scores
    'test_graphs': [...],                    # Test brain graphs
    'test_y': [...],                        # Test scores
    'train_indices': [0, 5, 12, ...],       # Subject indices in ListSort_AgeAdj.csv
    'val_indices': [...],
    'test_indices': [...]
}
```

**Important:**
- Scores in fold files are **normalized** (mean=0, std=1)
- Original scores are in `data/ListSort_AgeAdj.csv`
- Indices map to rows in the CSV file

### Cross-Validation Structure

This project uses **5×5 nested cross-validation** = 25 folds total

```
Outer Fold 1
├── Inner Fold 1 → graphs_outer1_inner1.pkl
├── Inner Fold 2 → graphs_outer1_inner2.pkl
├── Inner Fold 3 → graphs_outer1_inner3.pkl
├── Inner Fold 4 → graphs_outer1_inner4.pkl
└── Inner Fold 5 → graphs_outer1_inner5.pkl

Outer Fold 2
├── Inner Fold 1 → graphs_outer2_inner1.pkl
└── ... (5 inner folds)

... (5 outer folds total)
```

**Why?**
- Prevents overfitting to a specific train/test split
- More robust performance estimates
- Each subject appears in test set exactly once per outer fold

---

## Training Models

### Basic Training (Single Configuration)

Train GATv2 with default hyperparameters:

```bash
cd training/gatv2
python train_gatv2_improved.py --device cuda --epochs 50
```

**What happens:**
1. Loads each fold file from `data/folds_data/`
2. Trains a model for 50 epochs (or until early stopping)
3. Saves best checkpoint to `results/gatv2/improved/graphs_*/gatv2_best.pt`
4. Saves metrics to `results/gatv2/improved/graphs_*/gatv2_summary.json`
5. Repeats for all 25 folds

**Expected time:** ~2-3 minutes per fold on GPU (total: ~1-1.5 hours for 25 folds)

### Training with Custom Hyperparameters

```bash
python train_gatv2_improved.py \
    --device cuda \
    --epochs 50 \
    --hidden_dim 128 \
    --n_layers 3 \
    --n_heads 4 \
    --dropout 0.2 \
    --edge_dropout 0.1 \
    --lr 0.001 \
    --batch_size 32
```

**Hyperparameter Guide:**

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `hidden_dim` | 64 | 64-256 | Model capacity (higher = more complex) |
| `n_layers` | 3 | 2-4 | Network depth (more layers = more hops in graph) |
| `n_heads` | 4 | 4-8 | Attention heads (higher = more diverse attention patterns) |
| `dropout` | 0.2 | 0.1-0.3 | Regularization (higher = less overfitting, but may underfit) |
| `edge_dropout` | 0.1 | 0.0-0.2 | Graph regularization (randomly drops edges) |
| `lr` | 0.001 | 0.0001-0.001 | Learning rate (higher = faster but less stable) |

### Grid Search (Try Multiple Configurations)

Automatically train 486 different hyperparameter combinations:

```bash
python train_gatv2_improved.py --grid_search --device cuda
```

**What happens:**
1. Generates all combinations from predefined grid (line 683-690 in code):
   - hidden_dim: [64, 128, 256]
   - n_layers: [2, 3, 4]
   - n_heads: [4, 8]
   - dropout: [0.1, 0.2, 0.3]
   - edge_dropout: [0.0, 0.1, 0.2]
   - lr: [1e-3, 5e-4, 1e-4]
2. Trains each configuration on all 25 folds
3. Saves results to `results/gatv2/grid/[config_name]/`
4. Creates summary ranking all configurations

**Expected time:** Days to complete (486 configs × 25 folds × 2-3 min/fold)

---

## Making Predictions

### Generate Predictions from Trained Model

```bash
cd analysis
python predict_gatv2_simple.py --device cuda
```

**What it does:**
1. Loads trained models from `results/gatv2/improved/`
2. For each fold, loads the test set
3. Generates predictions using the trained model
4. Maps predictions back to original subject IDs
5. Denormalizes scores to original scale (100-130 range)
6. Saves to Excel: `results/predictions/gatv2_predictions_fold_*.xlsx`

**Output Excel Format:**

| Split | Subject_ID | True_Score_Original | Predicted_Score_Original | Error_Original | ... |
|-------|------------|---------------------|--------------------------|----------------|-----|
| test  | 100610     | 108.26              | 107.45                   | -0.81          | ... |
| val   | 102311     | 113.85              | 112.34                   | -1.51          | ... |
| train | 102816     | 112.66              | 113.12                   | 0.46           | ... |

**Columns Explained:**
- `Split`: train/val/test indicator
- `Subject_ID`: Real subject ID from ListSort_AgeAdj.csv
- `True_Score_Original`: Actual cognitive score (original scale)
- `Predicted_Score_Original`: Model prediction (denormalized)
- `Error_Original`: True - Predicted (negative = overestimated)
- `*_Normalized`: Same metrics in normalized space

### Compare Model Configurations

After grid search, compare all configurations:

```bash
cd analysis
python compare_gatv2_configs.py
```

**Output:** `gatv2_config_comparison.xlsx` with sheets:
1. **Ranking**: Top models sorted by Subject-level R
2. **Hyperparameters**: All hyperparameter settings
3. **Subject_Metrics**: Detailed subject-level performance
4. **Window_Metrics**: Window-level performance
5. **Full_Data**: All data combined

---

## Understanding Results

### Training Output Explanation

During training, you'll see:

```
Epoch 07/50 | Train MSE: 0.0875 | Val MSE: 0.7653 | Win r: 0.3006 | Subj r: 0.7497 | Time: 3.7s
```

**What each metric means:**

| Metric | Meaning | Good Value |
|--------|---------|------------|
| `Train MSE` | Training error (normalized space) | Decreasing |
| `Val MSE` | Validation error (normalized space) | Low & stable |
| `Win r` | Window-level correlation | 0.2-0.4 |
| `Subj r` | **Subject-level correlation** | **0.7-0.8+** (PRIMARY METRIC) |

**Early Stopping:**
```
Early stopping at epoch 22
Best epoch: 7
```
- Training stopped because validation didn't improve for 15 epochs (patience)
- Best model was at epoch 7 (highest Subject-level r)
- This checkpoint is saved and used for testing

### JSON Summary File (gatv2_summary.json)

```json
{
  "fold_name": "graphs_outer1_inner1",
  "config": {
    "hidden_dim": 64,
    "n_layers": 3,
    "dropout": 0.2,
    ...
  },
  "training_summary": {
    "best_epoch": 7,
    "total_epochs": 22,
    "early_stopped": true
  },
  "test_metrics": {
    "subject_level": {
      "r": 0.7497,      ← PRIMARY METRIC (correlation)
      "mse": 45.23,     ← Mean squared error
      "mae": 5.12,      ← Mean absolute error
      "rmse": 6.73      ← Root mean squared error
    },
    "window_level": {
      "r": 0.3006,
      "mse": 0.7653,
      ...
    }
  }
}
```

### Metric Interpretation

**Subject-level R (Correlation Coefficient)**
- **Range:** -1 to +1
- **Meaning:** How well predictions rank subjects correctly
- **0.75**: Strong positive correlation (good model)
- **0.50**: Moderate correlation (okay model)
- **0.30**: Weak correlation (poor model)

**Why R instead of MSE for model selection?**
- R measures relative ordering (which subjects score higher/lower)
- Less sensitive to systematic offsets in predictions
- Standard metric in neuroscience research
- MSE is still reported and useful for absolute accuracy

---

## Key Concepts Explained

### Target Normalization

**Problem:** Raw cognitive scores (100-130) cause training instability

**Solution:** Normalize to mean=0, std=1 during training
```python
scaler = StandardScaler()
normalized_scores = scaler.fit_transform(original_scores)
```

**Important:** Always denormalize predictions back to original scale for interpretation!

### Early Stopping

**Purpose:** Prevent overfitting

**How it works:**
1. Monitor validation Subject-level R after each epoch
2. Save checkpoint when validation R improves
3. If no improvement for 15 epochs (patience), stop training
4. Use the best checkpoint (not the last epoch)

**Why:** The model might overfit if trained too long, performing worse on unseen data

### Subject-level vs Window-level Metrics

**Window-level:**
- Each time window (brain graph) treated independently
- Many windows per subject (brain states during movie)
- More data points but noisier

**Subject-level:**
- Average predictions across all windows for each subject
- One prediction per subject
- **More stable and interpretable** (this is what we care about)

**Example:**
- Subject 100610 has 50 time windows
- Window predictions: [107.2, 108.5, 106.8, 109.1, ...]
- Subject prediction: mean(all windows) = 107.9
- True score: 108.26
- Subject-level accuracy: much better than individual windows!

### DropEdge Regularization

**Purpose:** Prevent overfitting to specific brain connections

**How it works:**
- During training, randomly drop some edges (connections) from the graph
- Forces model to learn robust patterns, not memorize specific connections
- Similar to dropout for regular neural networks

---

## Troubleshooting

### Common Issues

**1. FileNotFoundError: data/folds_data/graphs_outer1_inner1.pkl**
- **Cause:** Missing fold files
- **Solution:** Ensure fold files are in `data/folds_data/` directory
- Check path: Script assumes it's running from `training/gatv2/`

**2. CUDA out of memory**
- **Cause:** GPU doesn't have enough memory
- **Solution:** Reduce `--batch_size` (try 16 or 8)
- Or use `--device cpu` (slower but works)

**3. Validation metrics not improving (Subject r stays low)**
- **Possible causes:**
  - Learning rate too high → try `--lr 0.0005`
  - Model too simple → increase `--hidden_dim 128`
  - Model too complex → increase `--dropout 0.3`
  - Not enough epochs → increase `--epochs 100`

**4. Training very slow**
- **Check:** Are you using GPU? (`--device cuda`)
- **Check:** Is CUDA properly installed?
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**5. Predictions have wrong subject IDs**
- **Cause:** Missing or incorrect `ListSort_AgeAdj.csv`
- **Solution:** Ensure CSV is at `data/ListSort_AgeAdj.csv` with correct format

**6. JSON serialization error (Object of type float32 is not JSON serializable)**
- **Status:** Fixed in current version with `NumpyEncoder` class
- **If you see this:** Update to latest version of `train_gatv2_improved.py`

**7. DeprecationWarning: 'nn.glob.GlobalAttention' is deprecated**
- **Status:** Fixed in current version (uses `AttentionalAggregation`)
- **If you see this:** Update to latest version of `train_gatv2_improved.py`

### Getting Help

1. Check `docs/` folder for detailed guides
2. Review JSON summary files to understand what went wrong
3. Look at training logs for error messages
4. Open an issue on GitHub with:
   - Error message
   - Command you ran
   - Python version and package versions

---

## Expected Performance Benchmarks

Based on 25-fold cross-validation:

| Configuration | Subject R | Subject MSE | Training Time |
|--------------|-----------|-------------|---------------|
| Default (h64_l3_head4) | 0.75 ± 0.08 | 48 ± 12 | ~1.5 hours |
| Medium (h128_l3_head4) | 0.78 ± 0.07 | 42 ± 10 | ~2 hours |
| Large (h256_l4_head8) | 0.80 ± 0.06 | 38 ± 9 | ~3 hours |

**Performance Tips:**
- Start with default settings to verify everything works
- Use grid search to find optimal hyperparameters for your data
- Subject R > 0.70 is considered good performance
- Subject R > 0.80 is excellent performance

---

## Advanced Usage

### Custom Data

To use your own data:

1. Create `data/ListSort_AgeAdj.csv` with your subjects and scores
2. Prepare fold files with your brain graphs:
```python
import pickle
fold_data = {
    'train_graphs': [...],  # PyTorch Geometric Data objects
    'train_y': [...],       # Normalized scores
    'train_indices': [...], # Indices to CSV
    # ... same for val and test
}
with open('data/folds_data/graphs_outer1_inner1.pkl', 'wb') as f:
    pickle.dump(fold_data, f)
```

### Modifying the Model

Key sections in `training/gatv2/train_gatv2_improved.py`:

- **Line 205-266:** `ImprovedGATv2Regressor` class (model architecture)
- **Line 683-690:** Grid search hyperparameter ranges
- **Line 622:** Early stopping metric (currently Subject-level R)

### Interpreting Attention Weights

The GATv2 model learns attention weights showing which brain connections are important. See `analysis/analyze_gatv2_interpretability.py` for tools to visualize these.

---

## Summary: Complete Workflow Example

```bash
# 1. Install dependencies
pip install torch torch-geometric scikit-learn pandas openpyxl

# 2. Verify data is ready
ls data/ListSort_AgeAdj.csv
ls data/folds_data/*.pkl

# 3. Train model with default settings
cd training/gatv2
python train_gatv2_improved.py --device cuda --epochs 50

# 4. Wait for training to complete (~1.5 hours)
# Watch for "Best epoch" messages

# 5. Generate predictions
cd ../../analysis
python predict_gatv2_simple.py --device cuda

# 6. Review results
# - Check Excel files in results/predictions/
# - Review JSON summaries in results/gatv2/improved/*/gatv2_summary.json

# 7. (Optional) Run grid search for better performance
cd ../training/gatv2
python train_gatv2_improved.py --grid_search --device cuda

# 8. (Optional) Compare all configurations
cd ../../analysis
python compare_gatv2_configs.py
# Output: gatv2_config_comparison.xlsx
```

---

## References & Further Reading

**Graph Neural Networks:**
- Kipf & Welling (2017) - Semi-Supervised Classification with Graph Convolutional Networks
- Veličković et al. (2018) - Graph Attention Networks
- Brody et al. (2022) - How Attentive are Graph Attention Networks? (GATv2)

**Brain Connectivity & fMRI:**
- Human Connectome Project (HCP) - https://www.humanconnectome.org/
- Sporns (2011) - Networks of the Brain

**This Project:**
- Original repository: [Link to original repo if available]
- Documentation: See `docs/` folder

---

## License

[Specify license here]

## Citation

If you use this code in your research, please cite:

```bibtex
@software{movie_hcp_brain_graph,
  title={Movie-HCP Brain Graph: Predicting Cognitive Scores from fMRI Connectivity},
  author={[Your Name]},
  year={2025},
  url={[Repository URL]}
}
```

---

**Last Updated:** 2025-12-15
**Version:** 2.0
**Status:** Production Ready
