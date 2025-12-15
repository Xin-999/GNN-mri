# Quick Start: How to Run and Compare Models

## Simple 3-Step Process

### Step 1: Train Models

### Step 2: Compare Performance

### Step 3: View Results

---

## Detailed Instructions

### 🚀 Option 1: Train Enhanced Models (Recommended)

**Enhanced models have better performance (+3-5% accuracy)**

#### Train BrainGNN Enhanced

```bash
cd training/advanced
python train_enhanced_models.py --model braingnn --epochs 100 --device cuda
```

**What happens:**
- Trains on all 25 folds
- Saves results to: `results/enhanced/braingnn_enhanced/`
- Each fold creates: `braingnn_summary.json` and `braingnn_best.pt`
- Creates aggregate summary at end

---

#### Train BrainGT Enhanced

```bash
python train_enhanced_models.py --model braingt --epochs 100 --device cuda
```

**What happens:**
- Trains on all 25 folds
- Saves results to: `results/enhanced/braingt_enhanced/`
- Uses virtual nodes + GraphNorm enhancements

---

#### Train FBNetGen Enhanced

```bash
python train_enhanced_models.py --model fbnetgen --epochs 100 --device cuda
```

**What happens:**
- Trains on all 25 folds
- Saves results to: `results/enhanced/fbnetgen_enhanced/`
- Learns task-specific graph structure

---

### 🔧 Option 2: Train Base Models (For Comparison)

**If you want to compare base vs enhanced versions:**

#### Train BrainGNN Base

```bash
python train_advanced_models.py --model braingnn --epochs 100 --device cuda
```

Saves to: `results/advanced/braingnn/`

---

#### Train BrainGT Base

```bash
python train_advanced_models.py --model braingt --epochs 100 --device cuda
```

Saves to: `results/advanced/braingt/`

---

#### Train FBNetGen Base

```bash
python train_advanced_models.py --model fbnetgen --epochs 100 --device cuda
```

Saves to: `results/advanced/fbnetgen/`

---

### ⚡ Option 3: Train ALL Models at Once (Automated)

**Use the complete pipeline:**

```bash
cd pipelines
python run_complete_pipeline.py --epochs 100 --device cuda
```

**What it does:**
1. Trains BrainGT (enhanced)
2. Trains BrainGNN (enhanced)
3. Trains FBNetGen (enhanced)
4. Creates ensemble
5. Generates comparison report

**Output:** `results/complete_pipeline/comparison_report.txt`

**Quick test (fast, uses 10 epochs, 1 fold):**
```bash
python run_complete_pipeline.py --quick_test
```

---

## 📊 Step 2: Compare Model Performance

After training, compare results using these scripts:

### Compare All Advanced Models

```bash
cd analysis
python compare_models.py
```

**What it does:**
- Reads aggregate summaries from `results_*_advanced/` directories
- Creates comparison table
- Shows best model
- Prints to console

**Output:**
```
MODEL PERFORMANCE COMPARISON
============================================================
Model               Subject r    Subject MSE    Parameters
------------------------------------------------------------
BRAINGT            0.7890       38.45          180,234
BRAINGNN           0.7623       42.12          120,456
FBNETGEN           0.7401       45.23          150,789
ENSEMBLE           0.8124       35.67          N/A
============================================================

BEST MODEL: ENSEMBLE (Subject r = 0.8124)
```

---

### Compare Base vs Enhanced Versions

```bash
python compare_original_vs_enhanced.py
```

**What it does:**
- Compares base models (`results/advanced/`) vs enhanced (`results/enhanced/`)
- Shows improvement from enhancements
- Generates detailed Excel file

**Output:**
```
BrainGNN:
  Base:     r = 0.7201 ± 0.089
  Enhanced: r = 0.7623 ± 0.074  (+5.9% improvement)

BrainGT:
  Base:     r = 0.7512 ± 0.082
  Enhanced: r = 0.7890 ± 0.065  (+5.0% improvement)
```

---

### Compare Different GATv2 Configurations

```bash
python compare_gatv2_configs.py
```

**What it does:**
- Compares different hyperparameter configurations from grid search
- Ranks by performance
- Creates Excel: `gatv2_config_comparison.xlsx`

---

## 📁 Where Results Are Saved

```
results/
├── advanced/                    # Base model results
│   ├── braingnn/
│   │   ├── graphs_outer1_inner1/
│   │   │   ├── braingnn_best.pt
│   │   │   └── braingnn_summary.json
│   │   └── braingnn_aggregate_summary.json  ← Compare this
│   ├── braingt/
│   └── fbnetgen/
│
├── enhanced/                    # Enhanced model results
│   ├── braingnn_enhanced/
│   │   ├── graphs_outer1_inner1/
│   │   │   ├── braingnn_summary.json
│   │   │   └── braingnn_best.pt
│   │   └── braingnn_aggregate_summary.json  ← Compare this
│   ├── braingt_enhanced/
│   └── fbnetgen_enhanced/
│
├── gatv2/
│   ├── improved/                # GATv2 results
│   └── grid/                    # Grid search results
│
└── complete_pipeline/           # Pipeline results
    └── comparison_report.txt    ← Final comparison
```

---

## 🎯 Complete Example Workflow

### Scenario: Train and compare all enhanced models

**1. Train all models (takes ~8-10 hours total on GPU):**

```bash
# Terminal 1: Train BrainGNN
cd training/advanced
python train_enhanced_models.py --model braingnn --epochs 100 --device cuda

# Terminal 2: Train BrainGT (in parallel if you have multiple GPUs)
python train_enhanced_models.py --model braingt --epochs 100 --device cuda

# Terminal 3: Train FBNetGen
python train_enhanced_models.py --model fbnetgen --epochs 100 --device cuda
```

**OR use the automated pipeline (sequential):**

```bash
cd pipelines
python run_complete_pipeline.py --epochs 100
```

**2. Compare results:**

```bash
cd analysis
python compare_models.py
```

**3. View detailed comparison:**

Check the printed output or open:
- `results/complete_pipeline/comparison_report.txt`

---

## 🛠️ Customizing Training

### Change Hyperparameters

**Example: Train BrainGNN with larger model:**

```bash
python train_enhanced_models.py \
    --model braingnn \
    --epochs 150 \
    --hidden_dim 256 \
    --n_layers 4 \
    --dropout 0.25 \
    --lr 0.0001 \
    --device cuda
```

### Train on Specific Fold (Fast Testing)

```bash
python train_enhanced_models.py \
    --model braingnn \
    --fold_name graphs_outer1_inner2 \
    --epochs 50
```

### All Available Arguments

```bash
python train_enhanced_models.py --help
```

**Common arguments:**
- `--model`: braingt, braingnn, fbnetgen
- `--epochs`: Number of epochs (default: 100)
- `--hidden_dim`: Hidden dimension size (default: 128)
- `--n_layers`: Number of layers (default: 3)
- `--n_heads`: Number of attention heads (default: 8 for BrainGT, 4 for others)
- `--dropout`: Dropout probability (default: 0.2)
- `--edge_dropout`: Edge dropout probability (default: 0.1)
- `--lr`: Learning rate (default: 1e-4)
- `--device`: cuda or cpu
- `--fold_name`: Train specific fold only

---

## ❓ FAQ

**Q: Which models should I run?**

A: Start with **enhanced versions** - they perform 3-5% better than base versions.

**Q: Can I just run one model?**

A: Yes! Just run one training command. For example:
```bash
python train_enhanced_models.py --model braingnn --epochs 100
```

**Q: How long does training take?**

A:
- Single model, single fold: 5-10 minutes
- Single model, all 25 folds: 2-3 hours
- All 3 models, all folds: 8-10 hours

**Q: Can I run models in parallel?**

A: Yes, if you have multiple GPUs, open multiple terminals and run different models.

**Q: Where are the model files stored?**

A: Check the `models/` folder for base versions and `models_enhanced/` for enhanced versions. But you **don't need to edit** these files - just run the training scripts!

**Q: Do I need to run comparison scripts?**

A: No, they're optional. The training scripts already print performance metrics. Comparison scripts just make it easier to compare multiple models side-by-side.

**Q: What's the difference between the files?**

- `train_advanced_models.py` → Trains BASE versions
- `train_enhanced_models.py` → Trains ENHANCED versions (better performance)
- `run_complete_pipeline.py` → Trains ALL models + creates ensemble
- `compare_models.py` → Compares trained models
- `compare_original_vs_enhanced.py` → Shows base vs enhanced improvement

**Q: Can I train just GATv2?**

A: Yes! GATv2 has its own script:
```bash
cd training/gatv2
python train_gatv2_improved.py --device cuda --epochs 50
```

---

## 🎓 Recommended Workflow for Beginners

### Fastest Path (1 hour):

```bash
# 1. Quick test with one model
cd training/advanced
python train_enhanced_models.py \
    --model braingnn \
    --fold_name graphs_outer1_inner2 \
    --epochs 50 \
    --device cuda

# 2. Check results
ls -la ../../results/enhanced/braingnn_enhanced/graphs_outer1_inner2/
cat ../../results/enhanced/braingnn_enhanced/graphs_outer1_inner2/braingnn_summary.json
```

### Complete Workflow (8-10 hours):

```bash
# 1. Train all models using pipeline
cd pipelines
python run_complete_pipeline.py --epochs 100

# 2. View comparison report
cat results/complete_pipeline/comparison_report.txt

# 3. Compare in detail
cd analysis
python compare_models.py
```

### Research Workflow (Deep Analysis):

```bash
# 1. Train all enhanced models
cd training/advanced
python train_enhanced_models.py --model braingnn --epochs 150 --device cuda
python train_enhanced_models.py --model braingt --epochs 150 --device cuda
python train_enhanced_models.py --model fbnetgen --epochs 150 --device cuda

# 2. Train GATv2 with grid search
cd ../gatv2
python train_gatv2_improved.py --grid_search --device cuda

# 3. Compare everything
cd ../../analysis
python compare_models.py
python compare_gatv2_configs.py
python compare_original_vs_enhanced.py

# 4. Generate predictions
python predict_gatv2_simple.py
```

---

## Summary Cheat Sheet

| Task | Command |
|------|---------|
| **Train BrainGNN (enhanced)** | `python train_enhanced_models.py --model braingnn --epochs 100` |
| **Train BrainGT (enhanced)** | `python train_enhanced_models.py --model braingt --epochs 100` |
| **Train FBNetGen (enhanced)** | `python train_enhanced_models.py --model fbnetgen --epochs 100` |
| **Train ALL models** | `python run_complete_pipeline.py --epochs 100` (from pipelines/) |
| **Quick test** | `python run_complete_pipeline.py --quick_test` |
| **Compare models** | `python compare_models.py` (from analysis/) |
| **Compare base vs enhanced** | `python compare_original_vs_enhanced.py` (from analysis/) |

**Default working directory for training:** `training/advanced/`
**Default working directory for comparison:** `analysis/`

---

**Need more details?** See `MODEL_USAGE_GUIDE.md` for comprehensive model documentation.
