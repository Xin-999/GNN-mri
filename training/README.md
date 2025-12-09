# Training Scripts

All training scripts in this directory support being run from **either the project root OR from their subdirectories**. The scripts automatically detect the correct paths with fallbacks.

## ✅ Correct Usage - Both Methods Work

### Method 1: From Project Root (Recommended)
```bash
# From project root
python training/gatv2/train_gatv2_improved.py --device cuda --epochs 100
python training/advanced/train_advanced_models.py --model braingt --epochs 100
```

### Method 2: From Subdirectory
```bash
# Navigate to subdirectory first
cd training/gatv2
python train_gatv2_improved.py --device cuda --epochs 100

# Or for advanced models
cd training/advanced
python train_advanced_models.py --model braingt --epochs 100
```

The scripts automatically detect your working directory and find data/save results correctly using relative path fallbacks.

## 📁 Directory Structure

- `gatv2/` - GATv2 model variants
  - `train_gatv2_basic.py` - Basic GATv2 training
  - `train_gatv2_improved.py` - Improved GATv2 with target normalization (RECOMMENDED)
  - `train_gatv2_grid.py` - Hyperparameter grid search
  - `train_gatv2_with_excel.py` - Training with Excel prediction output
- `advanced/` - Advanced models (BrainGT, BrainGNN, FBNetGen)
  - `train_advanced_models.py` - Original advanced models
  - `train_enhanced_models.py` - Enhanced versions with extra techniques
  - `train_ensemble.py` - Ensemble multiple models
- `other/` - Experimental models (GIN, etc.)

## 🎯 Quick Start

### GATv2 Improved (Recommended)
```bash
python training/gatv2/train_gatv2_improved.py --device cuda --epochs 100 --hidden_dim 128
```

### Advanced Models
```bash
python training/advanced/train_enhanced_models.py --model braingt --epochs 100
```

## 📊 Data and Results Locations

### Data (Input)
Scripts automatically look for data in:
- Primary: `../../data/folds_data/` (when run from script directory)
- Fallback: `data/folds_data/` (when run from project root)

### Results (Output)
Results are automatically saved to:

**GATv2 Models:**
- `../../results/gatv2/basic/` - Basic GATv2 results
- `../../results/gatv2/improved/` - Improved GATv2 results (RECOMMENDED)
- `../../results/gatv2/grid/` - Grid search results

**Advanced Models:**
- `../../results/advanced/braingt/` - BrainGT results
- `../../results/advanced/braingnn/` - BrainGNN results
- `../../results/advanced/fbnetgen/` - FBNetGen results

**Enhanced Models:**
- `../../results/enhanced/braingt_enhanced/` - Enhanced BrainGT
- `../../results/enhanced/braingnn_enhanced/` - Enhanced BrainGNN
- `../../results/enhanced/fbnetgen_enhanced/` - Enhanced FBNetGen

**Other Results:**
- `../../results/ensemble/` - Ensemble model results
- `../../results/predictions/` - Prediction outputs

All paths use `../../` prefix when run from subdirectories, with automatic fallback to relative paths from project root.
