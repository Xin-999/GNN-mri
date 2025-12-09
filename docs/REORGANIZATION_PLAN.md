# Project Reorganization Plan

Your current structure has 20+ scripts scattered in the root directory. Here's a clean, organized structure:

---

## 🎯 Proposed Structure

```
Movie-HCP_Brain_Graph/
│
├── README.md                          # Main project README
├── requirements.txt                   # Dependencies
├── .gitignore
│
├── docs/                              # 📚 All documentation
│   ├── README_ADVANCED_MODELS.md
│   ├── ENHANCEMENTS_GUIDE.md
│   ├── GATV2_IMPROVEMENT_GUIDE.md
│   ├── JSON_TRACKING_GUIDE.md
│   ├── PREDICTION_GUIDE.md
│   └── mrimovie.pdf
│
├── data/                              # 📊 Data preparation
│   ├── raw/                           # Raw fMRI data (if applicable)
│   ├── folds_data/                    # Prepared fold data
│   └── ListSort_AgeAdj.csv           # Target scores
│
├── preprocessing/                     # 🔧 Data preprocessing scripts
│   ├── step1_compute_ldw.py          # Ledoit-Wolf covariance
│   ├── step2_prepare_data.py         # Graph preparation
│   └── plot_corr_matrix.py           # Visualization
│
├── models/                            # 🧠 Model architectures (ORIGINAL)
│   ├── __init__.py
│   ├── brain_gt.py
│   ├── brain_gnn.py
│   ├── fbnetgen.py
│   └── gatv2.py                      # Rename from 1modgatv2.py
│
├── models_enhanced/                   # 🚀 Enhanced model architectures
│   ├── __init__.py
│   ├── brain_gt_enhanced.py
│   ├── brain_gnn_enhanced.py
│   └── fbnetgen_enhanced.py
│
├── training/                          # 🏋️ Training scripts
│   ├── gatv2/
│   │   ├── train_gatv2_basic.py      # Rename from train_gatv2_interpretable.py
│   │   ├── train_gatv2_improved.py
│   │   ├── train_gatv2_grid.py
│   │   └── train_gatv2_with_excel.py
│   ├── advanced/
│   │   ├── train_advanced_models.py  # BrainGT, BrainGNN, FBNetGen
│   │   ├── train_enhanced_models.py
│   │   └── train_ensemble.py
│   ├── other/
│   │   ├── train_gin_grid.py
│   │   ├── train_gin_gru_sequence.py
│   │   └── train_gat_interpretable.py
│   └── hyperparameter_search.py
│
├── analysis/                          # 📊 Analysis and interpretability
│   ├── predict_with_trained_model.py
│   ├── analyze_gatv2_interpretability.py
│   ├── explain_gatv2_gnnexplainer.py
│   ├── compare_models.py
│   └── compare_original_vs_enhanced.py
│
├── pipelines/                         # 🔄 Complete workflows
│   └── run_complete_pipeline.py
│
├── utils/                             # 🛠️ Utility functions
│   ├── __init__.py
│   └── data_utils.py
│
├── notebooks/                         # 📓 Jupyter notebooks
│   └── plot_corr_matrix.ipynb
│
├── results/                           # 📈 Training results
│   ├── gatv2/
│   │   ├── basic/                    # From train_gatv2_basic.py
│   │   └── improved/                 # From train_gatv2_improved.py
│   ├── advanced/
│   │   ├── braingt/
│   │   ├── braingnn/
│   │   └── fbnetgen/
│   ├── enhanced/
│   │   ├── braingt_enhanced/
│   │   ├── braingnn_enhanced/
│   │   └── fbnetgen_enhanced/
│   └── predictions/                  # Prediction outputs
│
├── interpretability/                  # 🔍 Interpretability results
│   └── gatv2_improved/
│       ├── graphs_outer1_inner1/
│       └── aggregate_interpretability.json
│
└── scripts/                           # 🔧 Helper scripts
    ├── reorganize_project.py         # Script to reorganize automatically
    └── cleanup_old_files.py          # Archive old files
```

---

## 📁 Detailed File Mapping

### Current → New Location

#### Documentation
```
ENHANCEMENTS_GUIDE.md                    → docs/ENHANCEMENTS_GUIDE.md
GATV2_IMPROVEMENT_GUIDE.md               → docs/GATV2_IMPROVEMENT_GUIDE.md
JSON_TRACKING_GUIDE.md                   → docs/JSON_TRACKING_GUIDE.md
PREDICTION_GUIDE.md                      → docs/PREDICTION_GUIDE.md
README_ADVANCED_MODELS.md                → docs/README_ADVANCED_MODELS.md
mrimovie.pdf                             → docs/mrimovie.pdf
```

#### Preprocessing
```
step1_compute_ldw.py                     → preprocessing/step1_compute_ldw.py
step2_prepare_data.py                    → preprocessing/step2_prepare_data.py
plot_corr_matrix.py                      → preprocessing/plot_corr_matrix.py
plot_corr_matrix.ipynb                   → notebooks/plot_corr_matrix.ipynb
```

#### Models
```
1modgatv2.py                             → models/gatv2.py
(models/ already exists)
(models_enhanced/ already exists)
```

#### Training Scripts - GATv2
```
train_gatv2_interpretable.py             → training/gatv2/train_gatv2_basic.py
train_gatv2_improved.py                  → training/gatv2/train_gatv2_improved.py
train_gatv2_grid.py                      → training/gatv2/train_gatv2_grid.py
train_gatv2_with_excel.py                → training/gatv2/train_gatv2_with_excel.py
```

#### Training Scripts - Advanced Models
```
train_advanced_models.py                 → training/advanced/train_advanced_models.py
train_enhanced_models.py                 → training/advanced/train_enhanced_models.py
train_ensemble.py                        → training/advanced/train_ensemble.py
```

#### Training Scripts - Other
```
train_gin_grid.py                        → training/other/train_gin_grid.py
train_gin_gru_sequence.py                → training/other/train_gin_gru_sequence.py
train_gat_interpretable.py               → training/other/train_gat_interpretable.py
hyperparameter_search.py                 → training/hyperparameter_search.py
```

#### Analysis
```
predict_with_trained_model.py            → analysis/predict_with_trained_model.py
analyze_gatv2_interpretability.py        → analysis/analyze_gatv2_interpretability.py
explain_gatv2_gnnexplainer.py            → analysis/explain_gatv2_gnnexplainer.py
compare_models.py                        → analysis/compare_models.py
compare_original_vs_enhanced.py          → analysis/compare_original_vs_enhanced.py
```

#### Pipelines
```
run_complete_pipeline.py                 → pipelines/run_complete_pipeline.py
```

#### Results Directories
```
results_gatv2_interpretable/             → results/gatv2/basic/
results_gatv2_improved/                  → results/gatv2/improved/ (when created)
results_gat_interpretable/               → results/gatv2/other/
results_braingt_advanced/                → results/advanced/braingt/
results_gatv2_predictions/               → results/predictions/gatv2/
complete_pipeline_results/               → results/complete_pipeline/
```

#### Temporary/Deprecated Files
```
gat_save.py                              → DELETE or archive (deprecated)
temp_file.txt                            → DELETE
```

---

## 🚀 Reorganization Script

I'll create an automated script to reorganize everything safely.

### Option 1: Manual Reorganization (Safer)

1. **Create new directories**:
```bash
mkdir -p docs preprocessing training/gatv2 training/advanced training/other
mkdir -p analysis pipelines notebooks scripts
mkdir -p results/gatv2/{basic,improved} results/advanced/{braingt,braingnn,fbnetgen}
mkdir -p results/enhanced/{braingt_enhanced,braingnn_enhanced,fbnetgen_enhanced}
mkdir -p results/predictions interpretability
```

2. **Move files manually** following the mapping above

3. **Update import paths** in moved files

### Option 2: Automated Reorganization (Faster)

Use the `reorganize_project.py` script I'll create below.

---

## ⚠️ Important Notes

### Before Reorganizing

1. **Commit current state**:
```bash
git add .
git commit -m "Checkpoint before reorganization"
```

2. **Backup**:
```bash
# Create a backup
cp -r . ../Movie-HCP_Brain_Graph-BACKUP
```

3. **Test after reorganization**:
- Update import paths
- Test at least one training script
- Verify results directories work

### Import Path Updates

After moving files, you'll need to update imports:

**Before** (in root):
```python
from models.brain_gt import BrainGT
from utils.data_utils import load_graphs
```

**After** (in training/advanced/):
```python
import sys
sys.path.insert(0, '../..')  # Add project root to path

from models.brain_gt import BrainGT
from utils.data_utils import load_graphs
```

**Better approach** - Install as package:
```bash
# In project root, create setup.py
pip install -e .
```

Then imports work from anywhere:
```python
from models.brain_gt import BrainGT  # Works from any subdirectory
```

---

## 📊 Benefits of Reorganization

### Before (Current)
- ❌ 20+ scripts in root directory
- ❌ Hard to find specific scripts
- ❌ Documentation mixed with code
- ❌ Results directories scattered
- ❌ No clear structure

### After (Proposed)
- ✅ Clear separation of concerns
- ✅ Easy to navigate
- ✅ Documentation in dedicated folder
- ✅ Consistent results structure
- ✅ Scalable for future additions
- ✅ Professional project structure

---

## 🔄 Recommended Workflow After Reorganization

### 1. Data Preparation
```bash
# In preprocessing/
python step1_compute_ldw.py
python step2_prepare_data.py
```

### 2. Training
```bash
# In training/gatv2/
python train_gatv2_improved.py --device cuda --epochs 100

# In training/advanced/
python train_advanced_models.py --model braingt --epochs 100
python train_enhanced_models.py --model braingt --epochs 100
```

### 3. Analysis
```bash
# In analysis/
python predict_with_trained_model.py --model_dir ../results/gatv2/improved
python analyze_gatv2_interpretability.py --model_dir ../results/gatv2/improved
python compare_original_vs_enhanced.py
```

### 4. Complete Pipeline
```bash
# In pipelines/
python run_complete_pipeline.py
```

---

## 🎯 Next Steps

1. **Review this plan** - Make sure you agree with the structure
2. **Backup your work** - `git commit` or copy entire directory
3. **Run reorganization script** - Use the automated script below
4. **Update imports** - Fix any broken import paths
5. **Test** - Run at least one training script to verify
6. **Update README.md** - Document new structure

---

## 📝 Updated README Structure

After reorganization, your main README.md should have:

```markdown
# Movie-HCP Brain Graph Prediction

Predict cognitive scores from brain fMRI connectivity using Graph Neural Networks.

## 📁 Project Structure

- `docs/` - All documentation and guides
- `preprocessing/` - Data preparation scripts
- `models/` - Original model architectures
- `models_enhanced/` - Enhanced models with improvements
- `training/` - All training scripts organized by model type
- `analysis/` - Prediction and interpretability analysis
- `results/` - Training outputs and predictions

## 🚀 Quick Start

See `docs/GATV2_IMPROVEMENT_GUIDE.md` for detailed instructions.

### 1. Prepare Data
```bash
cd preprocessing
python step1_compute_ldw.py
python step2_prepare_data.py
```

### 2. Train Model
```bash
cd training/gatv2
python train_gatv2_improved.py --device cuda --epochs 100
```

### 3. Analyze Results
```bash
cd analysis
python analyze_gatv2_interpretability.py --model_dir ../../results/gatv2/improved
```
```

---

This reorganization will make your project much more professional and easier to maintain! 🎉
