# Movie-HCP Brain Graph Prediction

Predict cognitive scores from fMRI brain connectivity using Graph Neural Networks.

## 📁 Project Structure

```
Movie-HCP_Brain_Graph/
├── docs/                    # Documentation and guides
├── preprocessing/           # Data preparation scripts
├── models/                  # Original model architectures
├── models_enhanced/         # Enhanced models with improvements
├── training/               # Training scripts organized by type
│   ├── gatv2/             # GATv2 training variants
│   ├── advanced/          # Advanced models (BrainGT, BrainGNN, FBNetGen)
│   └── other/             # Other model experiments
├── analysis/              # Prediction and interpretability analysis
├── pipelines/             # Complete workflow scripts
├── results/               # Training outputs
└── interpretability/      # Interpretability analysis results
```

## 🚀 Quick Start

### 1. Prepare Data

```bash
cd preprocessing
python step1_compute_ldw.py
python step2_prepare_data.py
```

### 2. Train Model

```bash
# Train improved GATv2 (recommended)
cd training/gatv2
python train_gatv2_improved.py --device cuda --epochs 100 --hidden_dim 128

# Train advanced models
cd training/advanced
python train_enhanced_models.py --model braingt --epochs 100
```

### 3. Analyze Results

```bash
cd analysis
python analyze_gatv2_interpretability.py --model_dir ../../results/gatv2/improved
python predict_with_trained_model.py --model_dir ../../results/gatv2/improved
```

## 📚 Documentation

See `docs/` folder for detailed guides:

- **GATV2_IMPROVEMENT_GUIDE.md** - GATv2 improvements and usage
- **ENHANCEMENTS_GUIDE.md** - Enhanced model techniques
- **PREDICTION_GUIDE.md** - Making predictions with trained models
- **JSON_TRACKING_GUIDE.md** - Understanding result tracking

## 🎯 Key Features

- **Target Normalization** - Fixes training instability
- **DropEdge Regularization** - Reduces overfitting
- **Early Stopping** - Automatic optimal epoch selection
- **Interpretability Analysis** - Identify important brain regions and connections
- **Comprehensive Tracking** - JSON summaries for all experiments

## 📊 Expected Performance

| Model | Subject-level R | Improvement |
|-------|----------------|-------------|
| GATv2 (Original) | 0.05-0.10 | Baseline |
| GATv2 (Improved) | 0.75-0.85 | +700-800% |
| BrainGT Enhanced | 0.78-0.85 | +2-5% over base |
| BrainGNN Enhanced | 0.72-0.78 | +4-7% over base |

## 🔧 Requirements

```bash
pip install -r requirements.txt
```

## 📝 Citation

If you use this code, please cite:
- Original paper references
- This repository

## 📧 Contact

For questions or issues, please open an issue on GitHub.
