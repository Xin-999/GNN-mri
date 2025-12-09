# Model Prediction Guide

How to use trained models to make predictions and generate Excel reports.

---

## 🚀 Quick Start

### Basic Usage

```bash
# Predict on test set with trained GATv2 model
python predict_with_trained_model.py --model_dir results_gatv2_interpretable

# Predict on test set with enhanced BrainGT
python predict_with_trained_model.py --model_dir results_braingt_enhanced

# Predict on all splits (train, val, test)
python predict_with_trained_model.py --model_dir results_braingnn_enhanced --split all
```

---

## 📋 Command-Line Arguments

```bash
python predict_with_trained_model.py \
    --model_dir results_gatv2_interpretable \    # Required: Directory with trained models
    --graphs_dir graphs \                         # Data directory (default: graphs)
    --output_dir predictions_output \             # Output directory (default: predictions_output)
    --split test \                                # Which split: train/val/test/all (default: test)
    --device cuda \                               # Device: cuda or cpu (default: cuda)
    --fold graphs_outer1_inner1                   # Optional: Predict specific fold only
```

---

## 📂 Input Requirements

### Model Directory Structure

Your model directory should contain fold subdirectories with trained models:

```
results_gatv2_interpretable/
├── graphs_outer1_inner1/
│   ├── gatv2_best.pt           # Model checkpoint
│   └── gatv2_summary.json      # Training summary
├── graphs_outer1_inner2/
│   └── gatv2_best.pt
└── ...
```

### Data Directory Structure

The corresponding graph data directory:

```
graphs/
├── graphs_outer1_inner1/
│   ├── train_2d.pt             # Training graphs
│   ├── val_2d.pt               # Validation graphs
│   ├── test_2d.pt              # Test graphs
│   └── split_indices.pt        # Train/val/test indices
└── ...
```

---

## 📊 Output Format

### Excel Files Generated

#### Per-Fold Files

`predictions_output/gatv2_interpretable_graphs_outer1_inner1.xlsx`

Contains multiple sheets:

1. **test_windows** - Window-level predictions
   - Subject_ID
   - Window_ID
   - True_Original (denormalized true values)
   - Predicted_Original (denormalized predictions)
   - Error_Original (true - predicted)
   - Absolute_Error_Original
   - True_Normalized (normalized values)
   - Predicted_Normalized
   - Error_Normalized

2. **test_subjects** - Subject-level predictions (aggregated from windows)
   - Subject_ID
   - True_Original
   - Predicted_Original
   - Error_Original
   - Absolute_Error_Original
   - True_Normalized
   - Predicted_Normalized
   - Error_Normalized

3. **test_metrics** - Comprehensive metrics
   - MSE (window-level normalized)
   - MSE (window-level original)
   - MSE (subject-level normalized)
   - MSE (subject-level original)
   - MAE, RMSE, R, R², P-value for all levels

#### Aggregate Summary File

`predictions_output/gatv2_interpretable_aggregate_summary.xlsx`

Contains sheets for each split showing:
- Fold
- Window_MSE
- Window_R
- Subject_MSE
- Subject_R
- MEAN and STD across folds

---

## 🎯 Example Usage

### Example 1: Predict Test Set Only

```bash
python predict_with_trained_model.py \
    --model_dir results_gatv2_interpretable \
    --split test
```

Output:
```
predictions_output/
├── gatv2_interpretable_graphs_outer1_inner1.xlsx
├── gatv2_interpretable_graphs_outer1_inner2.xlsx
├── ...
└── gatv2_interpretable_aggregate_summary.xlsx
```

### Example 2: Predict All Splits

```bash
python predict_with_trained_model.py \
    --model_dir results_braingt_enhanced \
    --split all \
    --output_dir my_predictions
```

Each Excel file will have sheets for:
- `train_windows`, `train_subjects`, `train_metrics`
- `val_windows`, `val_subjects`, `val_metrics`
- `test_windows`, `test_subjects`, `test_metrics`

### Example 3: Predict Specific Fold

```bash
python predict_with_trained_model.py \
    --model_dir results_braingnn_enhanced \
    --fold graphs_outer1_inner1 \
    --split test
```

### Example 4: Custom Paths

```bash
python predict_with_trained_model.py \
    --model_dir "C:\Users\USER\Documents\InternQX\Movie-HCP_Brain_Graph-master\Movie-HCP_Brain_Graph-master- test\Movie-HCP_Brain_Graph-master-test\results_gatv2_interpretable" \
    --graphs_dir "C:\Users\USER\Documents\InternQX\Movie-HCP_Brain_Graph-master\Movie-HCP_Brain_Graph-master- test\Movie-HCP_Brain_Graph-master-test\graphs" \
    --output_dir "C:\Users\USER\Documents\InternQX\predictions" \
    --split test
```

---

## 📈 Understanding the Output

### Window-Level vs Subject-Level

**Window-Level**: Each row is a single temporal window (sliding window from fMRI time-series)
- Subject 001 might have 10-20 windows
- Direct predictions from the model

**Subject-Level**: Aggregated predictions per subject
- Each subject has one row
- Computed as mean of all window predictions for that subject
- **More reliable metric for evaluation**

### Normalized vs Original Scale

**Normalized**: Values after StandardScaler normalization (mean=0, std=1)
- Used during model training
- Good for comparing across different targets

**Original**: Denormalized back to original scale (e.g., 94-131 for ListSort_AgeAdj)
- Real-world interpretable values
- **Use these for reporting**

### Key Metrics

| Metric | Description | Better Value |
|--------|-------------|--------------|
| MSE | Mean Squared Error | Lower |
| MAE | Mean Absolute Error | Lower |
| RMSE | Root Mean Squared Error | Lower |
| R | Pearson Correlation | Higher (closer to 1) |
| R² | Coefficient of Determination | Higher (closer to 1) |
| P-value | Statistical significance | Lower (< 0.05) |

---

## 🔍 Verification Example

Let's verify the predictions are correct:

```python
import pandas as pd

# Load predictions
df = pd.read_excel('predictions_output/gatv2_interpretable_graphs_outer1_inner1.xlsx',
                   sheet_name='test_subjects')

# Check MSE manually
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df['True_Original'], df['Predicted_Original'])
print(f"Subject-level MSE: {mse:.4f}")

# Check correlation
from scipy.stats import pearsonr
r, p = pearsonr(df['True_Original'], df['Predicted_Original'])
print(f"Subject-level R: {r:.4f}, p={p:.4e}")

# Show sample predictions
print("\nSample predictions:")
print(df[['Subject_ID', 'True_Original', 'Predicted_Original', 'Error_Original']].head(10))
```

---

## 🐛 Troubleshooting

### Error: "No model checkpoint found"

Check that your model directory contains `*_best.pt` files:

```bash
ls results_gatv2_interpretable/graphs_outer1_inner1/
# Should see: gatv2_best.pt
```

### Error: "split_indices.pt not found"

The script will warn but continue. Subject IDs will be inferred from graphs or generated sequentially.

To fix: Make sure your step 2 script saves indices:

```python
# In step2_prepare_data.py
torch.save({
    'train_indices': train_idx,
    'val_indices': val_idx,
    'test_indices': test_idx,
}, fold_dir / 'split_indices.pt')
```

### Error: "Could not create model"

The script tries to auto-detect model type. If it fails, you may need to:

1. Make sure model files are importable:
   - `models/brain_gt.py`
   - `models/brain_gnn.py`
   - `models/fbnetgen.py`
   - `models_enhanced/*.py`

2. Or specify model architecture manually in the code

### Scaler Not Found Warning

If you see "Warning: No scaler found", your predictions won't be denormalized.

To fix: Make sure your training script saves the scaler in the checkpoint:

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
    'scaler': scaler,  # Add this!
}, checkpoint_path)
```

---

## 📊 Comparing Multiple Models

To compare different models, run predictions for each and compare aggregate summaries:

```bash
# Predict all models
python predict_with_trained_model.py --model_dir results_gatv2_interpretable --split test
python predict_with_trained_model.py --model_dir results_braingt_enhanced --split test
python predict_with_trained_model.py --model_dir results_braingnn_enhanced --split test
python predict_with_trained_model.py --model_dir results_fbnetgen_enhanced --split test

# Compare aggregate summaries
# Open all *_aggregate_summary.xlsx files and compare Subject_R column
```

Or use the comparison tool:

```bash
python compare_original_vs_enhanced.py --html
```

---

## 💡 Tips

1. **Always use Subject-level metrics** for final evaluation (more reliable)

2. **Check both normalized and original scale** to ensure denormalization worked correctly

3. **Verify predictions make sense**:
   - Original scale values should be in expected range (e.g., 94-131)
   - Correlation should be positive
   - MSE should be reasonable

4. **Use aggregate summary** to compare model performance across folds

5. **Save predictions** for downstream analysis, visualization, or reporting

---

## 🎯 Expected Output Example

When you run:
```bash
python predict_with_trained_model.py --model_dir results_gatv2_interpretable --split test
```

You should see output like:
```
======================================================================
Predicting fold: graphs_outer1_inner1
======================================================================

Loading checkpoint: gatv2_best.pt
Model loaded successfully: GATv2Model
Parameters: 1,234,567

--- Predicting on test set ---
Denormalized predictions using saved scaler

Window-level metrics (Original scale):
  MSE:  45.23
  MAE:  5.12
  RMSE: 6.73
  R:    0.7845
  R2:   0.6154
  N:    520

Subject-level metrics (Original scale):
  MSE:  38.91
  MAE:  4.67
  RMSE: 6.24
  R:    0.8123
  R2:   0.6598
  N:    26

Saved: predictions_output/gatv2_interpretable_graphs_outer1_inner1.xlsx
...
```

Perfect! You now have detailed Excel outputs with all predictions and metrics! 📊
