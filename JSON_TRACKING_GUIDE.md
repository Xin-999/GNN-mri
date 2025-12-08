# JSON Summary Tracking Guide

All models now save comprehensive JSON summaries for easy performance tracking and comparison.

---

## 📁 File Structure

After training, you'll have:

```
results_braingt_advanced/
├── graphs_outer1_inner1/
│   ├── braingt_summary.json        # ← Per-fold detailed summary
│   ├── braingt_predictions.csv     # Predictions with actual values
│   └── braingt_best.pt             # Model checkpoint
├── graphs_outer2_inner1/
│   └── ...
└── braingt_aggregate_summary.json  # ← Aggregate across all folds

results_braingnn_advanced/
└── braingnn_aggregate_summary.json

results_fbnetgen_advanced/
└── fbnetgen_aggregate_summary.json
```

---

## 📊 Summary JSON Structure

### Per-Fold Summary (`{model}_summary.json`)

Contains everything about a single fold:

```json
{
  "model_name": "braingt",
  "fold": "graphs_outer1_inner1",
  "timestamp": "2025-01-15T14:30:45",
  "hostname": "your-machine",
  "device": "cuda",

  "config": {
    "epochs": 100,
    "lr": 0.0001,
    "hidden_dim": 128,
    "dropout": 0.2,
    ...
  },

  "model_parameters": 1234567,
  "trainable_parameters": 1234567,

  "training_summary": {
    "total_epochs": 45,
    "best_epoch": 38,
    "early_stopped": true,
    "final_lr": 0.000012
  },

  "best_validation": {
    "epoch": 38,
    "win_mse": 1.234,
    "win_r": 0.756,
    "subj_mse": 1.123,
    "subj_r": 0.782
  },

  "test_metrics": {
    "window_level": {
      "mse": 1.456,
      "mae": 0.987,
      "r": 0.745,
      "p_value": 1.23e-10,
      "r2": 0.555
    },
    "subject_level": {
      "mse": 1.234,
      "mae": 0.876,
      "r": 0.789,      # ← THIS IS THE KEY METRIC!
      "p_value": 3.45e-08,
      "r2": 0.623
    }
  },

  "data_info": {
    "n_train_samples": 1450,
    "n_val_samples": 362,
    "n_test_samples": 181,
    "n_train_subjects": 147,
    "n_val_subjects": 37,
    "n_test_subjects": 18,
    "input_dim": 268,
    "target_range_original": {
      "train": [94.5, 128.3],
      "test": [96.2, 125.7]
    }
  },

  "history": [
    {
      "epoch": 1,
      "train_loss": 2.345,
      "win_mse": 2.123,
      "win_r": 0.234,
      "subj_mse": 2.045,
      "subj_r": 0.345,
      "lr": 0.0001
    },
    ...
  ]
}
```

### Aggregate Summary (`{model}_aggregate_summary.json`)

Statistics across all folds:

```json
{
  "model_name": "braingt",
  "timestamp": "2025-01-15T18:45:32",
  "n_folds": 25,

  "config": { ... },

  "window_level_aggregate": {
    "mse": {
      "mean": 1.456,
      "std": 0.234,
      "min": 0.987,
      "max": 2.123,
      "median": 1.412
    },
    "r": {
      "mean": 0.745,
      "std": 0.089,
      "min": 0.543,
      "max": 0.876,
      "median": 0.756
    }
  },

  "subject_level_aggregate": {
    "mse": {
      "mean": 1.234,
      "std": 0.198,
      "min": 0.876,
      "max": 1.987,
      "median": 1.211
    },
    "r": {
      "mean": 0.789,    # ← MEAN CORRELATION ACROSS FOLDS
      "std": 0.076,
      "min": 0.645,
      "max": 0.897,
      "median": 0.798
    }
  },

  "training_statistics": {
    "mean_best_epoch": 42.3,
    "mean_total_epochs": 48.7,
    "n_early_stopped": 23
  },

  "model_info": {
    "mean_parameters": 1234567.0,
    "architecture": "braingt"
  },

  "per_fold_summary": [
    {
      "fold": "graphs_outer1_inner1",
      "test_subj_r": 0.789,
      "test_subj_mse": 1.234,
      "best_epoch": 38
    },
    ...
  ],

  "per_fold_detailed": [ ... ]  # Full details for each fold
}
```

---

## 🔍 Quick Comparison

### Command Line Tool

```bash
# Compare all models
python compare_models.py

# Output:
# MODEL PERFORMANCE COMPARISON
# ====================================
#
# Subject-Level Performance:
# Model      Subject r    Subject MSE    Parameters (M)
# BRAINGT    0.7890      1.2340         1.234
# BRAINGNN   0.7650      1.3210         0.567
# FBNETGEN   0.7720      1.2980         0.891
#
# 🏆 BEST MODEL: BRAINGT (Subject r = 0.7890)
```

### Generate HTML Report

```bash
python compare_models.py --html

# Opens: model_comparison.html
```

### Export to CSV

```bash
python compare_models.py --csv comparison.csv
python compare_models.py --json comparison.json
```

---

## 📈 Tracking Your Best Models

### Method 1: Quick Check

```bash
# Compare all models
python compare_models.py
```

### Method 2: Manual JSON Inspection

```python
import json

# Load BrainGT results
with open('results_braingt_advanced/braingt_aggregate_summary.json') as f:
    braingt = json.load(f)

# Get key metric
print(f"BrainGT Test r: {braingt['subject_level_aggregate']['r']['mean']:.4f}")
```

### Method 3: Excel/Sheets

1. Open any `{model}_aggregate_summary.json`
2. Look at `subject_level_aggregate.r.mean` ← **This is your main metric**
3. Compare across models

---

## 🎯 Key Metrics to Track

### Most Important (in order):

1. **`subject_level_aggregate.r.mean`** ← Primary metric
   - Subject-level correlation
   - Should be > 0.7 for good performance
   - Your old model: 0.055 ❌
   - Expected new: 0.75-0.85 ✅

2. **`subject_level_aggregate.r.std`**
   - Consistency across folds
   - Lower = more stable
   - Should be < 0.1

3. **`subject_level_aggregate.mse.mean`**
   - Prediction error (lower = better)

### Secondary Metrics:

- `training_statistics.mean_best_epoch` - convergence speed
- `model_parameters` - model size
- `window_level_aggregate.r.mean` - window-level performance

---

## 📊 Example Analysis

After training all models:

```bash
# 1. Quick comparison
python compare_models.py

# 2. Generate report
python compare_models.py --html

# 3. Open model_comparison.html in browser
```

Example output:
```
BEST MODEL: BRAINGT (Subject r = 0.7890)

Model Performance:
- BrainGT:   r = 0.789 ± 0.076  ← Best!
- FBNetGen:  r = 0.772 ± 0.082
- BrainGNN:  r = 0.765 ± 0.089

Recommendation: Use BrainGT or ensemble for production
```

---

## 🔧 Programmatic Access

### Python Script Example

```python
import json
from pathlib import Path

def get_best_model(models=['braingt', 'braingnn', 'fbnetgen']):
    """Find best performing model from summaries."""

    results = {}

    for model in models:
        path = Path(f'results_{model}_advanced/{model}_aggregate_summary.json')
        if not path.exists():
            continue

        with open(path) as f:
            data = json.load(f)

        # Get subject-level correlation
        if data['subject_level_aggregate']:
            r = data['subject_level_aggregate']['r']['mean']
            results[model] = r

    # Find best
    best_model = max(results, key=results.get)
    best_r = results[best_model]

    print(f"Best model: {best_model.upper()} (r = {best_r:.4f})")
    return best_model, best_r

# Usage
best_model, best_r = get_best_model()
```

---

## 📝 Summary Checklist

After training, check:

- ✅ Per-fold summaries exist: `results_{model}_advanced/*/​{model}_summary.json`
- ✅ Aggregate summary exists: `results_{model}_advanced/{model}_aggregate_summary.json`
- ✅ Subject-level r > 0.7
- ✅ Consistent across folds (std < 0.1)
- ✅ Compare across models with `compare_models.py`

---

## 🚀 Integration with Your Workflow

### After Each Training Run:

```bash
# Train model
python train_advanced_models.py --model braingt --epochs 100

# Immediately compare with previous runs
python compare_models.py

# Generate report for documentation
python compare_models.py --html --csv results_braingt.csv
```

### Weekly Summary:

```bash
# Compare all current models
python compare_models.py --html

# Email the HTML report to your team
```

---

## 💡 Tips

1. **Always check `subject_level_aggregate.r.mean`** - this is your key metric
2. **Save aggregate JSONs to Git** - track model improvements over time
3. **Use compare_models.py** - faster than manual inspection
4. **Generate HTML reports** - great for presentations/documentation
5. **Check training_statistics** - if early_stopped is high, increase patience

---

## ❓ FAQ

**Q: Which metric should I focus on?**
A: `subject_level_aggregate.r.mean` - this is correlation on test subjects.

**Q: My subject_level is null?**
A: Check if graphs have `subject_id` attribute. Should be set automatically.

**Q: How do I compare specific folds?**
A: `python compare_models.py --fold graphs_outer1_inner1`

**Q: Can I plot training curves?**
A: Yes! The `history` field has per-epoch metrics. Use matplotlib:
```python
import json
import matplotlib.pyplot as plt

with open('results_braingt_advanced/.../braingt_summary.json') as f:
    data = json.load(f)

epochs = [h['epoch'] for h in data['history']]
val_r = [h['subj_r'] for h in data['history']]
plt.plot(epochs, val_r)
plt.xlabel('Epoch')
plt.ylabel('Validation r')
plt.show()
```

**Q: How to export to Excel?**
A: Use the CSV option:
```bash
python compare_models.py --csv comparison.csv
# Open comparison.csv in Excel
```

---

**Now you can easily track which model performs best!** 🎯
