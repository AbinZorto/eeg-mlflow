# Statistics Comparison Feature: Predicted vs Ground Truth

## Overview

The diagnostic script now compares **predicted signal statistics** (mean, std) to **ground truth statistics** to verify if the model learns to reconstruct actual EEG signal characteristics.

## What It Does

### Computes Per-Window Statistics

For each masked window:
- **Predicted**: Mean and std of reconstructed signal (2048 samples)
- **Ground Truth**: Mean and std of actual signal (2048 samples)

### Compares Across Channels

- Aggregates statistics per channel
- Computes correlation between predicted and GT statistics
- Reports errors (absolute differences)

## Usage

```bash
# Standard usage (auto-detects signal reconstruction from checkpoint)
python diagnose_100pct_masking.py \
    --checkpoint eeg_analysis/checkpoints/mamba2_eeg_pretrained.pt \
    --num-samples 100

# Force signal space decoding
python diagnose_100pct_masking.py \
    --checkpoint eeg_analysis/checkpoints/mamba2_eeg_pretrained.pt \
    --decode-to-signal \
    --num-samples 100
```

## Output Sections

### 1. Overall Statistics (All Channels)

```
OVERALL STATISTICS (All Channels Combined)
============================================================

Mean (per window):
  Predicted: 0.000123 ± 0.001456
  Ground Truth: 0.000000 ± 0.001234
  Difference: 0.000123
  Correlation: 0.8234

Std (per window):
  Predicted: 0.998765 ± 0.012345
  Ground Truth: 1.000000 ± 0.010234
  Difference: 0.001235
  Correlation: 0.9123
```

**Interpretation**:
- **Correlation > 0.7**: Model accurately reconstructs statistics ✅
- **Correlation 0.5-0.7**: Partial reconstruction ⚠️
- **Correlation < 0.5**: Poor reconstruction ❌

### 2. Per-Channel Statistics

```
PER-CHANNEL STATISTICS
============================================================

Channel      Windows    Mean Error       Std Error        Mean Corr     Std Corr
--------------------------------------------------------------------------------
C3           1250       0.000234         0.001234         0.8234        0.9123
FP1          1180       0.000189         0.001456         0.7891        0.9012
...
```

**Columns**:
- **Windows**: Number of masked windows for this channel
- **Mean Error**: Average absolute difference in mean
- **Std Error**: Average absolute difference in std
- **Mean Corr**: Correlation between predicted and GT means
- **Std Corr**: Correlation between predicted and GT stds

### 3. Interpretation

```
INTERPRETATION
============================================================

Overall Mean Correlation: 0.8234
Overall Std Correlation:  0.9123

✅ EXCELLENT: Model accurately reconstructs signal statistics
   Predictions match ground truth mean and std patterns
   Model is learning actual EEG signal characteristics
```

## What Good Results Look Like

### ✅ Excellent Model (Learning Signal Structure)

```
Mean Correlation: > 0.7
Std Correlation:  > 0.7

Interpretation:
- Model predicts different means/stds for different windows
- Predictions correlate with actual signal characteristics
- Model is learning EEG signal patterns, not just positions
```

### ⚠️ Moderate Model (Partial Learning)

```
Mean Correlation: 0.5-0.7
Std Correlation:  0.5-0.7

Interpretation:
- Some correlation but not perfect
- Model partially learns signal structure
- May need more training or better mask ratio
```

### ❌ Poor Model (Positional Learning Only)

```
Mean Correlation: < 0.5
Std Correlation:  < 0.5

Interpretation:
- Low correlation = predictions don't match signal stats
- Model likely learning positional patterns only
- Not reconstructing actual signal characteristics
```

## Why This Matters

### Traditional Metrics Can Be Misleading

**Old approach**: Compare embedding vectors
- Embeddings are learned representations
- Hard to interpret what model learned
- Can have good loss but poor signal understanding

**New approach**: Compare signal statistics
- Direct comparison of signal characteristics
- Clear interpretation: Does model predict correct mean/std?
- Reveals if model learns signal structure vs. positional patterns

### Example: Why Statistics Matter

**Scenario A**: Model predicts constant values
```
Predicted mean: 0.0 for all windows
Ground truth mean: varies from -5 to +5
Correlation: 0.0 ❌
```

**Scenario B**: Model predicts varying means matching GT
```
Predicted mean: varies from -4.8 to +4.9
Ground truth mean: varies from -5.0 to +5.0
Correlation: 0.95 ✅
```

**Both might have similar MSE loss**, but Scenario B shows the model learned signal structure!

## Integration with Other Diagnostics

The statistics comparison complements existing diagnostics:

1. **Position Correlation**: Does model depend on position?
2. **Context Sensitivity**: Do predictions vary with masking?
3. **Statistics Comparison**: Do predictions match signal characteristics? ← NEW

**Combined interpretation**:
- Low position corr + High context sensitivity + High stats corr = ✅ Excellent
- High position corr + Low stats corr = ❌ Positional learning only
- Low position corr + Low stats corr = ⚠️ Learning something else (investigate)

## Technical Details

### Normalization Handling

The comparison accounts for per-window normalization:
- Both predictions and GT are normalized per-window (mean=0, std=1)
- Statistics are computed on **normalized** windows
- This focuses on **relative patterns**, not absolute scales

### Window-Level Statistics

For each masked window:
```python
pred_window = pred[b, idx]  # (2048,) - reconstructed signal
gt_window = windows[b, idx]  # (2048,) - actual signal

pred_mean = pred_window.mean()
pred_std = pred_window.std()
gt_mean = gt_window.mean()
gt_std = gt_window.std()

# Compare these statistics
```

### Correlation Computation

```python
# Across all windows for a channel:
mean_corr = corrcoef([pred_mean_1, pred_mean_2, ...], 
                     [gt_mean_1, gt_mean_2, ...])

# High correlation = model predicts correct mean for each window
# Low correlation = model predicts similar mean for all windows
```

## Use Cases

### 1. Control Experiment Validation

With `mask_ratio=1.0` and no positions:
- **Expected**: Low statistics correlation (can't learn signal)
- **If high correlation**: Signal leakage detected!

### 2. Normal Training Evaluation

With `mask_ratio=0.75`:
- **Expected**: High statistics correlation (learning from context)
- **If low correlation**: Model not learning signal structure effectively

### 3. Model Comparison

Compare two models:
```bash
# Model A (mask_ratio=1.0)
python diagnose_100pct_masking.py --checkpoint model_A.pt
# Mean corr: 0.15, Std corr: 0.12

# Model B (mask_ratio=0.75)
python diagnose_100pct_masking.py --checkpoint model_B.pt
# Mean corr: 0.82, Std corr: 0.89

# Conclusion: Model B learns signal structure, Model A does not
```

## Limitations

1. **Requires Signal Reconstruction**: Only works if `reconstruct_signal_space=true`
2. **Per-Window Normalization**: Statistics computed on normalized windows
3. **Sample Size**: Needs sufficient masked windows per channel for reliable correlation

## Summary

The statistics comparison feature provides a **direct, interpretable measure** of whether your model learns actual EEG signal characteristics or just positional patterns.

**Key metric**: Correlation between predicted and ground truth statistics
- **> 0.7**: Excellent signal learning ✅
- **0.5-0.7**: Moderate learning ⚠️
- **< 0.5**: Poor learning ❌

Use this alongside position correlation and context sensitivity tests for comprehensive model evaluation!

