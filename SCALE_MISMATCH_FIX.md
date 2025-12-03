# Scale Mismatch Fix: Raw EEG Magnitude Issue

## Problem Detected

From training logs:

```
pred_var=280.234711, target_var=3377368.750000  ← 12,000x difference!
train_loss=6706086205  ← Loss in billions!
val_loss=2600507655    ← Impossibly high!
```

### Root Cause

**Raw EEG samples have huge absolute values** (microvolts scale):
- Decoder output: Small values (~200-400 variance) - initialized with Xavier
- Raw EEG targets: MASSIVE values (1M-80B variance) - actual physiological data
- MSE loss: Comparing tiny predictions to huge targets → billions

### Why This Happened

Your pipeline correctly uses raw signal as targets (no learned encoder), but didn't account for **EEG amplitude scale**:

```python
# What was happening:
pred = decoder(embeddings)  # Output: ~[-10, 10] (small Xavier init)
target = raw_EEG_samples    # Values: ~[-1000, 1000] or more (microvolts)
loss = MSE(pred, target)    # Comparing incompatible scales!
```

**EEG voltage ranges** (typical):
- Amplitude: 10-100 μV (microvolts)
- Artifacts can be 1000+ μV
- DC offset can vary widely between channels
- Your data shows variance from 100K to 80 billion!

## ✅ Solution: Per-Sample Normalization

Added normalization to targets (NOT inputs - those stay raw for masking):

```python
# Normalize targets to zero mean, unit variance per window
target_mean = target.mean(dim=-1, keepdim=True)  # Per window
target_std = target.std(dim=-1, keepdim=True) + 1e-8
target_normalized = (target - target_mean) / target_std

# Now decoder learns to output normalized signal
# Loss compares normalized pred to normalized target
```

### Why Per-Sample (Per-Window) Normalization?

1. **Preserves signal structure**: Removes DC offset and amplitude differences
2. **Handles variability**: Different channels/subjects have different amplitudes
3. **Stabilizes training**: Decoder learns patterns, not absolute scales
4. **Standard in MAE**: Vision MAE normalizes pixel values similarly

### What This Achieves

**Before**:
```
Window 1: [-500, -480, -460, ...]  (high DC offset)
Window 2: [20, 40, 60, ...]         (low DC offset)
Window 3: [2000, 2100, 2200, ...]   (artifact)

Decoder tries to predict absolute values → fails on scale differences
```

**After**:
```
Window 1: [-0.5, 0.0, 0.5, ...]  (normalized)
Window 2: [-0.5, 0.0, 0.5, ...]  (normalized)
Window 3: [-0.5, 0.0, 0.5, ...]  (normalized)

Decoder predicts relative patterns → learns signal structure
```

## Expected Results

### Before Fix
```
Epoch 1: train_loss=6,700,000,000, val_loss=2,600,000,000
Epoch 2: train_loss=6,700,000,000, val_loss=2,600,000,000
→ Loss in billions, not learning anything useful
```

### After Fix (Expected)
```
Epoch 1: train_loss=~1.0, val_loss=~1.0
Epoch 2: train_loss=~0.8, val_loss=~0.9
Epoch 3: train_loss=~0.6, val_loss=~0.7
→ Loss in reasonable range, steady improvement
```

For control experiment (mask_ratio=1.0, no positions):
```
Epoch 1: loss=~1.0
Epoch 2: loss=~1.0 (±0.01, stays constant)
→ Confirms no leakage with normalized scale
```

## Important Notes

### ✅ This Is Still "Raw Signal" Reconstruction

**Q**: Doesn't normalization violate the "raw signal targets" requirement?

**A**: No! Normalization is a **fixed, deterministic transform** with no learnable parameters:

```python
# No model weights involved:
target_normalized = (target - target.mean()) / target.std()

# Still comparing to actual signal, just rescaled
# Like converting meters to kilometers - same data, different units
```

The key properties preserved:
- ✅ No learnable encoder used
- ✅ Targets computed from raw input data
- ✅ No circular dependency
- ✅ Deterministic transform (same input → same normalized output)

### This Is Standard Practice

**Vision MAE** (original paper):
- Normalizes pixel values to [0, 1] or [-1, 1]
- Computes loss on normalized space
- Still reconstructing "pixels", just normalized

**Audio MAE**:
- Normalizes waveforms per-sample
- Removes DC bias and amplitude differences
- Learns temporal patterns, not absolute loudness

**Your EEG MAE** (now fixed):
- Normalizes EEG windows per-sample
- Removes DC offset and amplitude differences  
- Learns EEG patterns, not absolute microvolt scales

## Implementation Details

### What Gets Normalized

```python
# Training loop:
windows = batch["windows"]  # (B, L, 2048) - raw EEG
windows_masked = batch["windows_masked"]  # (B, L, 2048) - with zeros

# Input stays raw (masking needs raw scale)
pred = model(windows_masked)  # Decoder outputs (B, L, 2048)

# Target gets normalized
target = windows  # Start with raw
target = (target - target.mean(-1, keepdim=True)) / target.std(-1, keepdim=True)

# Loss on normalized scale
loss = MSE(pred[masked], target[masked])
```

### Normalization Dimensions

```python
target.mean(dim=-1, keepdim=True)  # Mean over window (2048 samples)
target.std(dim=-1, keepdim=True)   # Std over window

# Normalizes each window independently
# Shape: (B, L, 2048) → mean/std per (B, L) → (B, L, 2048) normalized
```

**Why per-window?**
- Each window may have different baseline
- Different channels have different amplitudes
- Artifacts affect different windows differently
- Preserves within-window temporal patterns

## Restart Training

```bash
# Stop current training (Ctrl+C)

# Restart with normalization fix (already applied)
python eeg_analysis/src/training/pretrain_mamba.py \
    --config eeg_analysis/configs/pretrain.yaml
```

### Expected Observations

1. **Loss drops to ~1.0** (not billions)
2. **pred_var and target_var are similar** (both normalized to ~1.0)
3. **Steady improvement** or plateau (for control experiment)
4. **No scale mismatch** in diagnostic logs

## Verification

After 1-2 epochs, check logs:

**Good signs**:
```
pred_var=0.8-1.2, target_var=0.8-1.2  ← Similar scales!
train_loss=0.5-1.5                    ← Reasonable range!
```

**Bad signs** (would indicate other issues):
```
pred_var=0.0, target_var=1.0   ← Not learning
loss > 10                       ← Still scale issues
loss < 0.1 in epoch 1          ← Suspicious (too good)
```

## Summary

| Aspect | Before Fix | After Fix |
|--------|-----------|-----------|
| Target values | Raw μV (huge range) | Normalized (μ=0, σ=1) |
| Loss magnitude | Billions | ~0.5-1.5 |
| Decoder task | Learn absolute μV | Learn normalized patterns |
| Training | Unstable (scale issues) | Stable |
| Still "raw signal"? | Yes | Yes (deterministic transform) |
| Learnable params on targets? | No | No ✅ |

**The fix maintains MAE integrity while handling EEG's physiological scale.**

