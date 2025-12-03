# LEAKAGE ANALYSIS: Loss Reduction with Zero Information

## 🚨 Critical Finding

With the control configuration:
- `mask_ratio: 1.0` - All tokens masked (zeros)
- `disable_temporal_encoding: true` - No position information
- `disable_spatial_encoding: true` - No channel information

**Result**: Loss is STILL DROPPING

```
Epoch 1: train_loss=0.808, val_loss=0.723
Epoch 2: train_loss=0.683, val_loss=0.637
Epoch 3: train_loss=0.603, val_loss=0.564
```

## 🔍 Root Cause Analysis

### What the Model Receives (Forward Pass)

```python
# Input
windows_masked = [0, 0, 0, ..., 0]  # All zeros (B, L, 2048)

# Token encoding
token_emb = token_encoder(windows_masked)
          = LayerNorm(Linear([0,0,...,0]) + bias)
          = LayerNorm(bias)  # CONSTANT for all samples

# Positional encodings (DISABLED)
temporal = 0
spatial = 0

# Input to backbone
x = token_emb + temporal + spatial
  = LayerNorm(bias)  # SAME for every sample, every position, every channel

# Prediction
pred = backbone(x)  # Processes the constant
```

**Key point**: Input `x` is IDENTICAL for all samples!

### What the Model is Compared Against (Targets)

```python
# Target generation
target = token_encoder(windows)  # windows = ORIGINAL unmasked signal
       = LayerNorm(Linear(actual_signal) + bias)
       = VARIES based on actual EEG content
```

**Key point**: Targets are DIFFERENT for each sample (contain actual signal)!

### The "Leakage"

**The model is learning to predict the DATASET MEAN of target embeddings!**

```python
# Model learns:
pred_constant = mean(all_target_embeddings_in_dataset)

# This reduces loss:
# Initial (random): MSE(random, targets) = high
# After learning: MSE(dataset_mean, targets) = lower
```

## 📊 Why This Isn't Traditional "Leakage"

This is **NOT** leakage in the sense of "model sees unmasked signal content."

Instead, it's learning the **prior distribution** of EEG embeddings:

- Model doesn't see individual signals
- Model learns: "On average, what does an EEG embedding look like?"
- Predicting the mean is better than predicting random values
- Loss decreases even though model has no sample-specific information

**Analogy**:
```
Task: "Guess someone's height without seeing them"
Bad guess: Random number (anywhere from 0 to 300cm)
Good guess: Average height (~170cm)
Your guess is better on average, but you don't know the specific person's height
```

## 🎯 Confirming This Hypothesis

### Test 1: Check Prediction Variance

If the model is predicting dataset mean:
- Predictions should be VERY SIMILAR across all samples
- Low variance in predictions
- High variance in targets

```bash
# After a few more epochs, run:
python diagnose_100pct_masking.py \
    --checkpoint eeg_analysis/checkpoints/mamba2_eeg_pretrained.pt \
    --num-samples 100

# Look for:
# - Very low prediction variance
# - Very high consistency (>0.999)
# - Predictions essentially constant
```

### Test 2: Check if Predictions Match Target Mean

```python
# In training loop, add logging:
pred_mean = pred[mask_bool].mean()
target_mean = target[mask_bool].mean()
print(f"Pred mean: {pred_mean:.6f}, Target mean: {target_mean:.6f}")

# After learning, these should converge!
```

## 🔬 The Real Question

**Is the model learning ANYTHING USEFUL?**

Answer: **No** - It's just learning a single constant (the dataset mean).

**Evidence needed**:
- Can the model produce DIFFERENT predictions for different samples?
- Or does it always predict the same embedding regardless of input?

## 🛠️ How to Detect TRUE Signal Leakage

Current test is insufficient because:
- Model can reduce loss by learning dataset statistics
- This doesn't require access to individual signal content
- We need to test if model can distinguish between DIFFERENT samples

### Proposed Test: Prediction Consistency Check

```python
# Take same sample, mask it twice with different random masks
sample_1 = sample with mask pattern A
sample_2 = same sample with mask pattern B

pred_1 = model(sample_1)
pred_2 = model(sample_2)

# If model has access to underlying signal:
# pred_1 ≈ pred_2 (same sample → same prediction)

# If model only predicts mean:
# pred_1 ≈ pred_2 ≈ constant (always predicts dataset mean)

# Need to compare both against:
# Different sample prediction (should be different if signal leakage)
```

## 💡 Solution: Add Prediction Variance Test

The model should NOT be able to produce varying predictions if:
1. All inputs are identical (constant)
2. All positions disabled
3. All channels disabled

**Proposed fix**: Add assertion in training:

```python
# After getting predictions
if disable_temporal and disable_spatial and mask_ratio == 1.0:
    # All inputs are identical, predictions should be identical
    pred_std = pred.std()
    if pred_std > 0.01:  # Some small threshold
        logger.warning(f"Predictions vary (std={pred_std:.6f}) despite constant input!")
        logger.warning("This suggests model has access to sample-specific information")
```

## 🎯 Current Status: INCONCLUSIVE

**What we know**:
- ✅ Loss decreases (model is learning something)
- ❓ Is it learning dataset mean OR accessing signal content?

**What we need to check**:
1. Do predictions vary across samples? (Should be constant if learning mean)
2. Do predictions equal target mean? (Should converge if learning mean)
3. Can model distinguish between different samples? (Should fail if no signal access)

## 📋 Next Steps

### Option A: Let training continue and diagnose

```bash
# Let it train for ~10 epochs
# Then run diagnostic:
python diagnose_100pct_masking.py \
    --checkpoint eeg_analysis/checkpoints/mamba2_eeg_pretrained.pt
```

Look for:
- Prediction variance (should be near-zero if learning mean)
- Prediction mean vs target mean (should match if learning mean)

### Option B: Add real-time checks

Modify training script to log:
```python
# In training loop
pred_variance = pred[mask_bool].var()
pred_mean = pred[mask_bool].mean()
target_mean = target[mask_bool].mean()

mlflow.log_metric("pred_variance", float(pred_variance), step=epoch)
mlflow.log_metric("pred_mean", float(pred_mean), step=epoch)
mlflow.log_metric("target_mean", float(target_mean), step=epoch)
```

If `pred_mean` converges to `target_mean` and `pred_variance → 0`:
→ Model is learning dataset mean (not true leakage)

If `pred_variance` remains high:
→ Model has sample-specific information (TRUE LEAKAGE)

## 🧠 Theoretical Lower Bound

What's the minimum achievable loss by predicting dataset mean?

```python
# Theoretical minimum (predicting target mean):
min_loss = Var(targets)  # Variance of targets

# Random baseline:
random_loss = much higher

# Current loss after 3 epochs: 0.564
# Need to compare this to target variance
```

If loss is approaching `Var(targets)`, model is just learning mean.
If loss goes BELOW `Var(targets)`, model has sample-specific information.

## 🎯 Verdict

**Preliminary**: Loss reduction is likely from learning **dataset mean**, NOT true signal leakage.

**Confirmation needed**: Check if predictions vary across samples or are constant.

**Action**: Let training run a bit longer, then diagnose prediction variance.

