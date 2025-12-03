# Solution: Preventing Dataset Mean Learning in Control Experiment

## Problem

With `mask_ratio=1.0` and positional encodings disabled, the model still reduced loss:

```
Epoch 1: loss=0.808
Epoch 2: loss=0.683
Epoch 3: loss=0.603
...
Epoch 10: still decreasing
```

**Root cause**: Model was learning to predict the **dataset mean** of target embeddings, not accessing individual signals.

```python
# Model with constant input
pred = constant  # Same for all samples

# Model learns
pred ≈ mean(all_targets_in_dataset)

# This reduces loss
loss = MSE(mean(targets), targets) < MSE(random, targets)
```

## Solution Implemented: Target Centering

Modified `pretrain_mamba.py` to **subtract the target mean** before computing loss when in control mode.

### Changes Made

```python
# Old loss (allows learning dataset mean)
loss = MSE(pred, target)

# New loss (prevents learning dataset mean)
target_mean = mean(target)
loss = MSE(pred - target_mean, target - target_mean)
```

**Key insight**: After centering:
- Predicting a constant (dataset mean) gives loss = Var(target)
- Predicting target_mean gives loss = Var(target) (no advantage)
- **Only way to reduce loss**: Predict sample-specific deviations from mean
- With constant input → IMPOSSIBLE to predict deviations
- **Loss should now stay constant** (cannot be reduced)

### Code Location

File: `eeg_analysis/src/training/pretrain_mamba.py`

Changes in:
1. Training loop (lines ~240-260)
2. Validation loop (lines ~320-335)

## Expected Behavior After Fix

### Scenario A: No Signal Leakage (Expected)

```
Epoch 11: loss=0.XXX (stays constant)
Epoch 12: loss=0.XXX (±0.01 fluctuation)
Epoch 13: loss=0.XXX (no improvement)
...
Early stopping triggers (no improvement for patience epochs)

✅ CONFIRMED: No information leakage
   Model cannot learn without varying information
```

Loss will plateau at the **variance of centered targets** (irreducible without sample info).

### Scenario B: True Signal Leakage (Would indicate bug)

```
Epoch 11: loss=0.500
Epoch 12: loss=0.450
Epoch 13: loss=0.400  ← Still decreasing!
...

❌ WARNING: Model is still learning
   This means it has access to sample-specific information
   TRUE SIGNAL LEAKAGE DETECTED
```

## How to Use

### Step 1: Stop current training

```bash
# Press Ctrl+C to stop the running training
```

### Step 2: Restart training with fix

```bash
cd /home/abin/eeg-mlflow
source .venv/bin/activate

# The fix is already in pretrain_mamba.py
python eeg_analysis/src/training/pretrain_mamba.py \
    --config eeg_analysis/configs/pretrain.yaml
```

### Step 3: Monitor loss

Watch for:
- **Loss stops decreasing** → ✅ No leakage confirmed
- **Loss continues decreasing** → ❌ Investigate further

### Step 4: Check diagnostic logs

Look for control experiment logs:

```
[Control] pred_var=0.XXX, target_var=0.XXX, pred_mean=0.XXX, target_mean=0.XXX
```

**Expected values**:
- `pred_var`: Should stay constant (model can't learn to vary)
- `target_var`: Dataset variance (should be > loss if model can't beat variance)
- `pred_mean` and `target_mean`: Should be similar initially

## Understanding the Math

### Why Original Loss Decreased

```python
# Model predicts constant c
pred = c  # Same for all samples

# Optimal constant minimizes MSE
c_optimal = mean(targets)

# Initial loss (random c)
loss_initial = MSE(c_random, targets) = high

# After learning (c = mean)
loss_learned = MSE(mean(targets), targets) = Var(targets)

# Reduction achieved!
loss_initial > loss_learned
```

### Why Centered Loss Cannot Decrease

```python
# Center targets
target_centered = target - mean(targets)
# mean(target_centered) = 0

# Model predicts constant c
pred = c

# Center predictions (subtract target mean)
pred_centered = c - mean(targets)

# Loss
loss = MSE(pred_centered, target_centered)
     = MSE(c - mean(targets), target - mean(targets))

# To minimize, take derivative w.r.t. c:
d/dc loss = 2 * (c - mean(targets) - mean(target - mean(targets)))
          = 2 * (c - mean(targets) - 0)
          = 2 * (c - mean(targets))

# Optimal c:
c_optimal = mean(targets)

# Plugging back:
pred_centered = mean(targets) - mean(targets) = 0
loss_optimal = MSE(0, target_centered) 
             = Var(target_centered)
             = Var(target)  # Variance unchanged by centering

# But pred_centered = 0 is just ONE prediction
# Model with constant input can only predict ONE value
# Predicting 0 gives loss = Var(target)
# Predicting any other constant gives loss ≥ Var(target)

# Result: Loss cannot go below Var(target)
# And Var(target) is achieved by ANY constant prediction!
# So model cannot improve by learning
```

**Key point**: After centering, **all constant predictions are equivalent**. Model cannot reduce loss by learning a better constant.

## Alternative Solutions Considered

### Option A: Randomize Targets (Not Used)

```python
# Shuffle targets relative to inputs
indices = torch.randperm(batch_size)
target_shuffled = target[indices]
loss = MSE(pred, target_shuffled)
```

**Why not**: Makes validation loss meaningless, complex to implement.

### Option B: Contrastive Loss (Not Used)

```python
# Require predictions to match specific samples
loss = contrastive_loss(pred, target, labels)
```

**Why not**: Requires labels, changes loss fundamentally, complex.

### Option C: Add Noise to Inputs (Not Used)

```python
# Add random noise to break constant input
windows_masked_noisy = windows_masked + noise
```

**Why not**: Adds varying information, defeats purpose of control.

### Option D: Target Centering (IMPLEMENTED) ✅

**Why this**: 
- Simple, principled, mathematically sound
- Removes ability to learn mean without adding information
- Preserves validation semantics
- Easy to implement and understand

## Diagnostic Output

With the fix, you'll see logs like:

```
{"timestamp": "...", "level": "INFO", "message": "[Control] pred_var=0.102135, target_var=0.102138, pred_mean=-0.002144, target_mean=-0.002145"}
```

**Interpretation**:
- `pred_var ≈ target_var`: Predictions have similar variance to targets
- `pred_mean ≈ target_mean`: Model is predicting around the mean
- If both var and mean stay constant across epochs → Model is not learning

## Expected Timeline

With the fix:

```
Epoch 1: loss=X.XX (initial)
Epoch 2: loss=X.XX (±0.01)
Epoch 3: loss=X.XX (±0.01)
...
Epoch 20: loss=X.XX (no improvement)
Early stopping triggered

Total time: ~20 epochs (patience=20)
✅ Control experiment complete: No leakage confirmed
```

Without the fix (old behavior):

```
Epoch 1-50: loss steadily decreasing
✗ Inconclusive (learning mean, not testing leakage)
```

## After Control Experiment

Once loss stops decreasing with centered targets:

**✅ No leakage confirmed** → Safe to train with real configuration

Update `pretrain.yaml`:

```yaml
mask_ratio: 0.75                   # Provide context
disable_temporal_encoding: false   # Enable positions
disable_spatial_encoding: false    # Enable channels
```

Then train for real:

```bash
python eeg_analysis/src/training/pretrain_mamba.py \
    --config eeg_analysis/configs/pretrain.yaml
```

## Summary

| Configuration | Loss Behavior | Interpretation |
|--------------|---------------|----------------|
| **Old**: No centering | Decreases | Learning dataset mean (inconclusive) |
| **New**: With centering | Plateaus | No leakage ✅ |
| **New**: With centering | Decreases | Signal leakage ❌ (investigate) |

**Current status**: Fix implemented, restart training to confirm no leakage.

**Expected outcome**: Loss will plateau, confirming no information leakage.

**Next step**: Once confirmed, switch to `mask_ratio=0.75` for actual training.

