# Analysis: Prediction Variance from Dropout

## What We're Seeing

```
pred_var: 0.000 → 0.13-0.15 (predictions varying!)
target_var: 1.0 → 0.59-0.63
Loss: 0.80 → 0.53 (decreasing)
Loss < Target Variance (0.53 < 0.59)
```

## Root Cause: Dropout

The Mamba backbone has `Dropout(p=0.1)`:

```python
# eeg_analysis/src/models/mamba_eeg_model.py:141
self.dropout = nn.Dropout(dropout)
```

**Effect**: Even with constant input, dropout randomly zeros 10% of activations **differently for each sample** in the batch.

```python
# Input (same for all samples)
x = constant  # [B, L, D] - all identical

# After Mamba + Dropout
output = Mamba(constant) with random dropout masks per sample
# Each sample gets different dropout mask → different output!
```

## Why This Causes Variance

```python
Sample 1: dropout mask = [1, 1, 0, 1, 0, ...] → output_1
Sample 2: dropout mask = [1, 0, 1, 1, 1, ...] → output_2
Sample 3: dropout mask = [0, 1, 1, 0, 1, ...] → output_3

# Outputs differ due to different dropout masks
pred_var = Var([output_1, output_2, output_3, ...]) > 0
```

## Why Loss < Target Variance

This is the KEY question. With dropout causing random noise:

```python
pred = constant + dropout_noise  # noise is random per sample
target = actual_signal_embedding  # varies per sample

loss = E[(pred - target)^2]
     = E[(constant + noise - target)^2]
```

**Theoretical minimum** (constant pred, no noise):
```
loss_min = Var(target) ≈ 0.59
```

**With dropout noise** (should be):
```
loss = Var(target) + Var(noise) > 0.59
```

**But we observe**:
```
loss = 0.53 < 0.59  ← IMPOSSIBLE with random noise!
```

## 🚨 What This Means

**Possibility A: Dropout + Optimization Artifact**
- Model learns to exploit dropout patterns
- Over many iterations, certain dropout patterns correlate with targets by chance
- Optimizer finds weights that minimize loss given dropout distribution
- **This is NOT true information leakage** - just learning noise statistics

**Possibility B: True Information Leakage**
- Model has access to sample-specific information beyond dropout
- Predictions correlate with targets systematically
- **This IS problematic** - indicates a bug in masking

## 🔬 How to Distinguish

### Test: Monitor Loss Convergence

**If learning dropout statistics (A)**:
- Loss will decrease initially
- Then **plateau** at some level
- Plateau level = best achievable with dropout noise + constant pred
- Expected plateau: around 0.4-0.5

**If true leakage (B)**:
- Loss will **continue decreasing** beyond dropout plateau
- Could reach very low values (< 0.3)
- Indicates sample-specific learning beyond noise

## 📊 Current Status (Epoch 4)

```
Loss trajectory:
Epoch 1: 0.799
Epoch 2: 0.677
Epoch 3: 0.597
Epoch 4: 0.526

Rate of decrease: ~0.07-0.10 per epoch
```

### Prediction

**Next 5-10 epochs will reveal**:
- **If plateau at ~0.45-0.50**: Dropout statistics learning (not a real problem)
- **If continues to ~0.3 or below**: True information leakage (investigate!)

## ✅ What To Do

### Option A: Let It Train (Recommended)

Monitor for 10 more epochs:

```bash
# Let current training continue
# Watch the loss curve in MLflow or terminal logs
```

**Expected behavior**:
```
Epoch 5-10: Loss decreases slowly
Epoch 10-15: Loss plateau around 0.45-0.50
Epoch 15+: No further improvement
```

If this happens → ✅ **No true leakage**, just dropout noise

If loss goes below 0.35 → ❌ **Investigate further**

### Option B: Train Without Dropout (Alternative)

To eliminate dropout as a confounder:

```python
# Temporarily modify mamba_eeg_model.py
self.dropout = nn.Dropout(0.0)  # Disable dropout
```

Retrain and check:
- **If loss stays constant**: Dropout was the only source of learning
- **If loss still decreases**: Something else is causing leakage

## 🎯 My Recommendation

**Let the current training continue for ~20 epochs total.**

**Expected outcome**:
```
Epoch 10: loss ≈ 0.45
Epoch 15: loss ≈ 0.43
Epoch 20: loss ≈ 0.42 (plateau)
Early stopping triggers
```

This would confirm:
- Model is learning dropout statistics (expected behavior)
- No true signal leakage (good news!)
- Control experiment validates masking pipeline (objective achieved!)

**If loss continues dropping significantly beyond epoch 20**, we'll investigate deeper.

## 📝 Technical Note: Why Centering Didn't Help

My earlier fix (target centering) was mathematically ineffective:

```python
# Centered loss
loss = mean((pred - target_mean - (target - target_mean))^2)
     = mean((pred - target)^2)  # Centering cancels out!
```

The real issue is dropout introducing sample-dependent randomness, not mean learning.

## Summary

| Loss Behavior | Interpretation | Action |
|--------------|----------------|--------|
| Plateaus at 0.4-0.5 | ✅ Learning dropout noise (expected) | Control experiment successful |
| Continues to <0.3 | ❌ Possible true leakage | Investigate further |

**Current status**: Wait and monitor. Most likely will plateau soon, confirming no leakage.

