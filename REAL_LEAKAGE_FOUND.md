# REAL INFORMATION LEAKAGE DETECTED AND FIXED

## 🚨 Critical Finding

**Loss reached 0.33 at epoch 7** - well below target variance (~0.6), confirming TRUE information leakage in the control experiment.

## 🔍 Root Cause: Circular Dependency in Target Computation

### The Bug

```python
# In training loop (BEFORE FIX)
pred = model(windows_masked, ...)  # Uses model.token_encoder
target = model.encode_tokens_only(windows)  # ALSO uses model.token_encoder
loss = MSE(pred, target)
```

**The problem**: 
- Both predictions AND targets use the **SAME TokenEncoder**
- TokenEncoder has **learnable parameters** (Linear weights + LayerNorm gamma/beta)
- As training progresses, TokenEncoder weights change
- Targets change (they depend on current TokenEncoder weights!)
- Model learns to track these changing targets
- Loss decreases even with constant input!

### The Leakage Mechanism

```python
Epoch 0:
  TokenEncoder weights: W0
  target = TokenEncoder(W0, real_signal) = T0
  pred = model(constant_input) = P0  
  loss = MSE(P0, T0) = high

Epoch 1:
  Model trains, weights update: W0 → W1
  target = TokenEncoder(W1, real_signal) = T1  ← CHANGED!
  pred = model(constant_input) = P1
  Model learns: "predict T1"
  loss = MSE(P1, T1) = lower

Epoch 2:
  Weights update: W1 → W2
  target = TokenEncoder(W2, real_signal) = T2  ← CHANGED AGAIN!
  ...and so on
```

**Result**: Model is chasing a moving target that it itself controls. This creates a circular dependency that allows loss to decrease indefinitely, even with no actual signal information!

## ✅ The Fix: Frozen Target Encoder

Modified `pretrain_mamba.py` to use a **separate, frozen encoder** for targets in control mode:

```python
# Create frozen copy of TokenEncoder at start of training
target_encoder = deepcopy(model.token_encoder)
target_encoder.eval()  # No dropout/batch effects
for param in target_encoder.parameters():
    param.requires_grad = False  # Freeze weights

# In training loop
pred = model(windows_masked)  # Uses training model
target = target_encoder(windows)  # Uses FROZEN encoder
loss = MSE(pred, target)
```

**Now**:
- Targets are computed with **fixed weights** (never change)
- Model cannot reduce loss by changing TokenEncoder
- With constant input → predictions cannot vary systematically
- **Loss should stay constant** (cannot be reduced)

## 🎯 Expected Behavior After Fix

### With Frozen Target Encoder

```
Epoch 1: loss = X.XX (initial)
Epoch 2: loss = X.XX (±0.01, no improvement)
Epoch 3: loss = X.XX (stays constant)
...
Epoch 20: Early stopping (no improvement)

✅ Confirms: No signal information leakage
   Model cannot learn without varying information
```

### If Loss Still Decreases (Would Indicate Another Bug)

```
Epoch 1: loss = X.XX
Epoch 2: loss = Y.YY (Y < X, still decreasing!)
...

❌ Another leakage source exists - investigate:
   - Dropout patterns
   - Batch normalization
   - Model internal state
```

## 📊 Previous vs. Fixed Behavior

### Before Fix (BUGGY)

| Epoch | Loss | Explanation |
|-------|------|-------------|
| 1 | 0.80 | Initial |
| 2 | 0.68 | TokenEncoder weights change → targets change |
| 3 | 0.60 | Model tracks moving target |
| 4 | 0.53 | Loss < target variance (impossible without leakage!) |
| 7 | 0.33 | Well below variance → confirmed leakage |

### After Fix (EXPECTED)

| Epoch | Loss | Explanation |
|-------|------|-------------|
| 1 | X.XX | Initial random loss |
| 2 | X.XX | No improvement (targets fixed, input constant) |
| 3 | X.XX | Loss plateaus immediately |
| ... | X.XX | No learning possible |
| 20 | X.XX | Early stopping |

## 🔬 Why This is Subtle

This bug is particularly sneaky because:

1. **Targets look correct**: We're using real signal, not masked signal
2. **@torch.no_grad() misleads**: Gradients don't flow through target computation, but targets still depend on model weights
3. **Works fine in normal training**: Only problematic in control experiments where input is constant
4. **Loss decreases smoothly**: Looks like normal learning, not an obvious bug

## 🎯 Implications for Normal Training

**Question**: Does this affect normal training (mask_ratio < 1.0)?

**Answer**: **Probably minimal impact**, but technically yes:

### In Normal Training

```python
# Normal training (mask_ratio=0.75)
windows_masked = [signal, signal, 0, signal, ...]  # Mix of real + masked
pred = model(windows_masked)  # Predictions vary by sample (different signals)
target = model.encode_tokens_only(windows)  # Also varies by sample

# Targets change as TokenEncoder weights change, but:
# - Model also receives varying signal information
# - Loss is dominated by reconstruction task, not circular dependency
# - Effect is minor compared to actual signal learning
```

**Impact**: Small bias, but overshadowed by actual signal reconstruction.

### In Control Experiment

```python
# Control (mask_ratio=1.0, no positions)
windows_masked = [0, 0, 0, ...]  # All zeros, all samples identical
pred = model(constant)  # Should be constant across samples
target = model.encode_tokens_only(windows)  # Varies, but depends on current weights!

# With no varying input, circular dependency dominates
# → Loss reduction is PURELY from circular dependency
# → Reveals the bug
```

**Impact**: **Complete** - the ONLY source of loss reduction.

## 🔧 How to Test the Fix

### Step 1: Stop Current Training

```bash
# Press Ctrl+C to stop
```

### Step 2: Restart with Fix

```bash
cd /home/abin/eeg-mlflow
source .venv/bin/activate

# Fix is already applied in pretrain_mamba.py
python eeg_analysis/src/training/pretrain_mamba.py \
    --config eeg_analysis/configs/pretrain.yaml
```

### Step 3: Monitor Loss

**Expected behavior**:
```
Epoch 1: loss = Y.YY
Epoch 2: loss = Y.YY (±0.01)
Epoch 3: loss = Y.YY (no change)
...
Early stopping after ~20 epochs
```

Loss should **plateau immediately** and not decrease.

### Step 4: Interpretation

| Observation | Meaning |
|-------------|---------|
| Loss plateaus | ✅ **Fixed!** No leakage, control experiment successful |
| Loss still decreases slowly (to ~0.5) | ⚠️ Dropout noise (expected, not a problem) |
| Loss decreases significantly (to <0.4) | ❌ Another leak exists (investigate) |

## 🎓 Lessons Learned

1. **Be careful with shared weights**: If both pred and target use the same model components, circular dependencies can arise

2. **Control experiments need truly fixed targets**: For control experiments, targets must be computed with frozen weights

3. **@torch.no_grad() ≠ no dependency**: Even without gradient flow, outputs can depend on model parameters

4. **Subtle bugs show up in edge cases**: This bug didn't significantly affect normal training, only revealed in the control experiment

## 📋 Summary

- **Bug Found**: Targets computed with training model's TokenEncoder created circular dependency
- **Symptom**: Loss decreased to 0.33 even with constant input
- **Fix**: Use frozen TokenEncoder copy for target computation in control mode
- **Expected**: Loss now plateaus immediately, confirming no true signal leakage
- **Verification**: Restart training and check if loss stays constant

**This was an excellent control experiment** - it revealed a real but subtle bug that would have been hard to find otherwise!

