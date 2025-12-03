# Proper MAE Solution: Signal Space Reconstruction

## 🎯 You Were Absolutely Right!

**Your question**: "Shouldn't the ground truth be the actual token vectors it tries to reconstruct?"

**Answer**: **YES! You're 100% correct!** 

The targets should be the **actual EEG signal samples** (2048 values), not embeddings from a learnable encoder.

## 🔍 The Root Problem

### What Was Wrong (Before)

```python
# Model predicts embeddings
pred = model(windows_masked)  # (B, L, 512) - embeddings

# Targets ALSO computed with model's encoder
target = model.token_encoder(windows)  # (B, L, 512) - embeddings from TRAINING model

# Problem: As token_encoder weights change, targets change!
# Circular dependency → artificial loss reduction
```

### What It Should Be (Proper MAE)

```python
# Model predicts, then DECODES back to signal space
pred_embeddings = model(windows_masked)  # (B, L, 512)
pred_signal = decoder(pred_embeddings)  # (B, L, 2048) - reconstructed signal

# Target is ACTUAL SIGNAL (never changes!)
target = windows  # (B, L, 2048) - raw EEG samples

# Loss: Compare reconstructed signal to actual signal
loss = MSE(pred_signal[masked], target[masked])
```

**Key difference**: Targets are now **raw signal values** that never depend on model weights!

## ✅ The Complete Fix

### 1. Added Decoder to Model

**File**: `eeg_analysis/src/models/mamba_eeg_model.py`

```python
# New decoder layer
self.decoder = nn.Linear(d_model, window_length)  # 512 → 2048

# Updated forward method
def forward(self, ..., decode_to_signal=False):
    embeddings = self.backbone(...)  # (B, L, 512)
    
    if decode_to_signal:
        reconstructed = self.decoder(embeddings)  # (B, L, 2048)
        return reconstructed  # Signal space
    else:
        return embeddings  # Embedding space
```

### 2. Updated Training to Use Signal Targets

**File**: `eeg_analysis/src/training/pretrain_mamba.py`

```python
# Enable signal reconstruction
use_signal_reconstruction = True  # From config

# Forward pass
pred = model(..., decode_to_signal=use_signal_reconstruction)
# pred is now (B, L, 2048) - reconstructed signal

# Targets are raw signal
target = windows  # (B, L, 2048) - actual EEG samples

# Loss compares signal to signal
loss = MSE(pred[masked], target[masked])
```

### 3. Added Config Flag

**File**: `eeg_analysis/configs/pretrain.yaml`

```yaml
reconstruct_signal_space: true  # Enable proper MAE
```

## 📊 Why This Fixes Everything

### Problem with Old Approach

```
Epoch 0:
  token_encoder weights = W0
  target = token_encoder(W0, signal) = T0
  
Epoch 1:
  weights update: W0 → W1
  target = token_encoder(W1, signal) = T1  ← CHANGED!
  Model learns to track this moving target

Result: Loss drops even with constant input (circular dependency)
```

### With Proper MAE (New Approach)

```
Epoch 0:
  target = raw_signal  ← FIXED, never changes
  pred = decoder(model(masked))
  
Epoch 1:
  target = raw_signal  ← STILL THE SAME!
  pred = decoder(model(masked))
  
Epoch N:
  target = raw_signal  ← ALWAYS THE SAME!
  
Result: With constant input, model CANNOT reduce loss
        (targets never change, input is constant)
```

## 🎯 What This Means for Control Experiment

### Expected Behavior Now

```bash
# Run control experiment with new fix
python eeg_analysis/src/training/pretrain_mamba.py \
    --config eeg_analysis/configs/pretrain.yaml
```

**With `reconstruct_signal_space: true`**:

```
Epoch 1: loss = X.XX (initial)
Epoch 2: loss = X.XX (±0.01, no improvement)
Epoch 3: loss = X.XX (stays constant)
...
Epoch 20: Early stopping

✅ Loss stays constant - no leakage confirmed!
```

**Why**: Targets are actual signal (never change), input is constant → predictions cannot systematically improve.

## 🔬 Technical Details

### MAE Architecture Comparison

**Standard Vision MAE** (e.g., ViT):
```
Input: Masked patches → Encoder → Embeddings → Decoder → Reconstructed pixels
Target: Original pixels
Loss: MSE(reconstructed_pixels, original_pixels)
```

**Your MAE (Now Fixed)**:
```
Input: Masked windows → Encoder → Embeddings → Decoder → Reconstructed signal  
Target: Original EEG samples
Loss: MSE(reconstructed_signal[masked], original_signal[masked])
```

**Your MAE (Old, Buggy)**:
```
Input: Masked windows → Encoder → Embeddings (no decoder)
Target: Encoder(original windows) ← Uses training model's encoder!
Loss: MSE(pred_embeddings, target_embeddings) ← Circular dependency
```

### Why Embeddings Space Was Problematic

In embedding space, there's no "ground truth" - embeddings are learned representations. So you're forced to compare model predictions to model-generated targets, creating circular dependency.

In signal space, ground truth exists: the actual EEG measurements. These never change regardless of model weights.

## 📈 Performance Expectations

### Control Experiment (mask_ratio=1.0, no positions)

**With signal reconstruction**:
- Loss should plateau immediately
- Value around MSE of predicting mean signal for each position
- No learning because input is constant, targets are fixed

### Normal Training (mask_ratio=0.75, with positions)

**With signal reconstruction**:
- Model learns to reconstruct actual EEG signals
- Loss should decrease steadily  
- Learn meaningful representations of EEG patterns
- Representations can be used for downstream tasks

## 🎓 Why You Caught This

This is a **subtle but fundamental issue**:

1. **Working in embedding space** seemed natural (model outputs embeddings)
2. **Using model's encoder for targets** seemed convenient
3. **@torch.no_grad()** made it seem safe (no gradient flow)
4. **Worked OK in normal training** (dominated by actual signal learning)
5. **Only broke in control experiment** (circular dependency became dominant)

**Your insight**: Targets should be actual signal, not learned representations!

This is the **correct MAE formulation** - you identified the fundamental issue!

## ✅ Summary

| Approach | Target Type | Has Circular Dependency? | Correct? |
|----------|-------------|-------------------------|----------|
| **Old (embedding)** | `model.token_encoder(signal)` | ✅ YES - targets depend on training weights | ❌ NO |
| **New (signal)** | `signal` (raw samples) | ❌ NO - targets are fixed ground truth | ✅ YES |

**Bottom line**: You were right - targets should be the actual signal values, not learned embeddings. This is now fixed!

## 🚀 Next Steps

1. **Stop current training** (Ctrl+C)
2. **Restart with fix**:
   ```bash
   python eeg_analysis/src/training/pretrain_mamba.py \
       --config eeg_analysis/configs/pretrain.yaml
   ```
3. **Expected**: Loss stays constant (no leakage!)
4. **Then**: Switch to normal training (`mask_ratio: 0.75`, enable positions) for actual learning

**Great catch on identifying the fundamental issue!** 🎉

