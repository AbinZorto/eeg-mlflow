# MAE Pipeline Audit: Line-by-Line Analysis

**Date**: December 2, 2025  
**Auditor**: Comprehensive code review per user request  
**Configuration**: `reconstruct_signal_space: true`

---

## Executive Summary

### ✅ PASS: With `reconstruct_signal_space: true`

The pipeline is **CORRECT** when signal reconstruction is enabled:
- Targets are raw signal (2048 samples), not encoded
- No learnable modules used for target computation
- Masking applied before projection
- Loss computed only on masked positions
- Predictions decoded back to signal space for comparison

### ⚠️ WARNING: Legacy Code Paths

**Dangerous fallback paths exist** that use encoders for targets (lines 270-275, 380-383). These should be **REMOVED** to prevent accidental misuse.

---

## Detailed Audit

### 1. ✅ Target Computation (Training Loop)

**File**: `eeg_analysis/src/training/pretrain_mamba.py`  
**Lines**: 266-275

```python
# Targets
if use_signal_reconstruction:
    # Proper MAE: Target is actual signal (never changes!)
    target = windows  # (B, L, W) - raw signal
elif target_encoder is not None:
    # Embedding space + control mode: frozen encoder
    target = target_encoder(windows)  # (B, L, D) ← ENCODER USED!
else:
    # Embedding space + normal training: current encoder  
    target = (model.module if isinstance(model, DDP) else model).encode_tokens_only(windows)  # ← ENCODER USED!
```

**Status**: ✅ **PASS** (when `use_signal_reconstruction=True`)
- Line 269: `target = windows` - Direct assignment of raw signal
- No learnable modules involved
- `windows` is from collate function (verified below)

**⚠️ ISSUE**: Lines 270-275 contain **dangerous fallback paths**:
- Line 272: Uses `target_encoder(windows)` - frozen but still an encoder
- Line 275: Uses `model.encode_tokens_only(windows)` - training encoder!

**Recommendation**: Remove these paths entirely or add assertions:

```python
# Targets - MUST be raw signal for proper MAE
if use_signal_reconstruction:
    target = windows  # (B, L, W) - raw signal
else:
    raise ValueError("Embedding space targets are deprecated. Use reconstruct_signal_space=true")
```

---

### 2. ✅ Target Computation (Validation Loop)

**File**: `eeg_analysis/src/training/pretrain_mamba.py`  
**Lines**: 378-383

```python
# Targets
if use_signal_reconstruction:
    target = windows  # Raw signal
elif target_encoder is not None:
    target = target_encoder(windows)  # Frozen encoder ← ENCODER USED!
else:
    target = (model.module if isinstance(model, DDP) else model).encode_tokens_only(windows)  # ← ENCODER USED!
```

**Status**: ✅ **PASS** (when `use_signal_reconstruction=True`)
- Line 379: `target = windows` - Raw signal
- Consistent with training loop

**⚠️ ISSUE**: Same dangerous fallback paths as training loop.

**Recommendation**: Remove fallback paths.

---

### 3. ✅ Raw Signal Source (Collate Function)

**File**: `eeg_analysis/src/data/eeg_pretraining_dataset.py`  
**Lines**: 157-167, 194-200

```python
# Allocate tensors
orig = torch.zeros((B, max_len, window_length), dtype=torch.float32)
masked = torch.zeros((B, max_len, window_length), dtype=torch.float32)

# Fill
for i, b in enumerate(batch):
    L = b["seq_len"]
    orig[i, :L, :] = b["windows"]  # Direct copy from dataset
    masked[i, :L, :] = b["windows"]  # Start with original

return {
    "windows": orig,                # (B, L, W) - original unmasked
    "windows_masked": masked,       # (B, L, W) - masked input
    "mask_bool": mask_bool,         # (B, L) - True at masked positions
}
```

**Status**: ✅ **PASS**
- `orig` (returned as `windows`) is direct copy from dataset
- No projection, normalization, or encoding
- Pure tensor copy operation
- `b["windows"]` comes from dataset `__getitem__` (line 119 in dataset.py)

**Verification Chain**:
```python
# Dataset returns raw windows
return {
    "windows": windows_t,  # torch.from_numpy(windows_np)
    ...
}
# → Collate copies directly
orig[i, :L, :] = b["windows"]
# → Training uses directly  
target = windows
```

✅ **No learnable modules in the chain**

---

### 4. ✅ Masking Applied Before Projection

**File**: `eeg_analysis/src/data/eeg_pretraining_dataset.py`  
**Lines**: 174-177

```python
if masking_style == "mae":
    # MAE-style: Replace ALL masked positions with zeros
    # No information leakage - model must reconstruct from context only
    masked[i, idxs, :] = 0.0  # Masking on raw windows (shape: window_length=2048)
```

**Status**: ✅ **PASS**
- Masking applied to raw window tensors (2048 samples)
- BEFORE any model processing
- In collate function, not in model

**Flow**:
```
Dataset → Raw windows (2048)
    ↓
Collate → Mask applied (zeros inserted)
    ↓
Model → TokenEncoder projects (2048 → 512)
    ↓
    → Backbone processes
    ↓
    → Decoder reconstructs (512 → 2048)
```

✅ Masking happens at step 2, projection at step 3.

---

### 5. ⚠️ Loss Computation (Control Mode Issue)

**File**: `eeg_analysis/src/training/pretrain_mamba.py`  
**Lines**: 277-308

```python
mask_exp = mask_bool.unsqueeze(-1).expand_as(pred)  # (B, L, D)

# Control experiment: Check if model is learning dataset mean
if disable_temporal and disable_spatial:
    # ... centering logic ...
    pred_masked = pred[mask_exp].view(-1, pred.shape[-1])  # Extract masked
    target_masked = target[mask_exp].view(-1, target.shape[-1])  # Extract masked
    
    # Remove mean (model can't reduce loss by learning a constant)
    target_mean = target_masked.mean(dim=0, keepdim=True)  # (1, D)
    pred_centered = pred_masked - target_mean  # ← MANIPULATES targets!
    target_centered = target_masked - target_mean  # ← MANIPULATES targets!
    
    loss = (pred_centered - target_centered).pow(2).mean()
else:
    # Normal training (standard loss)
    diff = pred - target
    masked_diff = diff[mask_exp]  # (N_masked * D,)
    loss = masked_diff.pow(2).mean()
```

**Status**: ⚠️ **ISSUE WITH CONTROL MODE**

**Normal Mode (lines 305-308)**: ✅ **PASS**
- Line 306: `diff = pred - target` - Direct comparison
- Line 307: `masked_diff = diff[mask_exp]` - Only masked positions
- Line 308: MSE on masked positions only
- ✅ Compares predictions to **raw signal targets**

**Control Mode (lines 280-294)**: ⚠️ **CONCERN**
- Lines 290-292: Subtracts mean from targets
- This **modifies the ground truth**
- While mathematically it doesn't create new information, it obscures the direct signal comparison

**Why this is problematic**:
- The whole point of using raw signal targets is that they're **fixed ground truth**
- Centering transforms them: `target_centered = target - mean(target)`
- This is no longer comparing to actual EEG samples

**Recommendation**: For control experiments with raw signals, centering is **unnecessary and confusing**:

```python
# Control mode should just use standard loss
# If model can't learn (constant input + fixed targets), loss stays high
# No need to center - targets are already fixed!

# Remove centering logic entirely
diff = pred - target
masked_diff = diff[mask_exp]
loss = masked_diff.pow(2).mean()
```

---

### 6. ✅ Model Forward Pass

**File**: `eeg_analysis/src/models/mamba_eeg_model.py`  
**Lines**: 257-295

```python
def forward(
    self,
    windows_masked: torch.Tensor,  # (B, L, 2048) - raw signal with zeros
    ...,
    decode_to_signal: bool = False,
):
    token_emb = self.token_encoder(windows_masked)  # (B, L, 512) - Project
    
    # ... add positional encodings ...
    
    embeddings = self.backbone(x)  # (B, L, 512) - Process
    
    # Optionally decode back to signal space (proper MAE)
    if decode_to_signal:
        reconstructed = self.decoder(embeddings)  # (B, L, 2048) - Decode
        return reconstructed
    else:
        return embeddings
```

**Status**: ✅ **PASS** (when `decode_to_signal=True`)
- Line 272: Encoder processes **masked input** (with zeros)
- Line 288: Backbone processes embeddings
- Line 292: Decoder reconstructs to signal space (2048)
- Returns reconstructed signal, **same shape as targets**

**Key point**: When `decode_to_signal=True`:
- Predictions: `(B, L, 2048)` - reconstructed signal
- Targets: `(B, L, 2048)` - raw signal
- **Direct comparison in signal space** ✅

---

### 7. ✅ No Encoder Used for Targets (When Configured Correctly)

**File**: `eeg_analysis/src/training/pretrain_mamba.py`  
**Lines**: 168-176

```python
# Reconstruction mode: Proper MAE (predict signal) vs embedding prediction
use_signal_reconstruction = bool(cfg.get("reconstruct_signal_space", True))  # Default: True

# If reconstructing signal space: targets are raw signal (never change!)
if use_signal_reconstruction:
    target_encoder = None  # Targets are raw signal, no encoder needed
    if is_main:
        logger.info("Using signal space reconstruction (proper MAE)")
```

**Status**: ✅ **PASS**
- `target_encoder = None` when signal reconstruction enabled
- Comment explicitly states "no encoder needed"
- Falls through to line 269: `target = windows` (raw signal)

**Verification**: With config `reconstruct_signal_space: true`:
1. `use_signal_reconstruction = True` (line 169)
2. `target_encoder = None` (line 174)
3. Training: `if use_signal_reconstruction: target = windows` (line 267-269)
4. Validation: `if use_signal_reconstruction: target = windows` (line 378-379)

✅ **No encoder ever touches targets**

---

## Summary of Issues

### Critical Issues: ❌ NONE (when configured correctly)

With `reconstruct_signal_space: true`, the pipeline is **correct**.

### High Priority Warnings: ⚠️ 2 Issues

1. **Dangerous fallback paths** (lines 270-275, 380-383)
   - Should be removed to prevent accidental encoder use
   - Could cause circular dependency bug if config is changed

2. **Unnecessary centering in control mode** (lines 280-294)
   - Obscures direct signal comparison
   - Unnecessary with fixed targets
   - Should be removed for clarity

### Medium Priority: 📝 1 Issue

3. **Legacy embedding space code** still present
   - `encode_tokens_only()` method (line 237-245)
   - Frozen encoder logic (lines 177-185)
   - Should be deprecated or removed

---

## Recommendations

### 1. Remove Dangerous Fallback Paths ✅ CRITICAL

```python
# In pretrain_mamba.py, lines 266-275 (training) and 378-383 (validation)
# BEFORE (dangerous):
if use_signal_reconstruction:
    target = windows
elif target_encoder is not None:  # ← REMOVE THIS
    target = target_encoder(windows)
else:  # ← REMOVE THIS
    target = model.encode_tokens_only(windows)

# AFTER (safe):
if use_signal_reconstruction:
    target = windows  # Raw signal
else:
    raise ValueError(
        "Embedding space reconstruction is deprecated and causes circular dependency. "
        "Set reconstruct_signal_space=true in config."
    )
```

### 2. Remove Centering Logic ✅ HIGH PRIORITY

```python
# In pretrain_mamba.py, lines 277-308
# BEFORE (complex):
if disable_temporal and disable_spatial:
    # ... 25 lines of centering logic ...
else:
    diff = pred - target
    masked_diff = diff[mask_exp]
    loss = masked_diff.pow(2).mean()

# AFTER (simple):
# Direct loss - no special case needed
diff = pred - target
masked_diff = diff[mask_exp]  # Only masked positions
loss = masked_diff.pow(2).mean()

# With fixed targets + constant input, loss CANNOT decrease
# Centering is unnecessary and confusing
```

### 3. Deprecate Embedding Space Code ✅ MEDIUM PRIORITY

```python
# In mamba_eeg_model.py
@torch.no_grad()
@deprecated("Use decode_to_signal=True for proper MAE reconstruction")
def encode_tokens_only(self, windows: torch.Tensor) -> torch.Tensor:
    """DEPRECATED: Creates circular dependency in MAE training."""
    raise DeprecationWarning("This method should not be used. Enable decode_to_signal=True.")
```

### 4. Add Configuration Validation ✅ HIGH PRIORITY

```python
# At start of pretrain_mamba.py main()
use_signal_reconstruction = bool(cfg.get("reconstruct_signal_space", True))

if not use_signal_reconstruction:
    raise ValueError(
        "reconstruct_signal_space must be True. "
        "Embedding space reconstruction causes circular dependency bug."
    )
```

---

## Final Verdict

### Current Status: ✅ **CORRECT** (with caveats)

**When `reconstruct_signal_space: true` is set**:

| Requirement | Status | Notes |
|------------|--------|-------|
| 1. Targets are raw signal (2048) | ✅ PASS | `target = windows` |
| 2. No encoder operates on targets | ✅ PASS | Direct assignment |
| 3. Masking before projection | ✅ PASS | In collate function |
| 4. Loss only on masked positions | ✅ PASS | Boolean indexing |
| 5. Compare to raw signal, not embeddings | ✅ PASS | Decoder produces (B,L,2048) |
| 6. Targets are direct clone, not encoded | ✅ PASS | Tensor copy in collate |
| 7. No need for frozen encoders | ✅ PASS | `target_encoder = None` |

### Dangerous Code Exists: ⚠️ **WARNING**

**Fallback paths that use encoders** (lines 270-275, 380-383) still exist and could be accidentally triggered if:
- Config is changed
- Code is refactored
- Someone doesn't understand the significance of `reconstruct_signal_space`

**These should be REMOVED immediately** to prevent future bugs.

---

## Minimal Required Changes

To make the pipeline **bulletproof**, apply these changes:

```python
# Change 1: Remove fallback paths in training loop (line 266-275)
# Targets
assert use_signal_reconstruction, "Must use signal space reconstruction"
target = windows  # (B, L, W) - raw signal

# Change 2: Remove fallback paths in validation loop (line 378-383)
# Targets  
assert use_signal_reconstruction, "Must use signal space reconstruction"
target = windows  # Raw signal

# Change 3: Remove centering (replace lines 277-308)
# Loss computation - works for both control and normal modes
diff = pred - target
masked_diff = diff[mask_exp]  # Only masked positions
loss = masked_diff.pow(2).mean()

# Change 4: Add config validation (after line 169)
if not use_signal_reconstruction:
    raise ValueError("reconstruct_signal_space must be True")
```

With these 4 changes, the pipeline becomes **provably correct** with no dangerous paths.

---

## Conclusion

Your MAE pipeline is **fundamentally correct** when properly configured (`reconstruct_signal_space: true`).

The main issues are:
1. **Legacy code paths** that could accidentally use encoders
2. **Unnecessary complexity** in control mode centering
3. **Lack of validation** to prevent misconfiguration

**All issues are in the training script, not the core architecture.**

Apply the minimal changes above to make it bulletproof. 🎯

