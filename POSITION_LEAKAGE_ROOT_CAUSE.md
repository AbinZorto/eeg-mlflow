# Position Leakage Root Cause Analysis

## ✅ What We Fixed

### 1. Positional Encoding Leakage (FIXED)

**Problem**: Positional encodings were added AFTER masking, so masked tokens still received position information.

**Fix**: Zero positional encoding for masked positions:
```python
if mask_bool is not None:
    temporal = temporal * (~mask_bool).unsqueeze(-1)  # Zero for masked
    spatial = spatial * (~mask_bool).unsqueeze(-1)    # Zero for masked
```

**Result**: ✅ Temporal/Spatial norms are now 0.0 for masked positions

### 2. Position Correlation Improvement

**Before fix**: Position correlation = 0.71 (very high)
**After fix**: Position correlation = 0.26 (moderate, but still present)

**Progress**: 64% reduction in position correlation! ✅

## ❌ Remaining Issue: Mamba's Sequential Hidden State

### The Problem

Even with:
- ✅ Zero positional encoding for masked positions
- ✅ Identical token embeddings (all masked inputs are zeros)
- ✅ Zero spatial encoding for masked positions

**Predictions still differ by position** (similarity 0.94-0.99, not 1.0)

### Root Cause

**Mamba processes tokens sequentially**, and its hidden state accumulates position information:

```
Position 0: hidden_state_0 = f(token_0, initial_state)
Position 1: hidden_state_1 = f(token_1, hidden_state_0)  ← Depends on position 0!
Position 2: hidden_state_2 = f(token_2, hidden_state_1)  ← Depends on position 1!
```

Even with identical inputs:
- Position 0 sees: `f(identical_input, initial_state)`
- Position 1 sees: `f(identical_input, hidden_state_0)` ← Different hidden state!
- Position 2 sees: `f(identical_input, hidden_state_1)` ← Different hidden state!

**Result**: Different hidden states → Different predictions → Position correlation

## 🔍 Evidence from Diagnostics

### Token Embeddings (IDENTICAL ✅)
```
Pos 0: ['0.0845', '-1.1857', '0.1370', ...] (norm=14.287698)
Pos 1: ['0.0845', '-1.1857', '0.1370', ...] (norm=14.287698)  ← Same!
Pos 2: ['0.0845', '-1.1857', '0.1370', ...] (norm=14.287698)  ← Same!
```

### Temporal/Spatial Encodings (ZERO ✅)
```
Temporal norm: 0.0000000000 ✅
Spatial norm:  0.0000000000 ✅
```

### Predictions (DIFFERENT ❌)
```
Pos 0: ['-0.2511', '-0.7443', '0.2894', ...]
Pos 1: ['-0.2216', '-0.5551', '-0.0714', ...]  ← Different!
Pos 2: ['0.0475', '-0.6260', '-0.1538', ...]   ← Different!
```

**Conclusion**: Mamba's sequential processing is the source of remaining position correlation.

## 💡 Solutions Implemented

### Solution 1: Variance Penalty (NEW)

Force all masked predictions to be identical:
```python
if mask_ratio >= 0.95:
    pred_masked_var = pred_masked_all.var(dim=0).mean()
    variance_penalty = pred_masked_var * loss * 1000.0
    loss = loss + variance_penalty
```

**Effect**: Penalizes predictions that vary across masked positions
**Goal**: Force position correlation → 0.0

### Solution 2: Token Permutation (Already Implemented)

Randomize token order before Mamba processes them:
```python
perm = torch.randperm(seq_len)
windows_permuted = windows[perm]
pred_permuted = mamba(windows_permuted)
pred = pred_permuted[inv_perm]  # Unpermute
```

**Effect**: Breaks consistent sequential order
**Status**: Needs to be enabled in config (`shuffle_sequences_prob > 0`)

### Solution 3: Position Regularization (Already Implemented)

Penalize position correlation:
```python
position_penalty = |correlation| * weight * loss * 100
loss = loss + position_penalty
```

**Effect**: Directly penalizes position-dependent predictions
**Status**: Needs to be enabled in config (`prevent_position_only_learning: true`)

## 🎯 Recommended Configuration for 100% Masking

```yaml
mask_ratio: 1.0
prevent_position_only_learning: true
position_regularization_weight: 1.0
shuffle_sequences_prob: 1.0  # Always shuffle to break sequential order
```

## 📊 Expected Results

With all fixes enabled:
- **Position correlation**: < 0.1 (ideally < 0.05)
- **Masked prediction variance**: Near zero
- **Pairwise similarity**: > 0.99 (nearly identical predictions)

## 🔬 Why This Matters

With 100% masking, the model should:
- **NOT** learn position-dependent patterns
- **NOT** use sequential hidden state information
- Predict identical values for all masked positions (or learn nothing)

If position correlation remains > 0.1, it indicates:
1. Mamba's architecture inherently provides position information
2. We may need architectural changes (non-sequential processing for masked positions)
3. Or accept that some position learning is inevitable with sequential models

## 🚀 Next Steps

1. **Enable all anti-position strategies** in config
2. **Train with variance penalty** (newly added)
3. **Monitor position correlation** - should drop to < 0.1
4. **If still high**: Consider architectural changes or accept position learning

## Summary

✅ **Fixed**: Positional encoding leakage (norms = 0.0)
✅ **Improved**: Position correlation (0.71 → 0.26)
❌ **Remaining**: Mamba's sequential hidden state (position correlation = 0.26)
💡 **Solution**: Variance penalty + token permutation + position regularization

The diagnostic output confirms:
- Positional encodings are correctly zeroed ✅
- Token embeddings are identical ✅  
- But predictions still vary by position ❌
- **Root cause**: Mamba's sequential processing

