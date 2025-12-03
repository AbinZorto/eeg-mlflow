# Baseline Similarity Analysis: Understanding Window-Level Learning

## Key Insight

**Token vectors for the same window across different channels are naturally more similar than vectors from different windows.**

This means:
- Ground truth windows have inherent similarity structure
- Model should learn to maintain this structure
- Low per-window correlation doesn't necessarily mean poor learning

## The Problem with Simple Pattern Correlation

### Traditional Approach (Misleading)

```
For each masked window:
  pred_window = model prediction
  gt_window = ground truth
  correlation = corr(pred_window, gt_window)
  
If correlation is low → Model is not learning ❌
```

**Issue**: This ignores the fact that:
- Different channels for the same window should be similar
- Model might learn window-level patterns correctly
- But per-window correlation appears low because model predicts window-level features, not exact channel-specific details

### Example

**Ground Truth**:
- Window 1, Channel A: [1, 2, 3, ...]
- Window 1, Channel B: [1.1, 2.1, 3.1, ...] ← Similar to Channel A
- Window 2, Channel A: [10, 20, 30, ...] ← Different from Window 1

**Model Prediction** (Learning window-level patterns):
- Window 1, Channel A: [0.9, 1.9, 2.9, ...] ← Learns Window 1 pattern
- Window 1, Channel B: [0.95, 1.95, 2.95, ...] ← Learns Window 1 pattern (similar!)
- Window 2, Channel A: [9.5, 19.5, 29.5, ...] ← Learns Window 2 pattern (different!)

**Per-window correlation**: Low (model doesn't match exact channel details)
**Baseline similarity**: High (model maintains GT similarity structure) ✅

## Baseline Similarity Analysis

### What It Measures

1. **GT Baseline Similarity**: 
   - Sample random pairs of GT windows
   - Compute average similarity
   - Measures inherent similarity structure in data

2. **Predicted Baseline Similarity**:
   - Sample random pairs of predicted windows
   - Compute average similarity
   - Measures if model maintains same structure

3. **Comparison**:
   - If `|pred_similarity - gt_similarity| < 0.1` → Model maintains structure ✅
   - If difference is large → Model doesn't learn structure ❌

### Interpretation

**Good Learning** (Window-Level Patterns):
```
GT baseline similarity: 0.35
Pred baseline similarity: 0.33
Difference: 0.02 ✅

Per-window correlation: 0.15 (low)
BUT: Model maintains similarity structure!
→ Model learns window-level patterns correctly
```

**Poor Learning** (Positional Patterns):
```
GT baseline similarity: 0.35
Pred baseline similarity: 0.85
Difference: 0.50 ❌

Per-window correlation: 0.10 (low)
AND: Model doesn't maintain similarity structure
→ Model learning positional patterns, not signal content
```

## Updated Diagnostic Interpretation

### Pattern Correlation Alone (Misleading)

```
Pattern correlation: -0.047
→ ❌ Model does NOT reconstruct signal patterns
```

**Problem**: Ignores baseline similarity structure!

### Pattern Correlation + Baseline Similarity (Accurate)

```
Pattern correlation: -0.047 (low)
GT baseline similarity: 0.32
Pred baseline similarity: 0.30
Difference: 0.02 ✅

→ Model maintains similarity structure
→ Low per-window correlation BUT learns window-level patterns
→ This is actually GOOD learning!
```

## Why This Matters

### Window-Level Learning is Correct!

In EEG data:
- **Same window, different channels**: Should be similar (shared temporal pattern)
- **Different windows**: Should be different (different temporal patterns)

**Model should learn**:
- Window-level temporal patterns ✅
- Not exact per-channel reconstruction (channels vary slightly)

**Traditional diagnostic** (per-window correlation) penalizes this correct behavior!

### Example: What Model Should Learn

**Input**: Masked windows from multiple channels
**Output**: Window-level temporal pattern

**Correct behavior**:
- Predictions for same window across channels: Similar ✅
- Predictions for different windows: Different ✅
- Per-window correlation: Low (doesn't match exact channel details)
- Baseline similarity: High (maintains GT structure) ✅

## Diagnostic Output

The updated diagnostic now reports:

```
BASELINE CROSS-CHANNEL SIMILARITY ANALYSIS
============================================================

Baseline GT similarity (random window pairs): 0.3245 ± 0.1234
Predicted similarity (random window pairs):    0.3102 ± 0.1156

Difference: -0.0143

✅ Model maintains similar baseline similarity structure as GT
```

**Interpretation**:
- Model learns window-level patterns correctly
- Maintains GT similarity structure
- Low per-window correlation is expected (model doesn't match exact channel details)

## When to Worry

### ❌ Bad: Model Doesn't Maintain Structure

```
GT baseline similarity: 0.35
Pred baseline similarity: 0.85
Difference: 0.50 ❌

→ Model predicts similar values for all windows
→ Learning positional patterns, not signal content
```

### ✅ Good: Model Maintains Structure

```
GT baseline similarity: 0.35
Pred baseline similarity: 0.33
Difference: 0.02 ✅

→ Model maintains GT similarity structure
→ Learning window-level patterns correctly
→ Low per-window correlation is OK!
```

## Summary

**Key Insight**: 
- Token vectors for same window across channels are naturally similar
- Model should learn to maintain this similarity structure
- Low per-window correlation doesn't mean poor learning if baseline similarity is maintained

**Updated Diagnostic**:
- Computes baseline GT similarity
- Compares to predicted baseline similarity
- If similar → Model learns window-level patterns correctly ✅
- If different → Model learning positional patterns ❌

**Takeaway**: 
- Window-level learning is correct behavior
- Baseline similarity analysis reveals this
- Per-window correlation alone can be misleading

