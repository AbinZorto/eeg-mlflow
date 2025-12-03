# Diagnosis Results: 100% Masking Model

**Date**: December 2, 2025  
**Model**: mamba2_eeg_pretrained.pt (d_model=512, num_layers=2, mask_ratio=1.0)  
**Samples Analyzed**: 17,436 masked token predictions

---

## 🔍 Key Findings

### 1. ⚠️ Strong Position Dependence

**Position Correlation: 0.4028**

- **Finding**: Model predictions have a strong correlation (40%) with temporal position (t/T)
- **Interpretation**: The model is learning **"at time t, predict embedding X"** rather than learning from signal context
- **Implication**: Position alone is a major factor in predictions

**What this means:**
```
Instead of learning: "Given context [signals before/after], reconstruct masked signal"
Model is learning:   "At position 25% through sequence, predict this embedding"
```

---

### 2. 🚨 CRITICAL: Zero Channel Diversity

**Channel Diversity (std of channel means): 0.0000**

- **Finding**: ALL 65 CHANNELS predict nearly identical values
- **Channel means**: All approximately -0.0021 (identical to 4 decimal places)
- **Interpretation**: The model is **completely ignoring channel information**

**This is the smoking gun!**

The model has access to:
- ✅ Temporal position (t/T) - USED
- ❌ Channel identity (C3, FP1, etc.) - IGNORED
- ❌ Signal content - UNAVAILABLE (100% masked)

**What the model learned:**
```python
def predict(position_t, channel):
    # Ignore channel completely!
    return mean_embedding + position_offset(position_t)
```

---

### 3. ✅ High-Dimensional Representations

**Variance Explained by Top 3 Dimensions: 0.9%**

- **Finding**: Predictions span high-dimensional space (not collapsed to a few dimensions)
- **Interpretation**: Model is not just predicting a single mean vector
- **Good sign**: Representations have internal structure

**But combined with finding #2**, this tells us:
- Predictions vary smoothly with temporal position
- But are essentially identical across channels
- High dimensionality comes from temporal variation, not channel-specific patterns

---

## 🎯 Diagnosis: Position-Only Learning

### What the Model Learned

The model is implementing approximately:

```python
embedding = f(t/T)  # Function of temporal position only
# Channel information is ignored
# Signal content is unavailable (100% masked)
```

This is **NOT** learning useful EEG representations because:

1. ❌ **No channel specificity**: All electrodes treated identically
2. ❌ **No signal understanding**: Only position matters
3. ❌ **No context usage**: Can't learn from surrounding signals

### Why Loss Still Drops

Loss decreases because:
1. Random predictions → Very high loss
2. Predicting temporal mean → Lower loss (better than random)
3. Fine-tuning per-position → Even lower loss

**But**: This doesn't mean the model learned anything useful!

**Analogy**: 
- Task: "Predict what word comes next in a book"
- Model with context: Reads previous words, understands story
- Your model: Only knows "word at 25% through book" → predicts average word at that position
- Loss drops (better than random) but no real understanding

---

## 🔬 Root Cause Analysis

### Why Is Channel Information Ignored?

**Hypothesis**: Spatial encoding is too weak compared to temporal encoding

Looking at the model architecture:

1. **Token Embedding**: `LayerNorm(Linear([0,0,...,0]) + bias)` = same for all masked tokens
2. **Temporal Encoding**: `Linear(t/T)` - continuous, varies across sequence
3. **Spatial Encoding**: `Fixed_Projection(channel_coords)` - fixed per channel

**The problem:**
- Temporal encoding has **strong gradient signal** (varies within each sequence)
- Spatial encoding has **weak gradient signal** (constant within sequence, varies across batches)
- With 100% masking, temporal is the ONLY signal, so model learns to rely on it exclusively

**Spatial encoding gets drowned out** by temporal encoding dominance.

---

## 💡 Why This Happens with 100% Masking

### Comparison: Context-Based vs Position-Based Learning

**With Unmasked Context (mask_ratio < 1.0):**
```
Input: [signal, signal, MASKED, signal, signal]
Model learns: Use surrounding signals to reconstruct masked one
Spatial encoding: Helps identify which channel patterns to use
```

**With 100% Masking (mask_ratio = 1.0):**
```
Input: [ZERO, ZERO, ZERO, ZERO, ZERO]
Model learns: Only position available, just predict positional mean
Spatial encoding: Provides channel info but model has no reason to use it
                  (all zeros look the same regardless of channel!)
```

---

## 🎯 Recommendations

### Immediate Action: Reduce Mask Ratio

**Update `eeg_analysis/configs/pretrain.yaml`:**

```yaml
mask_ratio: 0.75  # or 0.5 for easier learning
```

**Why this will help:**

1. ✅ **Unmasked tokens provide context**: Model must learn channel-specific patterns
2. ✅ **Forces spatial encoding to matter**: Different channels will have different patterns in unmasked tokens
3. ✅ **Enables true reconstruction**: Model learns "use C3 context to predict C3 signal"

### Expected Results After Reducing Mask Ratio

When you retrain with `mask_ratio=0.75`:

**What should change:**
- Channel diversity should increase (different channels → different predictions)
- Position correlation should decrease (predictions depend more on context)
- Representations should be more useful for downstream tasks

**How to verify:**
```bash
# After retraining with mask_ratio=0.75
python diagnose_100pct_masking.py \
    --checkpoint eeg_analysis/checkpoints/mamba2_eeg_pretrained_0.75.pt \
    --num-samples 100

# Look for:
# - Channel diversity > 0.01 (channels differ)
# - Position correlation < 0.3 (less position-dependent)
```

---

## 📊 Comparison: What Good Learning Should Look Like

### Current Model (mask_ratio=1.0)
- Position correlation: **0.40** (high - bad)
- Channel diversity: **0.00** (none - very bad)
- Interpretation: Position-only learning

### Target Model (mask_ratio=0.75)
- Position correlation: **< 0.20** (low - good)
- Channel diversity: **> 0.05** (distinct channels - good)
- Interpretation: Context-based learning with channel awareness

---

## 🔬 Technical Deep Dive: Why Channels Are Ignored

### Gradient Flow Analysis

With 100% masking:

```python
# Forward pass
token_emb = token_encoder([0,0,...,0])  # Same for all masked tokens
spatial = spatial_encoder(channel)       # Different per channel
temporal = temporal_encoder(t/T)        # Different per position

x = token_emb + spatial + temporal       # Combine

# Backward pass (loss gradient)
# For within-sequence variations: temporal gets strong signal
# For across-batch variations: spatial gets weak signal (batch mixing)
```

**The gradient imbalance:**
- Temporal: Dense signal every sequence
- Spatial: Sparse signal across batches
- Result: Model learns to rely on temporal, ignores spatial

**With unmasked context:**
```python
token_emb = token_encoder(actual_signal)  # DIFFERENT for each token
# Now token_emb varies with channel characteristics
# Spatial encoding reinforces this channel-specific pattern
# Model learns: "use spatial encoding to interpret token patterns"
```

---

## Conclusion

**Your model is not experiencing information leakage.**

**Instead, it's learning a trivial solution:**
- Predict average embedding at each temporal position
- Ignore all channel information
- Loss drops but representations lack semantic meaning

**Solution:**
```yaml
# Change in pretrain.yaml
mask_ratio: 0.75  # Force context-based learning
```

This will force the model to:
1. Learn from unmasked signal context
2. Use channel-specific patterns
3. Develop representations useful for downstream tasks

**The 100% masking experiment was valuable** - it revealed that your model CAN learn, but without context, it learns only trivial positional patterns.

