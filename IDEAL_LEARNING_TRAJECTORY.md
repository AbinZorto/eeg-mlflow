# Ideal Learning Trajectory for MAE Pretraining

## The Learning Phases

### Phase 1: Position Learning (Early Epochs)
**What happens:**
- Model learns position-dependent patterns quickly
- Position correlation increases (e.g., 0.0 → 0.3-0.5)
- Easy to learn: Sequential processing naturally provides position

**Is this good?**
- ✅ **YES** - Position is useful for temporal understanding
- ✅ **YES** - It's a stepping stone to richer representations
- ⚠️ **BUT** - Shouldn't be the ONLY thing learned

**Expected:**
- Position correlation: 0.3-0.5 (moderate)
- Pattern correlation: Low initially (0.0-0.1)
- Context sensitivity: High initially (0.9+) - predictions don't vary with context

### Phase 2: Context Learning (Middle Epochs)
**What happens:**
- Model starts learning from unmasked context
- Context sensitivity decreases (e.g., 0.9 → 0.6-0.7)
- Predictions start varying with different masking patterns

**Is this good?**
- ✅ **YES** - Model learns from signal content
- ✅ **YES** - Predictions become context-dependent
- ⚠️ **BUT** - Pattern correlation still low

**Expected:**
- Position correlation: Stays moderate (0.3-0.5) or decreases
- Pattern correlation: Starts increasing (0.1 → 0.2-0.3)
- Context sensitivity: Decreases (0.9 → 0.6-0.7)

### Phase 3: Signal Pattern Learning (Late Epochs)
**What happens:**
- Model learns actual signal patterns
- Pattern correlation increases (e.g., 0.2 → 0.5-0.7)
- Predictions match ground truth waveforms

**Is this good?**
- ✅ **YES** - Model learns rich signal representations
- ✅ **YES** - Useful for downstream tasks
- ✅ **YES** - This is the goal!

**Expected:**
- Position correlation: Moderate (0.2-0.4) - position helps but isn't dominant
- Pattern correlation: High (0.5-0.7+) - learns signal patterns
- Context sensitivity: Low (0.4-0.6) - predictions vary with context

## The Problem We're Seeing

### Current State (mask_ratio=0.75)
- ✅ Position correlation: 0.31 (moderate - good!)
- ✅ Context sensitivity: 0.58 (low - good! Model learns from context)
- ❌ Pattern correlation: -0.045 (very low - bad! Not learning signal patterns)
- ❌ Baseline similarity: 0.965 (very high - bad! All predictions similar)

### What This Means
1. **Position learning**: ✅ Working (moderate correlation)
2. **Context learning**: ✅ Working (low context sensitivity)
3. **Signal pattern learning**: ❌ NOT working (low pattern correlation)

**The model is stuck between Phase 1 and Phase 2:**
- It learned position (Phase 1) ✅
- It learned to use context (Phase 2) ✅
- But it's NOT learning signal patterns (Phase 3) ❌

## Why Signal Pattern Learning Isn't Happening

### Possible Causes

1. **Decoder too simple** (FIXED: Now MLP)
   - Single Linear layer couldn't learn complex patterns
   - MLP should help

2. **Model capacity insufficient**
   - d_model=512, num_layers=4 might not be enough
   - May need larger model

3. **Loss function issue**
   - Per-window normalization might be removing signal structure
   - Model optimizes for normalized targets, not actual signals

4. **Training dynamics**
   - Learning rate too high/low
   - Model converges to local minimum (constant predictions)
   - Need more training or different hyperparameters

## The Ideal Trajectory

### What We Want to See

**Early Training (Epochs 1-10):**
```
Position correlation: 0.0 → 0.4 (learns position)
Pattern correlation: 0.0 → 0.1 (starts learning)
Context sensitivity: 1.0 → 0.8 (starts using context)
```

**Middle Training (Epochs 10-50):**
```
Position correlation: 0.4 → 0.3 (position helps but not dominant)
Pattern correlation: 0.1 → 0.3 (learns signal patterns)
Context sensitivity: 0.8 → 0.6 (predictions vary with context)
```

**Late Training (Epochs 50+):**
```
Position correlation: 0.3 → 0.2 (position is helper feature)
Pattern correlation: 0.3 → 0.6+ (learns rich signal patterns)
Context sensitivity: 0.6 → 0.4 (strong context dependence)
```

## Answer to Your Question

**Is the ideal to learn positions quickly then learn underlying representation?**

**YES, but with caveats:**

1. **Position learning first is OK** ✅
   - It's a natural stepping stone
   - Position is useful for temporal understanding
   - Sequential models naturally learn position

2. **BUT signal learning must follow** ⚠️
   - Position should HELP, not REPLACE signal learning
   - Model must learn signal patterns eventually
   - If stuck in position-only learning, that's a problem

3. **The ideal trajectory:**
   - **Early**: Learn position quickly (easy)
   - **Middle**: Learn from context (harder)
   - **Late**: Learn signal patterns (hardest)

4. **What we're seeing:**
   - ✅ Position learned (Phase 1)
   - ✅ Context learned (Phase 2)
   - ❌ Signal patterns NOT learned (Phase 3 - stuck!)

## What to Monitor

### Good Signs (Model Learning Correctly)
- Position correlation: Moderate (0.2-0.4) and stable
- Pattern correlation: **Increasing** over time (0.1 → 0.5+)
- Context sensitivity: Decreasing over time (0.9 → 0.4-0.6)
- Decoder variance: **Increasing** (predictions become diverse)

### Bad Signs (Model Stuck)
- Position correlation: High and increasing (0.6+)
- Pattern correlation: **Stuck low** (< 0.1) or decreasing
- Context sensitivity: High and stable (0.9+)
- Decoder variance: **Low and constant** (predictions collapse)

## Summary

**Yes, learning position quickly is fine**, but:
- It should be a stepping stone, not the end goal
- Signal pattern learning must follow
- Position should help, not dominate

**Current state**: Model learned position and context, but NOT signal patterns. The MLP decoder should help, but we need to monitor if pattern correlation increases during training.

