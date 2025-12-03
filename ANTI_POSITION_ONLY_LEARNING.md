# Anti-Position-Only Learning Strategy

## Problem

Even with explicit positional encodings disabled, **Mamba's sequential processing provides implicit positional information**:
- Mamba processes tokens sequentially (token 0, then 1, then 2...)
- Hidden state accumulates position information
- Model learns "position → prediction" instead of "context → prediction"

**Result**: Model learns positional patterns, not signal content.

## Solution: Multi-Strategy Approach

We implement **two complementary strategies** to prevent position-only learning while keeping Mamba and position:

### Strategy 1: Position Regularization

**What it does**: Penalizes predictions that correlate strongly with position.

**How it works**:
1. For each masked token, compute its normalized position (0 to 1)
2. Compute correlation between position and prediction summary
3. Add penalty: `loss += position_corr² * weight`

**Effect**: Model is discouraged from learning position-only patterns.

**Configuration**:
```yaml
prevent_position_only_learning: true
position_regularization_weight: 0.1  # 0.0-1.0, higher = more penalty
```

### Strategy 2: Sequence Shuffling

**What it does**: Randomly shuffles window order during training.

**How it works**:
1. For each sequence in batch, with probability `shuffle_prob`:
   - Randomly permute window order
   - Breaks position=time mapping
2. Model must learn from context (unmasked tokens), not position

**Effect**: Forces model to learn from signal context, not temporal position.

**Configuration**:
```yaml
shuffle_sequences_prob: 0.3  # 0.0-1.0
# 0.0 = never shuffle (preserve temporal order)
# 0.3 = shuffle 30% of sequences (balanced)
# 1.0 = always shuffle (breaks temporal structure)
```

## Why This Works

### Position Regularization

**Before**:
```
Position 0 → Predict A
Position 1 → Predict B
Position 2 → Predict C
Loss: Low (model learns position mapping) ✅
Position correlation: High (0.6+) ❌
```

**After**:
```
Position 0 → Predict A (but penalized if too correlated)
Position 1 → Predict B (but penalized if too correlated)
Loss: Slightly higher (penalty added)
Position correlation: Lower (<0.3) ✅
```

### Sequence Shuffling

**Before** (ordered):
```
Window 0 (time 0s) → Position 0
Window 1 (time 8s) → Position 1
Window 2 (time 16s) → Position 2
Model learns: position = time ✅ (but learns position, not signal)
```

**After** (shuffled 30%):
```
Window 0 (time 16s) → Position 0  ← Shuffled!
Window 1 (time 0s) → Position 1   ← Shuffled!
Window 2 (time 8s) → Position 2   ← Shuffled!
Model must learn: context → prediction (can't use position=time)
```

## Configuration

### Recommended Settings

**Balanced** (recommended):
```yaml
prevent_position_only_learning: true
position_regularization_weight: 0.1
shuffle_sequences_prob: 0.3
```

**Aggressive** (if still learning position):
```yaml
prevent_position_only_learning: true
position_regularization_weight: 0.3
shuffle_sequences_prob: 0.5
```

**Conservative** (minimal intervention):
```yaml
prevent_position_only_learning: true
position_regularization_weight: 0.05
shuffle_sequences_prob: 0.1
```

## Monitoring

### MLflow Metrics

The training script logs:
- **`position_correlation`**: Correlation between position and predictions
  - **Target**: < 0.3 (low position dependence)
  - **Warning**: > 0.5 (high position dependence)
- **`position_penalty`**: Current penalty value
  - Should decrease as model learns context instead of position

### Diagnostic Script

Run after training:
```bash
python diagnose_100pct_masking.py \
    --checkpoint eeg_analysis/checkpoints/mamba2_eeg_pretrained.pt \
    --num-samples 100
```

**Look for**:
- Position correlation < 0.3 ✅
- Pattern correlation > 0.3 ✅
- Baseline similarity maintained ✅

## Expected Behavior

### With Anti-Position-Only Learning

**Training**:
- Loss may be slightly higher (penalty added)
- Position correlation decreases over time
- Model learns from context, not just position

**Diagnostics**:
- Position correlation: < 0.3 ✅
- Pattern correlation: > 0.3 ✅
- Context sensitivity: < 0.9 ✅

### Without Anti-Position-Only Learning

**Training**:
- Loss decreases quickly
- Position correlation stays high (> 0.6)
- Model learns position mapping

**Diagnostics**:
- Position correlation: > 0.6 ❌
- Pattern correlation: < 0.1 ❌
- Context sensitivity: > 0.9 ❌

## Trade-offs

### Position Regularization

**Pros**:
- ✅ Keeps temporal order intact
- ✅ Allows position to be used (just not exclusively)
- ✅ Simple to implement

**Cons**:
- ⚠️ Adds hyperparameter (weight)
- ⚠️ May slow convergence slightly

### Sequence Shuffling

**Pros**:
- ✅ Forces context learning
- ✅ Breaks position=time mapping directly
- ✅ No hyperparameter tuning needed (just probability)

**Cons**:
- ⚠️ Breaks temporal structure (may hurt some tasks)
- ⚠️ May confuse model if overused

## Best Practices

1. **Start with balanced settings** (0.1 weight, 0.3 shuffle prob)
2. **Monitor position correlation** during training
3. **Adjust if needed**:
   - High position corr (>0.5) → Increase weight/shuffle prob
   - Low pattern corr (<0.1) → Decrease weight/shuffle prob
4. **Test on downstream tasks** to validate representations

## Summary

**Goal**: Learn signal content, not just position.

**Strategy**: 
- Position regularization (penalize position-only learning)
- Sequence shuffling (break position=time mapping)

**Result**: Model learns from context while still using position when appropriate.

**Keep**: Mamba architecture, positional encodings, temporal order (mostly)

**Prevent**: Position-only learning, ignoring signal content

