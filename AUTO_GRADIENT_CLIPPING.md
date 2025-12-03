# Auto Gradient Clipping Implementation

## Overview

Adaptive gradient clipping that automatically adjusts the clipping threshold based on gradient norm statistics during training.

## Features

### 1. Percentile-Based Adaptive Clipping

Instead of a fixed clipping threshold, the system:
- Tracks gradient norms from recent batches (last 1000 by default)
- Computes the 95th percentile of gradient norms
- Uses this as the clipping threshold

**Benefits**:
- Adapts to training dynamics (gradients change over time)
- Prevents rare gradient explosions without over-clipping
- More flexible than fixed thresholds

### 2. Dual Threshold System

```python
effective_clip = min(grad_clip_norm, adaptive_threshold)
```

- `grad_clip_norm`: Hard upper limit (safety net)
- `adaptive_threshold`: Learned from gradient statistics
- Uses the more conservative (smaller) of the two

**Example**:
```
grad_clip_norm: 1.0 (config)
Epoch 1: adaptive=0.3 → clips at 0.3 (adaptive is lower)
Epoch 10: adaptive=0.8 → clips at 0.8 (adaptive is lower)
Epoch 50: adaptive=1.5 → clips at 1.0 (hard limit is lower)
```

### 3. Gradient Statistics Logging

Every 50 steps, logs to MLflow:
- `grad_norm`: Current gradient norm (before clipping)
- `grad_clip_threshold`: Effective threshold used
- `grad_clipped`: Whether clipping was applied (1=yes, 0=no)

## Configuration

### In `pretrain.yaml`

```yaml
# Gradient Clipping
grad_clip_norm: 1.0           # Hard maximum (safety limit)
auto_grad_clip: true          # Enable adaptive clipping
grad_clip_percentile: 95.0    # Clip at 95th percentile
```

### Parameters Explained

**`grad_clip_norm`** (default: 1.0)
- Hard upper limit on gradient norm
- Safety net to prevent extreme explosions
- Typical values: 0.5-2.0 depending on model

**`auto_grad_clip`** (default: true)
- Enable/disable adaptive clipping
- `true`: Use percentile-based adaptive threshold
- `false`: Use only fixed `grad_clip_norm`

**`grad_clip_percentile`** (default: 95.0)
- Percentile of gradient history to use as threshold
- 95.0 = clip the top 5% of gradients
- Higher = more lenient (99 = clip only top 1%)
- Lower = more aggressive (90 = clip top 10%)

**Recommended values**:
- Conservative: `percentile: 90` (clip top 10%)
- Balanced: `percentile: 95` (clip top 5%) ← default
- Lenient: `percentile: 99` (clip top 1%)

## How It Works

### Algorithm

```python
# Initialization
grad_norm_history = []  # Track last 1000 gradient norms

# During training:
for batch in dataloader:
    loss.backward()
    
    # 1. Compute gradient norm (no clipping)
    total_norm = compute_grad_norm(model.parameters())
    
    # 2. Compute adaptive threshold (after warmup)
    if len(grad_norm_history) >= 100:
        adaptive_threshold = percentile(grad_norm_history, 95.0)
        effective_clip = min(grad_clip_norm, adaptive_threshold)
    else:
        effective_clip = grad_clip_norm  # Use fixed during warmup
    
    # 3. Apply clipping
    clip_grad_norm_(model.parameters(), max_norm=effective_clip)
    
    # 4. Update history
    grad_norm_history.append(total_norm)
    if len(grad_norm_history) > 1000:
        grad_norm_history.pop(0)  # Keep recent 1000
    
    optimizer.step()
```

### Warmup Period

- First 100 batches: Uses fixed `grad_clip_norm`
- After 100 batches: Switches to adaptive threshold
- Ensures stable statistics before adapting

### History Management

- Keeps last **1000 gradient norms**
- Rolling window: old values removed when limit reached
- Adapts to recent training dynamics

## Monitoring

### MLflow Metrics

Check these metrics to understand gradient behavior:

**`grad_norm`**:
- Current gradient norm before clipping
- Trend shows if gradients are exploding/vanishing
- Should be stable, not wildly varying

**`grad_clip_threshold`**:
- Effective clipping threshold used
- Should adapt over training (usually decreases)
- Shows how aggressive clipping is

**`grad_clipped`**:
- Binary: 1 if clipped, 0 if not
- Percentage clipped ≈ (100 - percentile)%
- With 95th percentile, ~5% should be clipped

### Healthy Training Signs

```
Early training:
  grad_norm: 2.0-5.0 (large, unstable)
  grad_clip_threshold: 1.0 (hard limit)
  grad_clipped: 0.8 (80% clipped)

Mid training:
  grad_norm: 0.5-1.0 (moderate, stable)
  grad_clip_threshold: 0.8 (adaptive)
  grad_clipped: 0.05 (5% clipped)

Late training:
  grad_norm: 0.1-0.3 (small, very stable)
  grad_clip_threshold: 0.3 (adaptive)
  grad_clipped: 0.05 (5% clipped)
```

### Problem Signs

**Gradient explosion**:
```
grad_norm: 100+ (increasing rapidly)
grad_clipped: 1.0 (always clipping)
→ Solution: Lower grad_clip_norm to 0.5 or reduce learning rate
```

**Over-clipping**:
```
grad_norm: 0.3-0.5 (stable)
grad_clip_threshold: 0.1 (too low)
grad_clipped: 0.9 (90% clipped)
→ Solution: Increase grad_clip_percentile to 99
```

**Vanishing gradients**:
```
grad_norm: 0.001 (very small)
grad_clipped: 0.0 (never clipping)
→ Solution: Check loss scale, might need higher learning rate
```

## Comparison: Fixed vs. Auto

### Fixed Clipping (Traditional)

```yaml
grad_clip_norm: 1.0
auto_grad_clip: false
```

**Pros**:
- Simple, predictable
- Easy to tune once

**Cons**:
- May over-clip in later training (gradients shrink)
- May under-clip in early training (gradients large)
- Same threshold throughout training

### Auto Clipping (Adaptive)

```yaml
grad_clip_norm: 1.0
auto_grad_clip: true
grad_clip_percentile: 95.0
```

**Pros**:
- Adapts to training dynamics
- More aggressive early (large gradients)
- More lenient late (small gradients)
- Better final performance

**Cons**:
- Slightly more complex
- Need to monitor percentile choice
- First 100 steps use fixed clipping

## Tuning Guide

### Start with Defaults

```yaml
grad_clip_norm: 1.0
auto_grad_clip: true
grad_clip_percentile: 95.0
```

### If Training is Unstable (loss spikes)

```yaml
grad_clip_norm: 0.5  # Lower hard limit
grad_clip_percentile: 90.0  # More aggressive
```

### If Training is Too Conservative (slow progress)

```yaml
grad_clip_norm: 2.0  # Higher hard limit
grad_clip_percentile: 99.0  # More lenient
```

### If You Want Pure Adaptive (no hard limit)

```yaml
grad_clip_norm: 100.0  # Very high (effectively disabled)
auto_grad_clip: true
grad_clip_percentile: 95.0  # Only adaptive threshold
```

## Example MLflow Query

To analyze gradient clipping behavior:

```python
import mlflow
import pandas as pd

run = mlflow.get_run(run_id)
metrics = mlflow.get_metric_history(run.info.run_id, "grad_norm")

df = pd.DataFrame([
    {"step": m.step, "grad_norm": m.value}
    for m in metrics
])

# Plot gradient norm over training
df.plot(x="step", y="grad_norm")

# Compute clipping statistics
threshold_metrics = mlflow.get_metric_history(run.info.run_id, "grad_clip_threshold")
clip_applied = mlflow.get_metric_history(run.info.run_id, "grad_clipped")

print(f"Average gradient norm: {df['grad_norm'].mean():.4f}")
print(f"Max gradient norm: {df['grad_norm'].max():.4f}")
print(f"% of steps clipped: {sum(m.value for m in clip_applied) / len(clip_applied) * 100:.1f}%")
```

## Implementation Details

### Why Percentile, Not Mean/Median?

**Percentile (95th)**:
- Explicitly targets outliers (top 5%)
- Robust to extreme values
- Directly controls clipping frequency

**Mean**:
- Affected by outliers (what we want to clip)
- Would clip too aggressively

**Median (50th percentile)**:
- Would clip 50% of gradients (too aggressive)
- Not targeting outliers

### Why Rolling Window?

- Training dynamics change over epochs
- Recent gradients more relevant than old ones
- 1000 steps ≈ balances responsiveness vs. stability
- With batch_size=16, 1000 steps ≈ 2-3 epochs

### AMP Compatibility

The implementation works with both:
- **AMP enabled**: Unscales gradients before computing norms
- **AMP disabled**: Computes norms directly

Both paths have identical clipping logic.

## Summary

**Auto gradient clipping** provides adaptive, intelligent gradient norm management:

✅ **Adapts** to training dynamics
✅ **Prevents** gradient explosions
✅ **Avoids** over-clipping in late training
✅ **Monitors** via MLflow metrics
✅ **Simple** to configure

**Default config works well for most cases** - just monitor the metrics and adjust if needed!

