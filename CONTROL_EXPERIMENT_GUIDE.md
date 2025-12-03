# Control Experiment Guide: Detecting True Signal Leakage

## Purpose

Use `mask_ratio=1.0` with disabled positional encodings as a **control experiment** to detect if there's any information leakage from the actual signal content.

## The Problem We Solved

Your original diagnostic showed loss reduction with `mask_ratio=1.0`, but this was **NOT from signal leakage** - it was from positional encodings:

```
Position correlation: 0.403  ← Learning from temporal encoding (t/T)
Channel diversity:    0.000  ← Not using spatial encoding effectively
Consistency:          0.9995 ← Predictions identical regardless of content
```

The model was learning: `prediction = f(position)` not `prediction = f(signal)`

## Control Experiment Design

### Configuration (Already Applied)

`eeg_analysis/configs/pretrain.yaml`:

```yaml
mask_ratio: 1.0                    # All tokens masked (zeros)
disable_temporal_encoding: true    # Remove position information
disable_spatial_encoding: true     # Remove channel information
```

### What the Model Sees

With this configuration:

```python
# Input to model
windows_masked = [0, 0, 0, ..., 0]  # All zeros (fully masked)

# Token encoding
token_emb = LayerNorm(Linear([0,0,...,0]) + bias)
          = LayerNorm(bias)  # Constant for ALL tokens

# Temporal encoding (DISABLED)
temporal = [0, 0, 0, ..., 0]  # No position info

# Spatial encoding (DISABLED)  
spatial = [0, 0, 0, ..., 0]   # No channel info

# Final input to backbone
x = token_emb + temporal + spatial
  = LayerNorm(bias) + 0 + 0
  = constant  # SAME for all positions, all channels, all samples!
```

**Result**: The model has **ZERO varying information**. Every input looks identical.

### Expected Behavior

#### If NO Leakage (Expected)

```
Epoch 1: loss=1.234 (random predictions)
Epoch 2: loss=1.231 (minimal fluctuation)
Epoch 3: loss=1.228
...
Epoch 10: loss=1.225 (stays high, no learning)

✅ Conclusion: No information leakage
   Model cannot learn without information
```

Loss should stay at **random baseline** (~1.0 or higher depending on embedding dimension).

#### If Leakage EXISTS (Would indicate a bug)

```
Epoch 1: loss=1.234
Epoch 2: loss=0.987  ← Dropping!
Epoch 3: loss=0.756
...
Epoch 10: loss=0.234

❌ Conclusion: TRUE SIGNAL LEAKAGE DETECTED
   Model is learning from leaked unmasked signal
```

This would mean there's a bug in the masking or forward pass that allows unmasked signal to leak through.

## How to Run the Control Experiment

### Step 1: Run Training

```bash
cd /home/abin/eeg-mlflow
source .venv/bin/activate

# Train with control configuration (already set in pretrain.yaml)
python eeg_analysis/src/training/pretrain_mamba.py \
    --config eeg_analysis/configs/pretrain.yaml
```

### Step 2: Monitor Loss

Watch for loss behavior:

```bash
# In another terminal, monitor MLflow
cd /home/abin/eeg-mlflow
source .venv/bin/activate
mlflow ui --port 5000

# Open browser: http://localhost:5000
# Check experiment: eeg_pretraining_mamba2
# Look at train_loss and val_loss curves
```

### Step 3: Interpret Results

**Scenario A: Loss Stays High (Expected)**

```
train_loss: 1.2 → 1.19 → 1.18 → 1.17 (slow random fluctuation)
val_loss:   1.21 → 1.20 → 1.19 → 1.18

✅ NO LEAKAGE CONFIRMED
```

**What this means:**
- Your masking pipeline is correct
- No signal information leaks through
- Model cannot learn without information (as expected)
- The previous learning with mask_ratio=1.0 was purely from positional encodings

**Scenario B: Loss Drops Significantly (Would indicate bug)**

```
train_loss: 1.2 → 0.9 → 0.6 → 0.3 (rapid decrease)
val_loss:   1.21 → 0.95 → 0.65 → 0.35

❌ LEAKAGE DETECTED - INVESTIGATE!
```

**What to check if this happens:**
1. Are masked tokens truly all zeros? (Check collate function)
2. Is the `windows` tensor being passed to forward somehow?
3. Are normalization layers sharing statistics across masked/unmasked?
4. Is DDP synchronizing unmasked content across ranks?

## Comparison: Control vs. Standard Training

### Control Experiment (Current Config)

```yaml
mask_ratio: 1.0
disable_temporal_encoding: true
disable_spatial_encoding: true

Purpose: Detect signal leakage
Expected: Loss stays high (no learning)
```

### Standard Training (After confirming no leakage)

```yaml
mask_ratio: 0.75
disable_temporal_encoding: false
disable_spatial_encoding: false

Purpose: Learn useful EEG representations
Expected: Loss decreases (learning from context)
```

## After Control Experiment

Once you confirm **no leakage** (loss stays high), update config for real training:

```yaml
# Update pretrain.yaml
mask_ratio: 0.75                   # Provide unmasked context
disable_temporal_encoding: false   # Enable position information
disable_spatial_encoding: false    # Enable channel information
```

Then retrain for actual representation learning.

## Diagnostic Script Update

The diagnostic script will also respect these flags:

```bash
# After control training completes
python diagnose_100pct_masking.py \
    --checkpoint eeg_analysis/checkpoints/mamba2_eeg_pretrained.pt \
    --num-samples 100

# With disable flags enabled, you should see:
# Position correlation: ~0.0 (no position info available)
# Channel diversity: ~0.0 (no channel info available)
# Consistency: ~1.0 (all predictions identical - pure noise)
```

## Summary

| Configuration | Purpose | Expected Loss | Interpretation |
|--------------|---------|---------------|----------------|
| mask_ratio=1.0<br>disable_temporal=false<br>disable_spatial=false | Baseline (your original) | Decreases | Learning from positions (not leakage) |
| mask_ratio=1.0<br>disable_temporal=true<br>disable_spatial=true | Control (current) | Stays high | No leakage confirmed |
| mask_ratio=0.75<br>disable_temporal=false<br>disable_spatial=false | Standard training | Decreases | Learning from signal context |

## Key Insight

**Your original question**: "Is there leakage causing loss to drop with mask_ratio=1.0?"

**Answer**: No leakage - the loss dropped because the model learned from **positional encodings** (temporal + spatial), not from leaked signal.

**This control experiment confirms**: With positions disabled, the model truly has no information, and loss should stay high, confirming your masking pipeline is correct.

---

## Quick Reference

**Run control experiment** (current config is already set):
```bash
python eeg_analysis/src/training/pretrain_mamba.py --config eeg_analysis/configs/pretrain.yaml
```

**Expected result**: Loss stays at ~1.0-1.5 (random baseline)

**If loss drops**: Signal leakage bug detected - investigate masking pipeline

**After confirming no leakage**: Switch to `mask_ratio=0.75` with encodings enabled for real training

