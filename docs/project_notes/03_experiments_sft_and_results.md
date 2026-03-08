# Experiments, SFT, And Results Notes

## One-Page Summary

### Scope
This document captures control-experiment protocol, SFT workflow, run configuration guidance, and observed training outcomes.

### Key Findings
- Control experiments are essential to separate true signal learning from positional or implementation leakage.
- The SFT pipeline is operationally straightforward once prerequisite artifacts are in place (pretrained checkpoint + closed_finetune dataset).
- Automated checkpoint discovery and MLflow integration reduce manual error in fine-tuning cycles.
- Reported results highlight both what improved and where failure modes remained, supporting iterative tuning rather than one-shot optimization.

### Recommended Actions (Now)
- Keep the control experiment as a required validation step before drawing conclusions from pretraining quality.
- Use the documented 3-step SFT workflow consistently across runs.
- Capture all run metadata in MLflow and keep split strategy/config explicit in experiment names.
- Reuse the troubleshooting checklist during failures to avoid ad hoc debugging drift.

### Recommended Actions (Next)
- Run a structured matrix of pretraining settings (for example mask ratio variants) and compare downstream SFT metrics.
- Standardize a minimal report template for each experiment cycle (config, diagnostics, final metrics, interpretation).
- Promote best-performing workflow settings into default config baselines.

### Risks And Assumptions
- Some performance expectations are run-specific and may not transfer without comparable data preprocessing and splits.
- Fine-tuning outcomes can be limited by upstream pretraining quality even with correct SFT configuration.
- Resource constraints (GPU memory/time) can bias which configurations were evaluated.

### Where To Dive Deeper
- Control methodology and interpretation: `CONTROL_EXPERIMENT_GUIDE.md`
- Practical SFT quickstart: `QUICK_START_SFT.md`
- End-to-end SFT implementation summary: `SFT_PIPELINE_SUMMARY.md`
- Empirical result interpretation: `TRAINING_RESULTS_ANALYSIS.md`

This file preserves full notes merged from the project archive.

## Included Sources

- CONTROL_EXPERIMENT_GUIDE.md
- QUICK_START_SFT.md
- SFT_PIPELINE_SUMMARY.md
- TRAINING_RESULTS_ANALYSIS.md

---

## Source: CONTROL_EXPERIMENT_GUIDE.md

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

# Train with control configuration (already set in eeg_analysis/configs/pretrain.yaml)
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
# Update eeg_analysis/configs/pretrain.yaml
mask_ratio: 0.75                   # Provide unmasked context
disable_temporal_encoding: false   # Enable position information
disable_spatial_encoding: false    # Enable channel information
```

Then retrain for actual representation learning.

## Diagnostic Script Update

The diagnostic script will also respect these flags:

```bash
# After control training completes
python scripts/diagnose_100pct_masking.py \
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

---

## Source: QUICK_START_SFT.md

# Quick Start: Mamba EEG Fine-Tuning

## Prerequisites

1. ✅ Pretrained Mamba model exists: `eeg_analysis/checkpoints/mamba2_eeg_pretrained.pt`
2. ✅ Raw EEG data available
3. ✅ Configs set up: `eeg_analysis/configs/pretrain.yaml`, `eeg_analysis/configs/finetune.yaml`, `eeg_analysis/configs/processing_config.yaml`

---

## Three-Step Workflow

### 1️⃣ Create Closed_finetune Dataset (5-10 minutes)

```bash
uv run python3 eeg_analysis/run_representation_pipeline.py \
  --config eeg_analysis/configs/processing_config.yaml \
  process-closed-finetune
```

**Output**: `eeg_analysis/data/processed/features/closed_finetune/8s_af7-af8-tp9-tp10_closed_finetune.parquet`

**Note**: Windows are automatically sorted by `Participant → parent_window_id → sub_window_id` to preserve temporal order (critical for sequence modeling).

---

### 2️⃣ Fine-Tune Model (~30-60 minutes for 50 epochs)

```bash
uv run python3 eeg_analysis/src/training/finetune_mamba.py \
  --config eeg_analysis/configs/finetune.yaml \
  --pretrain-config eeg_analysis/configs/pretrain.yaml \
  --data-path eeg_analysis/data/processed/features/closed_finetune/8s_af7-af8-tp9-tp10_closed_finetune.parquet \
  --output-dir eeg_analysis/finetuned_models
```

**Output**: `eeg_analysis/finetuned_models/mamba2_eeg_d256_l2_m20_mae_finetuned_best.pt`

---

### 3️⃣ View Results in MLflow

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Open browser: http://localhost:5000
Navigate to experiment: `eeg_finetuning_mamba2`

---

## Key Configuration Options

### `eeg_analysis/configs/finetune.yaml`

```yaml
lr: 1.0e-4              # ↓ Lower if training unstable
batch_size: 8           # ↓ Lower if out of memory
epochs: 50              # ↑ Increase for better performance
freeze_backbone: true   # false = train all layers (slower, may improve)
```

### `eeg_analysis/configs/pretrain.yaml`

Used to find the pretrained checkpoint:
```yaml
d_model: 256            # Model size
num_layers: 2           # Number of Mamba layers
mask_ratio: 0.2         # Pretraining mask ratio
masking_style: "mae"    # Masking strategy
```

**Checkpoint name**: `mamba2_eeg_d256_l2_m20_mae`

---

## Expected Results

### Good Performance Indicators:
- ✅ Val F1 > 0.7
- ✅ Test accuracy > 70%
- ✅ AUC > 0.75
- ✅ Training loss decreasing smoothly

### If Performance is Poor:
1. Try `freeze_backbone: false` (full fine-tuning)
2. Increase `epochs` to 100
3. Adjust `lr` (try 5e-5 or 2e-4)
4. Use different pretrained model (different mask_ratio)

---

## Model Naming

Models are automatically named based on pretraining config:

| Config | Model Name |
|--------|------------|
| d_model=256, layers=2, mask=20%, MAE | `mamba2_eeg_d256_l2_m20_mae` |
| d_model=128, layers=6, mask=40%, MAE | `mamba2_eeg_d128_l6_m40_mae` |
| d_model=256, layers=2, mask=20%, BERT | `mamba2_eeg_d256_l2_m20_bert` |

Fine-tuned models append `_finetuned`.

---

## Troubleshooting

### ❌ "Checkpoint not found"
**Fix**: Run pretraining first or check config parameters match

### ❌ "Data file not found"
**Fix**: Run `process-closed-finetune` command first

### ❌ Out of memory
**Fix**: Reduce `batch_size` in `eeg_analysis/configs/finetune.yaml`

### ❌ Poor performance
**Fix**: Try `freeze_backbone: false` or increase epochs

---

## Full Pipeline (From Scratch)

```bash
# 1. Pretrain Mamba model
uv run python3 eeg_analysis/src/training/pretrain_mamba.py \
  --config eeg_analysis/configs/pretrain.yaml

# 2. Create closed_finetune dataset
uv run python3 eeg_analysis/run_representation_pipeline.py \
  --config eeg_analysis/configs/processing_config.yaml \
  process-closed-finetune

# 3. Fine-tune
uv run python3 eeg_analysis/src/training/finetune_mamba.py \
  --config eeg_analysis/configs/finetune.yaml \
  --pretrain-config eeg_analysis/configs/pretrain.yaml \
  --data-path eeg_analysis/data/processed/features/closed_finetune/8s_af7-af8-tp9-tp10_closed_finetune.parquet \
  --output-dir eeg_analysis/finetuned_models

# 4. View results
mlflow ui --backend-store-uri file:./mlruns
```

---

## Next Steps

- Compare different pretrained models (different mask ratios)
- Experiment with frozen vs unfrozen backbone
- Try different learning rates and epochs
- Evaluate on held-out test set
- Use fine-tuned model for inference

For detailed documentation, see `SFT_PIPELINE_SUMMARY.md`.

---

## Source: SFT_PIPELINE_SUMMARY.md

# Supervised Fine-Tuning (SFT) Pipeline for Mamba EEG Models

## Overview

This document describes the complete pipeline for:
1. Creating closed_finetune datasets (windowed EEG data without feature extraction)
2. Fine-tuning pretrained Mamba models for remission classification
3. Automatic model discovery based on pretraining configuration

---

## 1. Closed_finetune Dataset Creation

### What is the Closed_finetune Dataset?

The **closed_finetune dataset** is raw windowed EEG data concatenated across all participants, ready for direct model input. Unlike the feature-extracted dataset, it preserves the raw signal data for each channel.

**Structure:**
- `Participant`: Participant ID
- `Remission`: Binary label (0=non-remission, 1=remission)
- `parent_window_id`, `sub_window_id`: **Window identifiers (CRITICAL for temporal order)**
- `window_start`, `window_end`: Window boundaries
- Channel columns (`AF7`, `AF8`, `TP9`, `TP10`): Raw signal vectors

**⚠️ IMPORTANT: Temporal Ordering**

Windows are sorted by `Participant → parent_window_id → sub_window_id` to preserve temporal relationships. This is critical because:
1. The Mamba backbone is a sequence model that depends on temporal order
2. Windows represent consecutive time segments from EEG recordings
3. Scrambling window order would destroy temporal patterns the model needs to learn

The closed_finetune dataset creation automatically sorts and verifies window ordering.

### Creating the Closed_finetune Dataset

```bash
uv run python3 eeg_analysis/run_representation_pipeline.py \
  --config eeg_analysis/configs/processing_config.yaml \
  process-closed-finetune
```

**What it does:**
1. Loads raw EEG data
2. Applies upsampling, filtering, downsampling, windowing, DC offset removal
3. **Skips feature extraction** (unlike the regular `process` command)
4. Concatenates remission and non-remission windowed data
5. Saves to `eeg_analysis/data/processed/features/closed_finetune/`
6. Logs dataset to MLflow with tags for discovery

**Output:**
- File: `{window_size}s_{channels}_closed_finetune.parquet`
- Example: `8s_af7-af8-tp9-tp10_closed_finetune.parquet`
- MLflow dataset name: `EEG_8s_af7-af8-tp9-tp10_{N}windows_closed_finetune`

---

## 2. Supervised Fine-Tuning (SFT)

### Architecture

**Mamba EEG Classifier:**
- **Backbone**: Pretrained Mamba-2 model (loaded from checkpoint)
- **Freezing**: All backbone layers frozen by default (only train classification head)
- **Classification Head**:
  ```
  LayerNorm(d_model)
  → Dropout
  → Linear(d_model → d_model/2)
  → GELU
  → Dropout
  → Linear(d_model/2 → num_classes)
  ```

### Automatic Checkpoint Discovery

The fine-tuning script **automatically finds the pretrained checkpoint** based on parameters in `eeg_analysis/configs/pretrain.yaml`:

```yaml
# eeg_analysis/configs/pretrain.yaml
d_model: 256
num_layers: 2
mask_ratio: 0.2
masking_style: "mae"
```

**Expected checkpoint**: `mamba2_eeg_d256_l2_m20_mae`
**Checkpoint file**: `eeg_analysis/checkpoints/mamba2_eeg_pretrained.pt`

### Running Fine-Tuning

```bash
uv run python3 eeg_analysis/src/training/finetune_mamba.py \
  --config eeg_analysis/configs/finetune.yaml \
  --pretrain-config eeg_analysis/configs/pretrain.yaml \
  --data-path eeg_analysis/data/processed/features/closed_finetune/8s_af7-af8-tp9-tp10_closed_finetune.parquet \
  --output-dir eeg_analysis/finetuned_models
```

**Arguments:**
- `--config`: Fine-tuning hyperparameters (lr, epochs, batch_size, etc.)
- `--pretrain-config`: Pretraining config (used to find checkpoint and load model architecture)
- `--data-path`: Path to closed_finetune dataset parquet file
- `--output-dir`: Where to save fine-tuned models

### Configuration (`eeg_analysis/configs/finetune.yaml`)

```yaml
# Training hyperparameters
lr: 1.0e-4              # Learning rate
batch_size: 8
epochs: 50
dropout: 0.1
weight_decay: 0.01

# Model
freeze_backbone: true   # Freeze pretrained layers
num_classes: 2          # Binary classification

# Data splits
val_ratio: 0.2          # 20% validation
test_ratio: 0.1         # 10% test
seed: 42

# MLflow
mlflow_tracking_uri: "mlruns"
mlflow_experiment: "eeg_finetuning_mamba2"
```

### Training Process

1. **Data Loading**:
   - Loads closed_finetune dataset
   - **Verifies temporal ordering** (windows sorted by participant → parent_window_id → sub_window_id)
   - Splits by participant into train/val/test (stratified)
   - Each participant's windows remain in temporal order within their split

2. **Model Initialization**:
   - Creates `MambaEEGClassifier`
   - Loads pretrained weights from checkpoint
   - Freezes backbone layers (if `freeze_backbone: true`)
   - Initializes classification head with Xavier initialization

3. **Training Loop**:
   - AdamW optimizer with cosine annealing schedule
   - Cross-entropy loss
   - Gradient clipping (max_norm=1.0)
   - Logs metrics to MLflow every epoch

4. **Evaluation**:
   - Validation after each epoch
   - Saves best model based on F1 score
   - Final test evaluation at end
   - Metrics: accuracy, precision, recall, F1, AUC

5. **Model Registration**:
   - Saves best checkpoint to `{output_dir}/{model_name}_finetuned_best.pt`
   - Logs model to MLflow
   - Registers as `{model_name}_finetuned` in MLflow Model Registry

---

## 3. Model Naming Convention

Models are named based on pretraining configuration:

**Format**: `mamba2_eeg_d{d_model}_l{num_layers}_m{mask_ratio_percent}_{masking_style}`

**Examples:**
- `mamba2_eeg_d256_l2_m20_mae` → d_model=256, 2 layers, 20% mask, MAE-style
- `mamba2_eeg_d128_l6_m40_mae` → d_model=128, 6 layers, 40% mask, MAE-style
- `mamba2_eeg_d256_l2_m20_bert` → d_model=256, 2 layers, 20% mask, BERT-style

**Fine-tuned models** append `_finetuned`:
- `mamba2_eeg_d256_l2_m20_mae_finetuned`

---

## 4. Complete Workflow Example

### Step 1: Pretrain Mamba Model

```bash
# Single GPU
uv run python3 eeg_analysis/src/training/pretrain_mamba.py \
  --config eeg_analysis/configs/pretrain.yaml

# Multi-GPU
uv run torchrun --standalone --nproc_per_node=2 \
  eeg_analysis/src/training/pretrain_mamba.py \
  --config eeg_analysis/configs/pretrain.yaml \
  --distributed
```

**Output**: `eeg_analysis/checkpoints/mamba2_eeg_pretrained.pt`

### Step 2: Create Closed_finetune Dataset

```bash
uv run python3 eeg_analysis/run_representation_pipeline.py \
  --config eeg_analysis/configs/processing_config.yaml \
  process-closed-finetune
```

**Output**: `eeg_analysis/data/processed/features/closed_finetune/8s_af7-af8-tp9-tp10_closed_finetune.parquet`

### Step 3: Fine-Tune for Classification

```bash
uv run python3 eeg_analysis/src/training/finetune_mamba.py \
  --config eeg_analysis/configs/finetune.yaml \
  --pretrain-config eeg_analysis/configs/pretrain.yaml \
  --data-path eeg_analysis/data/processed/features/closed_finetune/8s_af7-af8-tp9-tp10_closed_finetune.parquet \
  --output-dir eeg_analysis/finetuned_models
```

**Output**: 
- Checkpoint: `eeg_analysis/finetuned_models/mamba2_eeg_d256_l2_m20_mae_finetuned_best.pt`
- MLflow model: `mamba2_eeg_d256_l2_m20_mae_finetuned`

---

## 5. Files Created

### New Files

1. **`eeg_analysis/src/processing/closed_finetune_dataset.py`**
   - Creates closed_finetune dataset from windowed data
   - Concatenates remission and non-remission groups
   - Logs to MLflow

2. **`eeg_analysis/src/data/eeg_sft_dataset.py`**
   - PyTorch Dataset for supervised fine-tuning
   - Loads closed_finetune dataset
   - Groups windows by participant
   - Handles train/val/test splits

3. **`eeg_analysis/src/models/mamba_sft_model.py`**
   - `MambaEEGClassifier`: Mamba model with classification head
   - Loads pretrained weights
   - Supports backbone freezing
   - Aggregates across channels and windows

4. **`eeg_analysis/src/training/finetune_mamba.py`**
   - Fine-tuning training script
   - Automatic checkpoint discovery
   - MLflow integration
   - Evaluation and model registration

5. **`eeg_analysis/configs/finetune.yaml`**
   - Fine-tuning hyperparameters
   - Data split configuration
   - MLflow settings

### Modified Files

1. **`eeg_analysis/run_representation_pipeline.py`**
   - Added `process-closed-finetune` command
   - Creates closed_finetune dataset without feature extraction

2. **`README.md`**
   - Consolidated project documentation now includes the SFT workflow
   - Usage examples for closed_finetune dataset creation and fine-tuning

---

## 6. Key Features

### ✅ Automatic Model Discovery
- No need to manually specify checkpoint paths
- Finds pretrained model based on config parameters
- Validates model exists before training

### ✅ Stratified Splits
- Participant-level splits (no data leakage)
- Maintains class balance across train/val/test
- Reproducible with seed

### ✅ Flexible Backbone Freezing
- `freeze_backbone: true` → Only train classification head (fast, prevents overfitting)
- `freeze_backbone: false` → Full fine-tuning (slower, may improve performance)

### ✅ Comprehensive Metrics
- Accuracy, Precision, Recall, F1, AUC
- Logged to MLflow for comparison
- Best model saved based on F1 score

### ✅ MLflow Integration
- Logs all hyperparameters
- Tracks metrics per epoch
- Registers fine-tuned models
- Links to pretrained model

---

## 7. Troubleshooting

### Checkpoint Not Found

**Error**: `Pretrained checkpoint not found`

**Solution**:
1. Check that pretraining completed: `ls eeg_analysis/checkpoints/mamba2_eeg_pretrained.pt`
2. Verify config parameters match: `d_model`, `num_layers`, `mask_ratio`, `masking_style`
3. Run pretraining if checkpoint missing

### Closed_finetune Dataset Not Found

**Error**: `Data file not found`

**Solution**:
1. Run `process-closed-finetune` command first
2. Check path matches config: `{window_size}s_{channels}_closed_finetune.parquet`
3. Verify windowed data exists: `ls eeg_analysis/data/interim/windowed/`

### Out of Memory

**Solution**:
1. Reduce `batch_size` in `eeg_analysis/configs/finetune.yaml`
2. Use gradient accumulation (modify training script)
3. Use smaller model (reduce `d_model` or `num_layers` in pretraining)

### Poor Performance

**Solutions**:
1. Try unfreezing backbone: `freeze_backbone: false`
2. Increase training epochs
3. Adjust learning rate
4. Try different pretrained models (different mask_ratio or masking_style)
5. Check data quality and class balance

---

## 8. Next Steps

### Experiment with Different Pretrained Models

```bash
# Sweep different mask ratios during pretraining
uv run python3 eeg_analysis/src/training/sweep_mask_ratio.py \
  --config eeg_analysis/configs/pretrain.yaml \
  --torchrun

# Fine-tune each pretrained model
for mask_ratio in 20 30 40 50 60 70 80; do
  # Update eeg_analysis/configs/pretrain.yaml with mask_ratio
  # Run fine-tuning
  uv run python3 eeg_analysis/src/training/finetune_mamba.py \
    --config eeg_analysis/configs/finetune.yaml \
    --pretrain-config eeg_analysis/configs/pretrain.yaml \
    --data-path eeg_analysis/data/processed/features/closed_finetune/8s_af7-af8-tp9-tp10_closed_finetune.parquet \
    --output-dir eeg_analysis/finetuned_models
done
```

### Compare Models in MLflow

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Navigate to `eeg_finetuning_mamba2` experiment to compare:
- Different pretrained models
- Frozen vs unfrozen backbone
- Different hyperparameters

---

## Summary

The SFT pipeline provides:
1. **Closed_finetune dataset creation** - Raw windowed data for direct model input
2. **Automatic checkpoint discovery** - No manual path management
3. **Flexible fine-tuning** - Freeze or unfreeze backbone layers
4. **Comprehensive evaluation** - Multiple metrics, MLflow tracking
5. **Model registration** - Organized model versioning

This enables systematic exploration of how different pretraining strategies (mask ratio, masking style, model size) affect downstream classification performance.

---

## Source: TRAINING_RESULTS_ANALYSIS.md

# Training Results Analysis: mask_ratio=0.75

## Key Findings

### ✅ What's Working

1. **Context Learning** ✅
   - Context sensitivity: **0.58** (LOW = GOOD)
   - Predictions vary with different masking contexts
   - Model learns from unmasked tokens, not just position

2. **Position Fix Working** ✅
   - Position correlation: **0.31** (down from 0.69 with 100% masking)
   - Positional encoding zeroing is effective
   - Position is a helper, not the only signal

### ❌ What's Not Working

1. **Signal Pattern Learning** ❌
   - Pattern correlation: **-0.045** (near zero, even negative)
   - Model does NOT learn actual EEG signal patterns
   - Predictions don't match ground truth waveforms

2. **Prediction Diversity** ❌
   - Baseline similarity: **0.965** (very high)
   - All predictions are nearly identical
   - Model predicts similar values regardless of input

## Interpretation

The model is learning **something** (context sensitivity is low), but it's **not learning signal content**.

**What the model might be learning**:
- Context-dependent patterns (good!)
- But predicting similar/constant values (bad!)
- Not learning actual signal waveforms (bad!)

**Possible causes**:
1. **Loss function issue**: Model optimizes for something other than signal reconstruction
2. **Normalization issue**: Per-window normalization might be removing signal structure
3. **Decoder issue**: Decoder might be collapsing to constant predictions
4. **Training dynamics**: Model might need more training or different hyperparameters

## Recommendations

1. **Check loss values**: Is reconstruction loss actually decreasing?
2. **Inspect decoder outputs**: Are predictions actually varying?
3. **Test without normalization**: Try training without per-window normalization
4. **Increase training**: Model might need more epochs
5. **Check gradient flow**: Ensure gradients are flowing to decoder

## Next Steps

1. Verify the model is actually trying to reconstruct signals
2. Check if decoder is learning or collapsing
3. Test with different mask ratios (0.5, 0.25)
4. Inspect actual predictions vs ground truth visually

The good news: Context learning works! The bad news: Signal pattern learning doesn't.

---
