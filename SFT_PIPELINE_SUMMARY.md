# Supervised Fine-Tuning (SFT) Pipeline for Mamba EEG Models

## Overview

This document describes the complete pipeline for:
1. Creating primary datasets (windowed EEG data without feature extraction)
2. Fine-tuning pretrained Mamba models for remission classification
3. Automatic model discovery based on pretraining configuration

---

## 1. Primary Dataset Creation

### What is the Primary Dataset?

The **primary dataset** is raw windowed EEG data concatenated across all participants, ready for direct model input. Unlike the feature-extracted dataset, it preserves the raw signal data for each channel.

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

The primary dataset creation automatically sorts and verifies window ordering.

### Creating the Primary Dataset

```bash
uv run python3 eeg_analysis/run_pipeline.py \
  --config eeg_analysis/configs/processing_config.yaml \
  process-primary
```

**What it does:**
1. Loads raw EEG data
2. Applies upsampling, filtering, downsampling, windowing, DC offset removal
3. **Skips feature extraction** (unlike the regular `process` command)
4. Concatenates remission and non-remission windowed data
5. Saves to `eeg_analysis/data/processed/features/primary/`
6. Logs dataset to MLflow with tags for discovery

**Output:**
- File: `{window_size}s_{channels}_primary_dataset.parquet`
- Example: `8s_af7-af8-tp9-tp10_primary_dataset.parquet`
- MLflow dataset name: `EEG_8s_af7-af8-tp9-tp10_{N}windows_primary`

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

The fine-tuning script **automatically finds the pretrained checkpoint** based on parameters in `pretrain.yaml`:

```yaml
# pretrain.yaml
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
  --data-path eeg_analysis/data/processed/features/primary/8s_af7-af8-tp9-tp10_primary_dataset.parquet \
  --output-dir eeg_analysis/finetuned_models
```

**Arguments:**
- `--config`: Fine-tuning hyperparameters (lr, epochs, batch_size, etc.)
- `--pretrain-config`: Pretraining config (used to find checkpoint and load model architecture)
- `--data-path`: Path to primary dataset parquet file
- `--output-dir`: Where to save fine-tuned models

### Configuration (`configs/finetune.yaml`)

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
   - Loads primary dataset
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

### Step 2: Create Primary Dataset

```bash
uv run python3 eeg_analysis/run_pipeline.py \
  --config eeg_analysis/configs/processing_config.yaml \
  process-primary
```

**Output**: `eeg_analysis/data/processed/features/primary/8s_af7-af8-tp9-tp10_primary_dataset.parquet`

### Step 3: Fine-Tune for Classification

```bash
uv run python3 eeg_analysis/src/training/finetune_mamba.py \
  --config eeg_analysis/configs/finetune.yaml \
  --pretrain-config eeg_analysis/configs/pretrain.yaml \
  --data-path eeg_analysis/data/processed/features/primary/8s_af7-af8-tp9-tp10_primary_dataset.parquet \
  --output-dir eeg_analysis/finetuned_models
```

**Output**: 
- Checkpoint: `eeg_analysis/finetuned_models/mamba2_eeg_d256_l2_m20_mae_finetuned_best.pt`
- MLflow model: `mamba2_eeg_d256_l2_m20_mae_finetuned`

---

## 5. Files Created

### New Files

1. **`src/processing/primary_dataset.py`**
   - Creates primary dataset from windowed data
   - Concatenates remission and non-remission groups
   - Logs to MLflow

2. **`src/data/eeg_sft_dataset.py`**
   - PyTorch Dataset for supervised fine-tuning
   - Loads primary dataset
   - Groups windows by participant
   - Handles train/val/test splits

3. **`src/models/mamba_sft_model.py`**
   - `MambaEEGClassifier`: Mamba model with classification head
   - Loads pretrained weights
   - Supports backbone freezing
   - Aggregates across channels and windows

4. **`src/training/finetune_mamba.py`**
   - Fine-tuning training script
   - Automatic checkpoint discovery
   - MLflow integration
   - Evaluation and model registration

5. **`configs/finetune.yaml`**
   - Fine-tuning hyperparameters
   - Data split configuration
   - MLflow settings

### Modified Files

1. **`run_pipeline.py`**
   - Added `process-primary` command
   - Creates primary dataset without feature extraction

2. **`README_pretraining.md`**
   - Added SFT workflow documentation
   - Usage examples for primary dataset and fine-tuning

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

### Primary Dataset Not Found

**Error**: `Data file not found`

**Solution**:
1. Run `process-primary` command first
2. Check path matches config: `{window_size}s_{channels}_primary_dataset.parquet`
3. Verify windowed data exists: `ls eeg_analysis/data/interim/windowed/`

### Out of Memory

**Solution**:
1. Reduce `batch_size` in `finetune.yaml`
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
  # Update pretrain.yaml with mask_ratio
  # Run fine-tuning
  uv run python3 eeg_analysis/src/training/finetune_mamba.py \
    --config eeg_analysis/configs/finetune.yaml \
    --pretrain-config eeg_analysis/configs/pretrain.yaml \
    --data-path eeg_analysis/data/processed/features/primary/8s_af7-af8-tp9-tp10_primary_dataset.parquet \
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
1. **Primary dataset creation** - Raw windowed data for direct model input
2. **Automatic checkpoint discovery** - No manual path management
3. **Flexible fine-tuning** - Freeze or unfreeze backbone layers
4. **Comprehensive evaluation** - Multiple metrics, MLflow tracking
5. **Model registration** - Organized model versioning

This enables systematic exploration of how different pretraining strategies (mask ratio, masking style, model size) affect downstream classification performance.

