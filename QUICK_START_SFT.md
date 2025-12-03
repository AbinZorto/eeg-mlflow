# Quick Start: Mamba EEG Fine-Tuning

## Prerequisites

1. ✅ Pretrained Mamba model exists: `eeg_analysis/checkpoints/mamba2_eeg_pretrained.pt`
2. ✅ Raw EEG data available
3. ✅ Configs set up: `pretrain.yaml`, `finetune.yaml`, `processing_config.yaml`

---

## Three-Step Workflow

### 1️⃣ Create Primary Dataset (5-10 minutes)

```bash
uv run python3 eeg_analysis/run_pipeline.py \
  --config eeg_analysis/configs/processing_config.yaml \
  process-primary
```

**Output**: `eeg_analysis/data/processed/features/primary/8s_af7-af8-tp9-tp10_primary_dataset.parquet`

**Note**: Windows are automatically sorted by `Participant → parent_window_id → sub_window_id` to preserve temporal order (critical for sequence modeling).

---

### 2️⃣ Fine-Tune Model (~30-60 minutes for 50 epochs)

```bash
uv run python3 eeg_analysis/src/training/finetune_mamba.py \
  --config eeg_analysis/configs/finetune.yaml \
  --pretrain-config eeg_analysis/configs/pretrain.yaml \
  --data-path eeg_analysis/data/processed/features/primary/8s_af7-af8-tp9-tp10_primary_dataset.parquet \
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

### `configs/finetune.yaml`

```yaml
lr: 1.0e-4              # ↓ Lower if training unstable
batch_size: 8           # ↓ Lower if out of memory
epochs: 50              # ↑ Increase for better performance
freeze_backbone: true   # false = train all layers (slower, may improve)
```

### `configs/pretrain.yaml`

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
**Fix**: Run `process-primary` command first

### ❌ Out of memory
**Fix**: Reduce `batch_size` in `finetune.yaml`

### ❌ Poor performance
**Fix**: Try `freeze_backbone: false` or increase epochs

---

## Full Pipeline (From Scratch)

```bash
# 1. Pretrain Mamba model
uv run python3 eeg_analysis/src/training/pretrain_mamba.py \
  --config eeg_analysis/configs/pretrain.yaml

# 2. Create primary dataset
uv run python3 eeg_analysis/run_pipeline.py \
  --config eeg_analysis/configs/processing_config.yaml \
  process-primary

# 3. Fine-tune
uv run python3 eeg_analysis/src/training/finetune_mamba.py \
  --config eeg_analysis/configs/finetune.yaml \
  --pretrain-config eeg_analysis/configs/pretrain.yaml \
  --data-path eeg_analysis/data/processed/features/primary/8s_af7-af8-tp9-tp10_primary_dataset.parquet \
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

