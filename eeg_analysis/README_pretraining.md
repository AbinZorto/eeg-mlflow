### EEG Mamba-2 Self-Supervised Pretraining

This folder contains a self-supervised pretraining pipeline for EEG using a Mamba-2 backbone with fixed spatial and temporal encoders.

### Data layout
- Each Parquet file corresponds to one participant-run.
- Columns:
  - participant, run, parent_window_id, sub_window_id, window_start, window_end
  - One column per channel (uppercased, e.g., FZ, C3, O1). Each cell holds a vector of length `window_length` (default 2048).
- One file yields 66–67 sequences (one per channel). Sequence length is the number of windows (≈124).

### How sequences are extracted
- The dataset treats each (file, channel) pair as a sample.
- For a given channel, it stacks the per-window vectors into `[num_windows, window_length]`.
- Padding/truncation ensures each token has length `window_length`.
- Collate pads sequences in a batch to the max length for that batch.

### Model
- Input projection: Linear(window_length → d_model)
- Spatial encoding: fixed 3→d_model projection using unit-sphere electrode coordinates
- Temporal encoding: Linear(1→d_model) on normalized t/T
- Backbone: Mamba-2 with 6 layers and d_model=128 (Transformer fallback if `mamba-ssm` not installed)

### Self-supervised objective

Two masking strategies are available:

**MAE-style masking** (default, recommended for mask_ratio > 0.2):
- Select `mask_ratio * seq_len` positions randomly
- Replace **ALL** masked positions with zeros
- No information leakage - model must reconstruct from context only
- Harder task → better learned representations
- Set `masking_style: "mae"` in config

**BERT-style masking** (optional, for mask_ratio ≤ 0.2):
- Select `mask_ratio * seq_len` positions randomly
- Of masked positions: 80% zeros, 10% Gaussian noise, 10% unchanged
- Some information leakage - easier reconstruction task
- Set `masking_style: "bert"` in config

**Loss**: Mean squared error computed **only on masked positions**:
```python
loss = mse(pred[mask_bool], target[mask_bool])
```

**Why MAE-style is better for high mask ratios**:
- BERT-style with 10% "unchanged" tokens creates information leakage
- At high mask ratios (0.4-0.8), the model can interpolate from nearby unchanged tokens
- This makes the task too easy and reduces representation quality
- MAE-style forces true reconstruction from distant context

### ASA electrode mapping
- Provide an ASA electrode file path in the config (`asa_path`) to replace defaults.
- The parser normalizes coordinates to the unit sphere and uses them for spatial encodings.

### How to run

#### Single training run
1) Set paths in `eeg_analysis/configs/pretrain.yaml`:
   - `asa_path`: path to your ASA file (e.g., `eeg_analysis/secondarydata/raw/asa_electrodes.txt`)
   - `dataset_path`: folder with per-run Parquet files
   - `save_dir`: where checkpoints are written
   - `masking_style`: "mae" (default) or "bert"
   - `mask_ratio`: 0.2 to 0.8 (higher = harder task)

2) Launch pretraining (single GPU):
```bash
uv run python3 eeg_analysis/src/training/pretrain_mamba.py --config eeg_analysis/configs/pretrain.yaml
```

3) Launch pretraining (multi-GPU with DDP):
```bash
uv run torchrun --standalone --nproc_per_node=2 \
  eeg_analysis/src/training/pretrain_mamba.py \
  --config eeg_analysis/configs/pretrain.yaml \
  --distributed
```

4) Resume from checkpoint:
```bash
uv run python3 eeg_analysis/src/training/pretrain_mamba.py \
  --config eeg_analysis/configs/pretrain.yaml \
  --resume /path/to/mamba2_eeg_pretrained_last.pt
```

#### Hyperparameter sweep (mask_ratio)
Run multiple experiments with different mask ratios (0.2 to 0.8):

```bash
# Sequential sweep with multi-GPU per experiment
uv run python3 eeg_analysis/src/training/sweep_mask_ratio.py \
  --config eeg_analysis/configs/pretrain.yaml \
  --torchrun

# Sequential sweep with single GPU per experiment
uv run python3 eeg_analysis/src/training/sweep_mask_ratio.py \
  --config eeg_analysis/configs/pretrain.yaml
```

Each experiment will be logged to MLflow under the experiment `eeg_pretraining_mamba2_sweep_mask`.

### Outputs
- MLflow logs in `./mlruns` (configurable)
- Checkpoints in `save_dir`:
  - `mamba2_eeg_pretrained.pt` (best)
  - `mamba2_eeg_pretrained_last.pt` (most recent)

### Using the pretrained model

#### For inference or feature extraction:
```python
import torch
from src.models.mamba_eeg_model import MambaEEGModel

ckpt = torch.load("eeg_analysis/checkpoints/mamba2_eeg_pretrained.pt", map_location="cpu")
cfg = ckpt.get("config", {})
model = MambaEEGModel(
    d_model=cfg.get("d_model", 128),
    num_layers=cfg.get("num_layers", 6),
    window_length=cfg.get("window_length", 2048),
    asa_path=cfg.get("asa_path"),
)
model.load_state_dict(ckpt["model"])
model.eval()
```

#### For supervised fine-tuning (classification):

**Step 1: Create primary dataset (windowed data without feature extraction)**

```bash
uv run python3 eeg_analysis/run_pipeline.py \
  --config eeg_analysis/configs/processing_config.yaml \
  process-primary
```

This creates a primary dataset in `eeg_analysis/data/processed/features/primary/` with:
- Participant ID
- Remission label (0 or 1)
- Raw channel data (af7, af8, tp9, tp10)
- Window metadata

**Step 2: Fine-tune the pretrained model**

```bash
uv run python3 eeg_analysis/src/training/finetune_mamba.py \
  --config eeg_analysis/configs/finetune.yaml \
  --pretrain-config eeg_analysis/configs/pretrain.yaml \
  --data-path eeg_analysis/data/processed/features/primary/8s_af7-af8-tp9-tp10_primary_dataset.parquet \
  --output-dir eeg_analysis/finetuned_models
```

The fine-tuning script will:
- Automatically find the pretrained checkpoint based on config parameters (d_model, num_layers, mask_ratio, masking_style)
- Load the pretrained Mamba backbone
- Freeze all backbone layers (only train classification head)
- Train on remission classification task
- Log metrics and models to MLflow

**Configuration options** (`configs/finetune.yaml`):
- `freeze_backbone: true` - Freeze pretrained layers (recommended)
- `lr: 1e-4` - Learning rate for classification head
- `epochs: 50` - Number of training epochs
- `batch_size: 8` - Batch size
- `val_ratio: 0.2` - Validation split
- `test_ratio: 0.1` - Test split

**Model naming**: Fine-tuned models are named based on pretraining config:
- `mamba2_eeg_d256_l2_m20_mae_finetuned` - d_model=256, 2 layers, 20% mask, MAE-style
- `mamba2_eeg_d128_l6_m40_mae_finetuned` - d_model=128, 6 layers, 40% mask, MAE-style


