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
- Mask 20% of tokens per sequence
  - 80% replaced with zeros
  - 10% replaced with Gaussian noise
  - 10% kept
- Predict the projected token embedding for masked tokens
- Loss: mean squared error on masked positions only

### ASA electrode mapping
- Provide an ASA electrode file path in the config (`asa_path`) to replace defaults.
- The parser normalizes coordinates to the unit sphere and uses them for spatial encodings.

### How to run
1) Set paths in `eeg_analysis/configs/pretrain.yaml`:
   - `asa_path`: path to your ASA file (e.g., `eeg_analysis/secondarydata/raw/asa_electrodes.txt`)
   - `dataset_path`: folder with per-run Parquet files
   - `save_dir`: where checkpoints are written
2) Launch pretraining:
```bash
uv run python3 eeg_analysis/src/training/pretrain_mamba.py --config eeg_analysis/configs/pretrain.yaml
```
3) Resume:
```bash
uv run python3 eeg_analysis/src/training/pretrain_mamba.py --config eeg_analysis/configs/pretrain.yaml --resume /path/to/mamba2_eeg_pretrained_last.pt
```

### Outputs
- MLflow logs in `./mlruns` (configurable)
- Checkpoints in `save_dir`:
  - `mamba2_eeg_pretrained.pt` (best)
  - `mamba2_eeg_pretrained_last.pt` (most recent)

### Using the pretrained model
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


