# Secondary EEG Preprocessing

This document describes how to build the secondary EEG dataset used for Mamba2 pretraining.

## Command

Run the secondary preprocessing CLI with the provided config:

```bash
python3 eeg_analysis/build_secondary_dataset.py \
  --config eeg_analysis/configs/secondary_processing.yaml \
  build-secondary
```

If you use `uv`:

```bash
uv run python3 eeg_analysis/build_secondary_dataset.py \
  --config eeg_analysis/configs/secondary_processing.yaml \
  build-secondary
```

## Config

Edit the config at `eeg_analysis/configs/secondary_processing.yaml`:

- `paths.source_root`: root folder containing the raw secondary EEG data
- `paths.output_dir`: output folder for per-run artifacts
- `processing.target_sampling_rate`: resample rate (Hz)
- `processing.convert_to_microvolts`: convert to uV
- `processing.keep_all_channels`: keep all channels during export
- `processing.register_with_mlflow`: log outputs and metrics
- `mlflow.tracking_uri` / `mlflow.experiment_name`: MLflow tracking settings

## Output

On success the script prints a summary and writes per-run outputs under the
configured `output_dir`.
