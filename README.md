# EEG MLflow Pipeline

Unified documentation for this repository.

This project provides two connected workflows:

1. Traditional EEG feature pipeline (processing + feature extraction + classical/deep-learning classifiers).
2. Representation-learning workflow (dataset build/conversion + self-supervised pretraining + supervised fine-tuning).

All commands below assume you run from the repository root: `/home/abin/eeg-mlflow`.

## Repository Layout

- `eeg_analysis/run_pipeline.py`: main CLI for processing/training/evaluation.
- `eeg_analysis/configs/`: processing, training, pretraining, and fine-tuning configs.
- `eeg_analysis/src/processing/`: EEG preprocessing + feature extraction.
- `eeg_analysis/src/models/`: trainers and model code.
- `eeg_analysis/src/training/`: trainer scripts used by representation model profiles (Mamba by default).
- `eeg_analysis/run_representation_pipeline.py`: CLI for representation dataset prep + pretraining/fine-tuning orchestration.
- `scripts/build_open_pretrain_dataset.py`: wrapper for open_pretrain dataset build.
- `scripts/convert_open_pretrain_window_size.py`: conversion utility used by the representation CLI.
- `mlruns/`: MLflow tracking data.
- `models/`: local outputs (predictions/metadata, plus legacy artifacts).

## Setup

1. Create and activate an environment.
2. Install dependencies.

```bash
uv run -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Traditional EEG Feature Pipeline

### 1) Process raw EEG into feature dataset

```bash
uv run eeg_analysis/run_pipeline.py \
  --config eeg_analysis/configs/processing_config.yaml \
  process
```

What it does:

- Loads raw EEG from `eeg_analysis/configs/processing_config.yaml`.
- Runs upsampling, filtering, downsampling, window slicing, DC offset removal, and feature extraction.
- Writes feature parquet under `eeg_analysis/data/processed/features/`.
- Logs dataset metadata to MLflow (tags include `mlflow.dataset.logged=true`, `mlflow.dataset.context=training`).

Current default processing config (`eeg_analysis/configs/processing_config.yaml`):

- Channels: `af7, af8, tp9, tp10`
- Window size: `10s`
- Window ordering: `sequential` (preserve order enabled)

### 2) (Optional) Build representation dataset for sequence models

```bash
uv run eeg_analysis/run_representation_pipeline.py \
  --config eeg_analysis/configs/processing_config.yaml \
  process-closed-finetune
```

This creates a parquet representation dataset from windowed channel data (without feature vectors), used by fine-tuning code.

### 3) List available MLflow datasets

```bash
uv run eeg_analysis/run_pipeline.py \
  --config eeg_analysis/configs/window_model_config.yaml \
  list-datasets
```

### 4) Train models

Window-level example:

```bash
uv run eeg_analysis/run_pipeline.py \
  --config eeg_analysis/configs/window_model_config.yaml \
  train \
  --level window \
  --model-type random_forest
```

Patient-level example:

```bash
uv run eeg_analysis/run_pipeline.py \
  --config eeg_analysis/configs/patient_model_config.yaml \
  train \
  --level patient \
  --model-type random_forest
```

Notes:

- Training first tries MLflow dataset discovery; if not found, it falls back to `data.feature_path` from the training config.
- You can force a specific dataset with `--use-dataset-from-run <mlflow_run_id>`.
- Models are saved only in MLflow artifacts (use run IDs / model URIs for loading).

### 5) Evaluate a saved model

```bash
uv run eeg_analysis/run_pipeline.py \
  --config eeg_analysis/configs/window_model_config.yaml \
  evaluate \
  --run-id <mlflow_run_id> \
  --data-path eeg_analysis/data/processed/features/10s_af7_af8_tp9_tp10_window_features_seq.parquet
```

Alternative:

```bash
uv run eeg_analysis/run_pipeline.py \
  --config eeg_analysis/configs/window_model_config.yaml \
  evaluate \
  --model-uri runs:/<mlflow_run_id>/model \
  --level window
```

Passing `--data-path` explicitly is the safest option.

## Feature Selection and Feature Filtering

### Feature selection flags (training CLI)

- `--enable-feature-selection`
- `--n-features-select <int>`
- `--fs-method <method>`

Supported methods:

- `model_based`
- `select_k_best_f_classif`
- `select_k_best_mutual_info`
- `select_from_model_l1`
- `rfe`

### Feature category filtering (training CLI)

List available categories:

```bash
uv run eeg_analysis/run_pipeline.py \
  --config eeg_analysis/configs/window_model_config.yaml \
  train \
  --level window \
  --model-type random_forest \
  --feature-categories list
```

Train with selected categories:

```bash
uv run eeg_analysis/run_pipeline.py \
  --config eeg_analysis/configs/window_model_config.yaml \
  train \
  --level window \
  --model-type random_forest \
  --feature-categories "spectral_features,psd_statistics"
```

Implemented categories are defined in `eeg_analysis/src/utils/feature_filter.py`, including:

- `spectral_features`, `psd_statistics`
- `temporal_features`, `entropy_features`, `complexity_features`
- `connectivity_features`, `asymmetry_features`
- `cross_hemispheric_features`, `diagonal_features`, `coherence_features`

## Representation Workflow

### 1) Build pretraining EEG dataset (per-run parquet windows)

```bash
uv run eeg_analysis/run_representation_pipeline.py \
  --config eeg_analysis/configs/open_pretrain_processing.yaml \
  build-open-pretrain
```

Key config: `eeg_analysis/configs/open_pretrain_processing.yaml`

- `paths.source_root`: raw source EEG root for representation pretraining
- `paths.output_dir`: output base dir
- `processing.target_sampling_rate`
- `processing.convert_to_microvolts`
- `processing.register_with_mlflow`

### 2) (Optional) Convert pretraining window size

```bash
uv run eeg_analysis/run_representation_pipeline.py \
  --config eeg_analysis/configs/open_pretrain_processing.yaml \
  convert-open-pretrain \
  --input-root eeg_analysis/secondarydata/raw/sr256_ws10s_open_pretrain \
  --output-base eeg_analysis/secondarydata/raw \
  --factor 2
```

### 3) Self-supervised pretraining

```bash
uv run eeg_analysis/run_representation_pipeline.py \
  --config eeg_analysis/configs/pretrain.yaml \
  pretrain
```

Run every configured pretraining model profile from `eeg_analysis/configs/pretrain.yaml`:

```bash
uv run eeg_analysis/run_representation_pipeline.py \
  --config eeg_analysis/configs/pretrain.yaml \
  pretrain --all-models
```

Multi-GPU:

```bash
uv run eeg_analysis/run_representation_pipeline.py \
  --config eeg_analysis/configs/pretrain.yaml \
  pretrain --distributed
```

Mask-ratio sweep:

```bash
uv run eeg_analysis/src/training/sweep_mask_ratio.py \
  --config eeg_analysis/configs/pretrain.yaml
```

### 4) Supervised fine-tuning

```bash
uv run eeg_analysis/run_representation_pipeline.py \
  --config eeg_analysis/configs/pretrain.yaml \
  finetune \
  --finetune-config eeg_analysis/configs/finetune.yaml \
  --data-path eeg_analysis/data/processed/features/closed_finetune/10s_af7-af8-tp9-tp10_closed_finetune.parquet \
  --output-dir eeg_analysis/finetuned_models
```

If `--data-path` is omitted, the script will try to build/find the representation dataset.
Use `--model <profile>` or `--all-models` to fine-tune specific/all enabled model profiles from `pretrain.yaml`.

Closed pretraining profile default path:
- `datasets.closed_pretrain.dataset_path: /home/abin/eeg-closed-pretrain-data`

## Training Configs and Model Types

- Window-level config: `eeg_analysis/configs/window_model_config.yaml`
- Patient-level config: `eeg_analysis/configs/patient_model_config.yaml`

Model types are validated from the selected config file:

- Classical/GPU: `random_forest`, `gradient_boosting`, `xgboost_gpu`, `catboost_gpu`, `lightgbm_gpu`, `logistic_regression`, `logistic_regression_l1`, `svm_rbf`, `svm_linear`, `extra_trees`, `ada_boost`, `knn`, `decision_tree`, `sgd`
- Deep learning (window config): `pytorch_mlp`, `keras_mlp`, `hybrid_1dcnn_lstm`, `advanced_hybrid_1dcnn_lstm`, `efficient_tabular_mlp`, `advanced_lstm`

## MLflow Notes

- Tracking data lives in `mlruns/` (default local file backend in current configs).
- Processing and training runs log dataset lineage via MLflow dataset inputs.
- Helper scripts at repo root:
  - `scripts/run_all_processing.sh`
  - `scripts/run_all_experiments.sh`
  - `scripts/rerun_experiments.sh`

## Complete Run Command Reference

This section is the single command index for the current codebase.

### Core CLI (`eeg_analysis/run_pipeline.py`)

Show top-level help:

```bash
uv run eeg_analysis/run_pipeline.py --help
```

Show command-specific help (requires a config):

```bash
uv run eeg_analysis/run_pipeline.py \
  --config eeg_analysis/configs/window_model_config.yaml \
  train --help
```

Run pipeline stages:

```bash
uv run eeg_analysis/run_pipeline.py --config eeg_analysis/configs/processing_config.yaml process
uv run eeg_analysis/run_representation_pipeline.py --config eeg_analysis/configs/processing_config.yaml process-closed-finetune
uv run eeg_analysis/run_pipeline.py --config eeg_analysis/configs/window_model_config.yaml list-datasets
```

Training template:

```bash
uv run eeg_analysis/run_pipeline.py \
  --config eeg_analysis/configs/window_model_config.yaml \
  train \
  --level window \
  --model-type random_forest \
  --enable-feature-selection \
  --n-features-select 10 \
  --fs-method select_k_best_f_classif \
  --feature-categories "spectral_features,psd_statistics" \
  --use-dataset-from-run <mlflow_run_id>
```

Evaluation template:

```bash
uv run eeg_analysis/run_pipeline.py \
  --config eeg_analysis/configs/window_model_config.yaml \
  evaluate \
  --run-id <mlflow_run_id> \
  --data-path <features.parquet> \
  --window-size 10
```

### Dataset Processing + Experiment Orchestration

Process all configured window sizes:

```bash
bash scripts/run_all_processing.sh
bash scripts/run_all_processing.sh --dry-run
```

Run full experiment sweep across window sizes / models:

```bash
bash scripts/run_all_experiments.sh
bash scripts/run_all_experiments.sh --ordering sequential
bash scripts/run_all_experiments.sh --model xgboost_gpu
bash scripts/run_all_experiments.sh --dataset-run-id <run_id>
bash scripts/run_all_experiments.sh --dry-run --ordering completion
```

Rerun selected models with feature-selection controls:

```bash
bash scripts/rerun_experiments.sh --channels 'af7 af8 tp9 tp10' --window-size 10 --n-features 5 --fs-method select_k_best_f_classif
```

Traditional-only experiment script:

```bash
bash scripts/run_traditional_experiments.sh
bash scripts/run_traditional_experiments.sh --dry-run
```

Notes:
- `scripts/run_all_experiments.sh` and `scripts/run_traditional_experiments.sh` now assume `uv` is installed and available on `PATH`.

### Representation Commands

Representation dataset build from core pipeline:

```bash
uv run eeg_analysis/run_representation_pipeline.py --config eeg_analysis/configs/processing_config.yaml process-closed-finetune
```

Pretraining dataset builder:

```bash
uv run eeg_analysis/run_representation_pipeline.py --config eeg_analysis/configs/open_pretrain_processing.yaml build-open-pretrain
```

Pretraining window-size conversion:

```bash
uv run eeg_analysis/run_representation_pipeline.py \
  --config eeg_analysis/configs/open_pretrain_processing.yaml \
  convert-open-pretrain \
  --input-root eeg_analysis/secondarydata/raw/sr256_ws10s_open_pretrain \
  --output-base eeg_analysis/secondarydata/raw \
  --factor 2
```

Pretraining:

```bash
uv run eeg_analysis/run_representation_pipeline.py --config eeg_analysis/configs/pretrain.yaml pretrain
```

Distributed pretraining:

```bash
uv run eeg_analysis/run_representation_pipeline.py \
  --config eeg_analysis/configs/pretrain.yaml \
  pretrain --distributed
```

Mask-ratio sweep:

```bash
uv run eeg_analysis/src/training/sweep_mask_ratio.py --config eeg_analysis/configs/pretrain.yaml
uv run eeg_analysis/src/training/sweep_mask_ratio.py --config eeg_analysis/configs/pretrain.yaml --mask-ratios 0.2,0.4,0.6 --torchrun --num-gpus 2
```

SFT / fine-tuning:

```bash
uv run eeg_analysis/run_representation_pipeline.py \
  --config eeg_analysis/configs/pretrain.yaml \
  finetune \
  --finetune-config eeg_analysis/configs/finetune.yaml \
  --data-path eeg_analysis/data/processed/features/closed_finetune/10s_af7-af8-tp9-tp10_closed_finetune.parquet \
  --output-dir eeg_analysis/finetuned_models
```

### Position Leakage / Masking Diagnostics

```bash
uv run scripts/diagnose_100pct_masking.py
uv run scripts/diagnose_100pct_masking.py --checkpoint <checkpoint.pt> --num-samples 100 --diagnostic-mask-ratio 1.0
uv run scripts/diagnose_100pct_masking.py --checkpoint <checkpoint.pt> --decode-to-signal --mask-replacement gaussian_noise
```

### Dataset Helper Scripts

Find matching dataset run:

```bash
uv run scripts/find_dataset.py <window_seconds>
uv run scripts/find_dataset.py <window_seconds> sequential
uv run scripts/find_dataset.py <window_seconds> completion
```

Create filtered-channel dataset:

```bash
uv run scripts/filter_dataset.py <run_id> "af7 af8" <window_seconds>
```

### Random-State Sweep Scripts

```bash
uv run scripts/sweep_random_state.py --config eeg_analysis/configs/window_model_config
# Optional: override model from config by editing top-level model_type first
```

Common optional flags for all sweep scripts:
- `--start`
- `--min`
- `--target`
- `--max`
- `--output`
- `--config`

### Utility Commands

Clean up old model versions:

```bash
uv run scripts/cleanup_old_model_versions.py --model <registered_model_name> --keep 1
uv run scripts/cleanup_old_model_versions.py --all --keep 2
```

Count Mamba model parameters:

```bash
uv run scripts/count_model_parameters.py
```

MLflow UI helpers:

```bash
bash scripts/mlflow-server.sh start
bash scripts/mlflow-server.sh stop
mlflow ui --port 5000
```

## Tests

```bash
pytest eeg_analysis/tests
uv run scripts/test_dataset_logging.py
uv run scripts/test_feature_filtering.py
uv run scripts/test_feature_filtering.py list
uv run scripts/test_model_utils.py
```

## Important Path Note

Some config files contain absolute local paths (for example in `eeg_analysis/configs/processing_config.yaml`, `eeg_analysis/configs/pretrain.yaml`, and `eeg_analysis/configs/open_pretrain_processing.yaml`). Update those paths to match your machine before running.
