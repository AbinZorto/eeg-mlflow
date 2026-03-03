# EEG MLflow Pipeline

Unified documentation for this repository.

This project provides two connected workflows:

1. Traditional EEG feature pipeline (processing + feature extraction + classical/deep-learning classifiers).
2. Mamba-based sequence workflow (secondary dataset build + self-supervised pretraining + supervised fine-tuning).

All commands below assume you run from the repository root: `/home/abin/eeg-mlflow`.

## Repository Layout

- `eeg_analysis/run_pipeline.py`: main CLI for processing/training/evaluation.
- `eeg_analysis/configs/`: processing, training, pretraining, and fine-tuning configs.
- `eeg_analysis/src/processing/`: EEG preprocessing + feature extraction.
- `eeg_analysis/src/models/`: trainers and model code.
- `eeg_analysis/src/training/`: Mamba pretraining/fine-tuning scripts.
- `scripts/build_secondary_dataset.py`: secondary EEG dataset build CLI.
- `scripts/convert_secondary_window_size.py`: utility to up-convert secondary window sizes.
- `mlruns/`: MLflow tracking data.
- `models/`: serialized trained models and metadata.

## Setup

1. Create and activate an environment.
2. Install dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Traditional EEG Feature Pipeline

### 1) Process raw EEG into feature dataset

```bash
python3 eeg_analysis/run_pipeline.py \
  --config eeg_analysis/configs/processing_config.yaml \
  process
```

What it does:

- Loads raw EEG from `processing_config.yaml`.
- Runs upsampling, filtering, downsampling, window slicing, DC offset removal, and feature extraction.
- Writes feature parquet under `eeg_analysis/data/processed/features/`.
- Logs dataset metadata to MLflow (tags include `mlflow.dataset.logged=true`, `mlflow.dataset.context=training`).

Current default processing config (`eeg_analysis/configs/processing_config.yaml`):

- Channels: `af7, af8, tp9, tp10`
- Window size: `10s`
- Window ordering: `sequential` (preserve order enabled)

### 2) (Optional) Build primary dataset for sequence models

```bash
python3 eeg_analysis/run_pipeline.py \
  --config eeg_analysis/configs/processing_config.yaml \
  process-primary
```

This creates a primary parquet dataset from windowed channel data (without feature vectors), used by fine-tuning code.

### 3) List available MLflow datasets

```bash
python3 eeg_analysis/run_pipeline.py \
  --config eeg_analysis/configs/window_model_config.yaml \
  list-datasets
```

### 4) Train models

Window-level example:

```bash
python3 eeg_analysis/run_pipeline.py \
  --config eeg_analysis/configs/window_model_config.yaml \
  train \
  --level window \
  --model-type random_forest
```

Patient-level example:

```bash
python3 eeg_analysis/run_pipeline.py \
  --config eeg_analysis/configs/patient_model_config.yaml \
  train \
  --level patient \
  --model-type random_forest
```

Notes:

- Training first tries MLflow dataset discovery; if not found, it falls back to `data.feature_path` from the training config.
- You can force a specific dataset with `--use-dataset-from-run <mlflow_run_id>`.
- Model IDs are saved in metadata files under `models/window_level/metadata.json` or `models/patient_level/metadata.json`.

### 5) Evaluate a saved model

```bash
python3 eeg_analysis/run_pipeline.py \
  --config eeg_analysis/configs/window_model_config.yaml \
  evaluate \
  --model-id <model_uuid> \
  --data-path eeg_analysis/data/processed/features/10s_af7_af8_tp9_tp10_window_features_seq.parquet
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
python3 eeg_analysis/run_pipeline.py \
  --config eeg_analysis/configs/window_model_config.yaml \
  train \
  --level window \
  --model-type random_forest \
  --feature-categories list
```

Train with selected categories:

```bash
python3 eeg_analysis/run_pipeline.py \
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

## Secondary Dataset + Mamba Workflow

### 1) Build secondary EEG dataset (per-run parquet windows)

```bash
python3 scripts/build_secondary_dataset.py \
  --config eeg_analysis/configs/secondary_processing.yaml \
  build-secondary
```

Key config: `eeg_analysis/configs/secondary_processing.yaml`

- `paths.source_root`: raw secondary EEG source
- `paths.output_dir`: output base dir
- `processing.target_sampling_rate`
- `processing.convert_to_microvolts`
- `processing.register_with_mlflow`

### 2) (Optional) Convert secondary window size

```bash
python3 scripts/convert_secondary_window_size.py \
  --input-root eeg_analysis/secondarydata/raw/sr256_ws4s \
  --output-base eeg_analysis/secondarydata/raw \
  --factor 2
```

### 3) Self-supervised pretraining

```bash
python3 eeg_analysis/src/training/pretrain_mamba.py \
  --config eeg_analysis/configs/pretrain.yaml
```

Multi-GPU:

```bash
torchrun --standalone --nproc_per_node=2 \
  eeg_analysis/src/training/pretrain_mamba.py \
  --config eeg_analysis/configs/pretrain.yaml \
  --distributed
```

Mask-ratio sweep:

```bash
python3 eeg_analysis/src/training/sweep_mask_ratio.py \
  --config eeg_analysis/configs/pretrain.yaml
```

### 4) Supervised fine-tuning

```bash
python3 eeg_analysis/src/training/finetune_mamba.py \
  --config eeg_analysis/configs/finetune.yaml \
  --pretrain-config eeg_analysis/configs/pretrain.yaml \
  --data-path eeg_analysis/data/processed/features/primary/10s_af7-af8-tp9-tp10_primary_dataset.parquet \
  --output-dir eeg_analysis/finetuned_models
```

If `--data-path` is omitted, the script will try to build/find the primary dataset.

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
python3 eeg_analysis/run_pipeline.py --help
```

Show command-specific help (requires a config):

```bash
python3 eeg_analysis/run_pipeline.py \
  --config eeg_analysis/configs/window_model_config.yaml \
  train --help
```

Run pipeline stages:

```bash
python3 eeg_analysis/run_pipeline.py --config eeg_analysis/configs/processing_config.yaml process
python3 eeg_analysis/run_pipeline.py --config eeg_analysis/configs/processing_config.yaml process-primary
python3 eeg_analysis/run_pipeline.py --config eeg_analysis/configs/window_model_config.yaml list-datasets
```

Training template:

```bash
python3 eeg_analysis/run_pipeline.py \
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
python3 eeg_analysis/run_pipeline.py \
  --config eeg_analysis/configs/window_model_config.yaml \
  evaluate \
  --model-id <model_uuid> \
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

### Secondary Dataset + Mamba Commands

Secondary dataset builder:

```bash
python3 scripts/build_secondary_dataset.py --config eeg_analysis/configs/secondary_processing.yaml build-secondary
```

Secondary window-size conversion:

```bash
python3 scripts/convert_secondary_window_size.py \
  --input-root eeg_analysis/secondarydata/raw/sr256_ws4s \
  --output-base eeg_analysis/secondarydata/raw \
  --factor 2
```

Pretraining:

```bash
python3 eeg_analysis/src/training/pretrain_mamba.py --config eeg_analysis/configs/pretrain.yaml
```

Distributed pretraining:

```bash
torchrun --standalone --nproc_per_node=2 \
  eeg_analysis/src/training/pretrain_mamba.py \
  --config eeg_analysis/configs/pretrain.yaml \
  --distributed
```

Mask-ratio sweep:

```bash
python3 eeg_analysis/src/training/sweep_mask_ratio.py --config eeg_analysis/configs/pretrain.yaml
python3 eeg_analysis/src/training/sweep_mask_ratio.py --config eeg_analysis/configs/pretrain.yaml --mask-ratios 0.2,0.4,0.6 --torchrun --num-gpus 2
```

SFT / fine-tuning:

```bash
python3 eeg_analysis/src/training/finetune_mamba.py \
  --config eeg_analysis/configs/finetune.yaml \
  --pretrain-config eeg_analysis/configs/pretrain.yaml \
  --data-path eeg_analysis/data/processed/features/primary/10s_af7-af8-tp9-tp10_primary_dataset.parquet \
  --output-dir eeg_analysis/finetuned_models
```

### Position Leakage / Masking Diagnostics

```bash
python3 scripts/diagnose_100pct_masking.py
python3 scripts/diagnose_100pct_masking.py --checkpoint <checkpoint.pt> --num-samples 100 --diagnostic-mask-ratio 1.0
python3 scripts/diagnose_100pct_masking.py --checkpoint <checkpoint.pt> --decode-to-signal --mask-replacement gaussian_noise
```

### Dataset Helper Scripts

Find matching dataset run:

```bash
python3 scripts/find_dataset.py <window_seconds>
python3 scripts/find_dataset.py <window_seconds> sequential
python3 scripts/find_dataset.py <window_seconds> completion
```

Create filtered-channel dataset:

```bash
python3 scripts/filter_dataset.py <run_id> "af7 af8" <window_seconds>
```

### Random-State Sweep Scripts

```bash
python3 scripts/sweep_random_state.py --config eeg_analysis/configs/window_model_config
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
python3 scripts/cleanup_old_model_versions.py --model <registered_model_name> --keep 1
python3 scripts/cleanup_old_model_versions.py --all --keep 2
```

Count Mamba model parameters:

```bash
python3 scripts/count_model_parameters.py
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
python3 scripts/test_dataset_logging.py
python3 scripts/test_feature_filtering.py
python3 scripts/test_feature_filtering.py list
python3 scripts/test_model_utils.py
```

## Important Path Note

Some config files contain absolute local paths (for example in `processing_config.yaml`, `pretrain.yaml`, and `secondary_processing.yaml`). Update those paths to match your machine before running.
