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
- `eeg_analysis/build_secondary_dataset.py`: secondary EEG dataset build CLI.
- `eeg_analysis/convert_secondary_window_size.py`: utility to up-convert secondary window sizes.
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

Optional package install for local imports:

```bash
pip install -e eeg_analysis
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
  --config eeg_analysis/configs/window_model_config_ultra_extreme.yaml \
  list-datasets
```

### 4) Train models

Window-level example:

```bash
python3 eeg_analysis/run_pipeline.py \
  --config eeg_analysis/configs/window_model_config_ultra_extreme.yaml \
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
  --config eeg_analysis/configs/window_model_config_ultra_extreme.yaml \
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
  --config eeg_analysis/configs/window_model_config_ultra_extreme.yaml \
  train \
  --level window \
  --model-type random_forest \
  --feature-categories list
```

Train with selected categories:

```bash
python3 eeg_analysis/run_pipeline.py \
  --config eeg_analysis/configs/window_model_config_ultra_extreme.yaml \
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
python3 eeg_analysis/build_secondary_dataset.py \
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
python3 eeg_analysis/convert_secondary_window_size.py \
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

- Window-level config: `eeg_analysis/configs/window_model_config_ultra_extreme.yaml`
- Patient-level config: `eeg_analysis/configs/patient_model_config.yaml`

Model types are validated from the selected config file:

- Classical/GPU: `random_forest`, `gradient_boosting`, `xgboost_gpu`, `catboost_gpu`, `lightgbm_gpu`, `logistic_regression`, `logistic_regression_l1`, `svm_rbf`, `svm_linear`, `extra_trees`, `ada_boost`, `knn`, `decision_tree`, `sgd`
- Deep learning (window config): `pytorch_mlp`, `keras_mlp`, `hybrid_1dcnn_lstm`, `advanced_hybrid_1dcnn_lstm`, `efficient_tabular_mlp`, `advanced_lstm`

## MLflow Notes

- Tracking data lives in `mlruns/` (default local file backend in current configs).
- Processing and training runs log dataset lineage via MLflow dataset inputs.
- Helper scripts at repo root:
  - `run_all_processing.sh`
  - `run_all_experiments.sh`
  - `rerun_experiments.sh`

## Tests

```bash
pytest eeg_analysis/tests
python3 eeg_analysis/test_dataset_logging.py
python3 test_feature_filtering.py
```

## Important Path Note

Some config files contain absolute local paths (for example in `processing_config.yaml`, `pretrain.yaml`, and `secondary_processing.yaml`). Update those paths to match your machine before running.
