# MLflow Dataset Logging Workflow

This document describes the enhanced EEG analysis pipeline that uses MLflow dataset logging for improved data versioning, lineage tracking, and reproducibility.

## Overview

The enhanced pipeline now logs processed datasets as MLflow datasets instead of relying solely on file paths. This provides several advantages:

1. **Data Lineage**: Clear tracking of which dataset was used for each model training
2. **Reproducibility**: Models can be retrained with the exact same dataset
3. **Versioning**: Different versions of processed data are tracked automatically
4. **Metadata**: Rich metadata about datasets (schema, profile, digest) is preserved
5. **Discovery**: Training runs can automatically discover and use recently processed datasets

## Workflow Architecture

```
Raw Data → Processing Pipeline → MLflow Dataset → Model Training
    ↓              ↓                    ↓             ↓
  .mat file    Feature Extraction   Logged Dataset  Trained Model
                     ↓                    ↓             ↓
               Parquet File          Dataset Metadata  Model Metadata
                     ↓                    ↓             ↓
              MLflow Artifact      Dataset Lineage   Model Registry
```

## Key Components

### 1. Enhanced Feature Extractor (`feature_extractor.py`)

The feature extractor now:
- Creates MLflow datasets from processed features
- Logs datasets with proper metadata (name, digest, schema, profile)
- Maintains backward compatibility with file-based workflows
- Returns both the DataFrame and the MLflow dataset

```python
# New signature
features_df, mlflow_dataset = extract_eeg_features(config, windowed_data)
```

### 2. Updated Base Trainer (`base_trainer.py`)

All trainers now include:
- Methods to load datasets from MLflow (`_load_dataset_from_mlflow`)
- Unified data loading with priority: MLflow dataset > provided dataset > file path
- Automatic fallback to file-based loading if MLflow dataset unavailable

### 3. Enhanced Training Pipeline (`run_pipeline.py`)

The training command now:
- Automatically searches for recent processing runs with datasets
- Can use specific datasets via `--use-dataset-from-run` option
- Logs dataset usage and lineage information
- Maintains full backward compatibility

## Usage

### 1. Processing Data

Run the processing pipeline as usual. It will automatically log the processed dataset:

```bash
python run_pipeline.py --config configs/window_model_config_ultra_extreme.yaml process
```

This will:
- Process raw EEG data through the pipeline
- Extract features and save as parquet file
- Create and log MLflow dataset with metadata
- Set tags to mark the run as having a training dataset

### 2. Training Models

#### Option A: Automatic Dataset Discovery (Recommended)

The training pipeline will automatically find and use the most recent dataset:

```bash
python run_pipeline.py --config configs/window_model_config_ultra_extreme.yaml \
    train --level window --model-type random_forest
```

#### Option B: Specify Dataset from Specific Run

If you want to use a dataset from a specific processing run:

```bash
python run_pipeline.py --config configs/window_model_config_ultra_extreme.yaml \
    train --level window --model-type random_forest \
    --use-dataset-from-run <processing_run_id>
```

#### Option C: Fallback to File Path

If no MLflow dataset is found, the system automatically falls back to file path loading.

### 3. Running All Experiments

The existing `run_all_experiments.sh` script works unchanged and will automatically use the new dataset workflow:

```bash
./run_all_experiments.sh
```

## Dataset Information in MLflow UI

When you view runs in the MLflow UI, you'll now see:

### Processing Runs
- **Datasets Used**: Shows the logged training dataset
- **Parameters**: Dataset metadata (name, digest, file path)
- **Tags**: `mlflow.dataset.logged=true`, `mlflow.dataset.context=training`

### Training Runs
- **Datasets Used**: Shows the dataset used for training
- **Parameters**: 
  - `used_mlflow_dataset`: Whether an MLflow dataset was used
  - `dataset_name`: Name of the dataset
  - `dataset_digest`: Unique identifier for the dataset
  - `data_source`: Source of the data (mlflow_dataset, provided_dataset, or file_path)

## Benefits

### 1. Data Lineage Tracking
- Every model is linked to the exact dataset used for training
- You can trace back from a model to the specific processing run that created its training data
- Clear visibility of data transformations and processing steps

### 2. Reproducibility
- Models can be retrained with identical datasets months later
- Processing pipeline changes don't affect ability to reproduce previous results
- Dataset digests ensure bit-for-bit reproducibility

### 3. Experiment Organization
- Related processing and training runs are automatically linked
- Easy to find all models trained on a specific dataset
- Search and filter capabilities in MLflow UI

### 4. Metadata Preservation
- Rich schema information (column types, names)
- Statistical profiles (row counts, data distribution)
- Source information and processing parameters

## Technical Details

### MLflow Dataset Structure

Each logged dataset includes:

```python
{
    "name": "EEG_Features_20640_windows",
    "digest": "abc123def456",  # Unique hash
    "source": "/path/to/features.parquet",
    "targets": "Remission",
    "schema": {
        "columns": [
            {"name": "Participant", "type": "string"},
            {"name": "Remission", "type": "long"},
            {"name": "af7_power_delta", "type": "double"},
            # ... more features
        ]
    },
    "profile": {
        "num_rows": 20640,
        "num_elements": 2064000
    }
}
```

### Trainer Data Loading Priority

1. **MLflow Dataset from Current Run**: If dataset was logged in current MLflow run
2. **Provided Dataset Parameter**: If dataset passed directly to trainer
3. **MLflow Dataset from Specified Run**: If `--use-dataset-from-run` specified
4. **Automatic Discovery**: Search for recent processing runs with datasets
5. **File Path Fallback**: Load from configured file path

### Backward Compatibility

The system maintains full backward compatibility:
- Existing file-based workflows continue to work
- Old training scripts work without modification
- Configuration files don't need changes
- Shell scripts run unchanged

## Testing

Test the dataset logging functionality:

```bash
cd eeg_analysis
python test_dataset_logging.py
```

This will verify:
- Dataset creation and logging
- Dataset retrieval from MLflow
- Training workflow integration
- Automatic dataset discovery

## Troubleshooting

### Common Issues

1. **No MLflow dataset found**
   - Check if processing run completed successfully
   - Verify tags: `mlflow.dataset.logged=true`
   - System will automatically fall back to file path

2. **Dataset loading errors**
   - Ensure MLflow tracking URI is consistent
   - Check file permissions on parquet files
   - Verify experiment access permissions

3. **Feature mismatch**
   - Dataset schema is preserved and validated
   - Check for changes in feature extraction configuration
   - Use dataset digest to verify exact data version

### Debugging

Enable detailed logging to debug dataset loading:

```python
import logging
logging.getLogger('mlflow').setLevel(logging.DEBUG)
```

Check MLflow UI for dataset information:
- Navigate to experiment runs
- Look for "Datasets Used" section
- Verify dataset metadata in run parameters

## Future Enhancements

Planned improvements:
1. Dataset versioning with semantic versions
2. Data drift detection between training and inference
3. Automated dataset validation and quality checks
4. Integration with data catalogs
5. Multi-format dataset support (beyond parquet)

## Migration from File-Based Workflow

No migration is required! The enhanced system:
- Works alongside existing file-based workflows
- Automatically detects and uses MLflow datasets when available
- Falls back gracefully to file paths when needed
- Preserves all existing functionality

Simply start using the new workflow and datasets will be logged automatically going forward. 