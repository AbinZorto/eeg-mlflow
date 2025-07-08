# Patient-Level Trainer Updates

This document describes the major updates made to the `PatientLevelTrainer` to enable automatic model selection and enhanced functionality.

## Key Improvements

### 1. Multiple Classifier Support

The trainer now supports automatic evaluation of multiple classifiers:

- Random Forest
- Gradient Boosting
- Logistic Regression (L1 and L2)
- Extra Trees
- AdaBoost
- SVM (RBF and Linear kernels)
- K-Nearest Neighbors
- Decision Tree
- Stochastic Gradient Descent

### 2. Automatic Model Selection

- Set `model_type: "auto"` in your config to try all classifiers
- The best model is selected based on F1 score
- All classifiers are evaluated using Leave-One-Group-Out cross-validation
- Results for all classifiers are saved for comparison

### 3. Enhanced Aggregation

Patient-level features now include:

- Mean, std, min, max, median (as before)
- 25th and 75th percentiles
- Number of windows per patient
- All numeric columns converted to float64

### 4. Detailed Metrics and Logging

- Comprehensive confusion matrix reporting (TP, TN, FP, FN)
- Detailed per-classifier metrics logging
- Enhanced MLflow integration with fold-level metrics
- Misclassified patient reporting with confidence scores

### 5. Class Weight Handling

- Configurable class weights to handle imbalanced datasets
- Default weights favor the minority class (remission patients)

## Configuration Changes

### Model Type Selection

```yaml
# For automatic model selection (recommended)
model_type: "auto"

# For single model training
model_type: "random_forest"  # or any other supported classifier
```

### Class Weights

```yaml
use_class_weights: true # Enable class balancing
```

### Enhanced Model Parameters

The configuration now includes parameters for all supported classifiers:

```yaml
model:
  params:
    random_forest:
      n_estimators: 200
      min_samples_leaf: 2
      class_weight: "balanced"
      random_state: 42

    svm_rbf:
      kernel: "rbf"
      C: 1.0
      class_weight: "balanced"
      probability: true
      random_state: 42

    # ... other classifiers
```

## Usage Examples

### Basic Usage with Auto Selection

```python
from models.patient_trainer import PatientLevelTrainer
import yaml

# Load configuration
with open('configs/patient_model_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set to auto mode
config['model_type'] = 'auto'

# Initialize and train
trainer = PatientLevelTrainer(config)
best_model = trainer.train()
```

### Command Line Usage

```bash
# Train with automatic model selection
python examples/train_patient_model.py \
    --config configs/patient_model_config.yaml \
    --window_size 2

# Train specific model with feature selection
python examples/train_patient_model.py \
    --config configs/patient_model_config.yaml \
    --window_size 2 \
    --model_type random_forest \
    --feature_selection \
    --n_features 15
```

## Output Files

When using auto mode, the trainer saves:

- `{classifier}_predictions.csv` for each classifier tested
- `feature_importances.csv` for the best model (if available)
- `model_metadata.json` with best classifier information
- MLflow artifacts and metrics for all classifiers

## MLflow Integration

The trainer logs:

- Per-fold accuracy and confidence for each classifier
- Overall metrics (TP, TN, FP, FN, accuracy, precision, recall, F1) for each classifier
- Best classifier selection with F1 score
- Feature importance for tree-based models
- Model artifacts with proper signatures

## Backward Compatibility

The trainer maintains backward compatibility:

- Existing single-model configurations still work
- Previous model types are supported
- Configuration structure remains compatible

## Feature Selection Integration

The enhanced trainer works seamlessly with existing feature selection:

- Model-based selection uses the same type as the main model (when not in auto mode)
- All selection methods work with the enhanced aggregation
- Selected features are properly logged and tracked

## Performance Considerations

- Auto mode takes longer as it trains multiple models
- All models use standardized features (via pipelines)
- Cross-validation is performed once, with all models trained on the same splits
- Memory usage scales with number of classifiers when in auto mode

## Migration Guide

To upgrade existing code:

1. Update your configuration file to include the new `model_type` field
2. Optionally add `use_class_weights: true` for better class balance handling
3. Set `model_type: "auto"` to enable automatic model selection
4. Update any custom scripts to use the new configuration structure

No code changes are required for existing PatientLevelTrainer usage.
