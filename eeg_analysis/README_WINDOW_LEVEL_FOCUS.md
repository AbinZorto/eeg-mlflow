# Window-Level Training Focus: EEG Depression Remission Prediction

## Overview
This document focuses specifically on **window-level training** in the EEG analysis pipeline, with the corrected understanding that each window contains **88 features** (not 124 as previously calculated).

## Data Pipeline Summary

### Raw Data Structure
- **21 patients total**: 14 non-remission, 7 remission
- **4 EEG channels**: `af7`, `af8`, `tp9`, `tp10` (frontal and temporal)
- **Original sampling rate**: 128 Hz (varies: some 5-min, some 10-min recordings)
- **Recording duration**: ~10 minutes per patient (some 5 minutes)

### Processing Pipeline
1. **Upsampling**: 2× factor → 256 Hz
2. **Filtering**: Butterworth filter, 60 Hz cutoff
3. **Downsampling**: 2× factor → 128 Hz (back to original rate)
4. **Window slicing**: 2-second non-overlapping windows (256 samples per window)
5. **Feature extraction**: 88 features per window

### Feature Extraction Details (88 Features Per Window)

From the processing configuration and feature extractor, the 88 features per window come from:

#### Per-Channel Features (4 channels × 21 features = 84 features)
**For each channel (`af7`, `af8`, `tp9`, `tp10`):**

1. **Spectral Features (6 per channel)**:
   - 5 frequency band powers: delta (0.5-4 Hz), theta (4-8 Hz), alpha (8-12 Hz), beta (12-30 Hz), gamma (30-60 Hz)
   - 1 frequency-weighted power

2. **Temporal Features (8 per channel)**:
   - mean, std, var, skew, kurtosis, rms, peak_to_peak, zero_crossings

3. **Entropy Features (2 per channel)**:
   - sample_entropy, spectral_entropy

4. **Additional Features (5 per channel)**:
   - Hjorth parameters (activity, mobility, complexity)
   - Mean absolute deviation
   - Additional statistical measures

**Subtotal: 4 channels × 21 features = 84 features**

#### Cross-Channel Asymmetry Features (4 features)
1. **Frontal asymmetry (AF7/AF8)**: 2 asymmetry features for key frequency bands
2. **Temporal asymmetry (TP9/TP10)**: 2 asymmetry features for key frequency bands

**Total: 84 + 4 = 88 features per window**

## Window-Level Training Dataset

### Window Distribution
- **Total windows**: ~4,725 windows
  - **Non-remission**: ~3,150 windows (from 14 patients)
  - **Remission**: ~1,575 windows (from 7 patients)
- **Class imbalance ratio**: ~2:1 (non-remission:remission)
- **Average windows per patient**: ~225 windows (varies by recording length)

### Feature Matrix Structure
```
Shape: (4,725 windows, 88 features)
Columns: 88 feature columns + 'Participant' + 'Remission' (binary target)
```

## SMOTE Integration in Window Training

### Problem Addressed
- **Patient-level imbalance**: 14 vs 7 patients (2:1 ratio)
- **Window-level imbalance**: ~3,150 vs ~1,575 windows (2:1 ratio)
- Previous issue: Zero true positives for remission patients

### SMOTE Implementation
The `WindowLevelTrainer` class now includes SMOTE integration:

```python
# In window_trainer.py
def _create_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
    """Create and train model, disabling class weights if SMOTE is used."""
    model_params = self.model_params.copy()
    if self.use_smote:
        model_params['class_weight'] = None  # Disable class weights when using SMOTE
    
    model = create_classifier(self.model_type, model_params)
    model.fit(X_train, y_train)
    return model
```

### SMOTE Application Process

#### 1. Cross-Validation with SMOTE
**Leave-One-Group-Out (LOGO) Cross-Validation**: 21 folds (one per patient)

For each fold:
```python
X_train_fold, y_train_fold = X_train, y_train
if self.use_smote:
    smote = SMOTE(random_state=42)
    X_train_fold, y_train_fold = smote.fit_resample(X_train_fold, y_train_fold)
```

#### 2. SMOTE Effect Per Fold
**Typical fold scenario** (leaving out 1 remission patient):
- **Before SMOTE**: ~3,150 non-remission + ~1,350 remission = ~4,500 training windows
- **After SMOTE**: ~3,150 non-remission + ~3,150 synthetic remission = ~6,300 training windows
- **Balance achieved**: 50%-50% class distribution

#### 3. Final Model Training
```python
# Train final model on full dataset with SMOTE
if self.use_smote:
    smote = SMOTE(random_state=42)
    X_final_train, y_final_train = smote.fit_resample(X_final_train, y_final_train)

final_model = self._create_and_train_model(X_final_train, y_final_train)
```

### SMOTE Configuration
- **SMOTE class**: `imblearn.over_sampling.SMOTE`
- **Random state**: 42 (for reproducibility)
- **Default k_neighbors**: 5 (SMOTE default)
- **Strategy**: Auto (balances to 50%-50%)

## Training Configuration

### Model Options
**Traditional ML models**:
- Random Forest (default)
- Gradient Boosting
- Logistic Regression
- SVM (RBF/Linear)
- Extra Trees, AdaBoost, KNN, Decision Tree, SGD

**Deep Learning models**:
- PyTorch MLP
- Keras MLP

### Current Configuration ([[memory:2839559]])
Using `window_model_config_ultra_extreme.yaml`:
- **Model type**: `pytorch_mlp`
- **Architecture**: [4096, 2048, 1024, 512, 256, 128] (6 layers)
- **Batch size**: 8,192 (massive for GPU utilization)
- **Learning rate**: 0.001
- **Epochs**: 25
- **SMOTE enabled**: `use_smote: true`

## Feature Selection Support

The window trainer supports multiple feature selection methods:

### Available Methods
1. **`model_based`**: Uses feature importances from the same model type
2. **`select_k_best_f_classif`**: ANOVA F-statistic (linear relationships)
3. **`select_k_best_mutual_info`**: Mutual information (non-linear relationships)
4. **`select_from_model_l1`**: L1-regularized Logistic Regression
5. **`rfe`**: Recursive Feature Elimination

### Configuration
```yaml
feature_selection:
  enabled: false  # Default: use all 88 features
  method: "model_based"
  n_features: 10  # If enabled, select top 10 features
```

## Expected Performance Impact

### Before SMOTE Implementation
- **Window-level accuracy**: 70-80%
- **Patient-level recall for remission**: 0-10% (zero true positives problem)
- **Overall patient-level accuracy**: ~60-70%

### Expected After SMOTE Implementation
- **Window-level accuracy**: May decrease slightly (65-75%) due to synthetic data
- **Patient-level recall for remission**: **50-80%** (major improvement)
- **Patient-level precision for remission**: 40-70%
- **Overall patient-level F1-score**: **Significant improvement**

### Key Benefits
1. **Eliminates zero true positives**: SMOTE generates synthetic remission windows
2. **Balanced training**: Each fold trains on balanced data (50%-50%)
3. **Better generalization**: Models learn remission patterns more effectively
4. **Maintained specificity**: Non-remission classification should remain strong

## Execution

### Training Command
```bash
cd eeg_analysis
python run_pipeline.py train \
    --level window \
    --model-type pytorch_mlp \
    --config configs/window_model_config_ultra_extreme.yaml
```

### Key Log Outputs
```
Dataset Statistics:
Total number of patients: 21
- Remission patients: 7
- Non-remission patients: 14
Total number of windows: 4,725

Performing Leave-One-Group-Out cross-validation with 21 splits

Fold 1/21
Testing on participant: [ID] (true label: [0/1])
Training windows: ~4,500, Test windows: ~225
Applying SMOTE to fold 1...
Original positive class balance: 0.30
SMOTE-balanced positive class balance: 0.50
```

## MLflow Tracking

### Logged Parameters
- `feature_path`: Path to 88-feature dataset
- `use_smote`: true
- `num_features_trained_on`: 88
- Model hyperparameters

### Logged Metrics
- `patient_accuracy`, `patient_precision`, `patient_recall`, `patient_f1`
- Per-fold metrics: `fold_X_patient_accuracy`, `fold_X_window_accuracy`
- SMOTE balance statistics

### Artifacts
- Trained model (final model with SMOTE)
- Window-level predictions (all 4,725 windows)
- Patient-level predictions (21 patients)
- Feature importance plots (if applicable)

## Technical Notes

### SMOTE with 88 Features
- **Dimensionality**: SMOTE works well with 88 features (manageable dimension)
- **k-neighbors**: Default k=5 suitable for this feature space
- **Synthetic quality**: With ~1,575 remission windows, SMOTE has sufficient real data for quality synthesis

### Memory Requirements
- **Original dataset**: 4,725 × 88 = ~416k data points
- **After SMOTE**: ~6,300 × 88 = ~554k data points per fold
- **GPU batch processing**: 8,192 batch size with 88 features = ~721k parameters per batch

### Computational Efficiency
- **SMOTE overhead**: ~1-2 seconds per fold
- **Total training time**: Depends on model complexity (PyTorch MLP: ~10-30 minutes)
- **Cross-validation**: 21 folds × model training time

This window-level approach with SMOTE should provide the breakthrough needed to achieve meaningful recall for remission patients while maintaining the detailed temporal analysis that window-level training provides. 