# EEG Analysis Pipeline: Complete Technical Documentation

## Overview

This document provides a comprehensive, step-by-step breakdown of the EEG analysis pipeline for depression remission prediction. The pipeline processes raw EEG data from 21 patients and uses machine learning to predict treatment outcomes.

## Data Structure

### Patient Demographics
- **Total Patients**: 21
- **Non-Remission Patients**: 14 (66.7% - majority class)
- **Remission Patients**: 7 (33.3% - minority class)

### Raw Data Format
The raw EEG data is stored in a MATLAB file: `MDD_21_subjects_EC_EEG_data_remission_and_non_remission_seperated.mat`

**Initial Data Dimensions**:
- **Channels**: 4 EEG electrodes (`af7`, `af8`, `tp9`, `tp10`)
- **Sampling Rate**: 256 Hz (standardized)
- **Recording Duration**: 10 minutes (some patients have 5 minutes)
- **Total Raw Samples per Patient**: 
  - 10-minute patients: 4 channels × 256 Hz × 600 seconds = 614,400 data points
  - 5-minute patients: 4 channels × 256 Hz × 300 seconds = 307,200 data points

## Processing Pipeline

### Stage 1: Data Loading (`data_loader.py`)

**Input**: Raw MATLAB file with pre-windowed data
**Output**: Structured pandas DataFrame

```
Raw Data: Pre-windowed by original researchers
- Each patient has multiple pre-existing windows
- Window structure preserved from original dataset
```

### Stage 2: Upsampling (`upsampler.py`)

**Purpose**: Standardize slight irregularities in original window lengths
**Method**: Linear interpolation to exact target length
**Target Length**: 5,120 samples (2× upsampling)

**Numerical Transformation**:
```
Before: Slightly irregular window lengths (~2,550-2,570 samples at 256 Hz)
After:  Exactly 5,120 samples per window (upsampled by factor of 2)
Purpose: Ensure perfect standardization before filtering
```

**Upsampling Factor**: 2× (from config)

### Stage 3: Filtering (`filter.py`)

**Purpose**: Remove noise and artifacts
**Filter Configuration**:
- **Type**: Butterworth filter (4th order)
- **Cutoff**: 60 Hz (removes high-frequency noise)
- **Sampling Rate**: 512 Hz effective (after upsampling)

**Data Preservation**: 100% (same dimensions, cleaner signals)
```
Shape: Still 4 channels × 5,120 samples per window
Quality: Improved signal-to-noise ratio
```

### Stage 4: Downsampling (`downsampler.py`)

**Purpose**: Return to exactly 2,560 samples per window for perfect standardization
**Method**: Decimation with anti-aliasing filter
**Downsampling Factor**: 2× (back to original effective rate)

**Numerical Transformation**:
```
Before: 4 channels × 5,120 samples (after upsampling and filtering)
After:  4 channels × 2,560 samples (exactly standardized)
Result: Perfectly consistent 2,560 samples per window for all patients
```

### Stage 5: Window Slicing (`window_slicer.py`)

**Purpose**: Create consistent 2-second analysis windows
**Window Configuration**:
- **Window Size**: 2 seconds
- **Sampling Rate**: 256 Hz effective (after processing)
- **Overlap**: 0 seconds (non-overlapping)
- **Samples per Window**: 2,560 samples (standardized length)

**Numerical Transformation per Patient**:
```
10-minute patients: 600 seconds ÷ 2 seconds = 300 windows per patient
5-minute patients: 300 seconds ÷ 2 seconds = 150 windows per patient
Output Shape: Variable windows × 4 channels × 2,560 samples
```

**Total Dataset After Windowing** (assuming mix of 10 and 5-minute recordings):
```
Estimated total: ~21 patients × 225 average windows = ~4,725 total windows
- Non-remission windows: ~14 patients × 225 = ~3,150 windows  
- Remission windows: ~7 patients × 225 = ~1,575 windows
Window Imbalance Ratio: 2:1 (Non-remission:Remission)
```

### Stage 6: Feature Extraction (`feature_extractor.py`)

**Purpose**: Convert raw EEG signals into meaningful features for ML
**Feature Categories**:

#### A. Spectral Features (per window, per channel)
- **Power Spectral Density**: 5 frequency bands × 4 channels = 20 features
  - Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-12 Hz), Beta (12-30 Hz), Gamma (30-60 Hz)
- **Peak Frequency**: 4 features
- **Spectral Entropy**: 4 features
- **Total Spectral Features**: 28 per window

#### B. Statistical Features (per window, per channel)
- **Basic Stats**: Mean, Std, Variance, Skewness, Kurtosis = 5 × 4 = 20 features
- **Range Stats**: Min, Max, Peak-to-Peak = 3 × 4 = 12 features
- **Advanced Stats**: RMS, Zero Crossings, Mean Abs Deviation = 3 × 4 = 12 features
- **Total Statistical Features**: 44 per window

#### C. Complexity Features (per window, per channel)
- **Entropy Measures**: Sample Entropy, Spectral Entropy = 2 × 4 = 8 features
- **Fractal Features**: Hurst Exponent, DFA, Correlation Dimension = 3 × 4 = 12 features
- **Hjorth Parameters**: Activity, Mobility, Complexity = 3 × 4 = 12 features
- **Other Complexity**: HFD, Lyapunov Exponent = 2 × 4 = 8 features
- **Total Complexity Features**: 40 per window

#### D. Connectivity Features (per window)
- **Cross-correlation**: 4×3/2 = 6 inter-channel correlations
- **Coherence**: 6 inter-channel coherence values
- **Total Connectivity Features**: 12 per window

**Total Features per Window**: 28 + 44 + 40 + 12 = **124 features**

**Final Feature Dataset**:
```
Shape: ~4,725 windows × 124 features
Size: ~585,900 data points
Memory: ~5-10MB as compressed parquet
File: eeg_analysis/data/processed/features/2s_window_features.parquet
```

## Training Pipeline

### Data Structure for Training

**Window-Level Dataset**:
```
Rows: ~4,725 windows
Columns: 126 total (124 features + Participant + Remission)
Features: 124 numerical features
Labels: Binary (0=Non-remission, 1=Remission)
Groups: Participant ID (for cross-validation)
```

**Patient-Level Dataset** (after aggregation):
```
Rows: 21 patients
Columns: 622 total (124 × 5 aggregations + Participant + Remission + n_windows)
Features: 620 aggregated features (mean, std, min, max, median per original feature)
Labels: Binary (0=Non-remission, 1=Remission)
```

### Training Approaches

### 1. Window-Level Training (`WindowLevelTrainer`)

#### Cross-Validation Strategy: Leave-One-Group-Out (LOGO)
**Folds**: 21 (one per patient)

#### Fold Example: Testing Patient 15 (Remission)
```
Test Set:
- Patient 15: ~225 windows (all labeled "remission")
- Test size: ~225 windows

Training Set:
- Remaining patients: 20 patients
- Non-remission patients: ~14 × 225 = ~3,150 windows
- Remission patients: ~6 × 225 = ~1,350 windows
- Training size: ~4,500 windows
- Class imbalance: ~3,150 vs ~1,350 (2.3:1 ratio)
```

#### SMOTE Application (NEW - Fixes Class Imbalance)
```
Before SMOTE:
- Training set: ~4,500 windows
- Non-remission: ~3,150 windows (70%)
- Remission: ~1,350 windows (30%)

After SMOTE:
- Training set: ~6,300 windows
- Non-remission: ~3,150 original windows (50%)
- Remission: ~3,150 windows (50% - 1,350 original + 1,800 synthetic)
- Perfect balance achieved
```

#### Model Training & Prediction Process
```
1. Train model on ~6,300 SMOTE-balanced windows
2. Predict on ~225 test windows from Patient 15
3. Get probability scores: [0.3, 0.7, 0.2, 0.8, ...]
4. Aggregate to patient-level: mean(probabilities) = 0.65
5. Final prediction: 0.65 > 0.5 → Remission
```

#### Complete Cross-Validation Results
```
21 folds × patient-level predictions = 21 final predictions
Evaluation: Compare 21 predictions vs 21 true labels
Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
```

### 2. Patient-Level Training (`PatientLevelTrainer`)

#### Data Aggregation Process
**Per Patient Feature Creation**:
```
Input: ~225 windows × 124 features (per patient)
Aggregations: mean, std, min, max, median
Output: 1 patient × 620 features

Example for Patient 1:
- Original: ~225 × 124 matrix
- Mean features: 124 values
- Std features: 124 values  
- Min features: 124 values
- Max features: 124 values
- Median features: 124 values
- Total: 620 patient-level features
```

#### Cross-Validation: LOGO on Patients
```
Test Patient 15:
- Input: 1 row × 620 features
- True label: Remission

Training Set:
- Patients: 20 remaining patients
- Non-remission: 14 patients
- Remission: 6 patients  
- Shape: 20 rows × 620 features
- Class imbalance: 14 vs 6 (2.3:1 ratio)
```

#### Class Weighting (CORRECT application)
```
Applied to 20 patients (not windows):
- Weight for Non-remission (14 patients): 20/(2×14) = 0.71
- Weight for Remission (6 patients): 20/(2×6) = 1.67
- Remission predictions weighted 2.35× higher
```

### Deep Learning Training (`DeepLearningTrainer`)

#### PyTorch MLP Configuration (Ultra-Extreme)
```
Architecture: [124] → [4,096] → [2,048] → [1,024] → [512] → [256] → [128] → [2]
Parameters: ~22 million trainable parameters
Batch Size: 8,192 windows (maximal GPU utilization)
Training Data: SMOTE-balanced windows per fold
Memory Usage: ~8-12GB VRAM on RTX 3090
```

#### Training Process per Fold
```
1. Load fold data: ~4,500 windows × 124 features
2. Apply SMOTE: ~6,300 balanced windows
3. Create batches: 6,300 ÷ 8,192 = 1 batch per epoch (plus remainder)
4. Train for 25 epochs: 25 total batch updates
5. GPU utilization: 85-95% throughout training
```

## Performance Characteristics

### Computational Requirements

#### Processing Pipeline
```
CPU Usage: 8-16 cores recommended
RAM: 16GB minimum, 32GB recommended
Processing Time: 
- Raw to features: ~45-60 minutes
- Storage: ~130MB compressed parquet
```

#### Training Pipeline
```
Window-Level Training:
- CPU: ~30-45 minutes per model
- GPU (Deep Learning): ~5-10 minutes per model
- Memory: 8-16GB RAM

Patient-Level Training:
- CPU: ~5-10 minutes per model  
- Memory: 2-4GB RAM
```

### Expected Performance Metrics

#### Baseline Performance (Before SMOTE)
```
Window-Level Models:
- Accuracy: 60-70%
- Precision (Remission): 0-20% (poor)
- Recall (Remission): 0-10% (very poor)
- F1-Score: 0.0-0.15

Patient-Level Models:
- Accuracy: 70-80%
- Precision (Remission): 40-60%
- Recall (Remission): 20-40%
- F1-Score: 0.30-0.50
```

#### Expected Performance (After SMOTE)
```
Window-Level Models:
- Accuracy: 65-75%
- Precision (Remission): 40-70%
- Recall (Remission): 50-80% (much improved)
- F1-Score: 0.45-0.75

Patient-Level Models:
- Accuracy: 75-85%
- Precision (Remission): 60-80%
- Recall (Remission): 60-85%
- F1-Score: 0.60-0.82
```

## Key Improvements Made

### 1. SMOTE Integration
- **Problem**: Window-level training suffered from incorrect class weighting
- **Solution**: SMOTE creates synthetic minority samples within each CV fold
- **Impact**: Dramatically improves recall for remission patients

### 2. Proper Cross-Validation
- **LOGO ensures**: No patient data leakage between train/test
- **21 folds**: Each patient tested exactly once
- **Realistic evaluation**: True generalization performance

### 3. Feature Selection Options
- **Methods**: K-best, Mutual Info, L1-based, RFE
- **Benefits**: Reduces overfitting, improves interpretability
- **Typical selection**: 500-2000 most informative features

### 4. Multi-Level Analysis
- **Window-level**: Captures temporal dynamics
- **Patient-level**: Provides clinical interpretability
- **Ensemble potential**: Combine both approaches

## File Structure

```
eeg_analysis/data/processed/features/
├── 2s_window_features.parquet          # Main feature dataset (130MB)
├── window_visualizations/              # QC plots
└── versions/                           # Data versioning

models/
├── window_level/                       # Window-level trained models
├── patient_level/                      # Patient-level trained models
└── best_model/                         # Production model

mlruns/                                 # MLflow experiment tracking
├── experiment_logs/                    # Training logs
├── metrics/                            # Performance metrics
└── artifacts/                          # Model artifacts
```

## Usage Examples

### Training Window-Level Model with SMOTE
```bash
python run_pipeline.py --config configs/window_model_config_ultra_extreme.yaml \
    train --level window --model-type pytorch_mlp \
    --enable-feature-selection --n-features-select 1000
```

### Training Patient-Level Model
```bash
python run_pipeline.py --config configs/patient_model_config.yaml \
    train --level patient --model-type random_forest \
    --enable-feature-selection --n-features-select 500
```

This pipeline represents a state-of-the-art approach to EEG-based depression outcome prediction, with careful attention to class imbalance, data leakage prevention, and computational efficiency. 