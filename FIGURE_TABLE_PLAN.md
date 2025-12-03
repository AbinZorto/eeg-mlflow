# Figure and Table Planning Document

## FIGURE SPECIFICATIONS

---

## Figure 1: Pipeline Diagram
**Purpose**: Overview of entire pipeline, highlight DC offset removal

**Content**:
```
[Raw EEG Data] 
    ↓
[Data Loading] (4 channels, 21 patients)
    ↓
[Upsampling] (2× to 256 Hz)
    ↓
[Filtering] (Butterworth, 60 Hz cutoff)
    ↓
[Downsampling] (2× to 128 Hz)
    ↓
[Windowing] (10-second windows)
    ↓
[DC OFFSET REMOVAL] ← HIGHLIGHT THIS STEP
    ↓
[Feature Extraction] (5 selected features)
    ↓
[Models] (Tabular MLP / Hybrid CNN-LSTM)
    ↓
[Window-Level Predictions]
    ↓
[Patient-Level Aggregation] (mean probabilities)
    ↓
[Evaluation] (Patient-level metrics)
```

**Design Notes**:
- Use color coding: DC removal step in RED/BOLD
- Show data flow clearly
- Include key numbers (21 patients, 4 channels, 5 features)
- Two parallel paths for two models

**Location**: After Introduction, before Methodology

---

## Figure 2: DC Offset Removal Visualization
**Purpose**: Visual demonstration of DC offset removal effect

**Content**: 4 subplots (one per channel)
- **Subplot 1-4**: Each channel (AF7, AF8, TP9, TP10)
  - **Top**: Before DC removal (signal with offset)
  - **Bottom**: After DC removal (centered signal)
  - **Annotation**: Show DC offset magnitude

**Design Notes**:
- Use time series plots
- Show clear baseline shift before
- Show zero-centered signal after
- Include DC offset value annotation
- Use consistent y-axis scale for comparison

**Example Data Points**:
- Before: Signal oscillating around +500 μV
- After: Signal oscillating around 0 μV
- DC Offset: ~500 μV removed

**Location**: In Methodology section (DC Offset Removal subsection)

---

## Figure 3: Model Architectures
**Purpose**: Compare simple vs. complex architectures

**Content**: Two side-by-side architecture diagrams

**Left: Tabular MLP**
```
Input (5 features)
    ↓
Linear(5 → 1024) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Linear(1024 → 512) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Linear(512 → 256) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Linear(256 → 128) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Linear(128 → 2)
    ↓
Output (Binary Classification)
```

**Right: Hybrid CNN-LSTM**
```
Input (5 features)
    ↓
Feature Embedding (5 → 128)
    ↓
[CNN Branch]              [LSTM Branch]
    ↓                           ↓
CNN Block 1 (64 filters)    LSTM Layer 1 (128 units, bidirectional)
    ↓                           ↓
CNN Block 2 (128 filters)   LSTM Layer 2 (64 units, bidirectional)
    ↓                           ↓
CNN Block 3 (256 filters)   Attention (8 heads, 64 dim)
    ↓                           ↓
Feature Pyramid              Positional Encoding
    ↓                           ↓
    └─────────── Fusion ─────────┘
            ↓
    Dense(256) + GELU + Dropout(0.3)
            ↓
    Dense(128) + GELU + Dropout(0.4)
            ↓
    Output (Binary Classification)
```

**Design Notes**:
- Use different colors for different layer types
- Show complexity difference clearly
- Include key parameters (units, filters, heads)
- Use consistent styling

**Location**: In Methodology section (Model Architectures subsection)

---

## Figure 4: ROC Curves
**Purpose**: Visualize classification performance improvements

**Content**: 4 ROC curves on same plot
1. Tabular MLP - Before DC removal (ROC-AUC = 0.776)
2. Tabular MLP - After DC removal (ROC-AUC = 0.898)
3. Hybrid Model - Before improvements (ROC-AUC = 0.765)
4. Hybrid Model - After improvements (ROC-AUC = 0.893)

**Design Notes**:
- Use different line styles/colors for each curve
- Include diagonal reference line (random classifier)
- Annotate AUC values on plot
- Use legend to distinguish curves
- Highlight improvement (before → after)

**Legend**:
- Solid line: After improvements
- Dashed line: Before improvements
- Colors: Blue (Tabular MLP), Red (Hybrid Model)

**Location**: In Results section (Performance Improvements subsection)

---

## Figure 5: Confusion Matrices
**Purpose**: Show classification errors before/after improvements

**Content**: 4 confusion matrices (2×2 grid)

**Top Row: Tabular MLP**
- **Left**: Before DC removal
  ```
         Predicted
        Neg  Pos
  Neg   11   3
  Pos    2   5
  ```
- **Right**: After DC removal
  ```
         Predicted
        Neg  Pos
  Neg   11   3
  Pos    0   7  ← Perfect Recall!
  ```

**Bottom Row: Hybrid Model**
- **Left**: Before improvements
  ```
         Predicted
        Neg  Pos
  Neg   14   0  ← Perfect Precision
  Pos    3   4
  ```
- **Right**: After improvements
  ```
         Predicted
        Neg  Pos
  Neg   14   0  ← Perfect Precision
  Pos    2   5
  ```

**Design Notes**:
- Use color intensity for cell values (darker = higher)
- Highlight perfect metrics (bold borders)
- Include percentages in cells
- Show TP, TN, FP, FN clearly

**Location**: In Results section (Confusion Matrices subsection)

---

## Figure 6: Performance Comparison Bar Chart
**Purpose**: Compare all metrics across conditions

**Content**: Grouped bar chart

**X-axis**: Metrics (ROC-AUC, F1-Score, Recall, Precision, Accuracy)

**Y-axis**: Metric values (0 to 1)

**Groups**: 4 bars per metric
1. Tabular MLP - Before (light blue)
2. Tabular MLP - After (dark blue)
3. Hybrid Model - Before (light red)
4. Hybrid Model - After (dark red)

**Annotations**:
- Percentage improvements above bars
- Highlight perfect scores (1.0) with star
- Show error bars if available

**Example**:
```
ROC-AUC: [0.776] [0.898↑+15.7%] [0.765] [0.893↑+16.7%]
F1:      [0.667] [0.824↑+23.5%] [0.727] [0.833↑+14.6%]
Recall:  [0.714] [1.000*↑+40%]  [0.571] [0.714↑+25%]
```

**Design Notes**:
- Use consistent color scheme
- Group by metric, not by model
- Include improvement annotations
- Highlight perfect scores

**Location**: In Results section (Comparative Analysis subsection)

---

## Figure 7: Feature Importance
**Purpose**: Show which features were selected and their importance

**Content**: Horizontal bar chart

**Y-axis**: Features (5 selected features)
1. tp10_total_power
2. tp10_psd_mean
3. tp10_psd_std
4. tp10_psd_max
5. left_frontal_temporal_diff_beta

**X-axis**: Importance score (f_classif F-statistic)

**Design Notes**:
- Sort by importance (highest to lowest)
- Use color gradient (darker = more important)
- Include actual F-statistic values
- Annotate feature names clearly
- Show that tp10 features dominate (4/5 features)

**Location**: In Methodology section (Feature Selection subsection) or Results section

---

## TABLE SPECIFICATIONS

---

## Table 1: Dataset Characteristics
**Purpose**: Provide comprehensive dataset description

**Structure**:
| Characteristic | Value |
|----------------|-------|
| **Participants** | 21 patients |
| **Non-Remission** | 14 (66.7%) |
| **Remission** | 7 (33.3%) |
| **EEG Channels** | 4 (AF7, AF8, TP9, TP10) |
| **Channel Types** | Frontal (AF7, AF8), Temporal (TP9, TP10) |
| **Sampling Rate** | 256 Hz |
| **Window Size** | 10 seconds |
| **Samples per Window** | 2,560 |
| **Total Windows** | 1,203 |
| **Positive Windows** | 393 (32.7%) |
| **Negative Windows** | 810 (67.3%) |
| **Class Ratio** | 2.06:1 (negative:positive) |
| **Cross-Validation** | Leave-One-Patient-Out (21 folds) |
| **Evaluation Level** | Patient-level (aggregated from windows) |
| **Selected Features** | 5 (after feature selection) |

**Location**: In Methodology section (Dataset Description subsection)

---

## Table 2: Model Architectures
**Purpose**: Detailed architecture specifications

**Structure**: Two subtables (one per model)

### Tabular MLP Architecture
| Component | Specification |
|-----------|--------------|
| **Input Features** | 5 |
| **Hidden Layers** | [1024, 512, 256, 128] |
| **Activation** | ReLU |
| **Batch Normalization** | Yes (after each hidden layer) |
| **Dropout Rate** | 0.3 |
| **Output Layer** | Linear(128 → 2) |
| **Total Parameters** | ~700K |

### Hybrid CNN-LSTM Architecture
| Component | Specification |
|-----------|--------------|
| **Input Features** | 5 |
| **Feature Embedding** | Linear(5 → 128) + BatchNorm + ReLU |
| **CNN Blocks** | 3 blocks |
| **Block 1** | Filters: [64, 64], Kernels: [5, 3], Pool: 2 |
| **Block 2** | Filters: [128, 128], Kernels: [3, 3], Pool: 2 |
| **Block 3** | Filters: [256, 128], Kernels: [3, 1], Pool: 1 |
| **CNN Dropout** | [0.1, 0.2, 0.3] (progressive) |
| **Spatial Dropout** | Yes |
| **Gaussian Noise** | 0.005 |
| **LSTM Layers** | 2 layers, bidirectional |
| **LSTM Layer 1** | 128 units, return_sequences=True, dropout=0.2 |
| **LSTM Layer 2** | 64 units, return_sequences=False, dropout=0.3 |
| **Attention** | 8 heads, key_dim=64, positional encoding |
| **Fusion Strategy** | Concatenation |
| **Feature Pyramid** | Enabled |
| **Dense Layers** | [256, 128] units, GELU activation |
| **Dense Dropout** | [0.3, 0.4] |
| **Total Parameters** | ~2.5M |

**Location**: In Methodology section (Model Architectures subsection)

---

## Table 3: Performance Metrics (Main Results)
**Purpose**: Comprehensive performance comparison

**Structure**:
| Metric | Tabular MLP | Tabular MLP | Hybrid Model | Hybrid Model | Tabular MLP | Hybrid Model |
|        | Before      | After       | Before       | After        | Improvement | Improvement |
|--------|-------------|-------------|--------------|--------------|-------------|--------------|
| **ROC-AUC** | 0.776 | 0.898 | 0.765 | 0.893 | +0.122 (+15.7%) | +0.128 (+16.7%) |
| **F1-Score** | 0.667 | 0.824 | 0.727 | 0.833 | +0.157 (+23.5%) | +0.106 (+14.6%) |
| **Recall** | 0.714 | **1.000** | 0.571 | 0.714 | +0.286 (+40.0%) | +0.143 (+25.0%) |
| **Precision** | 0.625 | 0.700 | 1.000 | 1.000 | +0.075 (+12.0%) | 0.000 (0%) |
| **Accuracy** | 0.762 | 0.857 | 0.857 | 0.905 | +0.095 (+12.5%) | +0.048 (+5.6%) |
| **Window Accuracy** | 0.789 | 0.828 | - | - | +0.039 (+4.9%) | - |

**Design Notes**:
- Bold perfect scores (1.000)
- Highlight largest improvements
- Include percentage improvements
- Use consistent formatting

**Location**: In Results section (Performance Improvements subsection)

---

## Table 4: Confusion Matrices
**Purpose**: Detailed error analysis

**Structure**: 4 confusion matrices

### Tabular MLP - Before DC Removal
|        | Predicted Negative | Predicted Positive | Total |
|--------|-------------------|-------------------|-------|
| **Actual Negative** | 11 (TN) | 3 (FP) | 14 |
| **Actual Positive** | 2 (FN) | 5 (TP) | 7 |
| **Total** | 13 | 8 | 21 |

### Tabular MLP - After DC Removal
|        | Predicted Negative | Predicted Positive | Total |
|--------|-------------------|-------------------|-------|
| **Actual Negative** | 11 (TN) | 3 (FP) | 14 |
| **Actual Positive** | **0 (FN)** | **7 (TP)** | 7 |
| **Total** | 11 | 10 | 21 |

### Hybrid Model - Before Improvements
|        | Predicted Negative | Predicted Positive | Total |
|--------|-------------------|-------------------|-------|
| **Actual Negative** | **14 (TN)** | **0 (FP)** | 14 |
| **Actual Positive** | 3 (FN) | 4 (TP) | 7 |
| **Total** | 17 | 4 | 21 |

### Hybrid Model - After Improvements
|        | Predicted Negative | Predicted Positive | Total |
|--------|-------------------|-------------------|-------|
| **Actual Negative** | **14 (TN)** | **0 (FP)** | 14 |
| **Actual Positive** | 2 (FN) | 5 (TP) | 7 |
| **Total** | 16 | 5 | 21 |

**Design Notes**:
- Bold perfect metrics (0 FN or 0 FP)
- Include abbreviations (TN, TP, FP, FN)
- Show totals for clarity

**Location**: In Results section (Confusion Matrices subsection)

---

## Table 5: Architecture Changes (Hybrid Model)
**Purpose**: Document what changed between experiments

**Structure**:
| Component | Baseline | Enhanced | Rationale |
|-----------|----------|----------|-----------|
| **Activation (Dense)** | ReLU | GELU | Smoother gradients, better performance |
| **Attention Heads** | 4 | 8 | Increased capacity for feature learning |
| **Attention Key Dim** | 32 | 64 | Larger representation space |
| **Positional Encoding** | Disabled | Enabled | Temporal awareness |
| **Deterministic Training** | False | True | Reproducibility |
| **Random State** | 42 | 30 | Different initialization |

**Design Notes**:
- Highlight changes clearly
- Include rationale for each change
- Note that changes are confounded with DC removal

**Location**: In Methodology section (Model Architectures subsection) or Results section

---

## Table 6: Hyperparameters
**Purpose**: Complete hyperparameter specifications for reproducibility

**Structure**: Two subtables (one per model)

### Tabular MLP Hyperparameters
| Hyperparameter | Value |
|----------------|-------|
| **Hidden Layers** | [1024, 512, 256, 128] |
| **Dropout Rate** | 0.3 |
| **Batch Normalization** | Yes |
| **Optimizer** | AdamW |
| **Learning Rate** | 0.001 |
| **Weight Decay** | 0.01 |
| **Beta 1** | 0.9 |
| **Beta 2** | 0.999 |
| **Epsilon** | 1e-7 |
| **LR Schedule** | Cosine Annealing Warm Restarts |
| **Initial LR** | 0.001 |
| **Min LR** | 1e-6 |
| **Warmup Epochs** | 5 |
| **Cycle Length** | 20 |
| **Label Smoothing** | 0.05 |
| **Gradient Clip Norm** | 1.0 |
| **Batch Size** | 1024 |
| **Epochs** | 100 |
| **Early Stopping Patience** | 10 |
| **Early Stopping Monitor** | Loss |
| **Mixed Precision** | Yes |
| **Class Weight** | Balanced |
| **Random State** | 42 |

### Hybrid CNN-LSTM Hyperparameters
| Hyperparameter | Value |
|----------------|-------|
| **Feature Embedding** | Linear(5 → 128) |
| **CNN Blocks** | 3 (see Table 2) |
| **CNN Dropout** | [0.1, 0.2, 0.3] |
| **Spatial Dropout** | Yes |
| **Gaussian Noise** | 0.005 |
| **LSTM Units** | [128, 64] |
| **LSTM Bidirectional** | Yes |
| **LSTM Dropout** | [0.2, 0.3] |
| **LSTM Recurrent Dropout** | [0.1, 0.2] |
| **Attention Heads** | 8 |
| **Attention Key Dim** | 64 |
| **Attention Dropout** | 0.1 |
| **Positional Encoding** | Yes |
| **Fusion Strategy** | Concat |
| **Feature Pyramid** | Yes |
| **Dense Layers** | [256, 128] |
| **Dense Activation** | GELU |
| **Dense Dropout** | [0.3, 0.4] |
| **Optimizer** | AdamW |
| **Learning Rate** | 0.001 |
| **Weight Decay** | 0.01 |
| **LR Schedule** | Cosine Annealing Warm Restarts |
| **Label Smoothing** | 0.05 |
| **Gradient Clip Norm** | 1.0 |
| **Batch Size** | 1024 |
| **Epochs** | 100 |
| **Early Stopping Patience** | 10 |
| **Early Stopping Monitor** | val_f1_macro |
| **Mixed Precision** | Yes |
| **Deterministic Training** | Yes |
| **Random State** | 30 |

**Location**: In Methodology section (Training Configuration subsection) or Appendix

---

## FIGURE/TABLE PLACEMENT STRATEGY

### Introduction Section
- None (keep focused on problem/motivation)

### Related Work Section
- None (text-focused)

### Methodology Section
- **Figure 1**: Pipeline diagram (beginning)
- **Figure 2**: DC offset removal visualization (DC removal subsection)
- **Figure 3**: Model architectures (Model architectures subsection)
- **Figure 7**: Feature importance (Feature selection subsection)
- **Table 1**: Dataset characteristics (Dataset description subsection)
- **Table 2**: Model architectures (Model architectures subsection)
- **Table 5**: Architecture changes (Model architectures subsection)
- **Table 6**: Hyperparameters (Training configuration subsection)

### Results Section
- **Figure 4**: ROC curves (Performance improvements subsection)
- **Figure 5**: Confusion matrices (Confusion matrices subsection)
- **Figure 6**: Performance comparison bar chart (Comparative analysis subsection)
- **Table 3**: Performance metrics (Performance improvements subsection)
- **Table 4**: Confusion matrices (Confusion matrices subsection)

### Discussion Section
- None (text-focused, may reference figures/tables)

### Conclusion Section
- None (text-focused)

---

## DESIGN GUIDELINES

### Color Scheme
- **Primary Colors**: Blue (Tabular MLP), Red (Hybrid Model)
- **Shades**: Light (before), Dark (after)
- **Highlights**: Green (improvements), Yellow (perfect scores)

### Typography
- **Headers**: Bold, 12-14pt
- **Body Text**: Regular, 10-11pt
- **Annotations**: Italic, 9-10pt
- **Numbers**: Monospace font for alignment

### Consistency
- Same color scheme across all figures
- Same formatting style across all tables
- Consistent axis labels and legends
- Uniform figure sizes (where possible)

### Accessibility
- High contrast colors
- Clear labels and legends
- Descriptive captions
- Avoid color-only encoding (use patterns/labels too)

