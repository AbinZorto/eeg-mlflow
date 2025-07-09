# Deep Learning Models for EEG Analysis

This document describes the deep learning capabilities added to the EEG analysis pipeline, with comprehensive overfitting prevention measures.

## Overview

The deep learning extension provides neural network models that integrate seamlessly with the existing training architecture while implementing robust overfitting prevention strategies.

## Available Models

### 1. PyTorch MLP Classifier (`pytorch_mlp`)

- Multi-layer perceptron implemented in PyTorch
- Supports customizable architecture
- Built-in regularization and optimization features

### 2. Keras MLP Classifier (`keras_mlp`)

- Multi-layer perceptron implemented in TensorFlow/Keras
- Similar capabilities to PyTorch version
- Different regularization options (L1/L2)

## Overfitting Prevention Strategies

### 1. Cross-Validation Architecture (Inherited)

The deep learning models leverage the existing robust validation framework:

- **Leave-One-Group-Out (LOGO) Cross-Validation**: Ensures no patient data leakage
- **Patient-Level Grouping**: Prevents temporal data leakage within patients
- **Independent Test Sets**: Each fold uses completely unseen patient data

### 2. Model-Level Regularization

#### Dropout Regularization

- **Default**: 40% dropout rate
- **Purpose**: Prevents co-adaptation of neurons
- **Implementation**: Applied after each hidden layer

#### Weight Decay (L2 Regularization)

- **Default**: 0.01 weight decay
- **Purpose**: Prevents large weights that can lead to overfitting
- **Implementation**: Built into optimizer

#### Batch Normalization

- **Default**: Enabled
- **Purpose**: Stabilizes training and acts as implicit regularization
- **Implementation**: Applied before activation functions

### 3. Training Control

#### Early Stopping

- **Default**: 25 epochs patience
- **Purpose**: Stops training when validation loss stops improving
- **Implementation**: Monitors training loss with patience mechanism

#### Learning Rate Scheduling

- **Strategy**: Reduce on plateau
- **Factor**: 0.5 reduction when loss plateaus
- **Purpose**: Fine-tunes learning in later stages

#### Gradient Clipping

- **Threshold**: 1.0 max norm
- **Purpose**: Prevents exploding gradients
- **Implementation**: Applied during backpropagation

### 4. Data-Level Protection

#### Feature Scaling

- **Method**: StandardScaler (zero mean, unit variance)
- **Purpose**: Ensures stable gradient flow
- **Implementation**: Fitted on training data, applied to test data

#### Batch Size Adaptation

- **Small Datasets**: Automatically reduces batch size
- **Purpose**: Ensures stable gradient estimates
- **Implementation**: `min(batch_size, len(X) // 4)` for datasets < 100 samples

#### Class Balancing

- **Method**: Balanced class weights
- **Purpose**: Handles class imbalance common in medical data
- **Implementation**: Integrated into loss function

## Installation

1. Install base requirements:

```bash
pip install -r requirements.txt
```

2. Install deep learning dependencies:

```bash
pip install -r requirements_deep_learning.txt
```

### GPU Support (Optional)

For faster training, install CUDA-enabled versions:

#### PyTorch with CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### TensorFlow with GPU:

```bash
pip install tensorflow[and-cuda]
```

## Usage

### Basic Training

Train a PyTorch MLP model:

```bash
python run_pipeline.py --config configs/window_model_config.yaml train \
  --level window \
  --model-type pytorch_mlp
```

Train a Keras MLP model:

```bash
python run_pipeline.py --config configs/window_model_config.yaml train \
  --level window \
  --model-type keras_mlp
```

### With Feature Selection

Train with feature selection for better generalization:

```bash
python run_pipeline.py --config configs/window_model_config.yaml train \
  --level window \
  --model-type pytorch_mlp \
  --enable-feature-selection \
  --n-features-select 20 \
  --fs-method select_k_best_f_classif
```

### Run All Experiments

The experiment script now includes deep learning models:

```bash
./run_all_experiments.sh
```

This will run 42 total experiments (6 models × 7 configurations each).

## Configuration

### Model Parameters

Deep learning models are configured in `configs/window_model_config.yaml`:

```yaml
deep_learning:
  pytorch_mlp:
    hidden_layers: [128, 64, 32] # Network architecture
    dropout_rate: 0.4 # Dropout probability
    weight_decay: 0.01 # L2 regularization
    learning_rate: 0.0005 # Learning rate
    batch_size: 32 # Batch size
    epochs: 300 # Maximum epochs
    early_stopping_patience: 25 # Early stopping patience
    batch_norm: true # Batch normalization
    activation: "relu" # Activation function
    optimizer: "adam" # Optimizer
    class_weight: "balanced" # Handle class imbalance
    random_state: 42 # Reproducibility

  keras_mlp:
    hidden_layers: [128, 64, 32] # Network architecture
    dropout_rate: 0.4 # Dropout probability
    l1_reg: 0.005 # L1 regularization
    l2_reg: 0.01 # L2 regularization
    learning_rate: 0.0005 # Learning rate
    batch_size: 32 # Batch size
    epochs: 300 # Maximum epochs
    early_stopping_patience: 25 # Early stopping patience
    batch_norm: true # Batch normalization
    activation: "relu" # Activation function
    optimizer: "adam" # Optimizer
    class_weight: "balanced" # Handle class imbalance
    random_state: 42 # Reproducibility
```

### Recommended Settings by Dataset Size

#### Small Datasets (< 500 samples)

```yaml
hidden_layers: [32, 16]
dropout_rate: 0.5
epochs: 200
early_stopping_patience: 30
```

#### Medium Datasets (500-2000 samples)

```yaml
hidden_layers: [64, 32, 16]
dropout_rate: 0.4
epochs: 250
early_stopping_patience: 25
```

#### Large Datasets (> 2000 samples)

```yaml
hidden_layers: [128, 64, 32]
dropout_rate: 0.3
epochs: 300
early_stopping_patience: 20
```

## Architecture Details

### Network Design Principles

1. **Progressive Dimensionality Reduction**: Each layer reduces the feature space
2. **Sufficient Depth**: 3-4 layers for complex pattern learning
3. **Appropriate Width**: Balances capacity with overfitting risk

### Regularization Schedule

```
Input → Linear → BatchNorm → ReLU → Dropout →
       Linear → BatchNorm → ReLU → Dropout →
       Linear → BatchNorm → ReLU → Dropout →
       Linear (Output)
```

### Loss Function

- **CrossEntropyLoss** with class weights for imbalanced data
- **Automatic weight calculation** based on class frequencies

## Performance Monitoring

### MLflow Integration

All deep learning experiments are logged to MLflow with:

- **Model parameters**: Architecture, regularization settings
- **Training metrics**: Loss curves, early stopping info
- **Validation metrics**: Accuracy, precision, recall, F1, AUC
- **Feature information**: Selected features, feature importance
- **Model artifacts**: Trained models, predictions, metadata

### Overfitting Detection

Monitor these metrics to detect overfitting:

1. **Early stopping triggered**: Model stopped before max epochs
2. **High training vs. validation gap**: Indicates overfitting
3. **Feature selection effectiveness**: Reduced features improve generalization
4. **Cross-validation consistency**: Low variance across folds indicates good generalization

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:

   - Reduce `batch_size` in config
   - Use CPU-only versions

2. **Convergence Issues**:

   - Reduce `learning_rate`
   - Increase `early_stopping_patience`
   - Check feature scaling

3. **Poor Performance**:
   - Enable feature selection
   - Adjust regularization parameters
   - Try different architectures

### Performance Tips

1. **Feature Selection**: Almost always improves generalization
2. **Class Balancing**: Essential for medical datasets
3. **Early Stopping**: Prevents overfitting automatically
4. **Cross-Validation**: Use all folds for final evaluation

## Comparison with Traditional Models

### Advantages of Deep Learning

- **Non-linear patterns**: Can capture complex feature interactions
- **Automatic feature learning**: Reduces need for manual feature engineering
- **Scalability**: Performance improves with more data

### When to Use Traditional Models

- **Small datasets**: Random Forest often outperforms on < 500 samples
- **Interpretability**: Tree-based models provide clear feature importance
- **Fast training**: Traditional models train much faster

### Recommended Approach

1. **Start with traditional models** for baseline performance
2. **Try deep learning** if you have sufficient data (> 500 samples)
3. **Use ensemble methods** combining both approaches
4. **Always use cross-validation** for fair comparison

## Research Considerations

### Publication Guidelines

- **Report all hyperparameters** used in final models
- **Include cross-validation results** not just test set performance
- **Document overfitting prevention** measures taken
- **Compare with baseline methods** using same validation scheme

### Reproducibility

- **Fixed random seeds** in all models (set to 42 by default)
- **Version control** of data and code
- **Environment documentation** via requirements files
- **MLflow tracking** for experiment reproducibility
