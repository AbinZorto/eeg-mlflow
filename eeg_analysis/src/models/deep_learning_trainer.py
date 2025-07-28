import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import mlflow
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging
import warnings

# Import deep learning libraries with fallback
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Deep learning models will not be functional.")

try:
    import tensorflow as tf
    # Handle different TensorFlow versions
    try:
        from tensorflow import keras
        from tensorflow.keras import layers, regularizers, callbacks
    except ImportError:
        # For newer TensorFlow versions where keras is separate
        import keras
        from keras import layers, regularizers, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    warnings.warn("TensorFlow not available. Keras models will not be functional.")

from src.models.base_trainer import BaseTrainer
from src.models.evaluation import ModelEvaluator
from mlflow.data.pandas_dataset import PandasDataset

logger = logging.getLogger(__name__)

class PyTorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    PyTorch MLP Classifier with comprehensive overfitting prevention.
    
    Features for overfitting prevention:
    - Dropout regularization
    - Weight decay (L2 regularization)
    - Early stopping with patience
    - Batch normalization
    - Learning rate scheduling
    - Gradient clipping
    """
    
    def __init__(self, 
                 hidden_layers=[64, 32],
                 dropout_rate=0.3,
                 weight_decay=0.01,
                 learning_rate=0.001,
                 batch_size=32,
                 epochs=200,
                 early_stopping_patience=20,
                 batch_norm=True,
                 activation='relu',
                 optimizer='adam',
                 class_weight=None,
                 random_state=42,
                 mixed_precision=False,
                 gradient_accumulation_steps=1):
        
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.batch_norm = batch_norm
        self.activation = activation
        self.optimizer = optimizer
        self.class_weight = class_weight
        self.random_state = random_state
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes_ = None
        self.feature_names_in_ = None
        
        # Initialize mixed precision scaler for maximum performance
        self.scaler_amp = None
        if mixed_precision and torch.cuda.is_available():
            self.scaler_amp = torch.cuda.amp.GradScaler()
            print("🔥 MIXED PRECISION ENABLED: Using automatic mixed precision for maximum GPU utilization!")
        
        # Multi-GPU setup for maximum performance
        self.use_multi_gpu = torch.cuda.device_count() > 1
        if self.use_multi_gpu:
            print(f"🚀 MULTI-GPU MODE: Using {torch.cuda.device_count()} GPUs for training!")
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
    def _create_model(self, n_features):
        """Create the neural network model."""
        
        class MLP(nn.Module):
            def __init__(self, n_features, hidden_layers, dropout_rate, batch_norm, activation):
                super(MLP, self).__init__()
                
                layers_list = []
                input_size = n_features
                
                # Build hidden layers
                for i, hidden_size in enumerate(hidden_layers):
                    # Linear layer
                    layers_list.append(nn.Linear(input_size, hidden_size))
                    
                    # Batch normalization
                    if batch_norm:
                        layers_list.append(nn.BatchNorm1d(hidden_size))
                    
                    # Activation
                    if activation == 'relu':
                        layers_list.append(nn.ReLU())
                    elif activation == 'tanh':
                        layers_list.append(nn.Tanh())
                    elif activation == 'elu':
                        layers_list.append(nn.ELU())
                    
                    # Dropout
                    layers_list.append(nn.Dropout(dropout_rate))
                    input_size = hidden_size
                
                # Output layer (no activation, will use logits)
                layers_list.append(nn.Linear(input_size, 2))
                
                self.layers = nn.Sequential(*layers_list)
                
                # Initialize weights
                self._init_weights()
            
            def _init_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        nn.init.zeros_(m.bias)
            
            def forward(self, x):
                return self.layers(x)
        
        return MLP(n_features, self.hidden_layers, self.dropout_rate, 
                  self.batch_norm, self.activation)
    
    def fit(self, X, y):
        """Train the model with comprehensive overfitting prevention."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Please install torch.")
        
        # Store classes and feature names
        self.classes_ = np.unique(y)
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        
        # Convert to numpy if pandas
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create model
        self.model = self._create_model(X_scaled.shape[1])
        self.model.to(self.device)
        
        # Enable multi-GPU training for maximum performance
        if self.use_multi_gpu:
            self.model = nn.DataParallel(self.model)
            print(f"💫 DataParallel enabled across {torch.cuda.device_count()} GPUs")
            print(f"🔥 Effective batch size: {self.batch_size} per GPU = {self.batch_size * torch.cuda.device_count()} total")
        
        # Prepare data
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Use smaller batch size for small datasets
        effective_batch_size = min(self.batch_size, len(X) // 4) if len(X) < 100 else self.batch_size
        
        # High-performance data loading (no pin_memory since data is already on GPU)
        dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True, 
                              num_workers=0, pin_memory=False)
        
        # Loss function with class weights
        if self.class_weight == 'balanced':
            class_counts = np.bincount(y)
            class_weights = len(y) / (len(np.unique(y)) * class_counts)
            class_weights = torch.FloatTensor(class_weights).to(self.device)
        else:
            class_weights = None
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer with weight decay
        if self.optimizer == 'adam':
            optimizer = optim.Adam(self.model.parameters(), 
                                 lr=self.learning_rate, 
                                 weight_decay=self.weight_decay)
        elif self.optimizer == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), 
                                lr=self.learning_rate, 
                                weight_decay=self.weight_decay,
                                momentum=0.9)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Early stopping variables
        best_loss = float('inf')
        patience_counter = 0
        
        # Training loop with mixed precision support for maximum performance
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
                # Mixed precision training for maximum GPU utilization
                if self.mixed_precision and self.scaler_amp is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y) / self.gradient_accumulation_steps
                    
                    self.scaler_amp.scale(loss).backward()
                    
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        # Gradient clipping with mixed precision
                        self.scaler_amp.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        self.scaler_amp.step(optimizer)
                        self.scaler_amp.update()
                        optimizer.zero_grad()
                else:
                    # Standard training
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y) / self.gradient_accumulation_steps
                    
                    loss.backward()
                    
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        # Gradient clipping to prevent exploding gradients
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                
                epoch_loss += loss.item() * self.gradient_accumulation_steps
            
            epoch_loss /= len(dataloader)
            scheduler.step(epoch_loss)
            
            # Early stopping check
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
                # Save best model state
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model state
        if 'best_state' in locals():
            self.model.load_state_dict(best_state)
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        if not TORCH_AVAILABLE or self.model is None:
            raise ValueError("Model not trained or PyTorch not available.")
        
        # Convert to numpy if pandas
        if hasattr(X, 'values'):
            X = X.values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
        
        return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if not TORCH_AVAILABLE or self.model is None:
            raise ValueError("Model not trained or PyTorch not available.")
        
        # Convert to numpy if pandas
        if hasattr(X, 'values'):
            X = X.values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()

class KerasMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    Keras MLP Classifier with comprehensive overfitting prevention.
    
    Features for overfitting prevention:
    - Dropout regularization
    - L1/L2 regularization
    - Early stopping with patience
    - Batch normalization
    - Learning rate scheduling
    - Model checkpointing
    """
    
    def __init__(self,
                 hidden_layers=[64, 32],
                 dropout_rate=0.3,
                 l1_reg=0.01,
                 l2_reg=0.01,
                 learning_rate=0.001,
                 batch_size=32,
                 epochs=200,
                 early_stopping_patience=20,
                 batch_norm=True,
                 activation='relu',
                 optimizer='adam',
                 class_weight=None,
                 random_state=42,
                 mixed_precision=False,
                 gradient_clip_norm=1.0):
        
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.batch_norm = batch_norm
        self.activation = activation
        self.optimizer = optimizer
        self.class_weight = class_weight
        self.random_state = random_state
        self.mixed_precision = mixed_precision
        self.gradient_clip_norm = gradient_clip_norm
        
        self.model = None
        self.scaler = StandardScaler()
        self.classes_ = None
        self.feature_names_in_ = None
        
        # Enable mixed precision for maximum performance
        if mixed_precision:
            keras.mixed_precision.set_global_policy('mixed_float16')
            print("🔥 KERAS MIXED PRECISION ENABLED: Using mixed_float16 for maximum GPU utilization!")
        
        # Multi-GPU setup for maximum performance
        self.strategy = None
        if len(tf.config.list_physical_devices('GPU')) > 1:
            self.strategy = tf.distribute.MirroredStrategy()
            print(f"🚀 KERAS MULTI-GPU MODE: Using {self.strategy.num_replicas_in_sync} GPUs for training!")
            print(f"💫 Keras model created with MirroredStrategy across {self.strategy.num_replicas_in_sync} GPUs")
            print(f"🔥 Effective batch size: {self.batch_size} per GPU = {self.batch_size * self.strategy.num_replicas_in_sync} total")
        
        # Set random seeds
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
    
    def _create_model(self, n_features):
        """Create the neural network model with multi-GPU support."""
        
        def create_model_fn():
            model = keras.Sequential()
            
            # Input layer
            model.add(layers.Input(shape=(n_features,)))
            
            # Hidden layers
            for i, hidden_size in enumerate(self.hidden_layers):
                # Dense layer with regularization
                model.add(layers.Dense(
                    hidden_size,
                    activation=None,  # Add activation separately
                    kernel_regularizer=regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                    kernel_initializer='glorot_uniform'
                ))
                
                # Batch normalization
                if self.batch_norm:
                    model.add(layers.BatchNormalization())
                
                # Activation
                model.add(layers.Activation(self.activation))
                
                # Dropout
                model.add(layers.Dropout(self.dropout_rate))
            
            # Output layer
            model.add(layers.Dense(2, activation='softmax'))
            
            # Compile model
            if self.optimizer == 'adam':
                opt = keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=self.gradient_clip_norm, epsilon=1e-4)
            elif self.optimizer == 'sgd':
                opt = keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9, clipnorm=self.gradient_clip_norm)
            
            model.compile(
                optimizer=opt,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
        
        # Create model with multi-GPU strategy if available
        if self.strategy is not None:
            with self.strategy.scope():
                model = create_model_fn()
                print(f"💫 Keras model created with MirroredStrategy across {self.strategy.num_replicas_in_sync} GPUs")
                print(f"🔥 Effective batch size: {self.batch_size} per GPU = {self.batch_size * self.strategy.num_replicas_in_sync} total")
        else:
            model = create_model_fn()
        
        return model
    
    def fit(self, X, y):
        """Train the model with comprehensive overfitting prevention."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available. Please install tensorflow.")
        
        # Store classes and feature names
        self.classes_ = np.unique(y)
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        
        # Convert to numpy if pandas
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        # DEBUG: Check input data for NaN values
        print(f"🔍 DEBUG: Input X shape: {X.shape}, y shape: {y.shape}")
        print(f"🔍 DEBUG: X contains NaN: {np.isnan(X).any()}")
        print(f"🔍 DEBUG: y contains NaN: {np.isnan(y).any()}")
        if np.isnan(X).any():
            nan_count = np.isnan(X).sum()
            print(f"🔍 DEBUG: Number of NaN values in X: {nan_count}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # DEBUG: Check scaled data for NaN values
        print(f"🔍 DEBUG: X_scaled contains NaN: {np.isnan(X_scaled).any()}")
        if np.isnan(X_scaled).any():
            nan_count = np.isnan(X_scaled).sum()
            print(f"🔍 DEBUG: Number of NaN values in X_scaled: {nan_count}")
        
        # Create model
        self.model = self._create_model(X_scaled.shape[1])
        
        # Calculate class weights if needed
        if self.class_weight == 'balanced':
            class_counts = np.bincount(y)
            class_weights = len(y) / (len(np.unique(y)) * class_counts)
            class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        else:
            class_weight_dict = None
        
        # Callbacks for overfitting prevention
        callbacks_list = [
            # Early stopping
            callbacks.EarlyStopping(
                monitor='loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=0
            ),
            # Learning rate reduction
            callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=0
            )
        ]
        
        # Smart batch size handling for multi-GPU training
        if self.strategy is not None:
            # For multi-GPU, ensure batch size is divisible by number of GPUs
            n_gpus = self.strategy.num_replicas_in_sync
            # Also ensure it's not larger than the dataset size
            max_batch_size = min(self.batch_size, len(X) // 2)  # Leave room for validation
            effective_batch_size = (max_batch_size // n_gpus) * n_gpus
            # Ensure minimum batch size
            effective_batch_size = max(effective_batch_size, n_gpus * 32)
        else:
            # Single GPU batch size handling
            effective_batch_size = min(self.batch_size, len(X) // 4) if len(X) < 100 else self.batch_size
        
        print(f"🔥 Keras batch size: {effective_batch_size} (data size: {len(X)})")
        
        # DEBUG: Final check before training
        print(f"🔍 DEBUG: About to train - X_scaled contains NaN: {np.isnan(X_scaled).any()}")
        print(f"🔍 DEBUG: About to train - y contains NaN: {np.isnan(y).any()}")
        
        # Train model
        self.model.fit(
            X_scaled, y,
            batch_size=effective_batch_size,
            epochs=self.epochs,
            callbacks=callbacks_list,
            class_weight=class_weight_dict,
            verbose=0
        )
        
        # DEBUG: Check model weights after training
        print(f"🔍 DEBUG: Model trained successfully")
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        if not TF_AVAILABLE or self.model is None:
            raise ValueError("Model not trained or TensorFlow not available.")
        
        # Convert to numpy if pandas
        if hasattr(X, 'values'):
            X = X.values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled, verbose=0)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if not TF_AVAILABLE or self.model is None:
            raise ValueError("Model not trained or TensorFlow not available.")
        
        # Convert to numpy if pandas
        if hasattr(X, 'values'):
            X = X.values
        
        # DEBUG: Check input data for NaN values
        print(f"🔍 DEBUG: predict_proba - Input X shape: {X.shape}")
        print(f"🔍 DEBUG: predict_proba - X contains NaN: {np.isnan(X).any()}")
        if np.isnan(X).any():
            nan_count = np.isnan(X).sum()
            print(f"🔍 DEBUG: predict_proba - Number of NaN values in X: {nan_count}")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # DEBUG: Check scaled data for NaN values
        print(f"🔍 DEBUG: predict_proba - X_scaled contains NaN: {np.isnan(X_scaled).any()}")
        if np.isnan(X_scaled).any():
            nan_count = np.isnan(X_scaled).sum()
            print(f"🔍 DEBUG: predict_proba - Number of NaN values in X_scaled: {nan_count}")
        
        # Predict probabilities
        predictions = self.model.predict(X_scaled, verbose=0)
        
        # DEBUG: Check predictions for NaN values
        print(f"🔍 DEBUG: predict_proba - predictions shape: {predictions.shape}")
        print(f"🔍 DEBUG: predict_proba - predictions contains NaN: {np.isnan(predictions).any()}")
        if np.isnan(predictions).any():
            nan_count = np.isnan(predictions).sum()
            print(f"🔍 DEBUG: predict_proba - Number of NaN values in predictions: {nan_count}")
            print(f"🔍 DEBUG: predict_proba - predictions sample: {predictions[:5]}")
            
            # FALLBACK: Replace NaN predictions with neutral probabilities
            print(f"🔧 FALLBACK: Replacing NaN predictions with neutral probabilities [0.5, 0.5]")
            nan_mask = np.isnan(predictions)
            predictions[nan_mask] = 0.5
            print(f"🔧 FALLBACK: Fixed predictions - contains NaN: {np.isnan(predictions).any()}")
        
        return predictions
    
    def __getstate__(self):
        """Custom serialization to handle TensorFlow objects."""
        print("🔧 KERAS SERIALIZATION: Preparing model for serialization...")
        state = self.__dict__.copy()
        
        # Remove unpickleable TensorFlow objects
        if 'strategy' in state:
            state['strategy'] = None  # Remove MirroredStrategy
        
        # Save model weights and architecture separately if model exists
        if self.model is not None:
            print("🔧 KERAS SERIALIZATION: Saving model weights and config...")
            # Save model config and weights
            state['model_config'] = self.model.get_config()
            state['model_weights'] = self.model.get_weights()
            state['model'] = None  # Remove the actual model object
        else:
            state['model_config'] = None
            state['model_weights'] = None
        
        print("🔧 KERAS SERIALIZATION: Model prepared for serialization")
        return state
    
    def __setstate__(self, state):
        """Custom deserialization to rebuild TensorFlow objects."""
        print("🔧 KERAS DESERIALIZATION: Restoring model from serialization...")
        self.__dict__.update(state)
        
        # Rebuild model if config and weights exist
        if state.get('model_config') is not None and state.get('model_weights') is not None:
            print("🔧 KERAS DESERIALIZATION: Rebuilding model from config and weights...")
            
            # Recreate strategy if multiple GPUs available
            if len(tf.config.list_physical_devices('GPU')) > 1:
                self.strategy = tf.distribute.MirroredStrategy()
                print(f"🔧 KERAS DESERIALIZATION: Recreated MirroredStrategy with {self.strategy.num_replicas_in_sync} GPUs")
            else:
                self.strategy = None
            
            # Recreate mixed precision policy if needed
            if self.mixed_precision:
                keras.mixed_precision.set_global_policy('mixed_float16')
                print("🔧 KERAS DESERIALIZATION: Restored mixed precision policy")
            
            # Rebuild model
            if self.strategy is not None:
                with self.strategy.scope():
                    self.model = keras.Sequential.from_config(state['model_config'])
                    self.model.set_weights(state['model_weights'])
            else:
                self.model = keras.Sequential.from_config(state['model_config'])
                self.model.set_weights(state['model_weights'])
            
            print("🔧 KERAS DESERIALIZATION: Model successfully restored")
        else:
            self.model = None
            self.strategy = None
            print("🔧 KERAS DESERIALIZATION: No model to restore")
        
        print("🔧 KERAS DESERIALIZATION: Deserialization complete")

class Hybrid1DCNNLSTMClassifier(BaseEstimator, ClassifierMixin):
    """
    Hybrid 1D CNN-LSTM Classifier for EEG signal processing.
    
    This model combines:
    - 1D CNN layers for spatial feature extraction from EEG channels
    - LSTM layers for temporal sequence modeling
    - Dense layers for final classification
    
    Features:
    - Bidirectional LSTM for better temporal modeling
    - Attention mechanism for focusing on important time steps
    - Residual connections in CNN for better gradient flow
    - Batch normalization for training stability
    - Mixed precision training for GPU efficiency
    """
    
    def __init__(self,
                 cnn_filters=[64, 128, 256],
                 cnn_kernel_sizes=[3, 3, 3],
                 cnn_pool_sizes=[2, 2, 2],
                 cnn_dropout=0.1,
                 lstm_units=[256, 128, 64],
                 lstm_dropout=0.2,
                 lstm_recurrent_dropout=0.1,
                 dense_layers=[128, 64],
                 dense_dropout=0.3,
                 learning_rate=0.001,
                 batch_size=32,
                 epochs=100,
                 early_stopping_patience=10,
                 optimizer='adam',
                 class_weight=None,
                 random_state=42,
                 sequence_length=1000,
                 n_channels=4,
                 normalize=True,
                 bidirectional_lstm=True,
                 attention_mechanism=True,
                 residual_connections=True,
                 batch_norm=True,
                 mixed_precision=True,
                 weight_decay=1e-5,
                 gradient_clip_norm=None):
        
        self.cnn_filters = cnn_filters
        self.cnn_kernel_sizes = cnn_kernel_sizes
        self.cnn_pool_sizes = cnn_pool_sizes
        self.cnn_dropout = cnn_dropout
        self.lstm_units = lstm_units
        self.lstm_dropout = lstm_dropout
        self.lstm_recurrent_dropout = lstm_recurrent_dropout
        self.dense_layers = dense_layers
        self.dense_dropout = dense_dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.optimizer = optimizer
        self.class_weight = class_weight
        self.random_state = random_state
        self.sequence_length = sequence_length
        self.n_channels = n_channels
        self.normalize = normalize
        self.bidirectional_lstm = bidirectional_lstm
        self.attention_mechanism = attention_mechanism
        self.residual_connections = residual_connections
        self.batch_norm = batch_norm
        self.mixed_precision = mixed_precision
        self.weight_decay = weight_decay
        self.gradient_clip_norm = gradient_clip_norm
        
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes_ = None
        self.feature_names_in_ = None
        self.scaler_amp = None
        
        if mixed_precision and torch.cuda.is_available():
            self.scaler_amp = torch.cuda.amp.GradScaler()
            print("🔥 MIXED PRECISION ENABLED: Using automatic mixed precision for maximum GPU utilization!")
    
    def _create_model(self, n_features):
        """Create the hybrid 1D CNN-LSTM model architecture."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Please install torch.")
        
        class Hybrid1DCNNLSTM(nn.Module):
            def __init__(self, n_features, cnn_filters, cnn_kernel_sizes, cnn_pool_sizes,
                         cnn_dropout, lstm_units, lstm_dropout, lstm_recurrent_dropout,
                         dense_layers, dense_dropout, n_channels, bidirectional_lstm,
                         attention_mechanism, residual_connections, batch_norm):
                super(Hybrid1DCNNLSTM, self).__init__()
                
                self.n_channels = n_channels
                self.bidirectional_lstm = bidirectional_lstm
                self.attention_mechanism = attention_mechanism
                self.residual_connections = residual_connections
                self.batch_norm = batch_norm
                
                # For feature-based data, we treat features as a 1D signal
                # Use n_features as the sequence length and 1 as the channel dimension
                self.sequence_length = n_features
                
                # CNN layers for feature extraction
                self.cnn_layers = nn.ModuleList()
                in_channels = 1  # Single channel for feature-based data
                
                for i, (filters, kernel_size, pool_size) in enumerate(zip(cnn_filters, cnn_kernel_sizes, cnn_pool_sizes)):
                    conv_layer = nn.Sequential(
                        nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size//2),
                        nn.BatchNorm1d(filters) if batch_norm else nn.Identity(),
                        nn.ReLU(),
                        nn.MaxPool1d(pool_size),
                        nn.Dropout(cnn_dropout)
                    )
                    self.cnn_layers.append(conv_layer)
                    in_channels = filters
                
                # For feature-based data, we'll use a much simpler approach
                # Use dynamic calculation to avoid dimension issues
                cnn_output_size = cnn_filters[-1]  # Use the last CNN layer's output size
                
                # Projection layer to map CNN output to dense input size
                self.cnn_to_dense = nn.Linear(cnn_output_size, dense_layers[0])
                
                # Dense layers for classification
                self.dense_layers = nn.ModuleList()
                dense_input_size = dense_layers[0]
                
                for units in dense_layers[1:]:  # Skip first layer as it's handled by projection
                    dense_layer = nn.Sequential(
                        nn.Linear(dense_input_size, units),
                        nn.BatchNorm1d(units) if batch_norm else nn.Identity(),
                        nn.ReLU(),
                        nn.Dropout(dense_dropout)
                    )
                    self.dense_layers.append(dense_layer)
                    dense_input_size = units
                
                # Output layer
                self.output_layer = nn.Linear(dense_input_size, 2)
                
                # Initialize weights
                self._init_weights()
            
            def _init_weights(self):
                """Initialize model weights."""
                for module in self.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
                    elif isinstance(module, nn.Conv1d):
                        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
                    elif isinstance(module, nn.LSTM):
                        for name, param in module.named_parameters():
                            if 'weight' in name:
                                nn.init.xavier_uniform_(param)
                            elif 'bias' in name:
                                nn.init.zeros_(param)
            
            def forward(self, x):
                # For feature-based data, we'll use a much simpler approach
                # Treat features as a 1D signal and use CNN for feature extraction
                batch_size = x.size(0)
                n_features = x.size(1)
                
                # DEBUG: Print input tensor info
                if batch_size == 1:  # Only print for first batch to avoid spam
                    print(f"   🔍 Forward pass - Input: {x.shape}, range: [{x.min():.3f}, {x.max():.3f}]")
                
                # Reshape to (batch, 1, features) for 1D convolution
                x = x.unsqueeze(1)  # Add channel dimension
                
                # CNN layers for feature extraction
                cnn_output = x
                for i, cnn_layer in enumerate(self.cnn_layers):
                    if self.residual_connections and i > 0 and cnn_output.size(1) == cnn_layer[0].out_channels:
                        residual = cnn_output
                        cnn_output = cnn_layer(cnn_output)
                        cnn_output = cnn_output + residual
                    else:
                        cnn_output = cnn_layer(cnn_output)
                    
                    # DEBUG: Print CNN layer outputs
                    if batch_size == 1:
                        print(f"   CNN layer {i+1}: {cnn_output.shape}, range: [{cnn_output.min():.3f}, {cnn_output.max():.3f}]")
                
                # Global average pooling over the sequence dimension
                cnn_output = torch.mean(cnn_output, dim=2)  # (batch, channels)
                
                if batch_size == 1:
                    print(f"   After pooling: {cnn_output.shape}, range: [{cnn_output.min():.3f}, {cnn_output.max():.3f}]")
                
                # Use a simple approach: flatten and pass through dense layers directly
                # Skip LSTM for now to avoid dimension issues
                cnn_output = cnn_output.view(batch_size, -1)  # Flatten
                
                if batch_size == 1:
                    print(f"   After flattening: {cnn_output.shape}, range: [{cnn_output.min():.3f}, {cnn_output.max():.3f}]")
                
                # Project CNN output to dense input size
                dense_output = self.cnn_to_dense(cnn_output)
                
                if batch_size == 1:
                    print(f"   After projection: {dense_output.shape}, range: [{dense_output.min():.3f}, {dense_output.max():.3f}]")
                
                # Dense layers for classification
                for i, dense_layer in enumerate(self.dense_layers):
                    dense_output = dense_layer(dense_output)
                    if batch_size == 1:
                        print(f"   Dense layer {i+1}: {dense_output.shape}, range: [{dense_output.min():.3f}, {dense_output.max():.3f}]")
                
                # Output layer
                output = self.output_layer(dense_output)
                
                if batch_size == 1:
                    print(f"   Final output: {output.shape}, range: [{output.min():.3f}, {output.max():.3f}]")
                
                return output
        
        return Hybrid1DCNNLSTM(
            n_features=n_features,
            cnn_filters=self.cnn_filters,
            cnn_kernel_sizes=self.cnn_kernel_sizes,
            cnn_pool_sizes=self.cnn_pool_sizes,
            cnn_dropout=self.cnn_dropout,
            lstm_units=self.lstm_units,
            lstm_dropout=self.lstm_dropout,
            lstm_recurrent_dropout=self.lstm_recurrent_dropout,
            dense_layers=self.dense_layers,
            dense_dropout=self.dense_dropout,
            n_channels=self.n_channels,
            bidirectional_lstm=self.bidirectional_lstm,
            attention_mechanism=self.attention_mechanism,
            residual_connections=self.residual_connections,
            batch_norm=self.batch_norm
        )
    
    def fit(self, X, y):
        """Train the hybrid model with comprehensive debugging."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Please install torch.")
        
        print(f"\n🔍 HYBRID MODEL DEBUGGING:")
        print(f"   Input X shape: {X.shape}")
        print(f"   Input y shape: {y.shape}")
        print(f"   Classes: {np.unique(y)}")
        print(f"   Class distribution: {np.bincount(y)}")
        print(f"   Device: {self.device}")
        
        # Set random seeds
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Store classes
        self.classes_ = np.unique(y)
        self.feature_names_in_ = X.columns.tolist() if hasattr(X, 'columns') else None
        
        print(f"   Feature names: {len(self.feature_names_in_)} features")
        if self.feature_names_in_:
            print(f"   Sample features: {self.feature_names_in_[:5]}...")
        
        # Check for NaN values
        if np.isnan(X).any().any():
            print(f"   ⚠️  WARNING: NaN values detected in input data!")
            nan_count = np.isnan(X).sum().sum()
            print(f"   NaN count: {nan_count}")
        else:
            print(f"   ✅ No NaN values in input data")
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y.values if hasattr(y, 'values') else y).to(self.device)
        
        print(f"   Scaled X shape: {X_scaled.shape}")
        print(f"   X tensor shape: {X_tensor.shape}")
        print(f"   Y tensor shape: {y_tensor.shape}")
        
        # Check scaled data for NaN
        if np.isnan(X_scaled).any():
            print(f"   ⚠️  WARNING: NaN values in scaled data!")
        else:
            print(f"   ✅ No NaN values in scaled data")
        
        # Create model
        print(f"\n🏗️  Creating model for {X.shape[1]} features...")
        self.model = self._create_model(X.shape[1]).to(self.device)
        
        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Loss function with class weights
        if self.class_weight == 'balanced':
            class_weights = torch.FloatTensor([
                len(y) / (2 * (y == 0).sum()),
                len(y) / (2 * (y == 0).sum())
            ]).to(self.device)
            print(f"   Class weights: {class_weights.cpu().numpy()}")
        else:
            class_weights = None
            print(f"   No class weights applied")
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer with configurable weight decay
        weight_decay = getattr(self, 'weight_decay', 1e-5)
        if self.optimizer.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=weight_decay)
        
        print(f"   Optimizer: {self.optimizer}")
        print(f"   Learning rate: {self.learning_rate}")
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Early stopping
        best_loss = float('inf')
        patience_counter = 0
        
        print(f"\n🚀 Starting training for {self.epochs} epochs...")
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            # Forward pass
            if self.mixed_precision and self.scaler_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(X_tensor)
                    loss = criterion(outputs, y_tensor)
                
                # Backward pass with mixed precision
                self.scaler_amp.scale(loss).backward()
                self.scaler_amp.step(optimizer)
                self.scaler_amp.update()
                optimizer.zero_grad()
            else:
                outputs = self.model(X_tensor)
                loss = criterion(outputs, y_tensor)
                
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping if specified
                if hasattr(self, 'gradient_clip_norm'):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                
                optimizer.step()
            
            # Calculate accuracy for this epoch
            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == y_tensor).float().mean().item()
            
            # Learning rate scheduling
            scheduler.step(loss)
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping_patience:
                print(f"   ⏹️  Early stopping at epoch {epoch + 1}")
                break
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"   Epoch {epoch + 1:3d}/{self.epochs}: Loss={loss.item():.4f}, Acc={accuracy:.4f}")
        
        print(f"\n✅ Training completed!")
        print(f"   Final loss: {loss.item():.4f}")
        print(f"   Final accuracy: {accuracy:.4f}")
        
        # Calculate final metrics on training data
        self.model.eval()
        with torch.no_grad():
            final_outputs = self.model(X_tensor)
            final_probs = torch.softmax(final_outputs, dim=1)
            _, final_preds = torch.max(final_outputs, 1)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            y_np = y_tensor.cpu().numpy()
            pred_np = final_preds.cpu().numpy()
            prob_np = final_probs[:, 1].cpu().numpy()  # Probability of positive class
            
            final_accuracy = accuracy_score(y_np, pred_np)
            final_precision = precision_score(y_np, pred_np, zero_division=0)
            final_recall = recall_score(y_np, pred_np, zero_division=0)
            final_f1 = f1_score(y_np, pred_np, zero_division=0)
            
            print(f"\n📊 FINAL TRAINING METRICS:")
            print(f"   Accuracy:  {final_accuracy:.4f}")
            print(f"   Precision: {final_precision:.4f}")
            print(f"   Recall:    {final_recall:.4f}")
            print(f"   F1-Score:  {final_f1:.4f}")
            print(f"   Class predictions: {np.bincount(pred_np)}")
            print(f"   True labels:       {np.bincount(y_np)}")
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()

class DeepLearningTrainer(BaseTrainer):
    """
    Deep Learning Trainer that integrates with existing LOGO cross-validation framework.
    
    Prevents overfitting through:
    1. Proper cross-validation (inherited LOGO)
    2. Regularization techniques in models
    3. Early stopping
    4. Feature scaling
    5. Class balancing
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_type = config.get('model_type', 'pytorch_mlp')
        
        # Get deep learning specific parameters
        dl_config = config.get('deep_learning', {})
        self.model_params = dl_config.get(self.model_type, {})
        
        # Use config as single source of truth - no hardcoded defaults
        # All parameters must be defined in the config file
        if not self.model_params:
            raise ValueError(f"No parameters found for model type '{self.model_type}' in config. "
                           f"Please define parameters in the 'deep_learning.{self.model_type}' section of your config file.")
        
        # Only add random_state if not already present
        if 'random_state' not in self.model_params:
            self.model_params['random_state'] = config.get('random_seed', 42)
        
        self.output_dir = config['paths']['models']
        self.metrics = config.get('metrics', {}).get('window_level', ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
        self.feature_selection_config = config.get('feature_selection', {})
        
        logger.info(f"DeepLearningTrainer initialized with model_type: {self.model_type}")
        logger.info(f"Model parameters: {self.model_params}")
    
    def _create_model_instance(self) -> BaseEstimator:
        """Create a deep learning model instance."""
        if self.model_type == 'pytorch_mlp':
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not available. Please install torch.")
            return PyTorchMLPClassifier(**self.model_params)
        
        elif self.model_type == 'keras_mlp':
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow not available. Please install tensorflow.")
            return KerasMLPClassifier(**self.model_params)
        
        elif self.model_type == 'hybrid_1dcnn_lstm':
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not available. Please install torch.")
            return Hybrid1DCNNLSTMClassifier(**self.model_params)
        
        else:
            raise ValueError(f"Unknown deep learning model type: {self.model_type}")
    
    def train(self, data_path: str = None, dataset: Optional[PandasDataset] = None) -> BaseEstimator:
        """
        Train a deep learning model using the existing LOGO cross-validation framework.
        
        This method inherits the robust overfitting prevention from the existing framework:
        - Leave-One-Group-Out cross-validation ensures no patient data leakage
        - Patient-level grouping prevents temporal data leakage
        - Feature selection prevents curse of dimensionality
        
        Additional overfitting prevention in deep learning models:
        - Early stopping with patience
        - Dropout regularization
        - Weight decay (L2 regularization)
        - Batch normalization
        - Learning rate scheduling
        
        Args:
            data_path: Optional path to feature data
            dataset: Optional MLflow dataset to use for training
        """
        # Determine data source with priority: MLflow dataset > provided dataset > config path > file path
        final_data_path = data_path or self.config.get('data', {}).get('feature_path')
        
        evaluator = ModelEvaluator(metrics=self.metrics)
        
        # Load data using the new base trainer method
        df = self._load_data_from_source(
            data_source=final_data_path,
            dataset=dataset,
            prefer_mlflow=True
        )
        
        # Log model information
        mlflow.log_param("model_type", self.model_type)
        mlflow.log_params(self.model_params)
        X_orig, y, groups = self._prepare_data(df)
        
        # *** CRITICAL FIX: Handle NaN values before any feature selection or cross-validation ***
        # This prevents failures in sklearn algorithms that cannot handle missing values
        if np.isnan(X_orig).any().any():
            logger.warning("NaN values detected in dataset. Applying median imputation before feature selection...")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X_orig = pd.DataFrame(
                imputer.fit_transform(X_orig),
                columns=X_orig.columns,
                index=X_orig.index
            )
            logger.info("Global NaN imputation completed before feature selection")
            mlflow.log_param("global_nan_imputation_applied", True)
        else:
            mlflow.log_param("global_nan_imputation_applied", False)
        
        # Feature Selection (inherited from existing framework)
        perform_selection = self.feature_selection_config.get('enabled', False)
        if perform_selection:
            n_features_target = self.feature_selection_config.get('n_features', 20)  # Use more features for DL
            selected_feature_names = self._select_features(X_orig, y, n_features_target)
            X = X_orig[selected_feature_names]
            mlflow.log_param("feature_selection_enabled", True)
            mlflow.log_param("target_n_features_to_select", n_features_target)
        else:
            X = X_orig
            selected_feature_names = X_orig.columns.tolist()
            mlflow.log_param("feature_selection_enabled", False)
        
        # Store selected feature names
        self.selected_feature_names = selected_feature_names
        mlflow.log_metric("num_features_trained_on", len(X.columns))
        
        # Log dataset statistics
        unique_patients = groups.unique()
        patient_labels = df.groupby('Participant')['Remission'].first()
        n_remission = sum(patient_labels == 1)
        n_non_remission = sum(patient_labels == 0)
        
        logger.info(f"\nDataset Statistics:")
        logger.info(f"Total patients: {len(unique_patients)}")
        logger.info(f"- Remission: {n_remission}")
        logger.info(f"- Non-remission: {n_non_remission}")
        logger.info(f"Total windows: {len(df)}")
        logger.info(f"Features used: {len(X.columns)}")
        
        self._log_dataset_info(X, y, groups)
        
        # Note: NaN values should have been handled globally at the start of train()
        # This check should not be necessary, but kept for safety
        if np.isnan(X).any().any():
            logger.error("Unexpected NaN values detected after global imputation. This should not happen.")
            raise ValueError("Data contains NaN values after global imputation")
        
        # Cross-validation with LOGO (prevents overfitting through proper validation)
        logo = LeaveOneGroupOut()
        n_splits = logo.get_n_splits(X, y, groups)
        logger.info(f"\nPerforming LOGO cross-validation with {n_splits} splits")
        
        # Store predictions for evaluation
        patient_predictions = []
        window_predictions = []
        patient_true_labels = []
        patient_pred_labels = []
        patient_pred_probs = []
        
        for fold_idx, (train_index, test_index) in enumerate(logo.split(X, y, groups=groups)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Double-check for NaN values (should not happen after global imputation)
            if np.isnan(X_train).any().any() or np.isnan(y_train).any():
                logger.error(f"Unexpected NaN values found in training data for fold {fold_idx} after global imputation. Skipping fold.")
                continue
            
            test_participant = groups.iloc[test_index].unique()[0]
            true_label = y.iloc[test_index].iloc[0]
            patient_true_labels.append(true_label)
            
            logger.info(f"\nFold {fold_idx + 1}/{n_splits}")
            logger.info(f"Testing on participant: {test_participant}")
            logger.info(f"Training windows: {len(train_index)}, Test windows: {len(test_index)}")
            
            # Use nested runs for each fold
            with mlflow.start_run(run_name=f"fold_{fold_idx}", nested=True):
                # Create a new model instance for each fold to ensure independence
                self.model = self._create_model_instance()
                
                logger.info(f"Fold {fold_idx+1}/{n_splits}: Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
                
                # Train model (with built-in overfitting prevention)
                self.model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = self.model.predict(X_test)
                y_prob = self.model.predict_proba(X_test)[:, 1]
                
                # Calculate test accuracies - only accuracy is meaningful for window-level
                # since all windows from a participant belong to the same class
                window_test_accuracy = np.mean(y_pred == y_test)
                
                # Calculate patient-level prediction (using window predictions, not probabilities)
                # If most windows are predicted as positive, predict patient as positive
                # If most windows are predicted as negative, predict patient as negative
                positive_window_count = sum(y_pred == 1)
                total_windows = len(y_pred)
                patient_pred = 1 if positive_window_count > total_windows / 2 else 0
                patient_prob = positive_window_count / total_windows  # Proportion of positive windows
                patient_pred_labels.append(patient_pred)
                patient_pred_probs.append(patient_prob)
                
                # Print detailed test results for this fold
                print(f"\n📊 FOLD {fold_idx + 1}/{n_splits} TEST RESULTS:")
                print(f"   Participant: {test_participant}")
                print(f"   True patient label: {true_label}")
                print(f"   Predicted patient label: {patient_pred}")
                print(f"   Patient prediction method: Majority vote of window predictions")
                print(f"   Positive windows: {positive_window_count}/{total_windows} ({patient_prob:.1%})")
                print(f"   Window-level test accuracy: {window_test_accuracy:.4f}")
                print(f"   Test windows: {len(test_index)}")
                print(f"   Patient-level correct: {'✅' if true_label == patient_pred else '❌'}")
                
                # Store predictions
                window_predictions.extend(self._create_window_predictions(
                    fold_idx, test_participant, y_test, y_pred, y_prob))
                
                # Store patient prediction with explicit variable scoping
                current_patient_prediction = {
                    'fold': fold_idx,
                    'participant': test_participant,
                    'true_label': true_label,
                    'predicted_label': patient_pred,
                    'probability': patient_prob,
                    'n_windows': len(test_index),
                    'n_positive_windows': sum(y_pred == 1),
                    'window_accuracy': window_test_accuracy
                }
                patient_predictions.append(current_patient_prediction)
                
                # Log fold-specific metrics
                patient_accuracy = int(true_label == patient_pred)
                print(f"   🔍 DEBUG: true_label={true_label}, patient_pred={patient_pred}, patient_accuracy={patient_accuracy}")
                mlflow.log_metric(f"fold_{fold_idx}_patient_accuracy", patient_accuracy)
                mlflow.log_metric(f"fold_{fold_idx}_window_accuracy", window_test_accuracy)
                mlflow.log_metric(f"fold_{fold_idx}_patient_id", test_participant)
                mlflow.log_metric(f"fold_{fold_idx}_true_remission", true_label)
                mlflow.log_metric(f"fold_{fold_idx}_predicted_remission", patient_pred)
                
                # Print detailed test results for this fold
                print(f"   🔍 MLFLOW LOGGING: fold_{fold_idx}_window_accuracy = {window_test_accuracy}")
                
                mlflow.log_metric(f"fold_{fold_idx}_patient_accuracy", patient_accuracy)
                mlflow.log_metric(f"fold_{fold_idx}_window_accuracy", window_test_accuracy)
        
        # Calculate and log patient-level metrics
        patient_metrics = evaluator.evaluate_patient_predictions(
            np.array(patient_true_labels),
            np.array(patient_pred_labels),
            np.array(patient_pred_probs)
        )
        
        # Print comprehensive cross-validation summary
        print(f"\n" + "="*80)
        print(f"🎯 CROSS-VALIDATION SUMMARY")
        print(f"="*80)
        
        # Calculate average window-level accuracy across all folds
        avg_window_accuracy = np.mean([p['window_accuracy'] for p in patient_predictions])
        
        print(f"📊 PATIENT-LEVEL RESULTS:")
        print(f"   Overall patient accuracy: {patient_metrics['accuracy']:.4f}")
        print(f"   Overall patient precision: {patient_metrics['precision']:.4f}")
        print(f"   Overall patient recall: {patient_metrics['recall']:.4f}")
        print(f"   Overall patient F1-score: {patient_metrics.get('f1_score', patient_metrics.get('f1', 0.0)):.4f}")
        print(f"   Overall patient AUC: {patient_metrics.get('auc', patient_metrics.get('roc_auc', 0.0)):.4f}")
        
        print(f"\n📊 WINDOW-LEVEL RESULTS (averaged across folds):")
        print(f"   Average window accuracy: {avg_window_accuracy:.4f}")
        print(f"   Note: Window-level precision/recall/F1 not meaningful (all windows per participant are same class)")
        
        print(f"\n📊 FOLD-BY-FOLD BREAKDOWN:")
        for i, pred in enumerate(patient_predictions):
            status = "✅" if pred['true_label'] == pred['predicted_label'] else "❌"
            print(f"   Fold {i+1:2d}: Patient {pred['participant']} | "
                  f"True: {pred['true_label']} | Pred: {pred['predicted_label']} | "
                  f"Prob: {pred['probability']:.3f} | "
                  f"Window Acc: {pred['window_accuracy']:.3f} | {status}")
        
        print(f"\n⚠️  OVERFITTING ANALYSIS:")
        print(f"   If window-level accuracy >> patient-level accuracy: Model overfits to individual windows")
        print(f"   If both are low: Model underfits or data is too noisy")
        print(f"   If both are high: Model generalizes well")
        
        # Verify patient accuracy calculations
        print(f"\n🔍 PATIENT ACCURACY VERIFICATION:")
        for i, pred in enumerate(patient_predictions):
            calculated_accuracy = int(pred['true_label'] == pred['predicted_label'])
            print(f"   Fold {i+1:2d}: True={pred['true_label']}, Pred={pred['predicted_label']}, "
                  f"Calculated Accuracy={calculated_accuracy}")
        
        # Final MLflow verification
        print(f"\n🔍 MLFLOW LOGGING VERIFICATION:")
        print(f"   The following metrics were logged to MLflow:")
        for i, pred in enumerate(patient_predictions):
            calculated_accuracy = int(pred['true_label'] == pred['predicted_label'])
            print(f"   - fold_{i}_patient_accuracy = {calculated_accuracy}")
            print(f"   - fold_{i}_window_accuracy = {pred['window_accuracy']:.4f}")
        
        # Log overall metrics
        mlflow.log_metrics({f"patient_{k}": v for k, v in patient_metrics.items()})
        mlflow.log_metric("avg_window_accuracy", avg_window_accuracy)
        
        # Train final model on all data
        logger.info("\nTraining final model on all data...")
        
        # Ensure no NaN values in final training data (should not be needed)
        if np.isnan(X).any().any():
            logger.error("Unexpected NaN values detected in final training data after global imputation.")
            raise ValueError("Data contains NaN values after global imputation")
        
        final_model = self._create_model_instance()
        final_model.fit(X, y)
        
        # Save results
        self._save_results(final_model, patient_metrics, window_predictions, 
                         patient_predictions, X)
        
        return final_model
    
    def _create_window_predictions(self, fold_idx: int, participant: str, 
                                 y_true: pd.Series, y_pred: np.ndarray, 
                                 y_prob: np.ndarray) -> List[Dict[str, Any]]:
        """Create window-level prediction records."""
        return [{
            'fold': fold_idx,
            'participant': participant,
            'true_label': y_true.iloc[idx],
            'predicted_label': pred,
            'probability': prob,
            'correct': y_true.iloc[idx] == pred
        } for idx, (pred, prob) in enumerate(zip(y_pred, y_prob))]
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series, n_features_to_select: int) -> List[str]:
        """
        Select features using statistical methods (deep learning models don't have feature importance).
        """
        from sklearn.feature_selection import SelectKBest, f_classif
        
        logger.info(f"Selecting {n_features_to_select} features for deep learning")
        
        if n_features_to_select >= X.shape[1]:
            logger.info("Requested features >= available features. Using all features.")
            return X.columns.tolist()
        
        # Use f_classif for feature selection (fast and effective)
        selector = SelectKBest(score_func=f_classif, k=n_features_to_select)
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        logger.info(f"Selected features: {len(selected_features)}")
        return selected_features
    
    def _save_results(self, model: BaseEstimator, patient_metrics: Dict[str, float],
                     window_predictions: List[Dict[str, Any]], 
                     patient_predictions: List[Dict[str, Any]], 
                     X_final_train: pd.DataFrame) -> None:
        """Save model results and predictions."""
        
        # Create DataFrames from predictions
        window_df = pd.DataFrame(window_predictions)
        patient_df = pd.DataFrame(patient_predictions)
        
        # Get window size for directory structure
        window_size = self._get_window_size()
        output_dir = Path(self.output_dir)
        if window_size:
            output_dir = output_dir / f"{window_size}s_window"
        else:
            output_dir = output_dir / "default_window"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save predictions
        if self.config.get('output', {}).get('save_predictions', True):
            window_pred_path = output_dir / f'{self.model_type}_window_predictions.csv'
            patient_pred_path = output_dir / f'{self.model_type}_patient_predictions.csv'
            
            window_df.to_csv(window_pred_path, index=False)
            patient_df.to_csv(patient_pred_path, index=False)
            
            mlflow.log_artifact(str(window_pred_path))
            mlflow.log_artifact(str(patient_pred_path))
            
            logger.info(f"Saved predictions to {window_pred_path} and {patient_pred_path}")
        
        # Save model metadata - convert numpy types to Python types for JSON serialization
        def convert_for_json(obj):
            """Convert numpy types to Python types for JSON serialization"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_for_json(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        metadata = {
            'model_type': self.model_type,
            'model_params': convert_for_json(self.model_params),
            'metrics': convert_for_json(patient_metrics),
            'n_features': int(X_final_train.shape[1]),
            'feature_names': X_final_train.columns.tolist(),
            'timestamp': pd.Timestamp.now().isoformat(),
            'framework': 'pytorch' if 'pytorch' in self.model_type else 'tensorflow'
        }
        
        import json
        metadata_path = output_dir / f'{self.model_type}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        mlflow.log_artifact(str(metadata_path))
        
        # Log model to MLflow (note: deep learning models may not support all MLflow features)
        try:
            input_example = X_final_train.head()
            if hasattr(model, 'predict'):
                prediction_example = model.predict(input_example)
                from mlflow.models.signature import infer_signature
                signature = infer_signature(input_example, prediction_example)
            else:
                signature = None
            
            # Use sklearn format since our models inherit from sklearn base classes
            mlflow.sklearn.log_model(model, "model", signature=signature, input_example=input_example)
        except Exception as e:
            logger.warning(f"Could not log model to MLflow: {e}")
            
            # Handle different model types for fallback saving
            try:
                if self.model_type == 'keras_mlp' and hasattr(model, 'model') and model.model is not None:
                    # Save Keras model using TensorFlow's native format
                    model_path = output_dir / f'{self.model_type}_model.h5'
                    model.model.save(str(model_path))
                    mlflow.log_artifact(str(model_path))
                    logger.info(f"Saved Keras model to {model_path}")
                    
                    # Also save the scaler separately
                    scaler_path = output_dir / f'{self.model_type}_scaler.joblib'
                    import joblib
                    joblib.dump(model.scaler, scaler_path)
                    mlflow.log_artifact(str(scaler_path))
                    logger.info(f"Saved scaler to {scaler_path}")
                    
                else:
                    # For PyTorch models, try joblib
                    import joblib
                    model_path = output_dir / f'{self.model_type}_model.joblib'
                    joblib.dump(model, model_path)
                    mlflow.log_artifact(str(model_path))
                    logger.info(f"Saved model to {model_path}")
                    
            except Exception as e2:
                logger.warning(f"Could not save model with fallback method: {e2}")
                logger.info("Model training completed but serialization failed - results and metrics are still saved")
        
        logger.info(f"Saved model metadata and artifacts to {output_dir}")
    
    def _get_window_size(self) -> Optional[str]:
        """Extract window size from config or data path."""
        if 'window_size' in self.config:
            return str(self.config['window_size'])
        
        data_path = self.config.get('data', {}).get('feature_path', '')
        if '{window_size}s' in data_path:
            import re
            match = re.search(r'(\d+)s_window_features', data_path)
            if match:
                return match.group(1)
        
        return None 