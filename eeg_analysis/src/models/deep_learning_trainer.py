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

# Import SMOTE and NearMiss for handling class imbalance
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import NearMiss
    IMBALANCE_AVAILABLE = True
except ImportError:
    IMBALANCE_AVAILABLE = False
    warnings.warn("SMOTE/NearMiss not available. Install imbalanced-learn for better handling of class imbalance.")

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
                 gradient_accumulation_steps=1,
                 use_smote=False,  # Add SMOTE parameter
                 use_nearmiss=False,  # Add NearMiss parameter
                 nearmiss_version=1):  # NearMiss version (1, 2, or 3)
        
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
        self.use_smote = use_smote  # Store SMOTE setting
        self.use_nearmiss = use_nearmiss  # Store NearMiss setting
        self.nearmiss_version = nearmiss_version  # Store NearMiss version
        
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
        
        # Apply SMOTE for class balance if enabled
        if self.use_smote and SMOTE_AVAILABLE:
            print(f"   🔄 SMOTE ENABLED: Balancing classes for better remission detection...")
            print(f"   Original class distribution: {np.bincount(y)}")
            
            try:
                smote = SMOTE(random_state=self.random_state)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                
                # Convert back to DataFrame/Series if needed
                if hasattr(X, 'columns'):
                    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
                if hasattr(y, 'name'):
                    y_resampled = pd.Series(y_resampled, name=y.name)
                
                print(f"   ✅ SMOTE applied successfully!")
                print(f"   New class distribution: {np.bincount(y_resampled)}")
                print(f"   Original samples: {len(X)}, Resampled samples: {len(X_resampled)}")
                
                # Update X and y with resampled data
                X = X_resampled
                y = y_resampled
                
            except Exception as e:
                print(f"   ⚠️  SMOTE failed: {e}. Continuing without SMOTE...")
                self.use_smote = False
        elif self.use_smote and not SMOTE_AVAILABLE:
            print(f"   ⚠️  SMOTE requested but not available. Install imbalanced-learn package.")
            self.use_smote = False
        else:
            print(f"   ℹ️  SMOTE disabled. Using original class distribution: {np.bincount(y)}")
        
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
        
        # Loss function with class weights - SMOTE/NearMiss-aware
        if self.use_smote or self.use_nearmiss:
            print(f"   ℹ️  {'SMOTE' if self.use_smote else 'NearMiss'} enabled - disabling class weights in loss function")
            class_weights = None
        elif self.class_weight == 'balanced':
            class_counts = np.bincount(y)
            class_weights = len(y) / (len(np.unique(y)) * class_counts)
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            print(f"   ⚖️  Using balanced class weights: {class_weights.cpu().numpy()}")
        else:
            class_weights = None
            print(f"   ℹ️  No class weights applied")
        
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
                 gradient_clip_norm=1.0,
                 use_smote=False,  # Add SMOTE parameter
                 use_nearmiss=False,  # Add NearMiss parameter
                 nearmiss_version=1):  # NearMiss version (1, 2, or 3)
        
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
        self.use_smote = use_smote  # Store SMOTE setting
        self.use_nearmiss = use_nearmiss  # Store NearMiss setting
        self.nearmiss_version = nearmiss_version  # Store NearMiss version
        
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
        
        # Apply SMOTE for class balance if enabled
        if self.use_smote and SMOTE_AVAILABLE:
            print(f"   🔄 SMOTE ENABLED: Balancing classes for better remission detection...")
            print(f"   Original class distribution: {np.bincount(y)}")
            
            try:
                smote = SMOTE(random_state=self.random_state)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                
                # Convert back to DataFrame/Series if needed
                if hasattr(X, 'columns'):
                    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
                if hasattr(y, 'name'):
                    y_resampled = pd.Series(y_resampled, name=y.name)
                
                print(f"   ✅ SMOTE applied successfully!")
                print(f"   New class distribution: {np.bincount(y_resampled)}")
                print(f"   Original samples: {len(X)}, Resampled samples: {len(X_resampled)}")
                
                # Update X and y with resampled data
                X = X_resampled
                y = y_resampled
                
            except Exception as e:
                print(f"   ⚠️  SMOTE failed: {e}. Continuing without SMOTE...")
                self.use_smote = False
        elif self.use_smote and not SMOTE_AVAILABLE:
            print(f"   ⚠️  SMOTE requested but not available. Install imbalanced-learn package.")
            self.use_smote = False
        else:
            print(f"   ℹ️  SMOTE disabled. Using original class distribution: {np.bincount(y)}")
        
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
        
        # Calculate class weights if needed - SMOTE/NearMiss-aware
        if self.use_smote or self.use_nearmiss:
            print(f"   ℹ️  {'SMOTE' if self.use_smote else 'NearMiss'} enabled - disabling class weights in loss function")
            class_weight_dict = None
        elif self.class_weight == 'balanced':
            class_counts = np.bincount(y)
            class_weights = len(y) / (len(np.unique(y)) * class_counts)
            class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
            print(f"   ⚖️  Using balanced class weights: {class_weights}")
        else:
            class_weight_dict = None
            print(f"   ℹ️  No class weights applied")
        
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

class EfficientTabularMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    Efficient tabular MLP optimized for window-level features.
    - AdamW (or Adam) optimizer
    - Optional cosine warm restarts
    - Label smoothing and gradient clipping
    - Early stopping on training loss (no validation split here)
    - Mixed precision support
    - SMOTE/NearMiss-aware class weighting
    """
    
    def __init__(self,
                 hidden_layers=[512, 256, 128],
                 dropout_rate=0.3,
                 batch_norm=True,
                 batch_size=1024,
                 epochs=100,
                 optimizer_config=None,
                 lr_schedule=None,
                 regularization=None,
                 early_stopping=None,
                 mixed_precision=True,
                 class_weight='balanced',
                 random_state=42,
                 use_smote=False,
                 use_nearmiss=False,
                 nearmiss_version=1):
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer_config = optimizer_config or {
            'name': 'adamw',
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-7
        }
        self.lr_schedule = lr_schedule or {
            'type': 'cosine_annealing_warm_restarts',
            'initial_lr': 0.001,
            'min_lr': 1e-6,
            'cycle_length': 20,
            'cycle_mult': 1
        }
        self.regularization = regularization or {
            'label_smoothing': 0.05,
            'gradient_clip_norm': 1.0
        }
        self.early_stopping = early_stopping or {
            'patience': 10,
            'restore_best_weights': True,
            'monitor': 'loss',
            'min_delta': 0.001
        }
        self.mixed_precision = mixed_precision
        self.class_weight = class_weight
        self.random_state = random_state
        self.use_smote = use_smote
        self.use_nearmiss = use_nearmiss
        self.nearmiss_version = nearmiss_version
        
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu')
        self.scaler_amp = None
        if self.mixed_precision and TORCH_AVAILABLE and torch.cuda.is_available():
            self.scaler_amp = torch.cuda.amp.GradScaler()
        
        # Seeds
        np.random.seed(self.random_state)
        if TORCH_AVAILABLE:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)
    
    def _create_model(self, n_features: int):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Please install torch.")
        
        class MLP(nn.Module):
            def __init__(self, n_in: int, layers: List[int], dropout: float, use_bn: bool):
                super().__init__()
                modules = []
                in_features = n_in
                for units in layers:
                    modules.append(nn.Linear(in_features, units))
                    if use_bn:
                        modules.append(nn.BatchNorm1d(units))
                    modules.append(nn.ReLU())
                    modules.append(nn.Dropout(dropout))
                    in_features = units
                modules.append(nn.Linear(in_features, 2))
                self.net = nn.Sequential(*modules)
                self._init_weights()
            
            def _init_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
            
            def forward(self, x):
                return self.net(x)
        
        return MLP(n_features, self.hidden_layers, self.dropout_rate, self.batch_norm)
    
    def fit(self, X, y):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Please install torch.")
        
        # Optional SMOTE
        if self.use_smote and IMBALANCE_AVAILABLE:
            try:
                smote = SMOTE(random_state=self.random_state)
                X, y = smote.fit_resample(X, y)
            except Exception:
                self.use_smote = False
        elif self.use_nearmiss and IMBALANCE_AVAILABLE:
            try:
                nm = NearMiss(version=self.nearmiss_version)
                X, y = nm.fit_resample(X, y)
            except Exception:
                self.use_nearmiss = False
        
        # Pandas to numpy
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Model
        self.model = self._create_model(X_scaled.shape[1]).to(self.device)
        
        # Loss (label smoothing + optional class weights)
        if self.use_smote or self.use_nearmiss:
            weight_tensor = None
        elif self.class_weight == 'balanced':
            class_counts = np.bincount(y)
            class_weights = len(y) / (len(np.unique(y)) * class_counts)
            weight_tensor = torch.FloatTensor(class_weights).to(self.device)
        else:
            weight_tensor = None
        criterion = nn.CrossEntropyLoss(weight=weight_tensor, 
                                        label_smoothing=float(self.regularization.get('label_smoothing', 0.0)))
        
        # Optimizer
        if self.optimizer_config.get('name') == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=float(self.optimizer_config.get('learning_rate', 0.001)),
                weight_decay=float(self.optimizer_config.get('weight_decay', 0.01)),
                betas=(float(self.optimizer_config.get('beta_1', 0.9)), float(self.optimizer_config.get('beta_2', 0.999))),
                eps=float(self.optimizer_config.get('epsilon', 1e-7))
            )
        else:
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=float(self.optimizer_config.get('learning_rate', 0.001)),
                weight_decay=float(self.optimizer_config.get('weight_decay', 0.01))
            )
        
        # Scheduler
        if self.lr_schedule and self.lr_schedule.get('type') == 'cosine_annealing_warm_restarts':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=int(self.lr_schedule.get('cycle_length', 20)),
                T_mult=int(self.lr_schedule.get('cycle_mult', 1)),
                eta_min=float(self.lr_schedule.get('min_lr', 1e-6))
            )
            use_plateau = False
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=False)
            use_plateau = True
        
        # Dataloader
        effective_batch_size = min(self.batch_size, max(16, len(X_tensor) // 2)) if len(X_tensor) < 100 else self.batch_size
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_tensor, y_tensor),
            batch_size=effective_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        # Early stopping
        best_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                if self.scaler_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                    self.scaler_amp.scale(loss).backward()
                    # clip
                    clip_norm = float(self.regularization.get('gradient_clip_norm', 0.0) or 0.0)
                    if clip_norm > 0:
                        self.scaler_amp.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
                    self.scaler_amp.step(optimizer)
                    self.scaler_amp.update()
                    optimizer.zero_grad()
                else:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    optimizer.zero_grad()
                    loss.backward()
                    clip_norm = float(self.regularization.get('gradient_clip_norm', 0.0) or 0.0)
                    if clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
                    optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= max(1, len(dataloader))
            if use_plateau:
                scheduler.step(epoch_loss)
            else:
                scheduler.step()
            if epoch_loss + float(self.early_stopping.get('min_delta', 0.0)) < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
                if self.early_stopping.get('restore_best_weights', True):
                    best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= int(self.early_stopping.get('patience', 10)):
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self
    
    def predict(self, X):
        if not TORCH_AVAILABLE or self.model is None:
            raise ValueError("Model not trained or PyTorch not available.")
        if hasattr(X, 'values'):
            X = X.values
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, preds = torch.max(outputs, 1)
        return preds.cpu().numpy()
    
    def predict_proba(self, X):
        if not TORCH_AVAILABLE or self.model is None:
            raise ValueError("Model not trained or PyTorch not available.")
        if hasattr(X, 'values'):
            X = X.values
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = F.softmax(outputs, dim=1)
        return probs.cpu().numpy()

class AdvancedHybrid1DCNNLSTMClassifier(BaseEstimator, ClassifierMixin):
    """
    OPTIMIZED Advanced Hybrid 1D CNN-LSTM Classifier for maximum speed and GPU utilization.
    
    Key optimizations:
    - Simplified architecture for faster training
    - Proper CUDA utilization with mixed precision
    - Optimized batch processing
    - Reduced debugging output
    - Consistent random seed handling
    """
    
    def __init__(self,
                 cnn_blocks=None,
                 normalization='layer_norm',
                 cnn_dropout=None,
                 spatial_dropout=True,
                 gaussian_noise=0.01,
                 lstm_architecture=None,
                 attention_config=None,
                 fusion_strategy='concat_attention',
                 feature_pyramid=True,
                 dense_architecture=None,
                 optimizer_config=None,
                 lr_schedule=None,
                 batch_size=32,
                 epochs=200,
                 early_stopping=None,
                 augmentation=None,
                 preprocessing=None,
                 regularization=None,
                 ensemble=None,
                 loss_config=None,
                 architecture_enhancements=None,
                 validation=None,
                 monitoring=None,
                 hardware_optimization=None,
                 random_state=42,
                 deterministic_training=True,
                 post_training=None,
                 use_smote=False,  # Add SMOTE parameter
                 use_nearmiss=False,  # Add NearMiss parameter
                 nearmiss_version=1):  # NearMiss version (1, 2, or 3)
        
        # Store all configurations
        self.cnn_blocks = cnn_blocks
        self.normalization = normalization
        self.cnn_dropout = cnn_dropout
        self.spatial_dropout = spatial_dropout
        self.gaussian_noise = gaussian_noise
        self.lstm_architecture = lstm_architecture
        self.attention_config = attention_config
        self.fusion_strategy = fusion_strategy
        self.feature_pyramid = feature_pyramid
        self.dense_architecture = dense_architecture
        self.optimizer_config = optimizer_config
        self.lr_schedule = lr_schedule
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.regularization = regularization
        self.ensemble = ensemble
        self.loss_config = loss_config
        self.architecture_enhancements = architecture_enhancements
        self.validation = validation
        self.monitoring = monitoring
        self.hardware_optimization = hardware_optimization
        self.random_state = random_state
        self.deterministic_training = deterministic_training
        self.post_training = post_training
        self.use_smote = use_smote  # Store SMOTE setting
        self.use_nearmiss = use_nearmiss  # Store NearMiss setting
        self.nearmiss_version = nearmiss_version  # Store NearMiss version
        
        # Validate that only one class balancing technique is enabled
        if self.use_smote and self.use_nearmiss:
            raise ValueError("❌ ERROR: Both SMOTE and NearMiss cannot be enabled simultaneously. Please choose only one class balancing technique.")
        
        # Initialize model components
        self.model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes_ = None
        self.feature_names_in_ = None
        self.scaler_amp = None
        
        # CRITICAL: Set random seeds FIRST before any other operations
        self._set_random_seeds()
        
        # Set up CUDA optimizations
        self._setup_cuda_optimizations()
        
        # Print device info
        print(f"🚀 ADVANCED HYBRID MODEL - DEVICE: {self.device}")
        if torch.cuda.is_available():
            print(f"   CUDA Device: {torch.cuda.get_device_name()}")
            print(f"   CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"   Mixed Precision: {'ENABLED' if self.scaler_amp else 'DISABLED'}")
    
    def _set_random_seeds(self):
        """Set all random seeds consistently."""
        # Set Python random seed
        import random
        random.seed(self.random_state)
        
        # Set NumPy random seed
        np.random.seed(self.random_state)
        
        # Set PyTorch random seeds
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)
        
        # Set deterministic training if enabled
        if self.deterministic_training:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print(f"🔒 DETERMINISTIC TRAINING ENABLED (seed: {self.random_state})")
        else:
            torch.backends.cudnn.benchmark = True
            print(f"⚡ BENCHMARK MODE ENABLED for maximum speed")
    
    def _setup_cuda_optimizations(self):
        """Set up CUDA optimizations for maximum performance."""
        if torch.cuda.is_available():
            # Enable mixed precision for maximum speed
            if self.hardware_optimization and self.hardware_optimization.get('mixed_precision', True):
                self.scaler_amp = torch.cuda.amp.GradScaler()
                print("🔥 MIXED PRECISION ENABLED: Using automatic mixed precision for maximum GPU utilization!")
            
            # Set memory fraction to prevent OOM
            torch.cuda.empty_cache()
            
            # Enable memory efficient attention if available
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
                print("⚡ FLASH ATTENTION ENABLED: Using memory efficient attention!")
    
    def _create_model(self, n_features):
        """Create an OPTIMIZED advanced hybrid model architecture."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Please install torch.")
        
        class OptimizedHybridModel(nn.Module):
            def __init__(self, n_features, cnn_blocks, normalization, cnn_dropout, spatial_dropout,
                         gaussian_noise, lstm_architecture, attention_config, fusion_strategy,
                         feature_pyramid, dense_architecture, architecture_enhancements):
                super(OptimizedHybridModel, self).__init__()
                
                self.n_features = n_features
                self.cnn_blocks = cnn_blocks
                self.normalization = normalization
                self.cnn_dropout = cnn_dropout
                self.spatial_dropout = spatial_dropout
                self.gaussian_noise = gaussian_noise
                self.lstm_architecture = lstm_architecture
                self.attention_config = attention_config
                self.fusion_strategy = fusion_strategy
                self.feature_pyramid = feature_pyramid
                self.dense_architecture = dense_architecture
                self.architecture_enhancements = architecture_enhancements
                
                # OPTIMIZED: Simplified feature embedding
                self.feature_embedding = nn.Sequential(
                    nn.Linear(n_features, 128),  # Reduced from n_features * 2
                    nn.BatchNorm1d(128),
                    nn.ReLU(),  # Faster than SiLU
                    nn.Dropout(0.1)
                )
                
                # OPTIMIZED: Direct reshape to CNN input size
                self.reshape_layer = nn.Linear(128, 64)
                
                # OPTIMIZED: Simplified Gaussian noise
                if gaussian_noise > 0:
                    self.gaussian_noise_layer = lambda x: x + torch.randn_like(x) * gaussian_noise if self.training else x
                else:
                    self.gaussian_noise_layer = lambda x: x
                
                # OPTIMIZED: Simplified CNN blocks
                self.cnn_blocks_layers = nn.ModuleList()
                self.feature_pyramid_features = [] if feature_pyramid else None
                
                in_channels = 1
                for i, block_config in enumerate(cnn_blocks):
                    block = self._create_optimized_cnn_block(in_channels, block_config, i)
                    self.cnn_blocks_layers.append(block)
                    in_channels = block_config['filters'][-1]
                    
                    if feature_pyramid:
                        self.feature_pyramid_features.append(in_channels)
                
                # OPTIMIZED: Simplified LSTM layers
                self.lstm_layers = nn.ModuleList()
                lstm_input_size = in_channels
                
                for i, lstm_config in enumerate(lstm_architecture):
                    lstm_layer = self._create_optimized_lstm_layer(lstm_input_size, lstm_config, i)
                    self.lstm_layers.append(lstm_layer)
                    lstm_input_size = lstm_config['units'] * (2 if lstm_config['bidirectional'] else 1)
                
                # OPTIMIZED: Simplified attention mechanism
                if attention_config['attention_type'] == 'multi_head':
                    self.attention = nn.MultiheadAttention(
                        embed_dim=lstm_input_size,
                        num_heads=attention_config['num_heads'],
                        dropout=attention_config['dropout'],
                        batch_first=True
                    )
                    self.attention_norm = nn.LayerNorm(lstm_input_size)
                    self.attention_dropout = nn.Dropout(attention_config['dropout'])
                
                # OPTIMIZED: Simplified dense layers
                self.dense_layers = nn.ModuleList()
                dense_input_size = lstm_input_size
                
                # Account for feature pyramid fusion
                if feature_pyramid and len(cnn_blocks) > 1:
                    pyramid_features_size = sum(block['filters'][-1] for block in cnn_blocks)
                    dense_input_size += pyramid_features_size
                
                for i, dense_config in enumerate(dense_architecture):
                    dense_block = self._create_optimized_dense_block(dense_input_size, dense_config, i)
                    self.dense_layers.append(dense_block)
                    dense_input_size = dense_config['units']
                
                # Output layer
                self.output_layer = nn.Linear(dense_input_size, 2)
                
                # Initialize weights
                self._init_weights()
            
            def _create_optimized_cnn_block(self, in_channels, block_config, block_idx):
                """Create an optimized CNN block."""
                layers = []
                
                for i, (filters, kernel_size, dilation_rate) in enumerate(zip(
                    block_config['filters'], block_config['kernel_sizes'], block_config['dilation_rates'])):
                    
                    # OPTIMIZED: Use regular convolutions instead of separable for speed
                    conv = nn.Conv1d(in_channels, filters, kernel_size, 
                                   padding=kernel_size//2, dilation=dilation_rate)
                    layers.append(conv)
                    
                    # OPTIMIZED: Use BatchNorm1d for speed
                    layers.append(nn.BatchNorm1d(filters))
                    
                    # OPTIMIZED: Use ReLU for speed
                    layers.append(nn.ReLU())
                    
                    # OPTIMIZED: Simplified dropout
                    if self.spatial_dropout:
                        layers.append(nn.Dropout1d(self.cnn_dropout[block_idx]))
                    else:
                        layers.append(nn.Dropout(self.cnn_dropout[block_idx]))
                    
                    in_channels = filters
                
                # Pooling
                if block_config['pool_size'] > 1:
                    layers.append(nn.MaxPool1d(block_config['pool_size']))
                
                return nn.Sequential(*layers)
            
            def _create_optimized_lstm_layer(self, input_size, lstm_config, layer_idx):
                """Create an optimized LSTM layer."""
                lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=lstm_config['units'],
                    num_layers=1,
                    batch_first=True,
                    bidirectional=lstm_config['bidirectional'],
                    dropout=lstm_config['dropout'] if layer_idx < len(self.lstm_architecture) - 1 else 0,
                )
                return lstm
            
            def _create_optimized_dense_block(self, input_size, dense_config, block_idx):
                """Create an optimized dense block."""
                layers = []
                
                # OPTIMIZED: Single linear layer instead of double
                layers.append(nn.Linear(input_size, dense_config['units']))
                
                if dense_config.get('batch_norm', False):
                    layers.append(nn.BatchNorm1d(dense_config['units']))
                
                # OPTIMIZED: Use ReLU for speed
                layers.append(nn.ReLU())
                
                layers.append(nn.Dropout(dense_config['dropout']))
                
                return nn.Sequential(*layers)
            
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
                """OPTIMIZED forward pass - minimal debugging."""
                # Feature embedding
                x = self.feature_embedding(x)
                
                # Reshape tabular features to sequence format
                x = self.reshape_layer(x)
                
                # Gaussian noise for robustness
                x = self.gaussian_noise_layer(x)
                
                # Reshape for CNN - treat features as sequence
                x = x.unsqueeze(1)  # Add channel dimension: (batch, 1, 64)
                
                # Multi-scale CNN blocks with feature pyramid
                cnn_outputs = []
                for i, cnn_block in enumerate(self.cnn_blocks_layers):
                    x = cnn_block(x)
                    if self.feature_pyramid:
                        cnn_outputs.append(x)
                
                # Global average pooling over the sequence dimension
                cnn_output = torch.mean(x, dim=2)  # (batch, channels)
                
                # Hierarchical LSTM layers
                lstm_output = cnn_output.unsqueeze(1)  # Add sequence dimension
                
                for i, lstm_layer in enumerate(self.lstm_layers):
                    lstm_output, _ = lstm_layer(lstm_output)
                
                # Multi-head attention
                if self.attention_config['attention_type'] == 'multi_head':
                    attn_output, _ = self.attention(lstm_output, lstm_output, lstm_output)
                    attn_output = self.attention_norm(attn_output + lstm_output)  # Residual connection
                    attn_output = self.attention_dropout(attn_output)
                    lstm_output = attn_output
                
                # Global average pooling over sequence dimension
                lstm_output = torch.mean(lstm_output, dim=1)  # (batch, features)
                
                # Feature fusion if using feature pyramid
                if self.feature_pyramid and len(cnn_outputs) > 1:
                    # Use features from multiple CNN levels
                    pyramid_features = []
                    for cnn_out in cnn_outputs:
                        pooled = torch.mean(cnn_out, dim=2)  # Global average pooling
                        pyramid_features.append(pooled)
                    
                    # Concatenate pyramid features
                    fused_features = torch.cat(pyramid_features, dim=1)
                    lstm_output = torch.cat([lstm_output, fused_features], dim=1)
                
                # Dense layers
                dense_output = lstm_output
                for i, dense_layer in enumerate(self.dense_layers):
                    dense_output = dense_layer(dense_output)
                
                # Output layer
                output = self.output_layer(dense_output)
                
                return output
        
        return OptimizedHybridModel(
            n_features=n_features,
            cnn_blocks=self.cnn_blocks,
            normalization=self.normalization,
            cnn_dropout=self.cnn_dropout,
            spatial_dropout=self.spatial_dropout,
            gaussian_noise=self.gaussian_noise,
            lstm_architecture=self.lstm_architecture,
            attention_config=self.attention_config,
            fusion_strategy=self.fusion_strategy,
            feature_pyramid=self.feature_pyramid,
            dense_architecture=self.dense_architecture,
            architecture_enhancements=self.architecture_enhancements
        )
    
    def fit(self, X, y):
        """OPTIMIZED training with maximum speed and proper CUDA usage."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Please install torch.")
        
        print(f"\n🚀 OPTIMIZED ADVANCED HYBRID MODEL TRAINING:")
        print(f"   Input X shape: {X.shape}")
        print(f"   Input y shape: {y.shape}")
        print(f"   Classes: {np.unique(y)}")
        print(f"   Device: {self.device}")
        print(f"   Random seed: {self.random_state}")
        
        # Store classes
        self.classes_ = np.unique(y)
        self.feature_names_in_ = X.columns.tolist() if hasattr(X, 'columns') else None
        
        # Check for NaN values and handle them
        if np.isnan(X).any().any():
            print(f"   ⚠️  WARNING: NaN values detected in input data! Applying median imputation...")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = pd.DataFrame(
                imputer.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        
        # Apply class balancing techniques if enabled
        if self.use_smote and IMBALANCE_AVAILABLE:
            print(f"   🔄 SMOTE ENABLED: Balancing classes for better remission detection...")
            print(f"   Original class distribution: {np.bincount(y)}")
            
            try:
                smote = SMOTE(random_state=self.random_state)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                
                # Convert back to DataFrame/Series if needed
                if hasattr(X, 'columns'):
                    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
                if hasattr(y, 'name'):
                    y_resampled = pd.Series(y_resampled, name=y.name)
                
                print(f"   ✅ SMOTE applied successfully!")
                print(f"   New class distribution: {np.bincount(y_resampled)}")
                print(f"   Original samples: {len(X)}, Resampled samples: {len(X_resampled)}")
                
                # Update X and y with resampled data
                X = X_resampled
                y = y_resampled
                
            except Exception as e:
                print(f"   ⚠️  SMOTE failed: {e}. Continuing without SMOTE...")
                self.use_smote = False
        elif self.use_nearmiss and IMBALANCE_AVAILABLE:
            print(f"   🔄 NEARMISS ENABLED: Undersampling majority class for balanced dataset...")
            print(f"   Original class distribution: {np.bincount(y)}")
            
            try:
                # NearMiss doesn't support random_state in all versions, so we'll set it manually
                np.random.seed(self.random_state)
                nearmiss = NearMiss(version=self.nearmiss_version)
                X_resampled, y_resampled = nearmiss.fit_resample(X, y)
                
                # Convert back to DataFrame/Series if needed
                if hasattr(X, 'columns'):
                    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
                if hasattr(y, 'name'):
                    y_resampled = pd.Series(y_resampled, name=y.name)
                
                print(f"   ✅ NearMiss (v{self.nearmiss_version}) applied successfully!")
                print(f"   New class distribution: {np.bincount(y_resampled)}")
                print(f"   Original samples: {len(X)}, Resampled samples: {len(X_resampled)}")
                
                # Update X and y with resampled data
                X = X_resampled
                y = y_resampled
                
            except Exception as e:
                print(f"   ⚠️  NearMiss failed: {e}. Continuing without NearMiss...")
                self.use_nearmiss = False
        elif (self.use_smote or self.use_nearmiss) and not IMBALANCE_AVAILABLE:
            print(f"   ⚠️  SMOTE/NearMiss requested but not available. Install imbalanced-learn package.")
            self.use_smote = False
            self.use_nearmiss = False
        else:
            print(f"   ℹ️  No class balancing enabled. Using original class distribution: {np.bincount(y)}")
        
        # OPTIMIZED: Use StandardScaler for speed
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y.values if hasattr(y, 'values') else y).to(self.device)
        
        # Create model
        print(f"\n🏗️  Creating optimized model for {X.shape[1]} features...")
        self.model = self._create_model(X.shape[1]).to(self.device)
        
        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # OPTIMIZED: Simplified loss function with SMOTE-aware class weights
        if self.loss_config and self.loss_config.get('primary_loss') == 'focal_loss':
            from torch.nn import functional as F
            class FocalLoss(nn.Module):
                def __init__(self, alpha=1, gamma=2):
                    super(FocalLoss, self).__init__()
                    self.alpha = alpha
                    self.gamma = gamma
                
                def forward(self, inputs, targets):
                    ce_loss = F.cross_entropy(inputs, targets, reduction='none')
                    pt = torch.exp(-ce_loss)
                    focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
                    return focal_loss.mean()
            
            criterion = FocalLoss()
        else:
            # Standard cross-entropy with class weights
            # If SMOTE or NearMiss is used, disable class weights since data is already balanced
            if self.use_smote or self.use_nearmiss:
                print(f"   ℹ️  {'SMOTE' if self.use_smote else 'NearMiss'} enabled - disabling class weights in loss function")
                class_weights = None
            elif self.loss_config and self.loss_config.get('class_weights') == 'balanced':
                class_weights = torch.FloatTensor([
                    len(y) / (2 * (y == 0).sum()),
                    len(y) / (2 * (y == 1).sum())
                ]).to(self.device)
                print(f"   ⚖️  Using balanced class weights: {class_weights.cpu().numpy()}")
            else:
                class_weights = None
                print(f"   ℹ️  No class weights applied")
            
            criterion = nn.CrossEntropyLoss(weight=class_weights, 
                                          label_smoothing=self.regularization.get('label_smoothing', 0.0) if self.regularization else 0.0)
        
        # OPTIMIZED: Use AdamW for better performance
        if self.optimizer_config and self.optimizer_config.get('name') == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.optimizer_config.get('learning_rate', 0.001),
                weight_decay=self.optimizer_config.get('weight_decay', 0.01),
                betas=(self.optimizer_config.get('beta_1', 0.9), self.optimizer_config.get('beta_2', 0.999)),
                eps=float(self.optimizer_config.get('epsilon', 1e-7))
            )
        else:
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.optimizer_config.get('learning_rate', 0.001) if self.optimizer_config else 0.001,
                weight_decay=self.optimizer_config.get('weight_decay', 0.01) if self.optimizer_config else 0.01
            )
        
        # OPTIMIZED: Simplified learning rate scheduler
        if self.lr_schedule and self.lr_schedule.get('type') == 'cosine_annealing_warm_restarts':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.lr_schedule.get('cycle_length'),
                T_mult=self.lr_schedule.get('cycle_mult'),
                eta_min=float(self.lr_schedule.get('min_lr'))
            )
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, verbose=False
            )
        
        # Early stopping
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"\n🚀 Starting optimized training for {self.epochs} epochs...")
        
        # OPTIMIZED: Better batch size handling
        effective_batch_size = min(self.batch_size, len(X) // 2) if len(X) < 100 else self.batch_size
        effective_batch_size = max(effective_batch_size, 16)  # Minimum batch size
        
        # Create data loader with optimizations
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=effective_batch_size, 
            shuffle=True,
            num_workers=0,  # Keep 0 for simplicity
            pin_memory=False  # Disable since tensors are already on GPU
        )
        
        print(f"   Effective batch size: {effective_batch_size}")
        print(f"   Number of batches per epoch: {len(dataloader)}")
        
        # OPTIMIZED: Training loop with minimal overhead
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
                # OPTIMIZED: Mixed precision training
                if self.scaler_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                    
                    self.scaler_amp.scale(loss).backward()
                    
                    # Gradient clipping
                    if self.regularization and self.regularization.get('gradient_clip_norm'):
                        self.scaler_amp.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.regularization['gradient_clip_norm'])
                    
                    self.scaler_amp.step(optimizer)
                    self.scaler_amp.update()
                    optimizer.zero_grad()
                else:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    if self.regularization and self.regularization.get('gradient_clip_norm'):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.regularization['gradient_clip_norm'])
                    
                    optimizer.step()
                
                # Calculate accuracy for this batch
                with torch.no_grad():
                    _, predicted = torch.max(outputs, 1)
                    epoch_correct += (predicted == batch_y).sum().item()
                    epoch_total += batch_y.size(0)
                
                epoch_loss += loss.item()
            
            # Calculate epoch accuracy
            accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0
            
            # Learning rate scheduling
            if self.lr_schedule and self.lr_schedule.get('type') == 'cosine_annealing_warm_restarts':
                scheduler.step()
            else:
                scheduler.step(loss)
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                
            if self.early_stopping and patience_counter >= self.early_stopping.get('patience', 30):
                print(f"   ⏹️  Early stopping at epoch {epoch + 1}")
                break
            
            # OPTIMIZED: Less frequent logging
            if (epoch + 1) % 5 == 0 or epoch == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"   Epoch {epoch + 1:3d}/{self.epochs}: Loss={loss.item():.4f}, Acc={accuracy:.4f}, LR={current_lr:.6f}")
        
        # Load best model state
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        print(f"\n✅ Optimized training completed!")
        print(f"   Final loss: {loss.item():.4f}")
        print(f"   Final accuracy: {accuracy:.4f}")
        
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
        
        # Add SMOTE and NearMiss parameters from config - check both root level and model section
        self.use_smote = config.get('use_smote', config.get('model', {}).get('use_smote', False))
        self.use_nearmiss = config.get('use_nearmiss', config.get('model', {}).get('use_nearmiss', False))
        self.nearmiss_version = config.get('nearmiss_version', config.get('model', {}).get('nearmiss_version', 1))
        
        # Validate that only one class balancing technique is enabled
        if self.use_smote and self.use_nearmiss:
            raise ValueError("❌ ERROR: Both SMOTE and NearMiss cannot be enabled simultaneously in the config. Please set only one to 'true'.")
        
        self.model_params['use_smote'] = self.use_smote
        self.model_params['use_nearmiss'] = self.use_nearmiss
        self.model_params['nearmiss_version'] = self.nearmiss_version
        
        # Log class balancing configuration
        logger.info(f"SMOTE enabled: {self.use_smote}")
        logger.info(f"NearMiss enabled: {self.use_nearmiss}")
        if self.use_nearmiss:
            logger.info(f"NearMiss version: {self.nearmiss_version}")
        
        self.output_dir = config['paths']['models']
        self.metrics = config.get('metrics', {}).get('window_level', ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
        self.feature_selection_config = config.get('feature_selection', {})
        
        logger.info(f"DeepLearningTrainer initialized with model_type: {self.model_type}")
        logger.info(f"Model parameters: {self.model_params}")
        logger.info(f"SMOTE enabled: {self.use_smote}")
    
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
        
        elif self.model_type == 'efficient_tabular_mlp':
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not available. Please install torch.")
            return EfficientTabularMLPClassifier(**self.model_params)
        
        elif self.model_type == 'hybrid_1dcnn_lstm':
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not available. Please install torch.")
            return AdvancedHybrid1DCNNLSTMClassifier(**self.model_params)
        
        elif self.model_type == 'advanced_hybrid_1dcnn_lstm':
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not available. Please install torch.")
            return AdvancedHybrid1DCNNLSTMClassifier(**self.model_params)
        
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
        
        print(f"\n📊 PATIENT-LEVEL RESULTS:")
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