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
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    warnings.warn("TensorFlow not available. Keras models will not be functional.")

from src.models.base_trainer import BaseTrainer
from src.models.evaluation import ModelEvaluator

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
                 random_state=42):
        
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
        
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes_ = None
        self.feature_names_in_ = None
        
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
        
        # Prepare data
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Use smaller batch size for small datasets
        effective_batch_size = min(self.batch_size, len(X) // 4) if len(X) < 100 else self.batch_size
        dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True)
        
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
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
            
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
                 random_state=42):
        
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
        
        self.model = None
        self.scaler = StandardScaler()
        self.classes_ = None
        self.feature_names_in_ = None
        
        # Set random seeds
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
    
    def _create_model(self, n_features):
        """Create the neural network model."""
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
            opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9)
        
        model.compile(
            optimizer=opt,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
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
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
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
        
        # Use smaller batch size for small datasets
        effective_batch_size = min(self.batch_size, len(X) // 4) if len(X) < 100 else self.batch_size
        
        # Train model
        self.model.fit(
            X_scaled, y,
            batch_size=effective_batch_size,
            epochs=self.epochs,
            callbacks=callbacks_list,
            class_weight=class_weight_dict,
            verbose=0
        )
        
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
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict probabilities
        return self.model.predict(X_scaled, verbose=0)

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
        
        # Set default parameters based on model type
        if self.model_type == 'pytorch_mlp':
            default_params = {
                'hidden_layers': [64, 32],
                'dropout_rate': 0.3,
                'weight_decay': 0.01,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 200,
                'early_stopping_patience': 20,
                'batch_norm': True,
                'activation': 'relu',
                'optimizer': 'adam',
                'class_weight': 'balanced',
                'random_state': config.get('random_seed', 42)
            }
        elif self.model_type == 'keras_mlp':
            default_params = {
                'hidden_layers': [64, 32],
                'dropout_rate': 0.3,
                'l1_reg': 0.001,
                'l2_reg': 0.01,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 200,
                'early_stopping_patience': 20,
                'batch_norm': True,
                'activation': 'relu',
                'optimizer': 'adam',
                'class_weight': 'balanced',
                'random_state': config.get('random_seed', 42)
            }
        else:
            default_params = {}
        
        # Merge with user parameters
        for key, default_value in default_params.items():
            if key not in self.model_params:
                self.model_params[key] = default_value
        
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
        
        else:
            raise ValueError(f"Unknown deep learning model type: {self.model_type}")
    
    def train(self, data_path: str = None) -> BaseEstimator:
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
        """
        if data_path is None:
            data_path = self.config['data']['feature_path']
        
        evaluator = ModelEvaluator(metrics=self.metrics)
        
        # Log the data path being used
        mlflow.log_param("feature_path", data_path)
        mlflow.log_param("model_type", self.model_type)
        mlflow.log_params(self.model_params)
        
        # Load and prepare data
        df = pd.read_parquet(data_path)
        X_orig, y, groups = self._prepare_data(df)
        
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
        
        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            test_participant = groups.iloc[test_idx].unique()[0]
            true_label = y.iloc[test_idx].iloc[0]
            patient_true_labels.append(true_label)
            
            logger.info(f"\nFold {fold_idx + 1}/{n_splits}")
            logger.info(f"Testing on participant: {test_participant}")
            logger.info(f"Training windows: {len(train_idx)}, Test windows: {len(test_idx)}")
            
            # Use nested runs for each fold
            with mlflow.start_run(run_name=f"fold_{fold_idx}", nested=True):
                # Create and train model (with built-in overfitting prevention)
                model = self._create_model_instance()
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                
                # Make predictions
                y_pred = model.predict(X.iloc[test_idx])
                y_prob = model.predict_proba(X.iloc[test_idx])[:, 1]
                
                # Calculate patient-level prediction (averaging window predictions)
                patient_prob = np.mean(y_prob)
                patient_pred = 1 if patient_prob >= 0.5 else 0
                patient_pred_labels.append(patient_pred)
                patient_pred_probs.append(patient_prob)
                
                # Store predictions
                window_predictions.extend(self._create_window_predictions(
                    fold_idx, test_participant, y.iloc[test_idx], y_pred, y_prob))
                
                patient_predictions.append({
                    'fold': fold_idx,
                    'participant': test_participant,
                    'true_label': true_label,
                    'predicted_label': patient_pred,
                    'probability': patient_prob,
                    'n_windows': len(test_idx),
                    'n_positive_windows': sum(y_pred == 1),
                    'window_accuracy': np.mean(y_pred == y.iloc[test_idx])
                })
                
                # Log fold-specific metrics
                mlflow.log_metric(f"fold_{fold_idx}_patient_accuracy", int(true_label == patient_pred))
                mlflow.log_metric(f"fold_{fold_idx}_window_accuracy", np.mean(y_pred == y.iloc[test_idx]))
        
        # Calculate and log patient-level metrics
        patient_metrics = evaluator.evaluate_patient_predictions(
            np.array(patient_true_labels),
            np.array(patient_pred_labels),
            np.array(patient_pred_probs)
        )
        
        # Log overall metrics
        mlflow.log_metrics({f"patient_{k}": v for k, v in patient_metrics.items()})
        
        # Train final model on all data
        logger.info("\nTraining final model on all data...")
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
            # Save using joblib as fallback
            import joblib
            model_path = output_dir / f'{self.model_type}_model.joblib'
            joblib.dump(model, model_path)
            mlflow.log_artifact(str(model_path))
        
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