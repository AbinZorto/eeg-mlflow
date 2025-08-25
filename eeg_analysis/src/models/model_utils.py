import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# GPU-accelerated imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("🚀 XGBoost GPU support available!")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available. Install with: pip install xgboost")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
    print("🚀 CatBoost GPU support available!")
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost not available. Install with: pip install catboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print("🚀 LightGBM GPU support available!")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available. Install with: pip install lightgbm")
from sklearn.pipeline import Pipeline
import mlflow
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os

logger = logging.getLogger(__name__)

class ModelBuilder:
    @staticmethod
    def create_classifier(name: str, params: Dict[Any, Any] = None) -> Pipeline:
        """Create a classifier pipeline with StandardScaler and the specified model."""
        if params is None:
            user_params = {}
        else:
            user_params = params

        model_instance = None
        
        if name == 'random_forest':
            default_params = {
                'n_estimators': 200,
                'min_samples_leaf': 20,
                'class_weight': 'balanced',
                'random_state': 42
            }
            final_params = {**default_params, **user_params}
            model_instance = RandomForestClassifier(**final_params)
            
        elif name == 'gradient_boosting':
            default_params = {
                'n_estimators': 200,
                'min_samples_leaf': 10,
                'max_depth': 5,
                'random_state': 42
            }
            # Filter out unsupported parameters for GradientBoostingClassifier
            supported_params = {
                'n_estimators', 'learning_rate', 'max_depth', 'min_samples_split',
                'min_samples_leaf', 'min_weight_fraction_leaf', 'subsample',
                'max_features', 'max_leaf_nodes', 'min_impurity_decrease',
                'random_state', 'verbose', 'warm_start', 'validation_fraction',
                'n_iter_no_change', 'tol', 'ccp_alpha'
            }
            filtered_user_params = {k: v for k, v in user_params.items() if k in supported_params}
            final_params = {**default_params, **filtered_user_params}
            model_instance = GradientBoostingClassifier(**final_params)
            
        elif name == 'xgboost_gpu':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available. Install with: pip install xgboost")
            
            default_params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'tree_method': 'hist',  # Modern tree method
                'device': 'cuda:0',  # GPU acceleration (new API)
                'random_state': 42,
                'eval_metric': 'logloss',
                'use_label_encoder': False
            }
            
            # XGBoost parameter mapping from sklearn-style names
            param_mapping = {
                'min_samples_leaf': 'min_child_weight',
                'class_weight': 'scale_pos_weight'  # Handle class weights differently
            }
            
            # Convert sklearn-style parameters to XGBoost parameters
            xgb_params = {}
            for key, value in user_params.items():
                if key in param_mapping:
                    if key == 'class_weight' and value == 'balanced':
                        # Calculate scale_pos_weight automatically during fit
                        continue
                    else:
                        xgb_params[param_mapping[key]] = value
                elif key in ['n_estimators', 'max_depth', 'learning_rate', 'random_state', 
                           'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda']:
                    xgb_params[key] = value
            
            final_params = {**default_params, **xgb_params}
            print(f"🚀 Creating XGBoost GPU model with parameters: {final_params}")
            model_instance = xgb.XGBClassifier(**final_params)
            
        elif name == 'catboost_gpu':
            if not CATBOOST_AVAILABLE:
                raise ImportError("CatBoost not available. Install with: pip install catboost")
            
            default_params = {
                'iterations': 200,  # Reduced from 500 for speed
                'depth': 6,  # Reduced from 8 for faster training
                'learning_rate': 0.2,  # Increased to compensate for fewer iterations
                'task_type': 'GPU',  # GPU acceleration!
                'devices': '0',  # Use first GPU
                'loss_function': 'Logloss',
                'eval_metric': 'Logloss',  # Use Logloss instead of AUC (faster on GPU)
                'random_seed': 42,
                'verbose': False,
                'allow_writing_files': False,
                'bootstrap_type': 'Bernoulli',  # Better for GPU
                'subsample': 0.8,  # Slightly more aggressive subsampling for speed
                'border_count': 128,  # Reduced from default 254 for speed
                'max_ctr_complexity': 1,  # Reduce complexity for speed
                'gpu_ram_part': 0.8  # Use more GPU memory for better performance
                # Note: colsample_bylevel (rsm) not supported in GPU mode for binary classification
            }
            
            # CatBoost parameter mapping
            catboost_params = {}
            for key, value in user_params.items():
                if key == 'n_estimators':
                    catboost_params['iterations'] = value
                elif key == 'max_depth':
                    catboost_params['depth'] = value
                elif key == 'class_weight' and value == 'balanced':
                    catboost_params['auto_class_weights'] = 'Balanced'
                elif key in ['learning_rate', 'random_state', 'subsample']:
                    if key == 'random_state':
                        catboost_params['random_seed'] = value
                    else:
                        catboost_params[key] = value
            
            final_params = {**default_params, **catboost_params}
            print(f"🚀 Creating CatBoost GPU model with parameters: {final_params}")
            model_instance = cb.CatBoostClassifier(**final_params)
            
        elif name == 'lightgbm_gpu':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not available. Install with: pip install lightgbm")
            
            default_params = {
                'n_estimators': 500,
                'max_depth': 8,
                'learning_rate': 0.15,
                'device': 'gpu',  # GPU acceleration!
                'gpu_device_id': 0,  # Use first GPU
                'objective': 'binary',
                'metric': 'auc',
                'random_state': 42,
                'verbose': -1,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'num_leaves': 255,  # Optimized for GPU
                'min_child_samples': 20
            }
            
            # LightGBM parameter mapping
            lgb_params = {}
            for key, value in user_params.items():
                if key == 'class_weight' and value == 'balanced':
                    lgb_params['is_unbalance'] = True
                elif key in ['n_estimators', 'max_depth', 'learning_rate', 'random_state', 
                           'subsample', 'colsample_bytree', 'num_leaves', 'min_child_samples']:
                    lgb_params[key] = value
            
            final_params = {**default_params, **lgb_params}
            print(f"🚀 Creating LightGBM GPU model with parameters: {final_params}")
            model_instance = lgb.LGBMClassifier(**final_params)
            
        elif name == 'logistic_regression':
            default_params = {
                'max_iter': 1000,
                'class_weight': 'balanced',
                'random_state': 42
            }
            final_params = {**default_params, **user_params}
            model_instance = LogisticRegression(**final_params)
            
        elif name == 'logistic_regression_l1':
            default_params = {
                'penalty': 'l1',
                'solver': 'liblinear',
                'max_iter': 1000,
                'class_weight': 'balanced',
                'random_state': 42
            }
            final_params = {**default_params, **user_params}
            model_instance = LogisticRegression(**final_params)
            
        elif name == 'svm' or name == 'svm_rbf':
            default_params = {
                'kernel': 'rbf',
                'class_weight': 'balanced',
                'probability': True,
                'random_state': 42
            }
            final_params = {**default_params, **user_params}
            model_instance = SVC(**final_params)
            
        elif name == 'svm_linear':
            default_params = {
                'kernel': 'linear',
                'class_weight': 'balanced',
                'probability': True,
                'random_state': 42
            }
            final_params = {**default_params, **user_params}
            model_instance = SVC(**final_params)
            
        elif name == 'extra_trees':
            default_params = {
                'n_estimators': 200,
                'min_samples_leaf': 2,
                'class_weight': 'balanced',
                'random_state': 42
            }
            final_params = {**default_params, **user_params}
            model_instance = ExtraTreesClassifier(**final_params)
            
        elif name == 'ada_boost':
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'random_state': 42
            }
            # Filter out unsupported parameters for AdaBoostClassifier
            supported_params = {
                'n_estimators', 'learning_rate', 'algorithm', 'random_state'
            }
            filtered_user_params = {k: v for k, v in user_params.items() if k in supported_params}
            final_params = {**default_params, **filtered_user_params}
            model_instance = AdaBoostClassifier(**final_params)
            
        elif name == 'knn':
            default_params = {
                'n_neighbors': 3,
                'weights': 'distance'
            }
            # Filter out unsupported parameters for KNeighborsClassifier
            supported_params = {
                'n_neighbors', 'weights', 'algorithm', 'leaf_size', 'p',
                'metric', 'metric_params', 'n_jobs'
            }
            filtered_user_params = {k: v for k, v in user_params.items() if k in supported_params}
            final_params = {**default_params, **filtered_user_params}
            model_instance = KNeighborsClassifier(**final_params)
            
        elif name == 'decision_tree':
            default_params = {
                'min_samples_leaf': 2,
                'class_weight': 'balanced',
                'random_state': 42
            }
            final_params = {**default_params, **user_params}
            model_instance = DecisionTreeClassifier(**final_params)
            
        elif name == 'sgd':
            default_params = {
                'loss': 'modified_huber',  # Enables probability estimates
                'penalty': 'elasticnet',
                'class_weight': 'balanced',
                'max_iter': 1000,
                'random_state': 42
            }
            final_params = {**default_params, **user_params}
            model_instance = SGDClassifier(**final_params)
            
        elif name in ['pytorch_mlp', 'keras_mlp', 'hybrid_1dcnn_lstm', 'advanced_hybrid_1dcnn_lstm', 'advanced_1dcnn', 'advanced_lstm']:
            # Deep learning models are handled by DeepLearningTrainer
            # Return a placeholder that will be replaced by actual DL model
            from .deep_learning_trainer import PyTorchMLPClassifier, KerasMLPClassifier, AdvancedHybrid1DCNNLSTMClassifier
            
            if name == 'pytorch_mlp':
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
                    'random_state': 42
                }
                final_params = {**default_params, **user_params}
                return PyTorchMLPClassifier(**final_params)  # Return directly, not as pipeline
                
            elif name == 'keras_mlp':
                default_params = {
                    'hidden_layers': [64, 32],
                    'dropout_rate': 0.3,
                    'l1_reg': 0.01,
                    'l2_reg': 0.01,
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 200,
                    'early_stopping_patience': 20,
                    'batch_norm': True,
                    'activation': 'relu',
                    'optimizer': 'adam',
                    'class_weight': 'balanced',
                    'random_state': 42
                }
                final_params = {**default_params, **user_params}
                return KerasMLPClassifier(**final_params)  # Return directly, not as pipeline
                
            elif name in ['hybrid_1dcnn_lstm', 'advanced_hybrid_1dcnn_lstm', 'advanced_1dcnn', 'advanced_lstm']:
                # All advanced models use the same implementation with different configurations
                # Filter out incompatible parameters for advanced models
                if name in ['advanced_hybrid_1dcnn_lstm', 'advanced_1dcnn', 'advanced_lstm']:
                    # Remove parameters that don't belong in the advanced model
                    filtered_params = {k: v for k, v in user_params.items() 
                                     if k not in ['class_weight', 'optimizer', 'learning_rate', 
                                                 'batch_size', 'epochs', 'early_stopping_patience']}
                else:
                    filtered_params = user_params
                return AdvancedHybrid1DCNNLSTMClassifier(**filtered_params)  # Return directly, not as pipeline
            
        else:
            supported_models = [
                'random_forest', 'gradient_boosting', 'xgboost_gpu', 'catboost_gpu', 'lightgbm_gpu', 'logistic_regression', 
                'logistic_regression_l1', 'svm_rbf', 'svm_linear', 'extra_trees', 'ada_boost', 'knn', 'decision_tree', 'sgd', 
                'pytorch_mlp', 'keras_mlp', 'hybrid_1dcnn_lstm', 'advanced_1dcnn', 'advanced_lstm', 'advanced_hybrid_1dcnn_lstm'
            ]
            raise ValueError(f"Classifier {name} not supported. Choose from: {supported_models}")
        
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model_instance)
        ])

    @staticmethod
    def get_all_classifiers(use_class_weights: bool = True, config_params: Dict[str, Dict] = None) -> Dict[str, Pipeline]:
        """
        Get all available classifiers with their default configurations.
        
        Args:
            use_class_weights: Whether to use class weights for imbalanced datasets
            config_params: Optional dictionary of model-specific parameters from config
            
        Returns:
            Dictionary of classifier name -> Pipeline
        """
        # Set class weights
        class_weights = {0: 1, 1: 2} if use_class_weights else 'balanced'
        
        # Default parameters for all models
        default_configs = {
            'random_forest': {
                'n_estimators': 200,
                'min_samples_leaf': 2,
                'class_weight': class_weights,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 200,
                'min_samples_leaf': 1,
                'max_depth': 5,
                'random_state': 42
            },
            'logistic_regression': {
                'max_iter': 1000,
                'class_weight': class_weights,
                'random_state': 42
            },
            'extra_trees': {
                'n_estimators': 200,
                'min_samples_leaf': 2,
                'class_weight': class_weights,
                'random_state': 42
            },
            'ada_boost': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'random_state': 42
            },
            'svm_rbf': {
                'kernel': 'rbf',
                'class_weight': class_weights,
                'probability': True,
                'random_state': 42
            },
            'svm_linear': {
                'kernel': 'linear',
                'class_weight': class_weights,
                'probability': True,
                'random_state': 42
            },
            'knn': {
                'n_neighbors': 3,
                'weights': 'distance'
            },
            'decision_tree': {
                'min_samples_leaf': 2,
                'class_weight': class_weights,
                'random_state': 42
            },
            'logistic_regression_l1': {
                'penalty': 'l1',
                'solver': 'liblinear',
                'class_weight': class_weights,
                'max_iter': 1000,
                'random_state': 42
            },
            'sgd': {
                'loss': 'modified_huber',
                'penalty': 'elasticnet',
                'class_weight': class_weights,
                'max_iter': 1000,
                'random_state': 42
            }
        }
        
        # Merge with config parameters if provided
        if config_params:
            for model_name, params in config_params.items():
                if model_name in default_configs:
                    default_configs[model_name].update(params)
        
        # Create all classifiers
        classifiers = {}
        for name, params in default_configs.items():
            try:
                classifiers[name] = ModelBuilder.create_classifier(name, params)
            except Exception as e:
                logger.warning(f"Failed to create classifier {name}: {e}")
                continue
                
        return classifiers

    @staticmethod
    def get_classifier_names() -> List[str]:
        """Get list of all supported classifier names."""
        return [
            'random_forest', 'gradient_boosting', 'logistic_regression', 'logistic_regression_l1',
            'svm_rbf', 'svm_linear', 'extra_trees', 'ada_boost', 'knn', 'decision_tree', 'sgd'
        ]

class MLflowLogger:
    @staticmethod
    def log_metrics(metrics: Dict[str, float]):
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
    
    @staticmethod
    def log_feature_importance(model: Pipeline, feature_names: List[str], top_n: int = 20):
        classifier = model.named_steps['classifier']
        
        if hasattr(classifier, 'feature_importances_'):
            importance = classifier.feature_importances_
        elif hasattr(classifier, 'coef_'):
            importance = np.abs(classifier.coef_[0])
        else:
            logger.warning("Model does not support feature importance")
            return
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        for idx, row in feature_importance.head(top_n).iterrows():
            mlflow.log_metric(f"feature_importance_{row['feature']}", row['importance'])
        
        feature_importance.to_csv('feature_importance.csv', index=False)
        mlflow.log_artifact('feature_importance.csv')
    
    @staticmethod
    def save_model(model: Pipeline, model_name: str, metrics: Dict[str, float]):
        MLflowLogger.log_metrics(metrics)
        
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=model_name
        )
        
        model_info = {
            'model_type': type(model.named_steps['classifier']).__name__,
            'scaler_type': type(model.named_steps['scaler']).__name__,
            'metrics': metrics
        }
        
        mlflow.log_dict(model_info, "model_info.json")

# Convenience functions to maintain backward compatibility
def create_classifier(*args, **kwargs):
    return ModelBuilder.create_classifier(*args, **kwargs)

def log_feature_importance(model, feature_names, output_path=None):
    """
    Log feature importance for a model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        output_path: Optional path to save feature importance CSV
    """
    if hasattr(model, 'feature_importances_'):
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save to CSV if path provided - do this BEFORE logging to MLflow
        if output_path:
            # Make sure the directory exists
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the file
            importances.to_csv(output_path, index=False)
            
            # Verify the file was created
            if not os.path.exists(output_path):
                logger.error(f"Failed to create feature importance file at {output_path}")
                return None
        
        # Log to MLflow
        mlflow.log_table(importances, "feature_importance.json")
        
        # Create feature importance plot
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=importances.head(20))
        plt.title('Feature Importance')
        plt.tight_layout()
        
        # Save plot to a temporary file and log it
        with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
            plt.savefig(tmp.name)
            mlflow.log_artifact(tmp.name, "feature_importance.png")
        plt.close()
        
        # Log the CSV file to MLflow if it was created
        if output_path and os.path.exists(output_path):
            mlflow.log_artifact(output_path)
            
        return importances
    else:
        logger.warning("Model does not have feature_importances_ attribute")
        return None

def save_model_results(model, metrics, model_name, output_path):
    return MLflowLogger.save_model(model, model_name, metrics)