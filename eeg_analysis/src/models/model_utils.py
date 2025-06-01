import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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
            final_params = {**default_params, **user_params}
            model_instance = GradientBoostingClassifier(**final_params)
        elif name == 'logistic_regression':
            default_params = {
                'max_iter': 1000,
                'class_weight': 'balanced',
                'random_state': 42
            }
            final_params = {**default_params, **user_params}
            model_instance = LogisticRegression(**final_params)
        elif name == 'svm':
            default_params = {
                'kernel': 'rbf',
                'class_weight': 'balanced',
                'probability': True,
                'random_state': 42
            }
            final_params = {**default_params, **user_params}
            model_instance = SVC(**final_params)
        else:
            supported_models = ['random_forest', 'gradient_boosting', 'logistic_regression', 'svm']
            raise ValueError(f"Classifier {name} not supported. Choose from: {supported_models}")
        
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model_instance)
        ])

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