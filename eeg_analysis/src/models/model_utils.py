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

logger = logging.getLogger(__name__)

class ModelBuilder:
    @staticmethod
    def create_classifier(name: str, params: Dict[Any, Any] = None) -> Pipeline:
        if params is None:
            params = {}
        
        classifiers = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=20,
                class_weight='balanced',
                random_state=42,
                **params
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                min_samples_leaf=10,
                max_depth=5,
                random_state=42,
                **params
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                **params
            ),
            'svm': SVC(
                kernel='rbf',
                class_weight='balanced',
                probability=True,
                random_state=42,
                **params
            )
        }
        
        if name not in classifiers:
            raise ValueError(f"Classifier {name} not supported. Choose from: {list(classifiers.keys())}")
        
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', classifiers[name])
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

def log_feature_importance(*args, **kwargs):
    return MLflowLogger.log_feature_importance(*args, **kwargs)

def save_model_results(model, metrics, model_name, output_path):
    return MLflowLogger.save_model(model, model_name, metrics)