import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.base import BaseEstimator
import mlflow
from src.models.model_utils import create_classifier

class BaseTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('model_name', 'model')
        self.classifier_name = config.get('classifier', 'random_forest')
        self.classifier_params = config.get('classifier_params', {})

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        X = df.drop(['Participant', 'Remission'], axis=1)
        y = df['Remission']
        groups = df['Participant']
        return X, y, groups

    def _create_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> BaseEstimator:
        model = create_classifier(self.classifier_name, self.classifier_params)
        model.fit(X_train, y_train)
        return model

    def _log_dataset_info(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series):
        mlflow.log_params({
            'n_features': X.shape[1],
            'n_positive': sum(y == 1),
            'n_negative': sum(y == 0)
        })

    def _save_predictions(self, predictions: pd.DataFrame, filename: str):
        predictions.to_csv(filename, index=False)
        mlflow.log_artifact(filename)

    def train(self, data_path: str) -> BaseEstimator:
        raise NotImplementedError("Subclasses must implement train method")