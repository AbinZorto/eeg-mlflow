import numpy as np
import pandas as pd
from typing import Dict, Any
from src.utils.metrics import MetricsCalculator

class ModelEvaluator:
    def __init__(self):
        self.calculator = MetricsCalculator()

    def evaluate_window_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_prob: np.ndarray) -> Dict[str, float]:
        return self.calculator.calculate_classification_metrics(y_true, y_pred, y_prob)

    def evaluate_patient_predictions(self, groups: pd.Series, y_true: np.ndarray,
                                  y_prob: np.ndarray) -> Dict[str, float]:
        patient_preds = self._aggregate_to_patient_level(groups, y_true, y_prob)
        return self.calculator.calculate_classification_metrics(
            patient_preds['true_label'],
            patient_preds['predicted_label'],
            patient_preds['probability']
        )

    def _aggregate_to_patient_level(self, groups: pd.Series, y_true: np.ndarray,
                                 y_prob: np.ndarray) -> pd.DataFrame:
        predictions = []
        for participant in groups.unique():
            mask = groups == participant
            avg_prob = np.mean(y_prob[mask])
            predictions.append({
                'true_label': y_true[mask].iloc[0],
                'predicted_label': 1 if avg_prob >= 0.5 else 0,
                'probability': avg_prob
            })
        return pd.DataFrame(predictions)