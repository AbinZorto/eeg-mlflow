from typing import List, Dict, Any
import numpy as np

from src.utils.clinical_metrics import (
    PATIENT_METRIC_NAMES,
    WINDOW_METRIC_NAMES,
    compute_binary_classification_metrics,
)

class ModelEvaluator:
    """Class for evaluating model predictions."""
    
    def __init__(self, metrics: List[str] = None):
        """
        Initialize the evaluator with specified metrics.
        
        Args:
            metrics: List of metric names to compute. If None, computes all available metrics.
        """
        self.available_metrics = sorted(set(PATIENT_METRIC_NAMES + WINDOW_METRIC_NAMES))
        self.metrics = metrics if metrics is not None else list(self.available_metrics)
        
    def evaluate_window_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_prob: np.ndarray) -> Dict[str, float]:
        """
        Evaluate predictions at the window-based granularity.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            
        Returns:
            Dictionary of metric names and values
        """
        results = compute_binary_classification_metrics(
            y_true,
            y_pred,
            y_prob,
            metric_names=self.metrics,
            count_field_name="n_windows",
        )
        return {
            key: value
            for key, value in results.items()
            if key in set(self.metrics) | {"n_windows"}
        }
    
    def evaluate_patient_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_prob: np.ndarray) -> Dict[str, float]:
        """
        Evaluate predictions at the patient level.
        
        Args:
            y_true: True patient labels
            y_pred: Predicted patient labels
            y_prob: Prediction probabilities for patients
            
        Returns:
            Dictionary of metric names and values
        """
        results = compute_binary_classification_metrics(
            y_true,
            y_pred,
            y_prob,
            metric_names=self.metrics,
            count_field_name="n_patients",
        )
        keep = set(self.metrics) | {
            "true_positives",
            "true_negatives",
            "false_positives",
            "false_negatives",
            "n_patients",
        }
        return {key: value for key, value in results.items() if key in keep}

    def calculate_detailed_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Calculate detailed metrics with counts and proper handling of edge cases.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with detailed metrics including counts
        """
        metrics = compute_binary_classification_metrics(
            y_true,
            y_pred,
            None,
            metric_names=["accuracy", "balanced_accuracy", "precision", "recall", "sensitivity", "specificity", "f1", "npv", "mcc"],
            count_field_name="n_samples",
        )
        return {
            'TP': int(metrics['true_positives']),
            'TN': int(metrics['true_negatives']),
            'FP': int(metrics['false_positives']),
            'FN': int(metrics['false_negatives']),
            'accuracy': float(metrics['accuracy']) if metrics['accuracy'] is not None else 0.0,
            'balanced_accuracy': float(metrics['balanced_accuracy']) if metrics['balanced_accuracy'] is not None else 0.0,
            'precision': float(metrics['precision']) if metrics['precision'] is not None else 0.0,
            'recall': float(metrics['recall']) if metrics['recall'] is not None else 0.0,
            'specificity': float(metrics['specificity']) if metrics['specificity'] is not None else 0.0,
            'f1': float(metrics['f1']) if metrics['f1'] is not None else 0.0,
            'npv': float(metrics['npv']) if metrics['npv'] is not None else 0.0,
            'mcc': float(metrics['mcc']) if metrics['mcc'] is not None else 0.0,
            'n_samples': int(metrics['n_samples'])
        }
