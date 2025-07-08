from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ModelEvaluator:
    """Class for evaluating model predictions."""
    
    def __init__(self, metrics: List[str] = None):
        """
        Initialize the evaluator with specified metrics.
        
        Args:
            metrics: List of metric names to compute. If None, computes all available metrics.
        """
        self.available_metrics = {
            'accuracy': accuracy_score,
            'precision': precision_score,
            'recall': recall_score,
            'f1': f1_score,
            'roc_auc': roc_auc_score
        }
        
        self.metrics = metrics if metrics is not None else list(self.available_metrics.keys())
        
    def evaluate_window_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_prob: np.ndarray) -> Dict[str, float]:
        """
        Evaluate predictions at the window level.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            
        Returns:
            Dictionary of metric names and values
        """
        results = {}
        
        for metric_name in self.metrics:
            if metric_name == 'roc_auc':
                # ROC AUC needs probabilities
                score = self.available_metrics[metric_name](y_true, y_prob)
            else:
                # Other metrics use predicted labels
                score = self.available_metrics[metric_name](y_true, y_pred)
            results[metric_name] = float(score)
            
        return results
    
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
        results = {}
        
        for metric_name in self.metrics:
            if metric_name == 'roc_auc':
                # ROC AUC needs probabilities
                score = self.available_metrics[metric_name](y_true, y_prob)
            else:
                # Other metrics use predicted labels
                score = self.available_metrics[metric_name](y_true, y_pred)
            results[metric_name] = float(score)
        
        # Add confusion matrix counts
        TP = sum((y_true == 1) & (y_pred == 1))
        TN = sum((y_true == 0) & (y_pred == 0))
        FP = sum((y_true == 0) & (y_pred == 1))
        FN = sum((y_true == 1) & (y_pred == 0))
        
        results.update({
            'true_positives': TP,
            'true_negatives': TN,
            'false_positives': FP,
            'false_negatives': FN,
            'n_patients': len(y_true)
        })
        
        return results

    def calculate_detailed_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Calculate detailed metrics with counts and proper handling of edge cases.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with detailed metrics including counts
        """
        # Basic counts
        TP = sum((y_true == 1) & (y_pred == 1))
        TN = sum((y_true == 0) & (y_pred == 0))
        FP = sum((y_true == 0) & (y_pred == 1))
        FN = sum((y_true == 1) & (y_pred == 0))
        
        # Calculate metrics with proper handling of division by zero
        accuracy = (TP + TN) / len(y_true) if len(y_true) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'TP': int(TP),
            'TN': int(TN),
            'FP': int(FP),
            'FN': int(FN),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'n_samples': int(len(y_true))
        }