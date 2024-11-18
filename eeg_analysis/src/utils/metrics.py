import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import mlflow
from .logger import get_logger

logger = get_logger(__name__)

class MetricsCalculator:
    """Class for calculating and logging various metrics."""
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize metrics calculator.
        
        Args:
            threshold: Classification threshold for probabilities
        """
        self.threshold = threshold
    
    def calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Optional prediction probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
        
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Add detailed metrics
        metrics.update({
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
        })
        
        return metrics
    
    def calculate_window_metrics(
        self,
        participant_results: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
        """
        Calculate metrics for window-level predictions.
        
        Args:
            participant_results: List of prediction results by participant
            
        Returns:
            Tuple of (average metrics, per-participant metrics)
        """
        all_metrics = []
        
        for result in participant_results:
            metrics = self.calculate_classification_metrics(
                result['true_labels'],
                result['predicted_labels'],
                result.get('probabilities')
            )
            metrics['participant'] = result['participant']
            all_metrics.append(metrics)
        
        # Calculate average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            if key != 'participant':
                values = [m[key] for m in all_metrics]
                avg_metrics[key] = np.mean(values)
                avg_metrics[f"{key}_std"] = np.std(values)
        
        return avg_metrics, all_metrics
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        prefix: Optional[str] = None
    ) -> None:
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics
            prefix: Optional prefix for metric names
        """
        for name, value in metrics.items():
            if prefix:
                name = f"{prefix}_{name}"
            mlflow.log_metric(name, value)
    
    def calculate_cross_validation_metrics(
        self,
        cv_results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for cross-validation results.
        
        Args:
            cv_results: List of cross-validation results
            
        Returns:
            Dictionary of metric statistics
        """
        # Collect metrics across folds
        fold_metrics = {}
        
        for fold_result in cv_results:
            metrics = self.calculate_classification_metrics(
                fold_result['y_true'],
                fold_result['y_pred'],
                fold_result.get('y_prob')
            )
            
            for metric, value in metrics.items():
                if metric not in fold_metrics:
                    fold_metrics[metric] = []
                fold_metrics[metric].append(value)
        
        # Calculate statistics
        metric_stats = {}
        for metric, values in fold_metrics.items():
            metric_stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return metric_stats
    
    def log_cross_validation_metrics(
        self,
        cv_metrics: Dict[str, Dict[str, float]],
        prefix: Optional[str] = None
    ) -> None:
        """
        Log cross-validation metrics to MLflow.
        
        Args:
            cv_metrics: Dictionary of cross-validation metrics
            prefix: Optional prefix for metric names
        """
        for metric, stats in cv_metrics.items():
            for stat_name, value in stats.items():
                metric_name = f"{prefix}_{metric}_{stat_name}" if prefix else f"{metric}_{stat_name}"
                mlflow.log_metric(metric_name, value)
    
    def calculate_feature_importance_metrics(
        self,
        feature_names: List[str],
        importance_values: np.ndarray,
        top_k: int = 20
    ) -> Dict[str, Any]:
        """
        Calculate feature importance metrics.
        
        Args:
            feature_names: List of feature names
            importance_values: Array of importance values
            top_k: Number of top features to include
            
        Returns:
            Dictionary containing feature importance metrics
        """
        # Create sorted feature importance pairs
        importance_pairs = sorted(
            zip(feature_names, importance_values),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Extract top features
        top_features = importance_pairs[:top_k]
        
        # Calculate importance statistics
        importance_metrics = {
            'top_features': [
                {'name': name, 'importance': float(importance)}
                for name, importance in top_features
            ],
            'importance_statistics': {
                'mean': float(np.mean(importance_values)),
                'std': float(np.std(importance_values)),
                'max': float(np.max(importance_values)),
                'min': float(np.min(importance_values))
            },
            'feature_categories': self._categorize_important_features(top_features)
        }
        
        return importance_metrics
    
    def _categorize_important_features(
        self,
        importance_pairs: List[Tuple[str, float]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Categorize important features by type.
        
        Args:
            importance_pairs: List of (feature_name, importance) tuples
            
        Returns:
            Dictionary of categorized features
        """
        categories = {
            'spectral': ['bp_', 'rbp_', 'spectral_', 'sef_'],
            'temporal': ['mean', 'std', 'var', 'skew', 'hjorth_'],
            'complexity': ['entropy', 'correlation_dim', 'hurst', 'lyap_', 'dfa'],
            'connectivity': ['xcorr_']
        }
        
        categorized_features = {category: [] for category in categories}
        
        for feature_name, importance in importance_pairs:
            categorized = False
            for category, patterns in categories.items():
                if any(pattern in feature_name.lower() for pattern in patterns):
                    categorized_features[category].append({
                        'name': feature_name,
                        'importance': float(importance)
                    })
                    categorized = True
                    break
            
            if not categorized:
                if 'other' not in categorized_features:
                    categorized_features['other'] = []
                categorized_features['other'].append({
                    'name': feature_name,
                    'importance': float(importance)
                })
        
        return categorized_features
    
    def calculate_model_calibration_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Calculate model calibration metrics.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            n_bins: Number of bins for calibration curve
            
        Returns:
            Dictionary of calibration metrics
        """
        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        binids = np.digitize(y_prob, bins) - 1
        
        # Calculate calibration metrics
        bin_metrics = []
        for bin_id in range(n_bins):
            mask = binids == bin_id
            if np.any(mask):
                bin_probs = y_prob[mask]
                bin_true = y_true[mask]
                bin_metrics.append({
                    'bin_id': bin_id,
                    'bin_start': float(bins[bin_id]),
                    'bin_end': float(bins[bin_id + 1]),
                    'samples': int(np.sum(mask)),
                    'mean_predicted': float(np.mean(bin_probs)),
                    'mean_actual': float(np.mean(bin_true))
                })
        
        # Calculate calibration error metrics
        predictions = np.array([m['mean_predicted'] for m in bin_metrics])
        actuals = np.array([m['mean_actual'] for m in bin_metrics])
        
        calibration_metrics = {
            'bin_metrics': bin_metrics,
            'calibration_error': float(np.mean((predictions - actuals) ** 2)),
            'max_calibration_error': float(np.max(np.abs(predictions - actuals)))
        }
        
        return calibration_metrics
    
    def calculate_threshold_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        thresholds: Optional[np.ndarray] = None
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        Calculate metrics across different classification thresholds.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            thresholds: Optional array of thresholds to evaluate
            
        Returns:
            Dictionary of metrics at each threshold
        """
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 9)
        
        threshold_metrics = []
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            metrics = self.calculate_classification_metrics(y_true, y_pred)
            
            threshold_metrics.append({
                'threshold': float(threshold),
                **{k: float(v) for k, v in metrics.items()}
            })
        
        return {'threshold_metrics': threshold_metrics}
    
    def generate_metrics_report(
        self,
        metrics_dict: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a detailed metrics report in markdown format.
        
        Args:
            metrics_dict: Dictionary containing all metrics
            output_path: Optional path to save report
            
        Returns:
            Markdown formatted report string
        """
        report = ["# Model Evaluation Report\n\n"]
        
        # Basic Classification Metrics
        if 'basic_metrics' in metrics_dict:
            report.append("## Classification Metrics\n")
            report.append("| Metric | Value |\n")
            report.append("|--------|-------|\n")
            for metric, value in metrics_dict['basic_metrics'].items():
                report.append(f"| {metric} | {value:.4f} |\n")
            report.append("\n")
        
        # Cross-validation Metrics
        if 'cv_metrics' in metrics_dict:
            report.append("## Cross-validation Results\n")
            report.append("| Metric | Mean ± Std | Min | Max |\n")
            report.append("|--------|------------|-----|-----|\n")
            for metric, stats in metrics_dict['cv_metrics'].items():
                report.append(
                    f"| {metric} | {stats['mean']:.4f} ± {stats['std']:.4f} | "
                    f"{stats['min']:.4f} | {stats['max']:.4f} |\n"
                )
            report.append("\n")
        
        # Feature Importance
        if 'feature_importance' in metrics_dict:
            report.append("## Top Important Features\n")
            report.append("| Feature | Importance |\n")
            report.append("|---------|------------|\n")
            for feature in metrics_dict['feature_importance']['top_features'][:10]:
                report.append(f"| {feature['name']} | {feature['importance']:.4f} |\n")
            report.append("\n")
        
        # Calibration Metrics
        if 'calibration' in metrics_dict:
            report.append("## Model Calibration\n")
            report.append(f"- Calibration Error: {metrics_dict['calibration']['calibration_error']:.4f}\n")
            report.append(f"- Max Calibration Error: {metrics_dict['calibration']['max_calibration_error']:.4f}\n\n")
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write("".join(report))
        
        return "".join(report)


def create_metrics_calculator(threshold: float = 0.5) -> MetricsCalculator:
    """
    Create a metrics calculator instance with default settings.
    
    Args:
        threshold: Classification threshold
        
    Returns:
        MetricsCalculator instance
    """
    return MetricsCalculator(threshold=threshold)