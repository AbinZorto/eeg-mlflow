import numpy as np
import pandas as pd
from src.models.base_trainer import BaseTrainer
from src.models.evaluation import ModelEvaluator
from sklearn.base import BaseEstimator
from src.models.model_utils import save_model_results, log_feature_importance
import mlflow
from sklearn.model_selection import LeaveOneGroupOut
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PatientLevelTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.model_type = config['model_type']
        self.model_params = config['model']['params'][self.model_type]
        self.output_dir = config['paths']['models']
        self.metrics = config.get('metrics', ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
        self.model_name = f"patient_level_{self.model_type}"
    
    def aggregate_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_cols = df.columns.difference(['Participant', 'Remission'])
        agg_funcs = {col: ['mean', 'std', 'min', 'max', 'median'] for col in feature_cols}
        agg_funcs['Remission'] = 'first'
        
        patient_df = df.groupby('Participant').agg(agg_funcs)
        patient_df.columns = [f"{col}_{agg}" if agg != 'first' else col 
                            for col, agg in patient_df.columns]
        patient_df['n_windows'] = df.groupby('Participant').size()
        return patient_df.reset_index()

    def train(self, data_path: str = None) -> BaseEstimator:
        """
        Train a patient-level model.
        
        Args:
            data_path: Optional path to feature data. If None, uses config path.
            
        Returns:
            Trained model
        """
        evaluator = ModelEvaluator()
        
        if data_path is None:
            data_path = self.config['data']['feature_path']
        
        with mlflow.start_run(run_name=self.model_name):
            # Log the data path being used
            mlflow.log_param("feature_path", data_path)
            
            df = pd.read_parquet(data_path)
            patient_df = self.aggregate_windows(df)
            X, y, groups = self._prepare_data(patient_df)
            
            self._log_dataset_info(X, y, groups)
            metrics = []
            predictions = []
            
            # Cross-validation
            logo = LeaveOneGroupOut()
            for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
                model = self._create_and_train_model(X.iloc[train_idx], y.iloc[train_idx])
                y_pred = model.predict(X.iloc[test_idx])
                y_prob = model.predict_proba(X.iloc[test_idx])[:, 1]
                
                fold_metrics = evaluator.evaluate_window_predictions(y.iloc[test_idx], y_pred, y_prob)
                metrics.append(fold_metrics)
                predictions.append(self._create_prediction_record(fold_idx, groups.iloc[test_idx], 
                                                               y.iloc[test_idx], y_pred, y_prob))
            
            # Final model and saving
            final_model = self._create_and_train_model(X, y)
            self._save_results(final_model, metrics, predictions, X.columns)
            return final_model

    def _create_prediction_record(self, fold_idx, groups, y_true, y_pred, y_prob):
        return {
            'fold': fold_idx,
            'participant': groups.iloc[0],
            'true_label': y_true.iloc[0],
            'predicted_label': y_pred[0],
            'probability': y_prob[0]
        }

    def _save_results(self, model, metrics, predictions_list, feature_names):
        """
        Save model results and predictions.
        
        Args:
            model: Trained model
            metrics: List of dictionaries with metrics for each fold
            predictions_list: List of prediction records
            feature_names: List of feature names
        """
        # Create DataFrame from predictions
        predictions_df = pd.DataFrame(predictions_list)
        
        # Calculate average metrics across folds
        avg_metrics = {k: np.mean([m[k] for m in metrics]) for k in metrics[0].keys()}
        
        # Log metrics to MLflow
        mlflow.log_metrics(avg_metrics)
        
        # Get window size from data path or config
        window_size = None
        if 'window_size' in self.config:
            window_size = self.config['window_size']
        else:
            # Try to extract from data path
            data_path = self.config['data']['feature_path']
            if '{window_size}s' in data_path:
                # The actual path should have the window size filled in
                import re
                match = re.search(r'(\d+)s_window_features', data_path)
                if match:
                    window_size = match.group(1)
        
        # Create output directory with window size
        output_dir = Path(self.output_dir)
        if window_size:
            output_dir = output_dir / f"{window_size}s_window"
        else:
            output_dir = output_dir / "default_window"
        
        # Create directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save predictions if configured
        if self.config['output']['save_predictions']:
            pred_path = output_dir / 'patient_predictions.csv'
            predictions_df.to_csv(pred_path, index=False)
            
            # Log to MLflow
            mlflow.log_artifact(str(pred_path))
            logger.info(f"Saved patient predictions to {pred_path}")
        
        # Log feature importance if configured
        if self.config['output']['feature_importance']:
            # Save feature importance to the same directory
            importance_path = output_dir / 'feature_importance.csv'
            log_feature_importance(model, feature_names, output_path=str(importance_path))
            mlflow.log_artifact(str(importance_path))
            logger.info(f"Saved feature importance to {importance_path}")
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model to disk
        model_path = output_dir / 'model.joblib'
        import joblib
        joblib.dump(model, model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Save model metadata
        metadata = {
            'window_size': window_size,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'metrics': avg_metrics,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        import json
        with open(output_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("\nMisclassified Patients:")
        misclassified = predictions_df[
            predictions_df['true_label'] != predictions_df['predicted_label']
        ]
        for _, row in misclassified.iterrows():
            logger.info(
                f"Participant {row['participant']}: "
                f"True={row['true_label']}, Pred={row['predicted_label']}, "
                f"Confidence={row['probability']:.3f}"
            )