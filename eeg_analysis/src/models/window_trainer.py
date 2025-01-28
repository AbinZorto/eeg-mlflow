import numpy as np
import pandas as pd
from src.models.base_trainer import BaseTrainer
from src.models.evaluation import ModelEvaluator
from sklearn.base import BaseEstimator
import mlflow
from sklearn.model_selection import LeaveOneGroupOut
from src.models.model_utils import save_model_results, log_feature_importance
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class WindowLevelTrainer(BaseTrainer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_type = config['model']['type']
        self.model_params = config['model']['params'][self.model_type]
        self.output_dir = config['paths']['models']
        self.metrics = config['metrics']['window_level']
    
    def train(self, data_path: str = None) -> BaseEstimator:
        if data_path is None:
            data_path = self.config['data']['feature_path']
            
        evaluator = ModelEvaluator(metrics=self.metrics)
        
        # Load and prepare data
        df = pd.read_parquet(data_path)
        X, y, groups = self._prepare_data(df)
        
        # Log detailed dataset statistics
        unique_patients = groups.unique()
        patient_labels = df.groupby('Participant')['Remission'].first()
        n_remission = sum(patient_labels == 1)
        n_non_remission = sum(patient_labels == 0)
        
        logger.info("\nDataset Statistics:")
        logger.info(f"Total number of patients: {len(unique_patients)}")
        logger.info(f"- Remission patients: {n_remission}")
        logger.info(f"- Non-remission patients: {n_non_remission}")
        logger.info(f"Total number of windows: {len(df)}")
        
        # Log windows per patient statistics
        windows_per_patient = df.groupby('Participant').size()
        logger.info("\nWindows per patient:")
        logger.info(f"- Mean: {windows_per_patient.mean():.1f}")
        logger.info(f"- Min: {windows_per_patient.min()}")
        logger.info(f"- Max: {windows_per_patient.max()}")
        logger.info(f"- Median: {windows_per_patient.median()}")
        
        self._log_dataset_info(X, y, groups)
        patient_predictions = []
        window_predictions = []
        
        # Cross-validation
        logo = LeaveOneGroupOut()
        n_splits = logo.get_n_splits(X, y, groups)
        logger.info(f"\nPerforming Leave-One-Group-Out cross-validation with {n_splits} splits")
        
        # Store true and predicted labels for patient-level evaluation
        patient_true_labels = []
        patient_pred_labels = []
        patient_pred_probs = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            test_participant = groups.iloc[test_idx].unique()[0]
            true_label = y.iloc[test_idx].iloc[0]
            patient_true_labels.append(true_label)
            
            logger.info(f"\nFold {fold_idx + 1}/{n_splits}")
            logger.info(f"Testing on participant: {test_participant} (true label: {true_label})")
            logger.info(f"Training windows: {len(train_idx)}")
            logger.info(f"Test windows: {len(test_idx)}")
            
            # Use nested runs for each fold
            with mlflow.start_run(run_name=f"fold_{fold_idx}", nested=True):
                model = self._create_and_train_model(X.iloc[train_idx], y.iloc[train_idx])
                y_pred = model.predict(X.iloc[test_idx])
                y_prob = model.predict_proba(X.iloc[test_idx])[:, 1]
                
                # Calculate patient-level prediction
                patient_prob = np.mean(y_prob)
                patient_pred = 1 if patient_prob >= 0.5 else 0
                patient_pred_labels.append(patient_pred)
                patient_pred_probs.append(patient_prob)
                
                # Store window-level predictions
                window_predictions.extend(self._create_window_predictions(
                    fold_idx, test_participant, y.iloc[test_idx], y_pred, y_prob))
                
                # Store patient-level prediction
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
                mlflow.log_metric(f"fold_{fold_idx}_patient_accuracy", 
                                int(true_label == patient_pred))
                mlflow.log_metric(f"fold_{fold_idx}_window_accuracy", 
                                np.mean(y_pred == y.iloc[test_idx]))
        
        # Calculate and log patient-level metrics
        patient_metrics = evaluator.evaluate_patient_predictions(
            np.array(patient_true_labels),
            np.array(patient_pred_labels),
            np.array(patient_pred_probs)
        )
        
        # Log overall metrics
        mlflow.log_metrics({f"patient_{k}": v for k, v in patient_metrics.items()})
        mlflow.log_params(self.model_params)
        
        # Train final model on all data
        final_model = self._create_and_train_model(X, y)
        self._save_results(final_model, patient_metrics, window_predictions, 
                         patient_predictions, X.columns)
        
        return final_model

    def _create_window_predictions(self, fold_idx: int, participant: str, 
                                 y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_prob: np.ndarray) -> List[Dict[str, Any]]:
        """
        Create window-level prediction records.
        
        Args:
            fold_idx: Index of the current fold
            participant: Participant ID
            y_true: True labels for windows
            y_pred: Predicted labels for windows
            y_prob: Prediction probabilities for windows
            
        Returns:
            List of dictionaries containing prediction information for each window
        """
        return [{
            'fold': fold_idx,
            'participant': participant,
            'true_label': y_true.iloc[idx],
            'predicted_label': pred,
            'probability': prob,
            'correct': y_true.iloc[idx] == pred
        } for idx, (pred, prob) in enumerate(zip(y_pred, y_prob))]

    def _create_patient_prediction(self, fold_idx, groups, y_true, y_pred, y_prob):
        avg_prob = np.mean(y_prob)
        return {
            'fold': fold_idx,
            'participant': groups.iloc[0],
            'true_label': y_true.iloc[0],
            'predicted_label': 1 if avg_prob >= 0.5 else 0,
            'probability': avg_prob,
            'n_windows': len(y_true),
            'n_positive_windows': sum(y_pred == 1),
            'is_window': False
        }

    def _save_results(self, model: BaseEstimator, patient_metrics: Dict[str, float],
                     window_predictions: List[Dict[str, Any]], 
                     patient_predictions: List[Dict[str, Any]], 
                     feature_names: List[str]) -> None:
        """
        Save model results and predictions.
        
        Args:
            model: Trained model
            patient_metrics: Dictionary of patient-level metrics
            window_predictions: List of window-level predictions
            patient_predictions: List of patient-level predictions
            feature_names: List of feature names
        """
        # Create DataFrames from predictions
        window_df = pd.DataFrame(window_predictions)
        patient_df = pd.DataFrame(patient_predictions)
        
        # Log metrics
        mlflow.log_metrics(patient_metrics)
        
        # Save predictions if configured
        if self.config['output']['save_predictions']:
            window_df.to_csv('window_predictions.csv', index=False)
            patient_df.to_csv('patient_predictions.csv', index=False)
            mlflow.log_artifact('window_predictions.csv')
            mlflow.log_artifact('patient_predictions.csv')
        
        # Log feature importance if configured
        if self.config['output']['feature_importance']:
            log_feature_importance(model, feature_names)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        logger.info("\nMisclassified Patients:")
        misclassified = patient_df[
            patient_df['true_label'] != patient_df['predicted_label']
        ]
        for _, row in misclassified.iterrows():
            logger.info(
                f"Participant {row['participant']}: "
                f"True={row['true_label']}, Pred={row['predicted_label']}, "
                f"Confidence={row['probability']:.3f}"
            )