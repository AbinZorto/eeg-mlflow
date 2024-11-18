import numpy as np
import pandas as pd
from src.models.base_trainer import BaseTrainer
from src.models.evaluation import ModelEvaluator
from sklearn.base import BaseEstimator
import mlflow
from sklearn.model_selection import LeaveOneGroupOut
from src.models.model_utils import save_model_results, log_feature_importance

class WindowLevelTrainer(BaseTrainer):
    def train(self, data_path: str) -> BaseEstimator:
        evaluator = ModelEvaluator()
        
        with mlflow.start_run(run_name=f"window_level_{self.classifier_name}"):
            df = pd.read_parquet(data_path)
            X, y, groups = self._prepare_data(df)
            
            self._log_dataset_info(X, y, groups)
            window_metrics = []
            window_predictions = []
            patient_predictions = []
            
            # Cross-validation
            logo = LeaveOneGroupOut()
            for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
                model = self._create_and_train_model(X.iloc[train_idx], y.iloc[train_idx])
                y_pred = model.predict(X.iloc[test_idx])
                y_prob = model.predict_proba(X.iloc[test_idx])[:, 1]
                
                test_groups = groups.iloc[test_idx]
                window_metrics.append(evaluator.evaluate_window_predictions(y.iloc[test_idx], y_pred, y_prob))
                
                window_predictions.extend(self._create_window_predictions(fold_idx, test_groups, 
                                                                       y.iloc[test_idx], y_pred, y_prob))
                patient_predictions.append(self._create_patient_prediction(fold_idx, test_groups, 
                                                                        y.iloc[test_idx], y_pred, y_prob))
            
            # Final model and saving
            final_model = self._create_and_train_model(X, y)
            self._save_results(final_model, window_metrics, window_predictions, 
                             patient_predictions, X.columns)
            return final_model

    def _create_window_predictions(self, fold_idx, groups, y_true, y_pred, y_prob):
        return [{
            'fold': fold_idx,
            'participant': groups.iloc[idx],
            'true_label': y_true.iloc[idx],
            'predicted_label': y_pred[idx],
            'probability': y_prob[idx],
            'is_window': True
        } for idx in range(len(y_true))]

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

    def _save_results(self, model, window_metrics, window_predictions, 
                     patient_predictions, feature_names):
        avg_metrics = {k: np.mean([m[k] for m in window_metrics]) for k in window_metrics[0].keys()}
        save_model_results(model, avg_metrics, self.model_name, self.config['output_path'])
        log_feature_importance(model, feature_names)
        
        self._save_predictions(pd.DataFrame(window_predictions), 'window_predictions.csv')
        self._save_predictions(pd.DataFrame(patient_predictions), 'patient_predictions.csv')