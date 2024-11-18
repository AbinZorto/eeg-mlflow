import numpy as np
import pandas as pd
from src.models.base_trainer import BaseTrainer
from src.models.evaluation import ModelEvaluator
from sklearn.base import BaseEstimator
from src.models.model_utils import save_model_results, log_feature_importance
import mlflow
from sklearn.model_selection import LeaveOneGroupOut

class PatientLevelTrainer(BaseTrainer):
    def aggregate_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_cols = df.columns.difference(['Participant', 'Remission'])
        agg_funcs = {col: ['mean', 'std', 'min', 'max', 'median'] for col in feature_cols}
        agg_funcs['Remission'] = 'first'
        
        patient_df = df.groupby('Participant').agg(agg_funcs)
        patient_df.columns = [f"{col}_{agg}" if agg != 'first' else col 
                            for col, agg in patient_df.columns]
        patient_df['n_windows'] = df.groupby('Participant').size()
        return patient_df.reset_index()

    def train(self, data_path: str) -> BaseEstimator:
        evaluator = ModelEvaluator()
        
        with mlflow.start_run(run_name=f"patient_level_{self.classifier_name}"):
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

    def _save_results(self, model, metrics, predictions, feature_names):
        avg_metrics = {k: np.mean([m[k] for m in metrics]) for k in metrics[0].keys()}
        save_model_results(model, avg_metrics, self.model_name, self.config['output_path'])
        log_feature_importance(model, feature_names)
        self._save_predictions(pd.DataFrame(predictions), 'predictions.csv')