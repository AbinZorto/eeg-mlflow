import numpy as np
import pandas as pd
from src.models.base_trainer import BaseTrainer
from src.models.evaluation import ModelEvaluator
from sklearn.base import BaseEstimator
import mlflow
from sklearn.model_selection import LeaveOneGroupOut
from src.models.model_utils import save_model_results, log_feature_importance, create_classifier
from pathlib import Path
from typing import Dict, Any, List
import logging
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SelectFromModel, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from mlflow.models.signature import infer_signature

logger = logging.getLogger(__name__)

class WindowLevelTrainer(BaseTrainer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_type = config['model_type']
        self.model_params = config['model']['params'][self.model_type]
        self.output_dir = config['paths']['models']
        self.metrics = config['metrics']['window_level']
        self.feature_selection_config = config.get('feature_selection', {})
        logger.info(f"WindowLevelTrainer initialized with feature_selection_config: {self.feature_selection_config}")
    
    def _create_model_instance(self) -> BaseEstimator:
        """Creates a model instance based on the trainer's configuration."""
        # Uses self.model_type and self.model_params which are specific to WindowLevelTrainer
        return create_classifier(self.model_type, self.model_params)

    def _get_feature_importances(self, model: BaseEstimator, feature_names: List[str]) -> pd.DataFrame:
        """Helper to get feature importances from a model."""
        importances_values = None
        final_estimator = None

        if hasattr(model, 'steps') and hasattr(model, 'named_steps'): # Check if it's a Pipeline
            final_estimator = model.steps[-1][1]
        else: # Assume it's a direct estimator
            final_estimator = model
        
        # Add specific check for SVM non-linear kernel
        if isinstance(final_estimator, SVC) and final_estimator.kernel != 'linear':
            logger.info(f"Model '{type(final_estimator).__name__}' with kernel '{final_estimator.kernel}' does not have direct feature_importances_ or coef_ for standard feature ranking. Permutation importance could be an alternative if detailed importances are needed.")
            return None  # Early return for non-linear SVM

        if hasattr(final_estimator, 'feature_importances_'):
            importances_values = final_estimator.feature_importances_
        elif hasattr(final_estimator, 'coef_'):
            # For linear models, coef_ can be used. Use absolute values.
            # If coef_ is 2D (e.g. multi-class LogisticRegression), take the mean across classes or handle appropriately.
            # For simplicity, assuming 1D coef_ or taking the first row for multi-class like in some selectors.
            if final_estimator.coef_.ndim == 1:
                importances_values = np.abs(final_estimator.coef_)
            elif final_estimator.coef_.ndim == 2: # e.g. OvR Logistic Regression
                 # Option 1: Use L2 norm across classes for each feature
                # importances_values = np.linalg.norm(final_estimator.coef_, axis=0, ord=2)
                # Option 2: Use mean of absolute coefficients across classes (simplistic, might not always be best)
                # importances_values = np.mean(np.abs(final_estimator.coef_), axis=0)
                # Option 3: Or, if it makes sense for the specific model (e.g. one class vs rest for a specific target class)
                # For now, let's try taking the L2 norm, which is a common approach for multi-class coef_ based importance.
                logger.info(f"Model '{type(final_estimator).__name__}' has 2D coef_ attribute. Using L2 norm for feature importance.")
                importances_values = np.linalg.norm(final_estimator.coef_, axis=0)
            else:
                logger.warning(f"Model '{type(final_estimator).__name__}' has coef_ attribute with unexpected shape: {final_estimator.coef_.shape}. Cannot determine importances.")

        if importances_values is not None:
            if len(importances_values) == len(feature_names):
                return pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances_values
                }).sort_values('importance', ascending=False)
            else:
                logger.warning(f"Length of importances ({len(importances_values)}) does not match number of features ({len(feature_names)}). Cannot determine importances.")
        return None

    def _select_features(self, X: pd.DataFrame, y: pd.Series, n_features_to_select: int) -> List[str]:
        """
        Select top N features based on various methods.

        Available methods:
        - 'model_based': Uses feature importances from a model of the same type as the main trainer.
                         Good for capturing interactions if the model type supports it (e.g., tree-based).
        - 'select_k_best_f_classif': Selects features using ANOVA F-value between label/feature for classification tasks.
                                     Fast and effective for linear relationships.
        - 'select_k_best_mutual_info': Selects features based on mutual information between each feature and the target.
                                         Can capture non-linear relationships. Might be more computationally intensive than f_classif.
        - 'select_from_model_l1': Uses Logistic Regression with L1 penalty (Lasso) to select features.
                                    Features with non-zero coefficients are selected. Good for sparse feature sets.
        - 'rfe': Recursive Feature Elimination. Recursively fits a model (Logistic Regression by default here)
                 and removes the weakest features until the target number of features is reached. 
                 Can be powerful but computationally more expensive.
        """
        logger.info(f"_select_features called with n_features_to_select: {n_features_to_select}")
        selection_method = self.feature_selection_config.get('method', 'model_based')
        logger.info(f"Performing feature selection using '{selection_method}'. Selecting top {n_features_to_select} features.")
        
        selected_features = []
        original_feature_names = X.columns.tolist()

        if n_features_to_select <= 0 or n_features_to_select > X.shape[1]:
            logger.warning(f"Invalid n_features_to_select ({n_features_to_select}). Must be between 1 and {X.shape[1]}. Selecting all features.")
            mlflow.log_param("num_selected_features", X.shape[1])
            mlflow.log_param("selected_features_list", original_feature_names)
            mlflow.log_param("feature_selection_method", selection_method)
            return original_feature_names

        if selection_method == 'model_based':
            # Trains a model (same type as specified in config) on the current data fold 
            # and uses its feature_importances_ attribute.
            # Pros: Considers feature interactions if the model supports it (e.g., tree ensembles).
            # Cons: Can be slower if model training is expensive. Importance type depends on the model.
            temp_model = self._create_model_instance()
            temp_model.fit(X, y)
            importances_df = self._get_feature_importances(temp_model, original_feature_names)
            
            if importances_df is None or importances_df.empty:
                logger.warning("Could not determine feature importances for model_based selection. Defaulting to all features.")
                selected_features = original_feature_names
            else:
                selected_features = importances_df['feature'].head(n_features_to_select).tolist()

        elif selection_method == 'select_k_best_f_classif':
            # Uses ANOVA F-statistic to select features. Tests for a linear relationship 
            # between each feature and the categorical target variable.
            # Pros: Fast, simple, easy to understand.
            # Cons: Only captures linear relationships, may miss features important in non-linear contexts.
            selector = SelectKBest(score_func=f_classif, k=n_features_to_select)
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            feature_scores = pd.DataFrame({'feature': X.columns, 'score': selector.scores_})
            feature_scores = feature_scores.sort_values('score', ascending=False)
            mlflow.log_dict(feature_scores.head(min(n_features_to_select * 2, X.shape[1])).to_dict('records'), "select_k_best_f_classif_scores.json")

        elif selection_method == 'select_k_best_mutual_info':
            # Uses mutual information between each feature and the target variable.
            # Mutual information measures the dependency between two variables, capturing non-linear relationships.
            # Pros: Can identify non-linear relationships. Model-agnostic.
            # Cons: Can be more computationally intensive than f_classif. Requires discrete or binned continuous features for some implementations, 
            #       but scikit-learn handles continuous features by estimating entropy.
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features_to_select)
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            feature_scores = pd.DataFrame({'feature': X.columns, 'score': selector.scores_})
            feature_scores = feature_scores.sort_values('score', ascending=False)
            mlflow.log_dict(feature_scores.head(min(n_features_to_select * 2, X.shape[1])).to_dict('records'), "select_k_best_mutual_info_scores.json")

        elif selection_method == 'select_from_model_l1':
            # Uses a model with L1 regularization (Lasso) to perform feature selection.
            # L1 regularization encourages sparse coefficients (some become zero).
            # Here, Logistic Regression with L1 penalty is used.
            # Pros: Embedded method, considers feature interactions if the model does. Effective for high-dimensional data.
            # Cons: Performance depends on the chosen model and its hyperparameters (e.g., C for LogisticRegression).
            #       The number of selected features is controlled by the threshold parameter of SelectFromModel or implicitly by C.
            #       We aim for n_features_to_select by adjusting C, but it might not be exact. Here we use a fixed C and rely on max_features.
            estimator = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', random_state=self.config.get('random_seed', 42), C=1.0)
            selector = SelectFromModel(estimator, max_features=n_features_to_select, threshold=-np.inf) # threshold=-np.inf ensures max_features is used
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            # To get importances, we can look at the coefficients of the fitted L1 model
            estimator.fit(X,y) # fit again to get coef_ on all original features
            if hasattr(estimator, 'coef_'):
                importances = np.abs(estimator.coef_[0])
                feature_importances_df = pd.DataFrame({'feature': X.columns, 'L1_coeff_abs': importances})
                feature_importances_df = feature_importances_df.sort_values('L1_coeff_abs', ascending=False)
                mlflow.log_dict(feature_importances_df.head(min(len(original_feature_names), X.shape[1])).to_dict('records'), "select_from_model_l1_coeffs.json")

        elif selection_method == 'rfe':
            # Recursive Feature Elimination (RFE).
            # It fits a model, ranks features (e.g., by coefficient magnitude or feature importance),
            # and recursively removes the least important ones until the desired number of features is reached.
            # Pros: Can be very effective in finding a good subset of features.
            # Cons: Computationally expensive, especially with many features or complex models.
            #       Performance depends on the estimator used.
            estimator = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=self.config.get('random_seed', 42))
            selector = RFE(estimator, n_features_to_select=n_features_to_select)
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            # RFE ranking can be logged
            if hasattr(selector, 'ranking_'):
                rankings_df = pd.DataFrame({'feature': X.columns, 'rfe_ranking': selector.ranking_})
                rankings_df = rankings_df.sort_values('rfe_ranking', ascending=True)
                mlflow.log_dict(rankings_df.head(min(len(original_feature_names),X.shape[1])).to_dict('records'), "rfe_rankings.json")

        else:
            logger.warning(f"Unknown feature selection method: {selection_method}. Defaulting to all features.")
            selected_features = original_feature_names
            
        if not selected_features: 
            logger.warning(f"Feature selection method '{selection_method}' resulted in no features. Defaulting to all features.")
            selected_features = original_feature_names

        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
        mlflow.log_param("num_selected_features", len(selected_features))
        mlflow.log_param("selected_features_list", selected_features) 
        mlflow.log_param("feature_selection_method", selection_method)
        return selected_features

    def train(self, data_path: str = None) -> BaseEstimator:
        """
        Train a window-level model.
        
        Args:
            data_path: Optional path to feature data. If None, uses config path.
            
        Returns:
            Trained model
        """
        if data_path is None:
            data_path = self.config['data']['feature_path']
            
        evaluator = ModelEvaluator(metrics=self.metrics)
        
        # Log the data path being used
        mlflow.log_param("feature_path", data_path)
        
        # Load and prepare data
        df = pd.read_parquet(data_path)
        X_orig, y, groups = self._prepare_data(df)
        
        # Feature Selection
        perform_selection = self.feature_selection_config.get('enabled', False)
        logger.info(f"In train method: perform_selection is {perform_selection}, feature_selection_config: {self.feature_selection_config}")
        
        if perform_selection:
            n_features_target = self.feature_selection_config.get('n_features', 10) 
            logger.info(f"In train method, feature selection enabled. Target n_features: {n_features_target}")
            
            selected_feature_names = self._select_features(X_orig, y, n_features_target)
            X = X_orig[selected_feature_names]

            mlflow.log_param("feature_selection_enabled", True)
            mlflow.log_param("target_n_features_to_select", n_features_target)
            # num_selected_features, selected_features_list, and feature_selection_method are logged by _select_features

            num_actually_selected = len(selected_feature_names)

            if num_actually_selected == X_orig.shape[1] and n_features_target < X_orig.shape[1]:
                # This case means: selection was intended to reduce features, but it resulted in all original features.
                # This typically happens if the selection method failed (e.g., model_based incompatible) and defaulted.
                logger.warning(
                    f"Feature selection was enabled targeting {n_features_target} features, "
                    f"but the method '{self.feature_selection_config.get('method')}' "
                    f"for model '{self.model_type}' resulted in all {num_actually_selected} features being used. "
                    f"This may indicate the selection method could not effectively reduce features or was incompatible. "
                    "Please check previous logs for details (e.g., 'Could not determine feature importances')."
                )
                mlflow.log_param("feature_selection_effective", False)
            elif num_actually_selected < X_orig.shape[1] or n_features_target >= X_orig.shape[1]:
                # Selection successfully reduced features OR selection targeted all/more features anyway.
                # If n_features_target >= X_orig.shape[1], _select_features might return all features, which is "effective".
                mlflow.log_param("feature_selection_effective", True)
            # Note: if num_actually_selected == X_orig.shape[1] because n_features_target was also X_orig.shape[1] (or greater),
            # then n_features_target < X_orig.shape[1] is false, so it falls into the elif and logs effective=True. Correct.
            
        else: # perform_selection is False
            X = X_orig
            selected_feature_names = X_orig.columns.tolist() # Ensure selected_feature_names is defined for consistency if needed later, though X is primary.
            mlflow.log_param("feature_selection_enabled", False)
            # No target_n_features_to_select, num_selected_features, selected_features_list, 
            # feature_selection_method, or feature_selection_effective if selection is off.
            # These are logged by _select_features only if it runs.

        # Store the selected feature names as an instance variable for external access
        self.selected_feature_names = selected_feature_names

        # Log the actual number of features used for training as a metric
        mlflow.log_metric("num_features_trained_on", len(X.columns))

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
                         patient_predictions, X)
        
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
                     X_final_train: pd.DataFrame) -> None:
        """
        Save model results and predictions.
        
        Args:
            model: Trained model
            patient_metrics: Dictionary of patient-level metrics
            window_predictions: List of window-level predictions
            patient_predictions: List of patient-level predictions
            X_final_train: DataFrame used to train the final model (for signature and feature names)
        """
        feature_names = X_final_train.columns.tolist() # Get feature names from X_final_train
        # Create DataFrames from predictions
        window_df = pd.DataFrame(window_predictions)
        patient_df = pd.DataFrame(patient_predictions)
        
        # Convert any numpy types in metrics to Python native types
        for key, value in patient_metrics.items():
            if hasattr(value, 'item'):  # Check if it's a numpy type
                patient_metrics[key] = value.item()  # Convert to Python native type
        
        # Log metrics
        mlflow.log_metrics(patient_metrics)
        
        

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
            window_pred_path = output_dir / 'window_predictions.csv'
            patient_pred_path = output_dir / 'patient_predictions.csv'
            
            window_df.to_csv(window_pred_path, index=False)
            patient_df.to_csv(patient_pred_path, index=False)
            
            # Log to MLflow
            mlflow.log_artifact(str(window_pred_path))
            mlflow.log_artifact(str(patient_pred_path))
            
            logger.info(f"Saved window predictions to {window_pred_path}")
            logger.info(f"Saved patient predictions to {patient_pred_path}")
        
        # Log feature importance if configured
        if self.config['output']['feature_importance']:
            # Try to get feature importances - handle both direct models and pipelines
            importances = None
            
            # Check if it's a pipeline with a final estimator that has feature_importances_
            if hasattr(model, 'steps') and hasattr(model, 'named_steps'):
                # It's a pipeline - get the last step (usually the estimator)
                final_estimator = model.steps[-1][1]
                if hasattr(final_estimator, 'feature_importances_') or hasattr(final_estimator, 'coef_'):
                    importances_df = self._get_feature_importances(model, feature_names)
                    if importances_df is not None:
                        importances = importances_df
            # Check if it's a direct model with feature_importances_ or coef_
            elif hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                importances_df = self._get_feature_importances(model, feature_names)
                if importances_df is not None:
                    importances = importances_df
            
            # Save importances if we found them
            if importances is not None and not importances.empty:
                importance_path = output_dir / 'feature_importance.csv'
                importances.to_csv(importance_path, index=False)
                
                # Log to MLflow
                mlflow.log_artifact(str(importance_path))
                logger.info(f"Saved feature importance to {importance_path}")
            else:
                logger.warning("Could not extract feature importances from the model")
        
        # Log model with signature
        input_example = X_final_train.head()
        try:
            # Predict on a small sample to help infer output schema
            # Ensure model is fitted if it's a freshly created one for some reason (though it should be fitted)
            if hasattr(model, "predict"):
                 prediction_example = model.predict(input_example)
                 signature = infer_signature(input_example, prediction_example)
            else: # Should not happen for sklearn models used here
                signature = infer_signature(input_example)
            mlflow.sklearn.log_model(model, "model", signature=signature, input_example=input_example)
        except Exception as e:
            logger.error(f"Failed to log model with signature: {e}. Logging model without signature.")
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
            'metrics': patient_metrics,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        import json
        with open(output_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
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