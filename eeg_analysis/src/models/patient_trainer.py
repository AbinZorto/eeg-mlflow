import numpy as np
import pandas as pd
from src.models.base_trainer import BaseTrainer
from src.models.evaluation import ModelEvaluator
from sklearn.base import BaseEstimator
from src.models.model_utils import save_model_results, log_feature_importance, create_classifier, ModelBuilder
import mlflow
from sklearn.model_selection import LeaveOneGroupOut
from pathlib import Path
from typing import Dict, Any, List
import logging
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SelectFromModel, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from mlflow.models.signature import infer_signature

logger = logging.getLogger(__name__)

class PatientLevelTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.model_type = config.get('model_type', 'auto')  # 'auto' for automatic selection
        self.model_params = config.get('model', {}).get('params', {})
        self.output_dir = config['paths']['models']
        self.metrics = config.get('metrics', ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
        self.model_name = f"patient_level_{self.model_type}"
        self.feature_selection_config = config.get('feature_selection', {})
        self.use_class_weights = config.get('use_class_weights', True)
        logger.info(f"PatientLevelTrainer initialized with model_type: {self.model_type}")
        logger.info(f"Feature selection config: {self.feature_selection_config}")
    
    def _get_classifiers(self):
        """Return dictionary of sklearn classifiers to try using centralized model_utils."""
        # If specific model type is requested, return only that one
        if self.model_type != 'auto' and self.model_type in ModelBuilder.get_classifier_names():
            model_params = self.model_params.get(self.model_type, {})
            return {self.model_type: create_classifier(self.model_type, model_params)}
        elif self.model_type != 'auto':
            # Fallback for backwards compatibility with older model types
            model_params = self.model_params.get(self.model_type, {})
            model = create_classifier(self.model_type, model_params)
            return {self.model_type: model}
        
        # Return all classifiers for auto mode
        return ModelBuilder.get_all_classifiers(
            use_class_weights=self.use_class_weights,
            config_params=self.model_params
        )

    def _calculate_detailed_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate detailed metrics with counts"""
        # Basic counts
        TP = sum((y_true == 1) & (y_pred == 1))
        TN = sum((y_true == 0) & (y_pred == 0))
        FP = sum((y_true == 0) & (y_pred == 1))
        FN = sum((y_true == 1) & (y_pred == 0))
        
        # Calculate metrics
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

    def _create_model_instance(self) -> BaseEstimator:
        """Creates a model instance based on the trainer's configuration."""
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
            if final_estimator.coef_.ndim == 1:
                importances_values = np.abs(final_estimator.coef_)
            elif final_estimator.coef_.ndim == 2: # e.g. OvR Logistic Regression
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
        - 'select_k_best_f_classif': Selects features using ANOVA F-value between label/feature for classification tasks.
        - 'select_k_best_mutual_info': Selects features based on mutual information between each feature and the target.
        - 'select_from_model_l1': Uses Logistic Regression with L1 penalty (Lasso) to select features.
        - 'rfe': Recursive Feature Elimination.
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
            temp_model = self._create_model_instance()
            temp_model.fit(X, y)
            importances_df = self._get_feature_importances(temp_model, original_feature_names)
            
            if importances_df is None or importances_df.empty:
                logger.warning("Could not determine feature importances for model_based selection. Defaulting to all features.")
                selected_features = original_feature_names
            else:
                selected_features = importances_df['feature'].head(n_features_to_select).tolist()

        elif selection_method == 'select_k_best_f_classif':
            selector = SelectKBest(score_func=f_classif, k=n_features_to_select)
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            feature_scores = pd.DataFrame({'feature': X.columns, 'score': selector.scores_})
            feature_scores = feature_scores.sort_values('score', ascending=False)
            mlflow.log_dict(feature_scores.head(min(n_features_to_select * 2, X.shape[1])).to_dict('records'), "select_k_best_f_classif_scores.json")

        elif selection_method == 'select_k_best_mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features_to_select)
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            feature_scores = pd.DataFrame({'feature': X.columns, 'score': selector.scores_})
            feature_scores = feature_scores.sort_values('score', ascending=False)
            mlflow.log_dict(feature_scores.head(min(n_features_to_select * 2, X.shape[1])).to_dict('records'), "select_k_best_mutual_info_scores.json")

        elif selection_method == 'select_from_model_l1':
            estimator = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', random_state=self.config.get('random_seed', 42), C=1.0)
            selector = SelectFromModel(estimator, max_features=n_features_to_select, threshold=-np.inf)
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            estimator.fit(X,y)
            if hasattr(estimator, 'coef_'):
                importances = np.abs(estimator.coef_[0])
                feature_importances_df = pd.DataFrame({'feature': X.columns, 'L1_coeff_abs': importances})
                feature_importances_df = feature_importances_df.sort_values('L1_coeff_abs', ascending=False)
                mlflow.log_dict(feature_importances_df.head(min(len(original_feature_names), X.shape[1])).to_dict('records'), "select_from_model_l1_coeffs.json")

        elif selection_method == 'rfe':
            estimator = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=self.config.get('random_seed', 42))
            selector = RFE(estimator, n_features_to_select=n_features_to_select)
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
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
    
    def aggregate_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced aggregation of window-level features to patient-level features.
        Includes mean, std, min, max, median, and percentiles.
        """
        feature_cols = df.columns.difference(['Participant', 'Remission'])
        
        # Define aggregation functions
        agg_funcs = {col: ['mean', 'std', 'min', 'max', 'median'] 
                     for col in feature_cols}
        agg_funcs['Remission'] = 'first'  # All windows for a patient have same label
        
        # Aggregate
        patient_df = df.groupby('Participant').agg(agg_funcs)
        
        # Add percentiles efficiently using pd.concat
        percentiles = [25, 75]
        percentile_dfs = []
        
        for p in percentiles:
            percentile_data = {}
            for col in feature_cols:
                percentile_data[(col, f'percentile_{p}')] = df.groupby('Participant')[col].quantile(p/100)
            percentile_df = pd.DataFrame(percentile_data)
            percentile_dfs.append(percentile_df)
        
        # Concatenate all percentile DataFrames at once
        if percentile_dfs:
            all_percentiles = pd.concat(percentile_dfs, axis=1)
            patient_df = pd.concat([patient_df, all_percentiles], axis=1)
        
        # Flatten column names
        patient_df.columns = [f"{col}_{agg}" if agg != 'first' else col 
                             for col, agg in patient_df.columns]
        
        # Add number of windows as a feature
        patient_df['n_windows'] = df.groupby('Participant').size()
        
        # Ensure all numeric columns are float64
        numeric_cols = patient_df.select_dtypes(include=['number']).columns
        patient_df[numeric_cols] = patient_df[numeric_cols].astype('float64')
        
        return patient_df.reset_index()

    def train(self, data_path: str = None) -> BaseEstimator:
        """
        Train a patient-level model with automatic classifier selection or single classifier.
        
        Args:
            data_path: Optional path to feature data. If None, uses config path.
            
        Returns:
            Best trained model
        """
        evaluator = ModelEvaluator()
        
        if data_path is None:
            data_path = self.config['data']['feature_path']
        
        # Check if there's already an active MLflow run
        active_run = mlflow.active_run()
        should_start_run = active_run is None
        
        def run_training():
            # Log the data path being used
            mlflow.log_param("feature_path", data_path)
            
            df = pd.read_parquet(data_path)
            patient_df = self.aggregate_windows(df)
            X_orig, y, groups = self._prepare_data(patient_df)
            
            # Feature Selection
            perform_selection = self.feature_selection_config.get('enabled', False)
            logger.info(f"Feature selection enabled: {perform_selection}")
            
            if perform_selection:
                n_features_target = self.feature_selection_config.get('n_features', 10) 
                logger.info(f"Target n_features: {n_features_target}")
                
                selected_feature_names = self._select_features(X_orig, y, n_features_target)
                X = X_orig[selected_feature_names]
                mlflow.log_param("feature_selection_enabled", True)
                mlflow.log_param("target_n_features_to_select", n_features_target)
            else:
                X = X_orig
                selected_feature_names = X_orig.columns.tolist()
                mlflow.log_param("feature_selection_enabled", False)

            # Store the selected feature names as an instance variable for external access
            self.selected_feature_names = selected_feature_names
            mlflow.log_metric("num_features_trained_on", len(X.columns))

            # Log detailed dataset statistics
            unique_patients = groups.unique()
            n_remission = sum(y == 1)
            n_non_remission = sum(y == 0)
            
            logger.info("\nDataset Statistics:")
            logger.info(f"Total number of patients: {len(unique_patients)}")
            logger.info(f"- Remission patients: {n_remission}")
            logger.info(f"- Non-remission patients: {n_non_remission}")
            
            self._log_dataset_info(X, y, groups)
            
            # Get classifiers to try
            classifiers = self._get_classifiers()
            
            # Cross-validation
            logo = LeaveOneGroupOut()
            n_splits = logo.get_n_splits(X, y, groups)
            logger.info(f"\nPerforming Leave-One-Group-Out cross-validation with {n_splits} splits")
            
            # Dictionary to store results for each classifier
            all_results = {name: [] for name in classifiers.keys()}
            
            # Perform LOGO cross-validation
            for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
                test_participant = groups.iloc[test_idx].iloc[0]
                true_label = y.iloc[test_idx].iloc[0]
                
                logger.info(f"\nFold {fold_idx + 1}/{n_splits}")
                logger.info(f"Testing on participant: {test_participant} (true label: {true_label})")
                logger.info(f"Training patients: {len(train_idx)}")
                logger.info(f"Test patients: {len(test_idx)}")
                
                # Split data
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Try each classifier
                for name, clf in classifiers.items():
                    logger.info(f"Training {name}")
                    
                    # Train the model
                    clf.fit(X_train, y_train)
                    
                    # Make predictions
                    pred_prob = clf.predict_proba(X_test)[:, 1][0]
                    pred_label = clf.predict(X_test)[0]
                    
                    # Store results
                    all_results[name].append({
                        'participant': test_participant,
                        'true_label': true_label,
                        'predicted_label': pred_label,
                        'confidence': pred_prob,
                        'correct_prediction': true_label == pred_label
                    })
                    
                    # Log fold metrics
                    mlflow.log_metric(f"{name}_fold_{fold_idx}_accuracy", 
                                    int(all_results[name][-1]['correct_prediction']))
                    mlflow.log_metric(f"{name}_fold_{fold_idx}_confidence", 
                                    all_results[name][-1]['confidence'])
            
            # Calculate and log overall metrics for each classifier
            best_classifier = None
            best_f1 = -1
            
            for name in classifiers.keys():
                # Combine results
                results_df = pd.DataFrame(all_results[name])
                
                # Calculate detailed metrics
                metrics = self._calculate_detailed_metrics(
                    results_df['true_label'].values,
                    results_df['predicted_label'].values
                )
                
                # Log detailed metrics for each classifier
                logger.info(f"\nDetailed Metrics for {name}:")
                logger.info(f"True Positives (Correct Remission): {metrics['TP']}")
                logger.info(f"True Negatives (Correct Non-Remission): {metrics['TN']}")
                logger.info(f"False Positives (Incorrect Remission): {metrics['FP']}")
                logger.info(f"False Negatives (Missed Remission): {metrics['FN']}")
                logger.info(f"Accuracy: {metrics['accuracy']:.3f}")
                logger.info(f"Precision: {metrics['precision']:.3f}")
                logger.info(f"Recall: {metrics['recall']:.3f}")
                logger.info(f"F1 Score: {metrics['f1']:.3f}")
                
                # Log metrics to MLflow
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):  # Only log numeric metrics
                        mlflow.log_metric(f"{name}_{metric_name}", value)
                
                # Track best classifier based on F1 score
                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    best_classifier = name
            
            logger.info(f"\nBest performing classifier: {best_classifier} (F1: {best_f1:.3f})")
            mlflow.log_param("best_classifier", best_classifier)
            mlflow.log_metric("best_f1_score", best_f1)
            
            # Train final model on all data using best classifier
            logger.info("\nTraining final model on all data...")
            final_model = classifiers[best_classifier]
            final_model.fit(X, y)
            
            # Save results with best classifier information
            self._save_results(final_model, all_results, best_classifier, X)
            return final_model
        
        # Execute training with proper MLflow run management
        if should_start_run:
            with mlflow.start_run(run_name=self.model_name):
                return run_training()
        else:
            # Use existing run (nested or continue current)
            logger.info("Using existing MLflow run")
            return run_training()

    def _save_results(self, model, all_results: Dict, best_classifier: str, X_final_train: pd.DataFrame):
        """
        Save model results and predictions with enhanced reporting.
        
        Args:
            model: Trained model
            all_results: Dictionary of results for all classifiers
            best_classifier: Name of the best performing classifier
            X_final_train: DataFrame used to train the final model
        """
        feature_names = X_final_train.columns.tolist()
        
        # Save results for all classifiers
        for name, results_list in all_results.items():
            results_df = pd.DataFrame(results_list)
            
            # Get window size from data path or config
            window_size = self._get_window_size()
            
            # Create output directory with window size
            output_dir = Path(self.output_dir)
            if window_size:
                output_dir = output_dir / f"{window_size}s_window"
            else:
                output_dir = output_dir / "default_window"
            
            # Create directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save predictions for this classifier
            pred_path = output_dir / f'{name}_predictions.csv'
            results_df.to_csv(pred_path, index=False)
            mlflow.log_artifact(str(pred_path))
        
        # Calculate metrics for the best classifier
        best_results_df = pd.DataFrame(all_results[best_classifier])
        best_metrics = self._calculate_detailed_metrics(
            best_results_df['true_label'].values,
            best_results_df['predicted_label'].values
        )
        
        # Log best model metrics to MLflow
        mlflow.log_metrics(best_metrics)
        if hasattr(model, 'named_steps') and 'clf' in model.named_steps:
            mlflow.log_params(model.named_steps['clf'].get_params())
        
        # Save feature importance if available
        if self.config['output']['feature_importance']:
            self._save_feature_importance(model, feature_names, output_dir)
        
        # Log model with signature
        input_example = X_final_train.head()
        try:
            prediction_example = model.predict_proba(input_example)[:, 1]
            signature = infer_signature(input_example, prediction_example)
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
            'window_size': self._get_window_size(),
            'best_classifier': best_classifier,
            'model_type': self.model_type,
            'metrics': best_metrics,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        import json
        with open(output_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Log misclassified patients
        logger.info("\nMisclassified Patients:")
        misclassified = best_results_df[
            best_results_df['true_label'] != best_results_df['predicted_label']
        ]
        for _, row in misclassified.iterrows():
            logger.info(
                f"Participant {row['participant']}: "
                f"True={row['true_label']}, Pred={row['predicted_label']}, "
                f"Confidence={row['confidence']:.3f}"
            )

    def _save_feature_importance(self, model, feature_names: List[str], output_dir: Path):
        """Save feature importance for the model"""
        try:
            if hasattr(model, 'named_steps') and 'clf' in model.named_steps:
                clf = model.named_steps['clf']
                if hasattr(clf, 'feature_importances_'):
                    feature_importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': clf.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    # Save feature importances
                    importance_path = output_dir / 'feature_importances.csv'
                    feature_importance_df.to_csv(importance_path, index=False)
                    mlflow.log_artifact(str(importance_path))
                    
                    # Log top features
                    logger.info("\nTop 10 most important features:")
                    for _, row in feature_importance_df.head(10).iterrows():
                        logger.info(f"{row['feature']}: {row['importance']:.4f}")
                        mlflow.log_metric(f"feature_importance_{row['feature']}", row['importance'])
                    
                    logger.info(f"Saved feature importance to {importance_path}")
                else:
                    logger.info("Model does not have feature_importances_ attribute")
        except Exception as e:
            logger.error(f"Error saving feature importance: {e}")

    def _get_window_size(self):
        """Extract window size from config or data path"""
        window_size = None
        if 'window_size' in self.config:
            window_size = self.config['window_size']
        else:
            # Try to extract from data path
            data_path = self.config['data']['feature_path']
            if '{window_size}s' in data_path:
                import re
                match = re.search(r'(\d+)s_window_features', data_path)
                if match:
                    window_size = match.group(1)
        return window_size