import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, Any, Tuple, Optional, Callable, List
from sklearn.base import BaseEstimator
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, GroupKFold
import mlflow
import mlflow.data
from mlflow.data.pandas_dataset import PandasDataset
from src.models.model_utils import create_classifier
from src.utils.feature_filter import FeatureFilter
import logging

logger = logging.getLogger(__name__)

class BaseTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('model_name', 'model')
        self.classifier_name = config.get('classifier', 'random_forest')
        self.classifier_params = config.get('classifier_params', {})

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        # Extract y and groups BEFORE dropping columns
        y = df['Remission']
        groups = df['Participant']
        
        # Drop metadata columns that should not be used as features
        # Note: We keep 'Participant' and 'Remission' for extraction above, but drop them from features
        metadata_columns_to_drop = ['Participant', 'Remission', 'sub_window_id', 'parent_window_id']
        existing_metadata_to_drop = [col for col in metadata_columns_to_drop if col in df.columns]
        
        if existing_metadata_to_drop:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Dropping metadata columns from features: {existing_metadata_to_drop}")
        
        X = df.drop(existing_metadata_to_drop, axis=1)
        return X, y, groups

    def _sanitize_features_for_training(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Ensure feature matrix is numeric and finite for sklearn/model training.
        Converts non-numeric values to NaN, replaces inf/-inf with NaN, then imputes
        missing values with per-column medians.
        """
        non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_columns:
            logger.warning(
                "Found %d non-numeric feature columns. Coercing to numeric with NaN on failure.",
                len(non_numeric_columns),
            )

        X_numeric = X.apply(pd.to_numeric, errors="coerce")
        values = X_numeric.to_numpy(dtype=np.float64, copy=True)

        inf_count = int(np.isinf(values).sum())
        nan_count_before = int(np.isnan(values).sum())
        non_finite_mask = ~np.isfinite(values)
        non_finite_count = int(non_finite_mask.sum())

        if non_finite_count:
            logger.warning(
                "Detected %d non-finite feature values (%d inf, %d NaN). Replacing with NaN.",
                non_finite_count,
                inf_count,
                nan_count_before,
            )
            values[non_finite_mask] = np.nan

        X_clean = pd.DataFrame(values, columns=X_numeric.columns, index=X_numeric.index)

        nan_count_for_impute = int(np.isnan(values).sum())
        imputation_applied = False
        all_nan_columns: list[str] = []

        if nan_count_for_impute:
            # Median imputation keeps distribution robust; all-NaN columns fall back to 0.0.
            medians = X_clean.median(axis=0, skipna=True)
            all_nan_columns = medians[medians.isna()].index.tolist()
            if all_nan_columns:
                logger.warning(
                    "Found %d all-NaN feature columns after coercion/sanitization. Filling with 0.0.",
                    len(all_nan_columns),
                )
                medians = medians.fillna(0.0)

            X_clean = X_clean.fillna(medians)
            imputation_applied = True
            logger.info("Applied global median imputation to training features.")

        final_values = X_clean.to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(final_values).all():
            raise ValueError("Feature sanitization failed: non-finite values remain after imputation.")

        sanitization_info = {
            "non_numeric_column_count": len(non_numeric_columns),
            "non_finite_count": non_finite_count,
            "inf_count": inf_count,
            "nan_count_before": nan_count_before,
            "nan_count_imputed": nan_count_for_impute,
            "all_nan_column_count": len(all_nan_columns),
            "imputation_applied": imputation_applied,
        }
        return X_clean, sanitization_info

    def _create_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> BaseEstimator:
        model = create_classifier(self.classifier_name, self.classifier_params)
        model.fit(X_train, y_train)
        return model

    def _get_cv_config(self) -> Dict[str, Any]:
        cv_config = self.config.get("cv", {})
        return cv_config if isinstance(cv_config, dict) else {}

    def _get_cv_random_state(self) -> int:
        cv_config = self._get_cv_config()
        return int(cv_config.get("random_state", self.config.get("random_seed", 42)))

    def _parse_optional_k(self, raw_value: Any, field_name: str) -> Optional[int]:
        if raw_value is None:
            return None
        if isinstance(raw_value, str) and raw_value.strip().lower() in {"", "none", "null"}:
            return None
        try:
            value = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be an integer >= 2 when provided.") from exc
        if value < 2:
            raise ValueError(f"{field_name} must be >= 2 when provided.")
        return value

    def _build_group_cv_splits(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        k: Optional[int] = None,
        scope: str = "outer",
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
        """
        Build group-aware CV splits.
        - Default (k=None): Leave-One-Group-Out
        - Requested k: StratifiedGroupKFold when feasible, GroupKFold fallback
        """
        cv_config = self._get_cv_config()
        requested_k = self._parse_optional_k(
            k if k is not None else cv_config.get(f"{scope}_k"),
            f"cv.{scope}_k",
        )
        shuffle = bool(cv_config.get("shuffle", True))
        random_state = self._get_cv_random_state()

        y_local = pd.Series(y).reset_index(drop=True)
        groups_local = pd.Series(groups).reset_index(drop=True)
        n_groups = int(groups_local.nunique())

        if n_groups < 2:
            raise ValueError(f"Need at least 2 groups for {scope} CV; found {n_groups}.")

        info: Dict[str, Any] = {
            "scope": scope,
            "strategy": "leave_one_group_out",
            "requested_k": requested_k,
            "effective_k": None,
            "n_samples": int(len(y_local)),
            "n_groups": n_groups,
            "shuffle": shuffle,
            "random_state": random_state,
            "stratified": False,
        }

        if requested_k is None:
            logo = LeaveOneGroupOut()
            splits = list(logo.split(X, y_local, groups_local))
            info["effective_k"] = len(splits)
            return splits, info

        if requested_k > n_groups:
            raise ValueError(
                f"cv.{scope}_k={requested_k} is larger than the number of groups ({n_groups})."
            )

        group_label_df = pd.DataFrame({"group": groups_local.values, "label": y_local.values})
        group_labels = group_label_df.groupby("group")["label"].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]
        )
        label_counts = group_labels.value_counts()
        min_groups_per_class = int(label_counts.min()) if not label_counts.empty else 0
        can_use_stratified = group_labels.nunique() > 1 and min_groups_per_class >= requested_k

        if can_use_stratified:
            splitter = StratifiedGroupKFold(
                n_splits=requested_k,
                shuffle=shuffle,
                random_state=random_state if shuffle else None,
            )
            splits = list(splitter.split(X, y_local, groups_local))
            info["strategy"] = "stratified_group_kfold"
            info["stratified"] = True
            info["effective_k"] = requested_k
            return splits, info

        splitter = GroupKFold(n_splits=requested_k)
        splits = list(splitter.split(X, y_local, groups_local))
        info["strategy"] = "group_kfold"
        info["effective_k"] = requested_k
        info["fallback_reason"] = (
            f"StratifiedGroupKFold unavailable for {scope} split: "
            f"min_groups_per_class={min_groups_per_class}, requested_k={requested_k}."
        )
        return splits, info

    def _select_features_with_inner_consensus(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        train_groups: pd.Series,
        n_features_to_select: int,
        selection_fn: Callable[[pd.DataFrame, pd.Series, int], List[str]],
        inner_k: Optional[int] = None,
        consensus_n_features: Optional[int] = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Run inner group CV on outer-train data and select features by frequency consensus
        across inner-train folds. Falls back to single-pass selection when inner CV is not
        configured or not feasible.
        """
        info: Dict[str, Any] = {
            "enabled": False,
            "applied": False,
            "scope": "inner",
            "requested_k": None,
            "strategy": "single_pass_selection",
            "n_inner_splits": 0,
            "inner_feature_sets": 0,
        }

        inner_selection_n_features = max(1, int(n_features_to_select))
        if consensus_n_features is None:
            consensus_n_features_effective = inner_selection_n_features
        else:
            consensus_n_features_effective = max(1, int(consensus_n_features))
        inner_selection_n_features = max(inner_selection_n_features, consensus_n_features_effective)
        info["inner_selection_n_features"] = inner_selection_n_features
        info["consensus_n_features"] = consensus_n_features_effective

        try:
            requested_inner_k = self._parse_optional_k(
                inner_k if inner_k is not None else self._get_cv_config().get("inner_k"),
                "cv.inner_k",
            )
        except ValueError as exc:
            logger.warning("Invalid inner_k configuration. Falling back to single-pass selection: %s", exc)
            info["reason"] = str(exc)
            selected_fallback = selection_fn(X_train, y_train, inner_selection_n_features)
            return selected_fallback[:consensus_n_features_effective], info

        if requested_inner_k is None:
            info["reason"] = "inner_k_not_set"
            selected_fallback = selection_fn(X_train, y_train, inner_selection_n_features)
            return selected_fallback[:consensus_n_features_effective], info

        try:
            inner_splits, inner_split_info = self._build_group_cv_splits(
                X_train, y_train, train_groups, k=requested_inner_k, scope="inner"
            )
        except ValueError as exc:
            logger.warning("Inner CV split setup failed. Falling back to single-pass selection: %s", exc)
            info["requested_k"] = requested_inner_k
            info["reason"] = str(exc)
            selected_fallback = selection_fn(X_train, y_train, inner_selection_n_features)
            return selected_fallback[:consensus_n_features_effective], info

        feature_sets: List[List[str]] = []
        for inner_train_idx, _ in inner_splits:
            if len(inner_train_idx) == 0:
                continue
            selected_inner = selection_fn(
                X_train.iloc[inner_train_idx],
                y_train.iloc[inner_train_idx],
                inner_selection_n_features,
            )
            if selected_inner:
                feature_sets.append(selected_inner)

        if not feature_sets:
            info.update({
                "enabled": True,
                "applied": False,
                "requested_k": requested_inner_k,
                "strategy": inner_split_info.get("strategy"),
                "n_inner_splits": len(inner_splits),
                "reason": "no_inner_feature_sets",
            })
            selected_fallback = selection_fn(X_train, y_train, inner_selection_n_features)
            return selected_fallback[:consensus_n_features_effective], info

        feature_counts: Counter = Counter()
        for feature_set in feature_sets:
            feature_counts.update(feature_set)
        selected_features = [name for name, _ in feature_counts.most_common(consensus_n_features_effective)]

        if not selected_features:
            info.update({
                "enabled": True,
                "applied": False,
                "requested_k": requested_inner_k,
                "strategy": inner_split_info.get("strategy"),
                "n_inner_splits": len(inner_splits),
                "reason": "consensus_empty_fallback",
            })
            selected_fallback = selection_fn(X_train, y_train, inner_selection_n_features)
            return selected_fallback[:consensus_n_features_effective], info

        info.update({
            "enabled": True,
            "applied": True,
            "requested_k": requested_inner_k,
            "strategy": inner_split_info.get("strategy"),
            "n_inner_splits": len(inner_splits),
            "inner_feature_sets": len(feature_sets),
            "top_feature_counts": [
                {"feature": name, "count": count}
                for name, count in feature_counts.most_common(min(2 * consensus_n_features_effective, len(feature_counts)))
            ],
        })
        return selected_features, info

    def _balance_groups_for_lopo(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        enabled: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
        """
        Randomly undersample majority-label groups so each class has the same number of groups.
        This keeps LOPO test folds class-balanced across the full CV run.
        """
        info: Dict[str, Any] = {
            "enabled": bool(enabled),
            "applied": False,
            "strategy": "random_undersample_majority_to_minority_group_count",
            "n_rows_before": int(len(X)),
            "n_groups_before": int(groups.nunique()),
        }

        if not enabled:
            info["reason"] = "disabled"
            info["n_rows_after"] = int(len(X))
            info["n_groups_after"] = int(groups.nunique())
            return X, y, groups, info

        group_label_df = pd.DataFrame({
            "group": groups.values,
            "label": y.values,
        })
        # Robustly derive one label per group (all windows for a participant should match).
        group_labels = group_label_df.groupby("group")["label"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
        label_counts_before = group_labels.value_counts().sort_index()
        info["group_label_counts_before"] = {str(k): int(v) for k, v in label_counts_before.items()}

        if group_labels.nunique() <= 1:
            info["reason"] = "single_class_groups"
            info["n_rows_after"] = int(len(X))
            info["n_groups_after"] = int(groups.nunique())
            info["group_label_counts_after"] = info["group_label_counts_before"]
            return X, y, groups, info

        min_group_count = int(label_counts_before.min())
        random_state = self._get_cv_random_state()
        rng = np.random.default_rng(random_state)

        selected_groups = []
        for label_value in label_counts_before.index.tolist():
            label_groups = group_labels[group_labels == label_value].index.to_numpy()
            if len(label_groups) > min_group_count:
                sampled = rng.choice(label_groups, size=min_group_count, replace=False)
                selected_groups.extend(sampled.tolist())
            else:
                selected_groups.extend(label_groups.tolist())

        selected_group_set = set(selected_groups)
        keep_mask = groups.isin(selected_group_set)

        X_bal = X.loc[keep_mask]
        y_bal = y.loc[keep_mask]
        groups_bal = groups.loc[keep_mask]

        group_labels_after = pd.DataFrame({
            "group": groups_bal.values,
            "label": y_bal.values,
        }).groupby("group")["label"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
        label_counts_after = group_labels_after.value_counts().sort_index()

        info["applied"] = True
        info["random_seed"] = random_state
        info["minority_group_count_before"] = min_group_count
        info["n_rows_after"] = int(len(X_bal))
        info["n_groups_after"] = int(groups_bal.nunique())
        info["group_label_counts_after"] = {str(k): int(v) for k, v in label_counts_after.items()}
        info["dropped_group_count"] = int(groups.nunique() - groups_bal.nunique())
        return X_bal, y_bal, groups_bal, info

    def _log_dataset_info(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series):
        mlflow.log_params({
            'n_features': X.shape[1],
            'n_positive': sum(y == 1),
            'n_negative': sum(y == 0)
        })

    def _save_predictions(self, predictions: pd.DataFrame, filename: str):
        predictions.to_csv(filename, index=False)
        mlflow.log_artifact(filename)

    def _load_dataset_from_mlflow(self, run_id: Optional[str] = None) -> Optional[PandasDataset]:
        """
        Load dataset from MLflow. If run_id is provided, load from that run.
        Otherwise, try to find a dataset in the current active run.
        
        Args:
            run_id: Optional run ID to load dataset from
            
        Returns:
            PandasDataset if found, None otherwise
        """
        try:
            if run_id:
                # Load from specific run
                run = mlflow.get_run(run_id)
                if hasattr(run, 'inputs') and run.inputs is not None and hasattr(run.inputs, 'dataset_inputs') and run.inputs.dataset_inputs:
                    dataset_input = run.inputs.dataset_inputs[0]  # Get first dataset
                    # Load the dataset
                    dataset_source = mlflow.data.get_source(dataset_input.dataset)
                    data_path = dataset_source.load()  # This should give us the local path
                    df = pd.read_parquet(data_path)
                    return mlflow.data.from_pandas(df, source=data_path, targets="Remission")
            else:
                # Try to get from current run
                active_run = mlflow.active_run()
                if active_run and hasattr(active_run, 'inputs') and active_run.inputs is not None and hasattr(active_run.inputs, 'dataset_inputs') and active_run.inputs.dataset_inputs:
                    dataset_input = active_run.inputs.dataset_inputs[0]
                    dataset_source = mlflow.data.get_source(dataset_input.dataset)
                    data_path = dataset_source.load()
                    df = pd.read_parquet(data_path)
                    return mlflow.data.from_pandas(df, source=data_path, targets="Remission")
                    
        except Exception as e:
            mlflow.log_param("dataset_load_from_mlflow_error", str(e))
            print(f"Could not load dataset from MLflow: {e}")
            
        return None

    def _load_data_from_source(self, data_source: Optional[str] = None, 
                               dataset: Optional[PandasDataset] = None,
                               prefer_mlflow: bool = True) -> pd.DataFrame:
        """
        Load data from various sources with priority: MLflow dataset > provided dataset > file path.
        
        Args:
            data_source: Optional file path to data
            dataset: Optional MLflow dataset 
            prefer_mlflow: Whether to prefer MLflow dataset over file path
            
        Returns:
            DataFrame with training data
        """
        # Try MLflow dataset first if preferred
        if prefer_mlflow:
            mlflow_dataset = self._load_dataset_from_mlflow()
            if mlflow_dataset is not None:
                mlflow.log_param("data_source", "mlflow_dataset")
                mlflow.log_param("dataset_name", mlflow_dataset.name)
                mlflow.log_param("dataset_digest", mlflow_dataset.digest)
                df = mlflow_dataset.df
            else:
                df = None
        else:
            df = None
        
        # Use provided dataset
        if df is None and dataset is not None:
            mlflow.log_param("data_source", "provided_dataset")
            mlflow.log_param("dataset_name", dataset.name)
            mlflow.log_param("dataset_digest", dataset.digest)
            df = dataset.df
            
        # Fall back to file path
        if df is None and data_source:
            df = pd.read_parquet(data_source)
            mlflow.log_param("data_source", "file_path")
            mlflow.log_param("data_file_path", data_source)
        
        # If we get here, we have no data source
        if df is None:
            raise ValueError("No data source provided: neither MLflow dataset, provided dataset, nor file path available")
        
        # Apply feature filtering if enabled
        feature_filtering_config = self.config.get('feature_filtering', {})
        if feature_filtering_config.get('enabled', False):
            try:
                channels = feature_filtering_config.get('channels', [])
                categories = feature_filtering_config.get('categories', [])
                
                if channels and categories:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"Applying feature filtering: channels={channels}, categories={categories}")
                    
                    feature_filter = FeatureFilter(channels, categories)
                    original_df = df.copy()
                    df = feature_filter.filter_features(original_df)
                    
                    mlflow.log_param("feature_filtering_applied", True)
                    mlflow.log_param("feature_categories", ",".join(categories))
                    mlflow.log_param("original_features", len(original_df.columns) - 2)
                    mlflow.log_param("filtered_features", len(df.columns) - 2)
                    
                    logger.info(f"Feature filtering applied: {len(original_df.columns)} -> {len(df.columns)} columns")
                    
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to apply feature filtering: {e}")
                mlflow.log_param("feature_filtering_error", str(e))
        
        return df

    def train(self, data_path: Optional[str] = None, dataset: Optional[PandasDataset] = None) -> BaseEstimator:
        """
        Train a model. This method should be overridden by subclasses.
        
        Args:
            data_path: Optional path to training data file
            dataset: Optional MLflow dataset to use for training
            
        Returns:
            Trained model
        """
        raise NotImplementedError("Subclasses must implement train method")
