import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.base import BaseEstimator
import mlflow
import mlflow.data
from mlflow.data.pandas_dataset import PandasDataset
from src.models.model_utils import create_classifier
from src.utils.feature_filter import FeatureFilter

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

    def _create_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> BaseEstimator:
        model = create_classifier(self.classifier_name, self.classifier_params)
        model.fit(X_train, y_train)
        return model

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