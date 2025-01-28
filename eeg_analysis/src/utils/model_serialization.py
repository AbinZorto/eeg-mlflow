import joblib
import json
import mlflow
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from sklearn.base import BaseEstimator
import cloudpickle
import os
import logging
import uuid
from mlflow.models import infer_signature
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class ModelSerializer:
    def __init__(self, base_path: str):
        """Initialize the model serializer.
        
        Args:
            base_path: Base directory for model storage
        """
        self.base_path = Path(base_path)
        os.makedirs(self.base_path, exist_ok=True)
        self.metadata_file = self.base_path / 'metadata.json'
        
        # Initialize or load metadata
        if not self.metadata_file.exists():
            self.metadata = {'models': {}}
            self._save_metadata()
        else:
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Corrupted metadata file found at {self.metadata_file}. Creating new metadata.")
                self.metadata = {'models': {}}
                self._save_metadata()

    def _save_metadata(self):
        """Save metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)

    def _load_metadata(self):
        """Load metadata from disk."""
        try:
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.metadata = {'models': {}}
            self._save_metadata()

    def _make_json_serializable(self, obj):
        """Convert a dictionary to a JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return {
                'type': 'DataFrame',
                'data': obj.to_dict(orient='records')
            }
        elif hasattr(obj, '__dict__'):
            # For objects with __dict__, extract their attributes
            return {
                'type': obj.__class__.__name__,
                'attributes': self._make_json_serializable(
                    {k: v for k, v in obj.__dict__.items() 
                     if not k.startswith('_')}
                )
            }
        elif pd.api.types.is_numeric_dtype(type(obj)):
            # Handle numpy/pandas numeric types
            return float(np.asarray(obj).item())
        else:
            # For other objects, convert to string representation
            return str(obj)

    def save_model(self, model, model_info: dict = None) -> str:
        """Save a model to disk with optional metadata.
        
        Args:
            model: The model object to save
            model_info: Optional dictionary of model metadata
            
        Returns:
            str: The model ID
        """
        model_id = str(uuid.uuid4())
        model_dir = self.base_path / model_id
        model_path = model_dir / 'model'
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            # Process X_sample and y_proba_sample before MLflow operations
            if model_info and 'X_sample' in model_info:
                X = model_info['X_sample']
                if isinstance(X, pd.DataFrame):
                    X = X.copy()  # Make a copy to avoid modifying original
                
            if model_info and 'y_proba_sample' in model_info:
                y_proba = model_info['y_proba_sample']
                if isinstance(y_proba, (np.ndarray, pd.Series)):
                    y_proba = np.asarray(y_proba)
            
            # Save model using MLflow format if possible
            if X is not None and y_proba is not None:
                signature = infer_signature(X, y_proba)
                mlflow.sklearn.save_model(
                    sk_model=model,
                    path=str(model_path),
                    signature=signature
                )
            else:
                logger.warning("No sample data provided for model signature inference")
                mlflow.sklearn.save_model(
                    sk_model=model,
                    path=str(model_path)
                )
            
            # Extract and process model parameters
            if hasattr(model, 'get_params'):
                params = self._make_json_serializable(model.get_params())
            elif hasattr(model, 'named_steps') and 'clf' in model.named_steps:
                params = self._make_json_serializable(
                    model.named_steps['clf'].get_params()
                )
            else:
                params = {}
                
        except Exception as e:
            logger.warning(f"Could not save model in MLflow format: {str(e)}. Falling back to joblib.")
            joblib.dump(model, model_path / 'model.joblib')
            params = {}
        
        # Process model_info to ensure JSON serializable
        safe_info = self._make_json_serializable(model_info or {})
        
        # Update metadata
        self.metadata['models'][model_id] = {
            'path': str(model_path),
            'created_at': datetime.now().isoformat(),
            'info': safe_info,
            'parameters': params
        }
        self._save_metadata()
        
        return model_id
    
    def load_model(self, model_id: str) -> BaseEstimator:
        """Load a model from disk.
        
        Args:
            model_id: ID of the model to load
            
        Returns:
            The loaded model
        """
        if model_id not in self.metadata['models']:
            raise ValueError(f"Model {model_id} not found")
        
        model_path = Path(self.metadata['models'][model_id]['path'])
        
        # Try loading with MLflow first
        try:
            return mlflow.sklearn.load_model(str(model_path))
        except Exception:
            # Fall back to joblib if MLflow fails
            return joblib.load(model_path / 'model.joblib')

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        return self.metadata['models'].get(model_id, {})

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        return self.metadata['models']

    def save_to_mlflow(self, 
                      model_id: str, 
                      model_name: str,
                      run_id: Optional[str] = None):
        if model_id not in self.metadata['models']:
            raise ValueError(f"Model {model_id} not found")
        
        model_info = self.metadata['models'][model_id]
        model = self.load_model(model_id)
        
        with mlflow.start_run(run_id=run_id):
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log metrics
            if model_info.get('metrics'):
                mlflow.log_metrics(model_info['metrics'])
            
            # Log model info
            mlflow.log_dict(model_info['info'], "model_info.json")
            
            # Register model if name provided
            if model_name:
                mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model",
                                    model_name)

    def load_from_mlflow(self, 
                        run_id: str, 
                        model_info: Optional[Dict[str, Any]] = None) -> str:
        # Load model from MLflow
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        
        # Get run information
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        
        # Save model locally
        return self.save_model(
            model=model,
            model_info=model_info or {},
            metrics=run.data.metrics
        )

class PipelineSerializer(ModelSerializer):
    def save_pipeline(self, 
                     pipeline: BaseEstimator,
                     pipeline_info: Dict[str, Any],
                     include_data: bool = False) -> str:
        if include_data:
            serializer = cloudpickle
        else:
            serializer = joblib
            
        timestamp = datetime.now().isoformat()
        pipeline_id = f"pipeline_{timestamp.replace(':', '-')}"
        pipeline_path = self.base_path / f"{pipeline_id}.pkl"
        
        # Save pipeline
        with open(pipeline_path, 'wb') as f:
            serializer.dump(pipeline, f)
        
        # Save pipeline structure
        if hasattr(pipeline, 'get_params'):
            structure = pipeline.get_params()
            steps = [str(step) for step in pipeline.steps] if hasattr(pipeline, 'steps') else []
        else:
            structure = {}
            steps = []

        # Update metadata
        self.metadata['models'][pipeline_id] = {
            'timestamp': timestamp,
            'path': str(pipeline_path),
            'info': pipeline_info,
            'type': 'pipeline',
            'structure': structure,
            'steps': steps,
            'includes_data': include_data
        }
        self._save_metadata()
        
        return pipeline_id

    def load_pipeline(self, pipeline_id: str) -> BaseEstimator:
        if pipeline_id not in self.metadata['models']:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        pipeline_info = self.metadata['models'][pipeline_id]
        pipeline_path = Path(pipeline_info['path'])
        
        if pipeline_info.get('includes_data', False):
            with open(pipeline_path, 'rb') as f:
                return cloudpickle.load(f)
        else:
            return joblib.load(pipeline_path)

def create_model_serializer(base_path: str) -> ModelSerializer:
    return ModelSerializer(base_path)

def create_pipeline_serializer(base_path: str) -> PipelineSerializer:
    return PipelineSerializer(base_path)