import joblib
import json
import mlflow
from pathlib import Path
from typing import Dict, Any, Optional
import datetime
from sklearn.base import BaseEstimator
import cloudpickle

class ModelSerializer:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.models_path = self.base_path / 'models'
        self.metadata_path = self.base_path / 'model_metadata.json'
        self.models_path.mkdir(parents=True, exist_ok=True)
        self._load_metadata()

    def _load_metadata(self):
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {'models': {}}

    def _save_metadata(self):
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def save_model(self, 
                  model: BaseEstimator, 
                  model_info: Dict[str, Any],
                  metrics: Optional[Dict[str, float]] = None) -> str:
        timestamp = datetime.datetime.now().isoformat()
        model_id = f"model_{timestamp.replace(':', '-')}"
        model_path = self.models_path / f"{model_id}.joblib"
        
        # Save model using joblib
        joblib.dump(model, model_path)
        
        # Save pipeline structure if available
        if hasattr(model, 'get_params'):
            pipeline_info = model.get_params()
        else:
            pipeline_info = {}

        # Update metadata
        self.metadata['models'][model_id] = {
            'timestamp': timestamp,
            'path': str(model_path),
            'info': model_info,
            'metrics': metrics or {},
            'pipeline': pipeline_info
        }
        self._save_metadata()
        
        return model_id

    def load_model(self, model_id: str) -> BaseEstimator:
        if model_id not in self.metadata['models']:
            raise ValueError(f"Model {model_id} not found")
        
        model_path = Path(self.metadata['models'][model_id]['path'])
        return joblib.load(model_path)

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
            
        timestamp = datetime.datetime.now().isoformat()
        pipeline_id = f"pipeline_{timestamp.replace(':', '-')}"
        pipeline_path = self.models_path / f"{pipeline_id}.pkl"
        
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