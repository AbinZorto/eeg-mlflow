import click
import yaml
import pandas as pd
import mlflow
import os
import time
from pathlib import Path
from src.utils.logger import setup_logger
from src.utils.config import load_config
from src.utils.data_versioning import create_data_versioner, DataVersioner
from src.utils.model_serialization import create_model_serializer

from src.processing.data_loader import load_eeg_data
from src.processing.filter import filter_eeg_data
from src.processing.upsampler import upsample_eeg_data
from src.processing.downsampler import downsample_eeg_data
from src.processing.window_slicer import slice_eeg_windows
from src.processing.feature_extractor import run_feature_extraction

from src.models.patient_trainer import PatientLevelTrainer
from src.models.window_trainer import WindowLevelTrainer
from src.models.evaluation import ModelEvaluator

logger = setup_logger(__name__)

@click.group()
@click.option('--config', type=click.Path(exists=True), help='Path to config file')
@click.pass_context
def cli(ctx, config):
    """EEG Analysis Pipeline CLI"""
    ctx.ensure_object(dict)
    ctx.obj['config'] = load_config(config)

@cli.command()
@click.pass_context
@click.option('--version-data', is_flag=True, default=False, help='Enable data versioning')
def process(ctx, version_data):
    """Process EEG data through the pipeline."""
    try:
        config = ctx.obj['config']
        
        # Load data
        raw_data = load_eeg_data(config)
        
        # Version data only if flag is set
        if version_data:
            data_versioner = DataVersioner()
            version_id = data_versioner.save_version(raw_data, {'stage': 'raw'})
            logger.info(f"Data version saved with ID: {version_id}")
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise

    
    # Set up MLflow tracking
    mlflow.set_tracking_uri("file:./mlruns")  # Store runs locally in ./mlruns directory
    experiment_name = "eeg_processing"
    
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except Exception:
        # If experiment already exists, get its ID
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    mlflow.set_experiment(experiment_name)
    
    # Create base interim directory path
    interim_base_path = Path(config['paths']['interim']['upsampled']).parent.parent
    data_versioner = create_data_versioner(str(interim_base_path))
    
    with mlflow.start_run(run_name="processing"):
        try:
            # Load data
            raw_data = load_eeg_data(config)
            version_id = data_versioner.save_version(raw_data, {'stage': 'raw'})
            
            # Process pipeline
            upsampled = upsample_eeg_data(config, raw_data)
            filtered = filter_eeg_data(config, upsampled)
            downsampled = downsample_eeg_data(config, filtered)
            windowed = slice_eeg_windows(config, downsampled)
            features = run_feature_extraction(config, windowed)
            
            # Save final features
            final_version = data_versioner.save_version(features, {
                'stage': 'features',
                'parent_version': version_id
            })
            
            mlflow.log_metric("processing_success", 1)
            logger.info("Processing completed successfully")
            
        except Exception as e:
            mlflow.log_metric("processing_success", 0)
            logger.error(f"Processing failed: {str(e)}")
            raise

@cli.command()
@click.option('--level', type=click.Choice(['patient', 'window']), required=True)
@click.pass_context
def train(ctx, level):
    """Train the model"""
    config = ctx.obj['config']
    model_serializer = create_model_serializer(config['paths']['models'])
    
    # Set up MLflow tracking with SQLite backend
    db_path = os.path.join(config['mlflow']['tracking_uri'], 'mlflow.db')
    tracking_uri = f"sqlite:///{db_path}"
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = config['mlflow']['experiment_name']
    
    # Create or get experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=config['mlflow']['artifact_location']
            )
        else:
            experiment_id = experiment.experiment_id
            
        mlflow.set_experiment(experiment_name)
        
    except Exception as e:
        logger.error(f"Failed to create/get MLflow experiment: {str(e)}")
        raise
    
    # Create appropriate trainer
    trainer_cls = PatientLevelTrainer if level == 'patient' else WindowLevelTrainer
    trainer = trainer_cls(config)
    
    try:
        with mlflow.start_run() as run:
            # Train model
            model = trainer.train()
            
            # Get sample data for model signature
            data_path = config['data']['feature_path']
            df = pd.read_parquet(data_path)
            X, y, _ = trainer._prepare_data(df)
            X_sample = X.iloc[:100]  # Take first 100 samples
            y_proba_sample = model.predict_proba(X_sample)[:, 1]
            
            # Save model
            model_id = model_serializer.save_model(
                model,
                model_info={
                    'level': level, 
                    'config': config,
                    'X_sample': X_sample,
                    'y_proba_sample': y_proba_sample
                }
            )
            
            logger.info(f"{level.capitalize()} training completed successfully")
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

@cli.command()
@click.option('--model-id', type=str, required=True)
@click.option('--data-path', type=click.Path(exists=True), required=True)
@click.pass_context
def evaluate(ctx, model_id, data_path):
    """Evaluate model performance"""
    config = ctx.obj['config']
    model_serializer = create_model_serializer(config['paths']['models'])
    evaluator = ModelEvaluator()
    
    with mlflow.start_run(run_name="evaluation"):
        try:
            # Load model and data
            model = model_serializer.load_model(model_id)
            model_info = model_serializer.get_model_info(model_id)
            test_data = pd.read_parquet(data_path)
            
            # Prepare data
            X = test_data.drop(['Participant', 'Remission'], axis=1)
            y = test_data['Remission']
            groups = test_data['Participant']
            
            # Get predictions
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]
            
            # Evaluate based on model level
            if model_info['level'] == 'patient':
                metrics = evaluator.evaluate_patient_predictions(groups, y, y_prob)
            else:
                metrics = evaluator.evaluate_window_predictions(y, y_pred, y_prob)
            
            # Log metrics
            for name, value in metrics.items():
                mlflow.log_metric(name, value)
                
            mlflow.log_metric("evaluation_success", 1)
            logger.info("Evaluation completed successfully")
            
        except Exception as e:
            mlflow.log_metric("evaluation_success", 0)
            logger.error(f"Evaluation failed: {str(e)}")
            raise

if __name__ == '__main__':
    cli(obj={})
