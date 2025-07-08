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
from src.processing.feature_extractor import extract_eeg_features as run_feature_extraction

from src.models.patient_trainer import PatientLevelTrainer
from src.models.window_trainer import WindowLevelTrainer
from src.models.evaluation import ModelEvaluator

logger = setup_logger(__name__)

def setup_mlflow_tracking(config):
    """Set up MLflow tracking with error handling and fallback for malformed experiments.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        experiment_id: ID of the MLflow experiment
    """
    experiment_name = config.get('mlflow', {}).get('experiment_name', "eeg_processing")
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is not None:
            logger.info(f"Found existing MLflow experiment: {experiment_name} (ID: {experiment.experiment_id})")
            # Try to set it to confirm it's usable
            mlflow.set_experiment(experiment_name=experiment.name) 
            logger.info(f"Successfully set experiment: {experiment.name}")
            return experiment.experiment_id
        else:
            # Experiment does not exist, create it
            logger.info(f"MLflow experiment '{experiment_name}' not found. Creating new one.")
            experiment_id = mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name) # Set the newly created one
            logger.info(f"Created and set new MLflow experiment: {experiment_name} (ID: {experiment_id})")
            return experiment_id
    except Exception as e: 
        logger.warning(f"Failed to get, create, or set experiment '{experiment_name}': {e}. Attempting to use/create a fallback.")
        # Create a fallback experiment name with timestamp
        fallback_name = f"{experiment_name}_fallback_{int(time.time())}"
        try:
            logger.info(f"Attempting to create or use fallback experiment: {fallback_name}")
            fb_experiment = mlflow.get_experiment_by_name(fallback_name)
            if fb_experiment is not None: # Fallback name already exists
                logger.info(f"Fallback experiment name '{fallback_name}' already exists. Using it.")
                mlflow.set_experiment(fallback_name)
                return fb_experiment.experiment_id
            else: # Fallback name does not exist, create it
                experiment_id = mlflow.create_experiment(fallback_name)
                mlflow.set_experiment(fallback_name)
                logger.info(f"Created and set fallback MLflow experiment: {fallback_name} (ID: {experiment_id})")
                return experiment_id
        except Exception as fallback_e:
            logger.error(f"Failed to create or set fallback experiment '{fallback_name}': {fallback_e}")
            # If fallback also fails, raise an error that indicates this.
            raise Exception(f"MLflow setup failed for both primary ('{experiment_name}') and fallback ('{fallback_name}') experiments. Original error: {e}, Fallback error: {fallback_e}")

@click.group()
@click.option('--config', type=click.Path(exists=True), required=True, help='Path to config file')
@click.pass_context
def cli(ctx, config):
    """EEG Analysis Pipeline CLI"""
    ctx.ensure_object(dict)
    ctx.obj['config'] = load_config(config)

@cli.command()
@click.pass_context
def process(ctx):
    """Process EEG data through the pipeline."""
    config = ctx.obj['config']

    # Set up MLflow tracking
    tracking_uri = config.get('mlflow', {}).get('tracking_uri', "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    
    try:
        experiment_id = setup_mlflow_tracking(config)
        logger.info(f"MLflow experiment ID: {experiment_id} selected for processing.")
    except Exception as e:
        logger.error(f"Critical error setting up MLflow for processing: {e}")
        raise
    
    # Create base interim directory path for data versioning
    interim_base_path = Path(config['paths']['interim']['upsampled']).parent.parent
    data_versioner = create_data_versioner(str(interim_base_path))
    
    with mlflow.start_run(run_name="processing"):
        mlflow.log_params(config.get('processing_params', {})) # Log processing parameters if available
        try:
            # Load data
            logger.info("Loading raw EEG data...")
            raw_data = load_eeg_data(config)
            version_id = data_versioner.save_version(raw_data, {'stage': 'raw'})
            logger.info(f"Raw data loaded and versioned. Version ID: {version_id}")
            
            # Process pipeline
            logger.info("Starting EEG processing pipeline...")
            upsampled = upsample_eeg_data(config, raw_data)
            logger.info("Upsampling complete.")
            filtered = filter_eeg_data(config, upsampled)
            logger.info("Filtering complete.")
            downsampled = downsample_eeg_data(config, filtered)
            logger.info("Downsampling complete.")
            windowed = slice_eeg_windows(config, downsampled)
            logger.info("Window slicing complete.")
            features = run_feature_extraction(config, windowed)
            logger.info("Feature extraction complete.")
            
            # Save final features
            final_version_metadata = {
                'stage': 'features',
                'parent_version': version_id,
                'processing_config': config # Log the config used for this feature set
            }
            final_version = data_versioner.save_version(features, final_version_metadata)
            logger.info(f"Final features saved and versioned. Version ID: {final_version}")
            
            mlflow.log_metric("processing_success", 1)
            mlflow.log_param("final_feature_version_id", final_version)
            logger.info("Processing completed successfully and logged to MLflow.")
            
        except Exception as e:
            mlflow.log_metric("processing_success", 0)
            logger.error(f"Processing pipeline failed: {str(e)}")
            # Log the exception to MLflow if possible
            mlflow.log_param("error_message", str(e))
            raise

@cli.command()
@click.option('--level', type=click.Choice(['patient', 'window']), required=True)
@click.option('--window-size', type=int, help='Window size in seconds (overrides config)')
@click.option('--model-type', type=click.Choice(['random_forest', 'gradient_boosting', 'logistic_regression', 'svm']), required=True)
@click.option('--enable-feature-selection', is_flag=True, help='Enable feature selection.')
@click.option('--n-features-select', type=int, default=10, help='Number of features to select if feature selection is enabled.')
@click.option('--fs-method', 
              type=click.Choice(['model_based', 'select_k_best_f_classif', 'select_k_best_mutual_info', 'select_from_model_l1', 'rfe']), 
              default='model_based', 
              help='Feature selection method.')
@click.pass_context
def train(ctx, level, window_size, model_type, enable_feature_selection, n_features_select, fs_method):
    """Train the model"""
    config = ctx.obj['config']
    logger.info(f"CLI train inputs: level='{level}', window_size={window_size}, model_type='{model_type}', enable_feature_selection={enable_feature_selection}, n_features_select={n_features_select}, fs_method='{fs_method}'")

    model_serializer = create_model_serializer(config['paths']['models'])
    
    # Load processing config to get window size if not provided
    if window_size is None:
        try:
            script_dir = Path(__file__).parent
            config_path = script_dir / 'configs' / 'processing_config.yaml'
            with open(config_path, 'r') as f:
                processing_config = yaml.safe_load(f)
                window_size = processing_config['window_slicer']['window_seconds']
        except Exception as e:
            logger.error(f"Failed to load window size from processing config: {str(e)}")
            raise
    
    logger.info(f"Using window size: {window_size}s")
    logger.info(f"Using model type: {model_type}")
    
    # Format the feature path with the window size
    if '{window_size}' in config['data']['feature_path']:
        config['data']['feature_path'] = config['data']['feature_path'].format(window_size=window_size)
        logger.info(f"Feature path: {config['data']['feature_path']}")
    
    # Add window_size to config so trainers can access it directly
    config['window_size'] = window_size
    config['model_type'] = model_type
    
    # Add feature selection config
    config['feature_selection'] = {
        'enabled': enable_feature_selection,
        'n_features': n_features_select,
        'method': fs_method
    }
    logger.info(f"Config for feature selection set to: {config['feature_selection']}")
    
    # Set up MLflow tracking
    tracking_uri = config.get('mlflow', {}).get('tracking_uri', "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    try:
        experiment_id = setup_mlflow_tracking(config)
        logger.info(f"MLflow experiment ID: {experiment_id} selected for training.")
    except Exception as e:
        logger.error(f"Critical error setting up MLflow for training: {e}")
        raise
    
    # Create appropriate trainer
    trainer_cls = PatientLevelTrainer if level == 'patient' else WindowLevelTrainer
    trainer = trainer_cls(config)
    
    try:
        run_name_suffix = ""
        if enable_feature_selection:
            run_name_suffix = f"_fs_{n_features_select}_{fs_method}"
            
        with mlflow.start_run(run_name=f"{model_type}_{window_size}s{run_name_suffix}") as run:
            # Train model
            model = trainer.train()
            
            # Get sample data for model signature - use the aggregated data like in training
            data_path = config['data']['feature_path']
            df = pd.read_parquet(data_path)
            
            if level == 'patient':
                # For patient-level training, aggregate the data first like in training
                patient_df = trainer.aggregate_windows(df)
                X, y, _ = trainer._prepare_data(patient_df)
            else:
                X, y, _ = trainer._prepare_data(df)
            
            # Use the same feature selection that was applied during training
            if hasattr(trainer, 'selected_feature_names'):
                X_sample = X[trainer.selected_feature_names].iloc[:100]  # Use selected features only
            else:
                X_sample = X.iloc[:100]  # Fallback to all features if no selection was applied
                
            y_proba_sample = model.predict_proba(X_sample)[:, 1]
            
            # Log window size as a parameter
            mlflow.log_param("window_size", window_size)
            
            # Save model
            model_id = model_serializer.save_model(
                model,
                model_info={
                    'level': level, 
                    'config': config,
                    'window_size': window_size,
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
@click.option('--data-path', type=click.Path(exists=True), required=False)
@click.option('--window-size', type=int, help='Window size in seconds for feature path')
@click.pass_context
def evaluate(ctx, model_id, data_path, window_size):
    """Evaluate model performance"""
    config = ctx.obj['config']
    model_serializer = create_model_serializer(config['paths']['models'])
    evaluator = ModelEvaluator()
    
    # If data path not provided, construct it from config and window size
    if data_path is None:
        # Get window size from model info if not provided
        if window_size is None:
            model_info = model_serializer.get_model_info(model_id)
            window_size = model_info.get('window_size')
            
            if window_size is None:
                # Try to get from processing config
                try:
                    with open('configs/processing_config.yaml', 'r') as f:
                        processing_config = yaml.safe_load(f)
                        window_size = processing_config['window_slicer']['window_seconds']
                except Exception as e:
                    logger.error(f"Failed to determine window size: {str(e)}")
                    raise
        
        # Format the feature path with the window size
        data_path = config['data']['feature_path'].format(window_size=window_size)
        logger.info(f"Using feature path: {data_path}")
    
    # Set up MLflow tracking
    tracking_uri = config.get('mlflow', {}).get('tracking_uri', "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    try:
        experiment_id = setup_mlflow_tracking(config) # Use the same config as it contains experiment name
        logger.info(f"MLflow experiment ID: {experiment_id} selected for evaluation.")
    except Exception as e:
        logger.error(f"Critical error setting up MLflow for evaluation: {e}")
        raise

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
