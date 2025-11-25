import click
import yaml
import pandas as pd
import mlflow
import mlflow.data
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
from src.processing.dc_offset import remove_dc_offset_eeg_data

from src.models.patient_trainer import PatientLevelTrainer
from src.models.window_trainer import WindowLevelTrainer
from src.models.deep_learning_trainer import DeepLearningTrainer
from src.models.evaluation import ModelEvaluator
from src.utils.feature_filter import FeatureFilter

logger = setup_logger(__name__)

def setup_mlflow_tracking(config):
    """Set up MLflow tracking with error handling and fallback for malformed experiments.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        experiment_id: ID of the MLflow experiment
    """
    # Check for environment variable first (set by run_all_experiments.sh)
    experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME')
    
    # Fall back to config if environment variable not set
    if experiment_name is None:
        experiment_name = config.get('mlflow', {}).get('experiment_name', "eeg_processing")
        logger.info(f"Using experiment name from config: {experiment_name}")
    else:
        logger.info(f"Using experiment name from environment: {experiment_name}")
    
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
        fallback_name = f"{experiment_name}_fallback_1"
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

# Configuration-based dataset search function
def find_dataset_by_config(config, experiment_id=None):
    """
    Find MLflow dataset that matches the current processing configuration.
    
    Args:
        config: Configuration dictionary containing window and channel settings
        experiment_id: Optional experiment ID to search in
        
    Returns:
        PandasDataset if found, None otherwise
    """
    try:
        # Get configuration parameters for dataset name matching
        window_seconds = config.get('window_size')  # From CLI or config
        if window_seconds is None:
            # Try to get from processing config
            script_dir = Path(__file__).parent
            config_path = script_dir / 'configs' / 'processing_config.yaml'
            with open(config_path, 'r') as f:
                processing_config = yaml.safe_load(f)
                window_seconds = processing_config['window_slicer']['window_seconds']
                
        # Get channels from processing config
        script_dir = Path(__file__).parent
        config_path = script_dir / 'configs' / 'processing_config.yaml'
        with open(config_path, 'r') as f:
            processing_config = yaml.safe_load(f)
            channels = processing_config['data_loader']['channels']
            
        # Create expected dataset name pattern (without window count since it varies)
        channels_str = "-".join(channels)
        dataset_name_pattern = f"EEG_{window_seconds}s_{channels_str}_"
        
        logger.info(f"Searching for dataset matching pattern: {dataset_name_pattern}*")
        logger.info(f"Configuration: {window_seconds}s windows, channels: {channels}")
        
        # Search in processing experiment first, then current experiment
        search_experiment_ids = []
        
        # Try to find the eeg_processing experiment
        try:
            processing_experiment = mlflow.get_experiment_by_name("eeg_processing")
            if processing_experiment:
                search_experiment_ids.append(processing_experiment.experiment_id)
                logger.info(f"Will search in eeg_processing experiment: {processing_experiment.experiment_id}")
        except Exception as e:
            logger.warning(f"Could not find eeg_processing experiment: {e}")
            
        # Also search in current experiment if provided
        if experiment_id:
            search_experiment_ids.append(experiment_id)
            logger.info(f"Will also search in current experiment: {experiment_id}")
            
        if not search_experiment_ids:
            logger.warning("No experiments to search in")
            return None
            
        # Search for runs with datasets that match our configuration
        runs_with_datasets = mlflow.search_runs(
            experiment_ids=search_experiment_ids,
            filter_string="tags.`mlflow.dataset.logged` = 'true' AND tags.`mlflow.dataset.context` = 'training'",
            order_by=["start_time DESC"],
            max_results=20  # Get more results to find matching dataset
        )
        
        if runs_with_datasets.empty:
            logger.info("No runs with datasets found")
            return None
            
        logger.info(f"Found {len(runs_with_datasets)} runs with datasets, searching for configuration match...")
        
        # Check each run for dataset name match
        for idx, run in runs_with_datasets.iterrows():
            run_id = run['run_id']
            try:
                mlflow_run = mlflow.get_run(run_id)
                
                # Check if this run has dataset inputs
                if hasattr(mlflow_run, 'inputs') and mlflow_run.inputs.dataset_inputs:
                    dataset_input = mlflow_run.inputs.dataset_inputs[0]
                    dataset_name = dataset_input.dataset.name
                    
                    logger.info(f"Checking dataset: {dataset_name}")
                    
                    # Check if dataset name matches our configuration pattern
                    if dataset_name.startswith(dataset_name_pattern):
                        logger.info(f"✓ Found matching dataset: {dataset_name}")
                        logger.info(f"  From run: {run_id}")
                        logger.info(f"  Run start time: {run.get('start_time', 'unknown')}")
                        
                        # Load and return the dataset, preserving the original name
                        dataset_source = mlflow.data.get_source(dataset_input.dataset)
                        data_path = dataset_source.load()
                        df = pd.read_parquet(data_path)
                        dataset_to_use = mlflow.data.from_pandas(
                            df=df, 
                            source=data_path, 
                            targets="Remission",
                            name=dataset_name  # Preserve the original name
                        )
                        
                        logger.info(f"  Dataset shape: {df.shape}")
                        logger.info(f"  Dataset digest: {dataset_to_use.digest}")
                        
                        return dataset_to_use
                    else:
                        logger.debug(f"  Dataset name doesn't match pattern: {dataset_name}")
                        
            except Exception as e:
                logger.warning(f"Error checking run {run_id}: {e}")
                continue
                
        logger.info(f"No dataset found matching configuration: {window_seconds}s windows with {channels} channels")
        return None
        
    except Exception as e:
        logger.error(f"Error in configuration-based dataset search: {e}")
        return None

@click.group()
@click.option('--config', type=click.Path(exists=True), required=True, help='Path to config file')
@click.pass_context
def cli(ctx, config):
    """EEG Analysis Pipeline CLI"""
    ctx.ensure_object(dict)
    ctx.obj['config'] = load_config(config)
    ctx.obj['config_path'] = config

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
            #version_id = data_versioner.save_version(raw_data, {'stage': 'raw'})
            #logger.info(f"Raw data loaded and versioned. Version ID: {version_id}")
            
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
            dc_removed = remove_dc_offset_eeg_data(config, windowed)
            logger.info("DC offset removal complete.")
            features, mlflow_dataset = run_feature_extraction(config, dc_removed)
            logger.info("Feature extraction complete.")
            
            # Save final features with enhanced metadata
            final_version_metadata = {
                'stage': 'features',
                #'parent_version': version_id,
                'processing_config': config, # Log the config used for this feature set
                'mlflow_dataset_name': mlflow_dataset.name if mlflow_dataset else None,
                'mlflow_dataset_digest': mlflow_dataset.digest if mlflow_dataset else None
            }
            #final_version = data_versioner.save_version(features, final_version_metadata)
            #logger.info(f"Final features saved and versioned. Version ID: {final_version}")
            
            mlflow.log_metric("processing_success", 1)
            #mlflow.log_param("final_feature_version_id", final_version)
            
            # Store dataset info for potential use in training
            if mlflow_dataset:
                mlflow.log_param("dataset_available_for_training", True)
                # Set a tag to mark this run as having a training dataset available
                mlflow.set_tag("mlflow.dataset.logged", "true")
                mlflow.set_tag("mlflow.dataset.context", "training")
            else:
                mlflow.log_param("dataset_available_for_training", False)
                
            logger.info("Processing completed successfully and logged to MLflow.")
            
        except Exception as e:
            mlflow.log_metric("processing_success", 0)
            logger.error(f"Processing pipeline failed: {str(e)}")
            # Log the exception to MLflow if possible
            mlflow.log_param("error_message", str(e))
            raise

@cli.command()
@click.pass_context
def list_datasets(ctx):
    """List available MLflow datasets that match the current processing configuration."""
    config = ctx.obj['config']
    
    try:
        # Get configuration parameters
        script_dir = Path(__file__).parent
        config_path = script_dir / 'configs' / 'processing_config.yaml'
        with open(config_path, 'r') as f:
            processing_config = yaml.safe_load(f)
            window_seconds = processing_config['window_slicer']['window_seconds']
            channels = processing_config['data_loader']['channels']
            
        # Look for all datasets with this window size (regardless of channel configuration)
        window_pattern = f"EEG_{window_seconds}s_"
        
        print(f"\n🔍 Searching for all datasets with window size: {window_seconds}s")
        print(f"   Current config channels: {channels}")
        print(f"   Will look for 4-channel datasets to filter from")
        print(f"   Pattern: {window_pattern}*")
        print("-" * 80)
        
        # Set up MLflow tracking
        tracking_uri = config.get('mlflow', {}).get('tracking_uri', "file:./mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        
        # Search in processing experiment and others
        search_experiment_ids = []
        
        # Try to find experiments
        experiments_to_check = ["eeg_processing", "eeg_deep_learning_gpu_baseline", "eeg_boosting_gpu_baseline", "eeg_traditional_models_baseline"]
        
        for exp_name in experiments_to_check:
            try:
                experiment = mlflow.get_experiment_by_name(exp_name)
                if experiment:
                    search_experiment_ids.append(experiment.experiment_id)
                    print(f"✓ Will search in experiment: {exp_name}")
            except Exception:
                pass
                
        if not search_experiment_ids:
            print("❌ No MLflow experiments found to search in.")
            return
            
        # Search for runs with datasets
        runs_with_datasets = mlflow.search_runs(
            experiment_ids=search_experiment_ids,
            filter_string="tags.`mlflow.dataset.logged` = 'true' AND tags.`mlflow.dataset.context` = 'training'",
            order_by=["start_time DESC"],
            max_results=50
        )
        
        if runs_with_datasets.empty:
            print("❌ No runs with datasets found.")
            return
            
        print(f"\n📊 Found {len(runs_with_datasets)} runs with datasets. Checking for matches...")
        print("-" * 80)
        
        matching_datasets = []
        other_datasets = []
        
        # Check each run for dataset name match
        for idx, run in runs_with_datasets.iterrows():
            run_id = run['run_id']
            try:
                mlflow_run = mlflow.get_run(run_id)
                
                if hasattr(mlflow_run, 'inputs') and mlflow_run.inputs.dataset_inputs:
                    dataset_input = mlflow_run.inputs.dataset_inputs[0]
                    dataset_name = dataset_input.dataset.name
                    
                    # Get run start time
                    start_time = run.get('start_time', 'unknown')
                    if start_time != 'unknown':
                        start_time = pd.to_datetime(start_time).strftime('%Y-%m-%d %H:%M:%S')
                    
                    dataset_info = {
                        'name': dataset_name,
                        'run_id': run_id,
                        'start_time': start_time,
                        'experiment': mlflow_run.info.experiment_id
                    }
                    
                    # Try to get additional info
                    try:
                        dataset_source = mlflow.data.get_source(dataset_input.dataset)
                        data_path = dataset_source.load()
                        df = pd.read_parquet(data_path)
                        dataset_info['rows'] = len(df)
                        dataset_info['participants'] = df['Participant'].nunique() if 'Participant' in df.columns else 'unknown'
                        dataset_info['features'] = len(df.columns) - 2 if len(df.columns) >= 2 else len(df.columns)
                    except Exception:
                        dataset_info['rows'] = 'unknown'
                        dataset_info['participants'] = 'unknown'
                        dataset_info['features'] = 'unknown'
                    
                    # Check if this dataset matches the window size
                    if dataset_name.startswith(window_pattern):
                        # Further classify as 4-channel vs other
                        is_4_channel = any(all_4_ch in dataset_name for all_4_ch in ['af7-af8-tp9-tp10', 'af7-af8-tp10-tp9', 'af8-af7-tp9-tp10', 'af8-af7-tp10-tp9', 'tp9-tp10-af7-af8', 'tp10-tp9-af7-af8'])
                        if is_4_channel:
                            dataset_info['type'] = '4-channel (can be filtered)'
                            matching_datasets.append(dataset_info)
                        else:
                            dataset_info['type'] = 'filtered/subset'
                            matching_datasets.append(dataset_info)
                    else:
                        other_datasets.append(dataset_info)
                        
            except Exception as e:
                continue
                
        # Display matching datasets
        if matching_datasets:
            print(f"✅ Found {len(matching_datasets)} datasets with {window_seconds}s window size:")
            print()
            for i, ds in enumerate(matching_datasets, 1):
                print(f"{i}. 📈 {ds['name']} ({ds.get('type', 'unknown')})")
                print(f"   Run ID: {ds['run_id']}")
                print(f"   Created: {ds['start_time']}")
                print(f"   Rows: {ds['rows']}, Participants: {ds['participants']}, Features: {ds['features']}")
                print()
        else:
            print(f"❌ No datasets found with {window_seconds}s window size.")
            
        # Display other datasets for reference
        if other_datasets:
            print(f"\n🔍 Found {len(other_datasets)} other datasets (different configurations):")
            print()
            for i, ds in enumerate(other_datasets[:10], 1):  # Show first 10
                print(f"{i}. 📊 {ds['name']}")
                print(f"   Run ID: {ds['run_id']}")
                print(f"   Created: {ds['start_time']}")
                print()
            if len(other_datasets) > 10:
                print(f"   ... and {len(other_datasets) - 10} more")
                
        # Show usage instructions
        print("\n" + "=" * 80)
        print("💡 Usage instructions:")
        print()
        if matching_datasets:
            print("Dataset Selection Strategy:")
            print("1. 4-channel datasets can be automatically filtered for any channel subset")
            print("2. Filtered datasets are ready to use as-is")
            print()
            print("To use a specific dataset, run:")
            print(f"   python run_pipeline.py --config <config> train --model-type <model> --use-dataset-from-run <run_id>")
            print()
            print("To run all experiments with automatic dataset selection:")
            print("   ./run_all_experiments.sh")
            print("   (This will find a 4-channel dataset and filter it for your channel config)")
        else:
            print(f"No datasets found with {window_seconds}s window size. You may need to:")
            print("1. Run the processing pipeline first with all 4 channels:")
            print("   - Edit configs/processing_config.yaml: channels: ['af7', 'af8', 'tp9', 'tp10']")
            print(f"   - Run: python run_pipeline.py --config configs/processing_config.yaml process")
            print("2. Then you can filter this 4-channel dataset for any channel subset")
        print()
        
    except Exception as e:
        logger.error(f"Error listing datasets: {str(e)}")
        print(f"❌ Error: {str(e)}")

def get_available_models_from_config(config_path):
    """Extract available model types from the config file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        available_models = []
        
        # Get traditional models from model.params section
        if 'model' in config and 'params' in config['model']:
            available_models.extend(list(config['model']['params'].keys()))
        
        # Get deep learning models from deep_learning section
        if 'deep_learning' in config:
            available_models.extend(list(config['deep_learning'].keys()))
        
        return available_models
    except Exception as e:
        logger.warning(f"Failed to load models from config: {e}")
        # Return default models as fallback
        return [
                'random_forest', 'gradient_boosting', 'xgboost_gpu', 'catboost_gpu', 'lightgbm_gpu', 'logistic_regression', 
                'logistic_regression_l1', 'svm_rbf', 'svm_linear', 'extra_trees', 'ada_boost', 'knn', 'decision_tree', 'sgd', 
                'pytorch_mlp', 'keras_mlp', 'hybrid_1dcnn_lstm', 'advanced_1dcnn', 'advanced_lstm', 'advanced_hybrid_1dcnn_lstm'
            ]

@cli.command()
@click.option('--level', type=click.Choice(['patient', 'window']), required=True)
@click.option('--window-size', type=int, help='Window size in seconds (overrides config)')
@click.option('--model-type', type=str, required=True, help='Model type (will be validated against available models)')
@click.option('--enable-feature-selection', is_flag=True, help='Enable feature selection.')
@click.option('--n-features-select', type=int, default=10, help='Number of features to select if feature selection is enabled.')
@click.option('--fs-method', 
              type=click.Choice(['model_based', 'select_k_best_f_classif', 'select_k_best_mutual_info', 'select_from_model_l1', 'rfe']), 
              default='model_based', 
              help='Feature selection method.')
@click.option('--feature-categories', type=str, help='Comma-separated list of feature categories to include (e.g., "spectral_features,psd_statistics,temporal_features"). Use "list" to see available categories.')
@click.option('--use-dataset-from-run', type=str, help='MLflow run ID to load dataset from (optional)')
@click.pass_context
def train(ctx, level, window_size, model_type, enable_feature_selection, n_features_select, fs_method, feature_categories, use_dataset_from_run):
    """Train the model"""
    config = ctx.obj['config']
    logger.info(f"CLI train inputs: level='{level}', window_size={window_size}, model_type='{model_type}', enable_feature_selection={enable_feature_selection}, n_features_select={n_features_select}, fs_method='{fs_method}', use_dataset_from_run='{use_dataset_from_run}'")

    # Validate model type against available models from config
    config_path = ctx.obj.get('config_path', 'configs/window_model_config_ultra_extreme.yaml')
    available_models = get_available_models_from_config(config_path)
    
    if model_type not in available_models:
        logger.error(f"Invalid model type '{model_type}'. Available models: {', '.join(available_models)}")
        raise click.BadParameter(f"Model type '{model_type}' not found in config. Available models: {', '.join(available_models)}")
    
    logger.info(f"Model type '{model_type}' validated against available models: {', '.join(available_models)}")

    model_serializer = create_model_serializer(config['paths']['models'])
    
    # Load processing config to get window size, channels, and ordering if not provided
    try:
        script_dir = Path(__file__).parent
        processing_config_path = script_dir / 'configs' / 'processing_config.yaml'
        with open(processing_config_path, 'r') as f:
            processing_config = yaml.safe_load(f)
            
        if window_size is None:
            window_size = processing_config['window_slicer']['window_seconds']
        
        channels = processing_config['data_loader']['channels']
        ordering_method = processing_config['feature_extractor']['ordering_method']
        
        # Create channels string for path
        channels_str = "-".join(channels)
        
    except Exception as e:
        logger.error(f"Failed to load processing config: {str(e)}")
        raise
    
    logger.info(f"Using window size: {window_size}s")
    logger.info(f"Using channels: {channels}")
    logger.info(f"Using ordering method: {ordering_method}")
    logger.info(f"Using model type: {model_type}")
    
    # Format the feature path with dynamic parameters
    if '{window_size}' in config['data']['feature_path'] or '{channels}' in config['data']['feature_path']:
        config['data']['feature_path'] = config['data']['feature_path'].format(
            window_size=window_size,
            channels=channels_str
        )
        logger.info(f"Fallback feature path: {config['data']['feature_path']}")
    
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
    
    # Add feature filtering config
    if feature_categories and feature_categories.lower() != 'list':
        # Parse feature categories for config
        feature_categories_list = [cat.strip() for cat in feature_categories.split(',')]
        config['feature_filtering'] = {
            'enabled': True,
            'categories': feature_categories_list,
            'channels': channels
        }
        logger.info(f"Config for feature filtering set to: {config['feature_filtering']}")
    else:
        config['feature_filtering'] = {
            'enabled': False
        }
    
    # Set up MLflow tracking
    tracking_uri = config.get('mlflow', {}).get('tracking_uri', "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    try:
        experiment_id = setup_mlflow_tracking(config)
        logger.info(f"MLflow experiment ID: {experiment_id} selected for training.")
    except Exception as e:
        logger.error(f"Critical error setting up MLflow for training: {e}")
        raise
    
    # Try to find and load dataset from MLflow based on configuration or specific run
    dataset_to_use = None
    
    if use_dataset_from_run:
        logger.info(f"Attempting to load dataset from specific run: {use_dataset_from_run}")
        try:
            run = mlflow.get_run(use_dataset_from_run)
            if hasattr(run, 'inputs') and run.inputs.dataset_inputs:
                dataset_input = run.inputs.dataset_inputs[0]
                dataset_source = mlflow.data.get_source(dataset_input.dataset)
                data_path = dataset_source.load()
                df = pd.read_parquet(data_path)
                dataset_to_use = mlflow.data.from_pandas(
                    df=df, 
                    source=data_path, 
                    targets="Remission",
                    name=dataset_input.dataset.name # Preserve the original name
                )
                logger.info(f"Successfully loaded dataset from run {use_dataset_from_run}: {dataset_to_use.name}")
            else:
                logger.warning(f"No dataset found in run {use_dataset_from_run}")
        except Exception as e:
            logger.warning(f"Failed to load dataset from run {use_dataset_from_run}: {e}")
    
    # If no specific run provided, search for dataset that matches current configuration
    if dataset_to_use is None:
        logger.info("Searching for dataset matching current processing configuration...")
        dataset_to_use = find_dataset_by_config(config, experiment_id)
        
        if dataset_to_use:
            logger.info(f"✓ Found configuration-matching dataset: {dataset_to_use.name}")
            logger.info(f"  Dataset digest: {dataset_to_use.digest}")
            logger.info(f"  Dataset rows: {len(dataset_to_use.df)}")
            logger.info(f"  Dataset features: {len(dataset_to_use.df.columns) - 2}")  # Exclude Participant and Remission
        else:
            logger.info("No dataset found matching current configuration")
            
            # Fallback: try the old method of searching for recent processing runs
            logger.info("Falling back to searching for recent processing runs with datasets...")
            try:
                # First, try to find the eeg_processing experiment
                processing_experiment = None
                try:
                    processing_experiment = mlflow.get_experiment_by_name("eeg_processing")
                except Exception as e:
                    logger.warning(f"Could not find eeg_processing experiment: {e}")
                
                # Search for runs with datasets in the processing experiment
                search_experiment_ids = [processing_experiment.experiment_id] if processing_experiment else [experiment_id]
                runs_with_datasets = mlflow.search_runs(
                    experiment_ids=search_experiment_ids,
                    filter_string="tags.`mlflow.dataset.logged` = 'true' AND tags.`mlflow.dataset.context` = 'training'",
                    order_by=["start_time DESC"],
                    max_results=5
                )
                
                if not runs_with_datasets.empty:
                    latest_run_id = runs_with_datasets.iloc[0]['run_id']
                    experiment_source = "eeg_processing" if processing_experiment else "current"
                    logger.info(f"Found recent processing run with dataset in {experiment_source} experiment: {latest_run_id}")
                    
                    try:
                        run = mlflow.get_run(latest_run_id)
                        if hasattr(run, 'inputs') and run.inputs.dataset_inputs:
                            dataset_input = run.inputs.dataset_inputs[0]
                            dataset_source = mlflow.data.get_source(dataset_input.dataset)
                            data_path = dataset_source.load()
                            df = pd.read_parquet(data_path)
                            dataset_to_use = mlflow.data.from_pandas(
                                df=df, 
                                source=data_path, 
                                targets="Remission",
                                name=dataset_input.dataset.name # Preserve the original name
                            )
                            logger.info(f"⚠ Using fallback dataset from latest processing run: {dataset_to_use.name}")
                            logger.info(f"  Dataset digest: {dataset_to_use.digest}")
                            logger.info(f"  Dataset rows: {len(df)}")
                            logger.info(f"  Dataset features: {len(df.columns) - 2}")  # Exclude Participant and Remission
                            logger.warning(f"  Note: This dataset may not match your current configuration!")
                        else:
                            logger.warning(f"No dataset inputs found in run {latest_run_id}")
                    except Exception as e:
                        logger.warning(f"Failed to load dataset from latest run {latest_run_id}: {e}")
                else:
                    search_location = "eeg_processing experiment" if processing_experiment else "current experiment"
                    logger.info(f"No recent processing runs with datasets found in {search_location}")
                    
            except Exception as e:
                logger.warning(f"Failed to search for processing runs: {e}")
    
    # Handle feature categories filtering
    if feature_categories:
        if feature_categories.lower() == 'list':
            # Show available feature categories
            categories = FeatureFilter.list_available_categories()
            subcategories = FeatureFilter.list_available_subcategories()
            
            print("\n=== Available Feature Categories ===")
            for category, description in categories.items():
                print(f"\n{category}: {description}")
                if category in subcategories:
                    print("  Subcategories:")
                    for subcat, subdesc in subcategories[category].items():
                        print(f"    - {subcat}: {subdesc}")
            
            print(f"\n=== Usage Examples ===")
            print("--feature-categories=spectral_features,psd_statistics")
            print("--feature-categories=temporal_features,basic_statistics")
            print("--feature-categories=entropy_features,complexity_features")
            print("--feature-categories=spectral_features,temporal_features,entropy_features")
            return
        
        # Parse feature categories
        feature_categories_list = [cat.strip() for cat in feature_categories.split(',')]
        logger.info(f"Feature filtering enabled with categories: {feature_categories_list}")
        
        # Validate feature categories
        try:
            feature_filter = FeatureFilter(channels, feature_categories_list)
            logger.info("Feature categories validated successfully")
        except ValueError as e:
            logger.error(f"Invalid feature categories: {e}")
            print(f"\nError: {e}")
            print("Use --feature-categories=list to see available categories")
            return
        
        # Apply feature filtering to dataset if available
        if dataset_to_use:
            logger.info("Applying feature filtering to MLflow dataset...")
            try:
                original_df = dataset_to_use.df
                filtered_df = feature_filter.filter_features(original_df)
                
                # Create new filtered dataset
                filtered_dataset = mlflow.data.from_pandas(
                    df=filtered_df,
                    source=dataset_to_use.source,
                    targets="Remission",
                    name=f"{dataset_to_use.name}_filtered_{'_'.join(feature_categories_list)}"
                )
                
                dataset_to_use = filtered_dataset
                logger.info(f"Applied feature filtering: {len(original_df.columns)} -> {len(filtered_df.columns)} columns")
                
                # Store feature filtering info for later logging (don't log here to avoid MLflow run conflicts)
                feature_filtering_info = {
                    "feature_filtering_applied": True,
                    "feature_categories": feature_categories,
                    "original_features": len(original_df.columns) - 2,
                    "filtered_features": len(filtered_df.columns) - 2
                }
                
            except Exception as e:
                logger.error(f"Failed to apply feature filtering: {e}")
                raise
        else:
            logger.warning("Feature filtering requested but no dataset available. Will be applied during training.")
            feature_filtering_info = {
                "feature_filtering_applied": False,
                "feature_categories": feature_categories
            }
    else:
        logger.info("No feature filtering requested")
        feature_filtering_info = {
            "feature_filtering_applied": False
        }
    
    # Create appropriate trainer
    if model_type in ['pytorch_mlp', 'keras_mlp', 'hybrid_1dcnn_lstm', 'advanced_hybrid_1dcnn_lstm']:
        trainer = DeepLearningTrainer(config)
    else:
        trainer_cls = PatientLevelTrainer if level == 'patient' else WindowLevelTrainer
        trainer = trainer_cls(config)
    
    try:
        run_name_suffix = ""
        if enable_feature_selection:
            run_name_suffix = f"_fs_{n_features_select}_{fs_method}"
        
        # Add dataset info to run name if available
        dataset_suffix = ""
        if dataset_to_use:
            dataset_suffix = f"_ds_{dataset_to_use.digest[:8]}"
            
        with mlflow.start_run(run_name=f"{model_type}_{window_size}s{run_name_suffix}{dataset_suffix}") as run:
            # Log dataset usage info
            if dataset_to_use:
                mlflow.log_input(dataset_to_use, context="training")
                mlflow.log_param("used_mlflow_dataset", True)
                mlflow.log_param("dataset_name", dataset_to_use.name)
                mlflow.log_param("dataset_digest", dataset_to_use.digest)
                mlflow.log_param("dataset_selection_method", "configuration_match" if not use_dataset_from_run else "specific_run")
            else:
                mlflow.log_param("used_mlflow_dataset", False)
                mlflow.log_param("dataset_selection_method", "file_fallback")
                logger.info("No MLflow dataset available, trainer will fall back to file path")
            
            # Log feature filtering info
            for key, value in feature_filtering_info.items():
                mlflow.log_param(key, value)
            
            # Train model with dataset
            model = trainer.train(dataset=dataset_to_use)
            
            # Get sample data for model signature - prioritize dataset if available
            if dataset_to_use:
                df = dataset_to_use.df
            else:
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
            model_info = {
                'level': level, 
                'config': config,
                'window_size': window_size,
                'X_sample': X_sample,
                'y_proba_sample': y_proba_sample
            }
            
            # Add dataset info to model metadata if available
            if dataset_to_use:
                model_info['dataset_name'] = dataset_to_use.name
                model_info['dataset_digest'] = dataset_to_use.digest
                model_info['used_mlflow_dataset'] = True
            else:
                model_info['used_mlflow_dataset'] = False
            
            model_id = model_serializer.save_model(model, model_info=model_info)
            
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
                        channels = processing_config['data_loader']['channels']
                        channels_str = "-".join(channels)
                except Exception as e:
                    logger.error(f"Failed to determine window size: {str(e)}")
                    raise
            else:
                # Get channels from processing config
                try:
                    with open('configs/processing_config.yaml', 'r') as f:
                        processing_config = yaml.safe_load(f)
                        channels = processing_config['data_loader']['channels']
                        channels_str = "-".join(channels)
                except Exception as e:
                    logger.error(f"Failed to load channels from processing config: {str(e)}")
                    raise
        
        # Format the feature path with the window size and channels
        data_path = config['data']['feature_path'].format(
            window_size=window_size,
            channels=channels_str
        )
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
