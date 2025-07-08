#!/usr/bin/env python3
"""
Example script for training patient-level models with automatic model selection.

This script demonstrates how to use the updated PatientLevelTrainer that can:
1. Try multiple classifiers automatically
2. Select the best one based on F1 score
3. Use enhanced aggregation with percentiles
4. Provide detailed metrics and logging

Usage:
    python train_patient_model.py --config configs/patient_model_config.yaml --window_size 2
"""

import argparse
import yaml
import mlflow
import logging
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.patient_trainer import PatientLevelTrainer
from utils.config import load_config

def setup_logging(config):
    """Setup logging configuration."""
    log_config = config.get('logging', {})
    
    # Create logs directory if it doesn't exist
    log_file = log_config.get('file', 'logs/training.log')
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    level = getattr(logging, log_config.get('level', 'INFO'))
    format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    handlers = []
    
    # File handler
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    # Console handler
    if log_config.get('console', True):
        handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers,
        force=True
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train patient-level EEG classification models")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--window_size", 
        type=int, 
        default=2,
        help="Window size in seconds"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Specific model type to use (overrides config). Use 'auto' for automatic selection."
    )
    parser.add_argument(
        "--feature_selection",
        action="store_true",
        help="Enable feature selection"
    )
    parser.add_argument(
        "--n_features",
        type=int,
        default=10,
        help="Number of features to select (if feature selection enabled)"
    )
    return parser.parse_args()

def update_config_with_args(config, args):
    """Update configuration with command line arguments."""
    # Update window size in data path
    if 'data' in config and 'feature_path' in config['data']:
        config['data']['feature_path'] = config['data']['feature_path'].format(
            window_size=args.window_size
        )
    
    # Override model type if specified
    if args.model_type:
        config['model_type'] = args.model_type
    
    # Update feature selection settings
    if args.feature_selection:
        config['feature_selection']['enabled'] = True
        config['feature_selection']['n_features'] = args.n_features
    
    # Add window size to config for reference
    config['window_size'] = args.window_size
    
    return config

def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with command line arguments
    config = update_config_with_args(config, args)
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting patient-level model training")
    logger.info(f"Configuration file: {args.config}")
    logger.info(f"Window size: {args.window_size}s")
    logger.info(f"Model type: {config.get('model_type', 'auto')}")
    logger.info(f"Feature selection: {config.get('feature_selection', {}).get('enabled', False)}")
    
    # Setup MLflow
    mlflow_config = config.get('mlflow', {})
    mlflow.set_tracking_uri(mlflow_config.get('tracking_uri', 'file:./mlruns'))
    mlflow.set_experiment(mlflow_config.get('experiment_name', 'patient_training'))
    
    try:
        # Initialize trainer
        trainer = PatientLevelTrainer(config)
        
        # Check if data file exists
        data_path = config['data']['feature_path']
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Feature file not found: {data_path}")
        
        logger.info(f"Loading features from: {data_path}")
        
        # Train model
        logger.info("Starting training...")
        model = trainer.train()
        
        logger.info("Training completed successfully!")
        
        # Log final information
        if hasattr(trainer, 'selected_feature_names'):
            logger.info(f"Number of features used: {len(trainer.selected_feature_names)}")
        
        # Log model type information
        if config.get('model_type') == 'auto':
            logger.info("Used automatic model selection - check MLflow logs for best classifier")
        else:
            logger.info(f"Used single model type: {config.get('model_type')}")
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    
    logger.info("Training script completed")

if __name__ == "__main__":
    main() 