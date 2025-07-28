import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from .logger import get_logger
import mlflow

logger = get_logger(__name__)

class ConfigurationError(Exception):
    """Custom exception for configuration errors."""
    pass

def validate_paths(config: Dict[str, Any]) -> None:
    """
    Validate and create necessary paths in configuration.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ConfigurationError: If required paths are missing or invalid
    """
    required_paths = ['raw_data', 'interim', 'features']
    
    if 'paths' not in config:
        raise ConfigurationError("Configuration must contain 'paths' section")
    
    for path_type in required_paths:
        if path_type not in config['paths']:
            raise ConfigurationError(f"Missing required path: {path_type}")
        
        # Create directories if they don't exist
        if isinstance(config['paths'][path_type], str):
            Path(config['paths'][path_type]).parent.mkdir(parents=True, exist_ok=True)
        elif isinstance(config['paths'][path_type], dict):
            for path in config['paths'][path_type].values():
                Path(path).parent.mkdir(parents=True, exist_ok=True)

def validate_parameters(config: Dict[str, Any]) -> None:
    """
    Validate processing parameters in configuration.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ConfigurationError: If required parameters are missing or invalid
    """
    required_sections = [
        'data_loader',
        'upsampler',
        'filter',
        'downsampler',
        'window_slicer',
        'feature_extractor'
    ]
    
    for section in required_sections:
        if section not in config:
            raise ConfigurationError(f"Missing required configuration section: {section}")
    
    # Validate specific parameters
    if config['upsampler'].get('factor', 0) < 1:
        raise ConfigurationError("Upsampling factor must be greater than 0")
    
    if config['filter'].get('cutoff_frequency', 0) <= 0:
        raise ConfigurationError("Filter cutoff frequency must be positive")
    
    if config['window_slicer'].get('window_seconds', 0) <= 0:
        raise ConfigurationError("Window size must be positive")

def load_config(config_path: str, env: Optional[str] = None) -> Dict[str, Any]:
    """
    Load and validate configuration file.
    
    Args:
        config_path: Path to configuration file
        env: Optional environment name for environment-specific configs
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    try:
        # Load base configuration
        print(f"🔍 DEBUG: Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"🔍 DEBUG: Config loaded successfully from {config_path}")
        
        # Load environment-specific configuration if provided
        if env:
            env_config_path = Path(config_path).parent / f"config.{env}.yaml"
            if env_config_path.exists():
                with open(env_config_path, 'r') as f:
                    env_config = yaml.safe_load(f)
                    config = deep_update(config, env_config)
        
        # Log configuration
        log_config(config)
        
        return config
        
    except Exception as e:
        raise ConfigurationError(f"Error loading configuration: {str(e)}")

def deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update a dictionary.
    
    Args:
        base_dict: Base dictionary to update
        update_dict: Dictionary with updates
        
    Returns:
        Updated dictionary
    """
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key] = deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def log_config(config: Dict[str, Any]) -> None:
    """
    Log configuration to MLflow and local log.
    
    Args:
        config: Configuration dictionary
    """
    # Log to MLflow
    if mlflow.active_run():
        mlflow.log_params({
            "sampling_rate": config['data_loader'].get('sampling_rate'),
            "upsampling_factor": config['upsampler'].get('factor'),
            "filter_cutoff": config['filter'].get('cutoff_frequency'),
            "window_size": config['window_slicer'].get('window_seconds')
        })
        
        # Save full config as artifact
        with open("config.yaml", "w") as f:
            yaml.dump(config, f)
        mlflow.log_artifact("config.yaml")
    
    # Log to local logger
    logger.info("Loaded configuration", extra={"config": json.dumps(config)})

def get_config(config_path: Optional[str] = None, env: Optional[str] = None) -> Dict[str, Any]:
    """
    Get configuration with environment handling.
    
    Args:
        config_path: Optional path to configuration file
        env: Optional environment name
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = os.environ.get('CONFIG_PATH', 'configs/processing_config.yaml')
    
    if env is None:
        env = os.environ.get('ENV')
    
    return load_config(config_path, env)