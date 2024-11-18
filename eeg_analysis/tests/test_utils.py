import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import mlflow
import logging
from src.utils.logger import setup_logger, get_logger
from src.utils.config import load_config, validate_paths, validate_parameters
from src.utils.metrics import MetricsCalculator

class TestLogger:
    def test_logger_setup(self, tmp_path):
        """Test logger initialization and configuration."""
        log_dir = tmp_path / "logs"
        logger = setup_logger(
            name="test_logger",
            log_dir=str(log_dir),
            console_level=logging.INFO,
            file_level=logging.DEBUG,
            json_format=True
        )
        
        # Test logger level
        assert logger.level == logging.DEBUG
        
        # Test handlers
        assert len(logger.handlers) == 2  # Console and file handlers
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
        assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        
        # Test log file creation
        log_files = list(log_dir.glob("*.log"))
        assert len(log_files) == 1
    
    def test_logger_output(self, tmp_path):
        """Test logger output formatting."""
        log_dir = tmp_path / "logs"
        logger = setup_logger("test_logger", str(log_dir))
        
        # Log test message
        test_message = "Test log message"
        logger.info(test_message)
        
        # Check log file content
        log_file = next(log_dir.glob("*.log"))
        with open(log_file) as f:
            content = f.read()
            assert test_message in content
            assert "INFO" in content

class TestConfig:
    @pytest.fixture
    def sample_config(self):
        return {
            'paths': {
                'raw_data': 'data/raw/eeg_data.mat',
                'interim': {
                    'upsampled': 'data/processed/interim/upsampled',
                    'filtered': 'data/processed/interim/filtered'
                },
                'features': 'data/processed/features'
            },
            'data_loader': {'sampling_rate': 256},
            'upsampler': {'factor': 2},
            'filter': {'cutoff_frequency': 60},
            'downsampler': {'factor': 2},
            'window_slicer': {'window_seconds': 1},
            'feature_extractor': {'spectral_features': True}
        }
    
    def test_path_validation(self, sample_config, tmp_path):
        """Test configuration path validation."""
        # Modify paths to use temporary directory
        config = sample_config.copy()
        for key in ['raw_data', 'features']:
            config['paths'][key] = str(tmp_path / config['paths'][key])
        for key, value in config['paths']['interim'].items():
            config['paths']['interim'][key] = str(tmp_path / value)
        
        # Test validation
        validate_paths(config)
        
        # Check directory creation
        assert (tmp_path / "data" / "raw").exists()
        assert (tmp_path / "data" / "processed" / "interim").exists()
        assert (tmp_path / "data" / "processed" / "features").exists()
    
    def test_parameter_validation(self, sample_config):
        """Test configuration parameter validation."""
        # Valid configuration
        validate_parameters(sample_config)
        
        # Test invalid configurations
        invalid_configs = [
            {'upsampler': {'factor': 0}},
            {'filter': {'cutoff_frequency': 0}},
            {'window_slicer': {'window_seconds': 0}}
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises(ValueError):
                validate_parameters({**sample_config, **invalid_config})

class TestMetrics:
    @pytest.fixture
    def sample_predictions(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = y_true.copy()
        y_pred[np.random.choice(len(y_pred), 20)] = 1 - y_pred[np.random.choice(len(y_pred), 20)]
        y_prob = np.random.random(100)
        return y_true, y_pred, y_prob
    
    def test_classification_metrics(self, sample_predictions):
        """Test basic classification metrics calculation."""
        y_true, y_pred, y_prob = sample_predictions
        calculator = MetricsCalculator()
        
        metrics = calculator.calculate_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob
        )
        
        # Check metric existence
        required_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        for metric in required_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1
    
    def test_cross_validation_metrics(self, sample_predictions):
        """Test cross-validation metrics calculation."""
        y_true, y_pred, y_prob = sample_predictions
        calculator = MetricsCalculator()
        
        # Create mock CV results
        cv_results = [
            {'y_true': y_true[:50], 'y_pred': y_pred[:50], 'y_prob': y_prob[:50]},
            {'y_true': y_true[50:], 'y_pred': y_pred[50:], 'y_prob': y_prob[50:]}
        ]
        
        metrics = calculator.calculate_cross_validation_metrics(cv_results)
        
        # Check metric statistics
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            assert metric in metrics
            assert all(stat in metrics[metric] for stat in ['mean', 'std', 'min', 'max'])
    
    def test_feature_importance_metrics(self):
        """Test feature importance metrics calculation."""
        calculator = MetricsCalculator()
        
        feature_names = [f'feature_{i}' for i in range(10)]
        importance_values = np.random.random(10)
        
        metrics = calculator.calculate_feature_importance_metrics(
            feature_names=feature_names,
            importance_values=importance_values,
            top_k=5
        )
        
        # Check results
        assert len(metrics['top_features']) == 5
        assert all(key in metrics for key in ['top_features', 'importance_statistics'])
        assert all(key in metrics['importance_statistics'] 
                  for key in ['mean', 'std', 'max', 'min'])
    
    def test_calibration_metrics(self, sample_predictions):
        """Test model calibration metrics calculation."""
        y_true, _, y_prob = sample_predictions
        calculator = MetricsCalculator()
        
        metrics = calculator.calculate_model_calibration_metrics(
            y_true=y_true,
            y_prob=y_prob,
            n_bins=10
        )
        
        # Check results
        assert 'bin_metrics' in metrics
        assert 'calibration_error' in metrics
        assert 'max_calibration_error' in metrics
        assert len(metrics['bin_metrics']) <= 10  # Some bins might be empty