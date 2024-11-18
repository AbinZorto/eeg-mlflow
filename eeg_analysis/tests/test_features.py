import pytest
import numpy as np
import pandas as pd
from src.processing.feature_extractor import EEGFeatureExtractor

@pytest.fixture
def sample_features_data():
    """Generate sample data for feature extraction testing."""
    # Generate synthetic EEG windows
    n_windows = 10
    window_length = 256  # 1 second at 256 Hz
    
    windows = []
    for i in range(n_windows):
        # Create time vector
        t = np.linspace(0, 1, window_length)
        
        # Generate signal with multiple frequency components
        signal = (
            0.5 * np.sin(2 * np.pi * 2 * t) +   # Delta (2 Hz)
            0.3 * np.sin(2 * np.pi * 6 * t) +   # Theta (6 Hz)
            0.4 * np.sin(2 * np.pi * 10 * t) +  # Alpha (10 Hz)
            0.2 * np.sin(2 * np.pi * 20 * t)    # Beta (20 Hz)
        )
        
        # Add noise
        signal += 0.1 * np.random.randn(len(t))
        
        windows.append({
            'Participant': f'P{i+1:03d}',
            'af7': signal,
            'af8': signal * 1.1,
            'tp9': signal * 0.9,
            'tp10': signal * 1.2,
            'Remission': i % 2  # Alternate between classes
        })
    
    return pd.DataFrame(windows)

@pytest.fixture
def feature_config():
    """Generate feature extraction configuration."""
    return {
        'channels': ['af7', 'af8', 'tp9', 'tp10'],
        'sampling_rate': 256,
        'spectral_features': True,
        'temporal_features': True,
        'complexity_features': True,
        'connectivity_features': True,
        'frequency_bands': {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 30),
            'gamma': (30, 60)
        },
        'validation': {
            'check_nan': True,
            'check_infinite': True,
            'remove_invalid': False
        }
    }

class TestFeatureExtractor:
    def test_spectral_features(self, sample_features_data, feature_config):
        extractor = EEGFeatureExtractor(feature_config)
        features = extractor.extract_all_features(sample_features_data)
        
        # Test band power features
        for channel in feature_config['channels']:
            for band in feature_config['frequency_bands']:
                assert f"{channel}_bp_{band}" in features.columns
                assert f"{channel}_rbp_{band}" in features.columns
        
        # Test spectral entropy
        for channel in feature_config['channels']:
            assert f"{channel}_spectral_entropy" in features.columns
    
    def test_temporal_features(self, sample_features_data, feature_config):
        extractor = EEGFeatureExtractor(feature_config)
        features = extractor.extract_all_features(sample_features_data)
        
        temporal_features = ['mean', 'std', 'var', 'skew', 'kurtosis', 
                           'rms', 'zero_crossings', 'hjorth_activity',
                           'hjorth_mobility', 'hjorth_complexity']
        
        for channel in feature_config['channels']:
            for feature in temporal_features:
                assert f"{channel}_{feature}" in features.columns
    
    def test_complexity_features(self, sample_features_data, feature_config):
        extractor = EEGFeatureExtractor(feature_config)
        features = extractor.extract_all_features(sample_features_data)
        
        complexity_features = ['sample_entropy', 'correlation_dim',
                             'hurst', 'lyapunov', 'dfa']
        
        for channel in feature_config['channels']:
            for feature in complexity_features:
                assert f"{channel}_{feature}" in features.columns
    
    def test_connectivity_features(self, sample_features_data, feature_config):
        extractor = EEGFeatureExtractor(feature_config)
        features = extractor.extract_all_features(sample_features_data)
        
        # Test cross-correlation features
        for i, ch1 in enumerate(feature_config['channels']):
            for j, ch2 in enumerate(feature_config['channels'][i+1:], i+1):
                assert f"xcorr_{ch1}_{ch2}" in features.columns
    
    def test_feature_validation(self, sample_features_data, feature_config):
        extractor = EEGFeatureExtractor(feature_config)
        
        # Test with valid data
        features = extractor.extract_all_features(sample_features_data)
        assert not features.isna().any().any()
        
        # Test with invalid data
        invalid_data = sample_features_data.copy()
        invalid_data.loc[0, 'af7'] = np.array([np.nan] * len(invalid_data.loc[0, 'af7']))
        
        features = extractor.extract_all_features(invalid_data)
        assert features.isna().any().any()
    
    def test_feature_stability(self, sample_features_data, feature_config):
        """Test feature stability across similar signals."""
        extractor = EEGFeatureExtractor(feature_config)
        
        # Create slightly modified version of the data
        modified_data = sample_features_data.copy()
        noise_level = 0.01
        
        for channel in feature_config['channels']:
            modified_data[channel] = modified_data[channel].apply(
                lambda x: x + noise_level * np.random.randn(len(x))
            )
        
        # Extract features for both datasets
        original_features = extractor.extract_all_features(sample_features_data)
        modified_features = extractor.extract_all_features(modified_data)
        
        # Compare feature values
        for col in original_features.columns:
            if col not in ['Participant', 'Remission']:
                relative_diff = np.abs(
                    original_features[col] - modified_features[col]
                ) / np.abs(original_features[col])
                
                assert np.mean(relative_diff) < 0.1  # Less than 10% difference
    
    def test_feature_scaling(self, sample_features_data, feature_config):
        """Test feature scaling properties."""
        extractor = EEGFeatureExtractor(feature_config)
        
        # Create scaled version of the data
        scale_factor = 2.0
        scaled_data = sample_features_data.copy()
        
        for channel in feature_config['channels']:
            scaled_data[channel] = scaled_data[channel].apply(
                lambda x: x * scale_factor
            )
        
        # Extract features
        original_features = extractor.extract_all_features(sample_features_data)
        scaled_features = extractor.extract_all_features(scaled_data)
        
        # Test scaling properties
        for col in original_features.columns:
            if col not in ['Participant', 'Remission']:
                if 'rms' in col or 'mean' in col or 'std' in col:
                    # These features should scale linearly
                    ratio = scaled_features[col] / original_features[col]
                    np.testing.assert_array_almost_equal(
                        ratio, 
                        scale_factor,
                        decimal=1
                    )
                elif 'entropy' in col or 'correlation_dim' in col:
                    # These features should be scale-invariant
                    np.testing.assert_array_almost_equal(
                        scaled_features[col],
                        original_features[col],
                        decimal=1
                    )
    
    def test_feature_reproducibility(self, sample_features_data, feature_config):
        """Test feature extraction reproducibility."""
        extractor = EEGFeatureExtractor(feature_config)
        
        # Extract features multiple times
        features_1 = extractor.extract_all_features(sample_features_data)
        features_2 = extractor.extract_all_features(sample_features_data)
        
        # Compare results
        for col in features_1.columns:
            if col not in ['Participant', 'Remission']:
                np.testing.assert_array_equal(
                    features_1[col],
                    features_2[col]
                )