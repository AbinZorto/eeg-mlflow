import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.signal
from src.processing.data_loader import EEGDataLoader
from src.processing.upsampler import EEGUpsampler
from src.processing.filter import EEGFilter
from src.processing.downsampler import EEGDownsampler
from src.processing.window_slicer import EEGWindowSlicer
from src.processing.feature_extractor import EEGFeatureExtractor

@pytest.fixture
def sample_eeg_data():
    """Generate sample EEG data for testing."""
    # Create synthetic EEG-like signals
    t = np.linspace(0, 10, 2560)  # 10 seconds at 256 Hz
    
    # Generate different frequency components
    delta = 0.5 * np.sin(2 * np.pi * 2 * t)  # 2 Hz
    theta = 0.3 * np.sin(2 * np.pi * 6 * t)  # 6 Hz
    alpha = 0.4 * np.sin(2 * np.pi * 10 * t)  # 10 Hz
    beta = 0.2 * np.sin(2 * np.pi * 20 * t)  # 20 Hz
    
    # Combine components and add noise
    signal = delta + theta + alpha + beta + 0.1 * np.random.randn(len(t))
    
    # Create DataFrame with multiple channels
    df = pd.DataFrame({
        'Participant': ['P001'] * len(t),
        'af7': signal,
        'af8': signal * 1.1,  # Slightly different amplitude
        'tp9': signal * 0.9,  # Slightly different amplitude
        'tp10': signal * 1.2,  # Slightly different amplitude
        'Remission': [0] * len(t)
    })
    
    return df

@pytest.fixture
def processing_config():
    """Generate sample configuration for processing."""
    return {
        'channels': ['af7', 'af8', 'tp9', 'tp10'],
        'sampling_rate': 256,
        'upsampler': {
            'factor': 2,
            'method': 'linear'
        },
        'filter': {
            'type': 'butterworth',
            'order': 4,
            'cutoff_frequency': 60
        },
        'downsampler': {
            'factor': 2,
            'method': 'decimate'
        },
        'window_slicer': {
            'window_seconds': 1,
            'overlap_seconds': 0
        },
        'feature_extractor': {
            'spectral_features': True,
            'temporal_features': True,
            'complexity_features': True,
            'validation': {
                'check_nan': True,
                'check_infinite': True,
                'remove_invalid': False
            }
        }
    }

class TestDataLoader:
    def test_data_validation(self, sample_eeg_data, processing_config):
        loader = EEGDataLoader(processing_config)
        
        # Test valid data
        is_valid, message = loader.validate_data(sample_eeg_data['af7'].values)
        assert is_valid
        
        # Test invalid data
        invalid_data = np.array([np.nan, 1, 2])
        is_valid, message = loader.validate_data(invalid_data)
        assert not is_valid
        assert "NaN" in message
    
    def test_channel_extraction(self, sample_eeg_data, processing_config):
        loader = EEGDataLoader(processing_config)
        
        # Test that all channels are present
        for channel in processing_config['channels']:
            assert channel in sample_eeg_data.columns
            
        # Test channel data shape
        for channel in processing_config['channels']:
            assert len(sample_eeg_data[channel]) == len(sample_eeg_data)

class TestUpsampler:
    def test_upsampling(self, sample_eeg_data, processing_config):
        upsampler = EEGUpsampler(processing_config['upsampler'])
        upsampled_df = upsampler.upsample_data(sample_eeg_data)
        
        # Test output dimensions
        factor = processing_config['upsampler']['factor']
        expected_length = len(sample_eeg_data) * factor
        
        for channel in processing_config['channels']:
            assert len(upsampled_df[channel].iloc[0]) == expected_length
        
        # Test signal properties preservation
        original_power = np.mean(np.square(sample_eeg_data['af7'].values))
        upsampled_power = np.mean(np.square(upsampled_df['af7'].iloc[0]))
        assert np.abs(original_power - upsampled_power) < 0.1 * original_power

class TestFilter:
    def test_filter_design(self, processing_config):
        filter_obj = EEGFilter(processing_config['filter'])
        
        # Test filter coefficients
        assert len(filter_obj.b) == processing_config['filter']['order'] + 1
        assert len(filter_obj.a) == processing_config['filter']['order'] + 1
        
        # Test frequency response
        w, h = scipy.signal.freqz(filter_obj.b, filter_obj.a)
        freq = w * processing_config['sampling_rate'] / (2 * np.pi)
        cutoff_idx = np.argmin(np.abs(freq - processing_config['filter']['cutoff_frequency']))
        magnitude_db = 20 * np.log10(np.abs(h))
        
        # Check -3dB point near cutoff frequency
        assert np.abs(magnitude_db[cutoff_idx] + 3) < 1.0
    
    def test_filtering(self, sample_eeg_data, processing_config):
        filter_obj = EEGFilter(processing_config['filter'])
        filtered_df = filter_obj.filter_data(sample_eeg_data)
        
        # Test output dimensions
        assert len(filtered_df) == len(sample_eeg_data)
        
        # Test frequency content
        for channel in processing_config['channels']:
            freqs, psd = scipy.signal.welch(filtered_df[channel].iloc[0], 
                                          fs=processing_config['sampling_rate'])
            
            # Check that high frequencies are attenuated
            high_freq_power = np.mean(psd[freqs > processing_config['filter']['cutoff_frequency']])
            low_freq_power = np.mean(psd[freqs <= processing_config['filter']['cutoff_frequency']])
            assert high_freq_power < 0.1 * low_freq_power

class TestDownsampler:
    def test_downsampling(self, sample_eeg_data, processing_config):
        downsampler = EEGDownsampler(processing_config['downsampler'])
        downsampled_df = downsampler.downsample_data(sample_eeg_data)
        
        # Test output dimensions
        factor = processing_config['downsampler']['factor']
        expected_length = len(sample_eeg_data) // factor
        
        for channel in processing_config['channels']:
            assert len(downsampled_df[channel].iloc[0]) == expected_length
        
        # Test signal properties preservation
        original_power = np.mean(np.square(sample_eeg_data['af7'].values))
        downsampled_power = np.mean(np.square(downsampled_df['af7'].iloc[0]))
        assert np.abs(original_power - downsampled_power) < 0.1 * original_power

class TestWindowSlicer:
    def test_window_slicing(self, sample_eeg_data, processing_config):
        slicer = EEGWindowSlicer(processing_config['window_slicer'])
        windowed_df = slicer.slice_data(sample_eeg_data)
        
        # Test window dimensions
        window_length = int(processing_config['window_slicer']['window_seconds'] * 
                          processing_config['sampling_rate'])
        
        for channel in processing_config['channels']:
            assert len(windowed_df[channel].iloc[0]) == window_length
        
        # Test window count
        expected_windows = len(sample_eeg_data) // window_length
        assert len(windowed_df) >= expected_windows