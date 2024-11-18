import numpy as np
import pandas as pd
from scipy.signal import decimate
import mlflow
from pathlib import Path
from typing import Dict, Any, List
from ..utils.logger import get_logger
import time

logger = get_logger(__name__)

class EEGDownsampler:
    """EEG signal downsampling class."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize downsampler.
        
        Args:
            config: Configuration dictionary containing downsampling parameters
        """
        # Get downsampling parameters from the downsampler section of config
        self.factor = config['downsampler']['factor']
        self.method = config['downsampler']['method']
        
        # Get channels from data_loader config
        self.channels = config['data_loader']['channels']
        
        self.logger = get_logger(__name__)
        # Validate configuration
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.factor < 1:
            raise ValueError("Downsampling factor must be greater than 1")
            
        valid_methods = ['decimate', 'subsample']
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
    
    def downsample_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Downsample a single signal.
        
        Args:
            signal: Input signal array
            
        Returns:
            Downsampled signal array
        """
        try:
            if self.method == 'decimate':
                # Use scipy.signal.decimate (includes anti-aliasing filter)
                return decimate(signal, self.factor)
            else:
                # Simple subsampling
                return signal[::self.factor]
                
        except Exception as e:
            logger.error(f"Error downsampling signal: {str(e)}")
            return np.zeros(len(signal) // self.factor)
    
    def process_window(self, window_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Process a single window of EEG data.
        
        Args:
            window_data: Dictionary containing channel data
            
        Returns:
            Dictionary with downsampled channel data
        """
        processed_data = {}
        
        # Copy non-channel data
        for key, value in window_data.items():
            if key not in self.channels:
                processed_data[key] = value
        
        # Process each channel
        for channel in self.channels:
            channel_data = window_data.get(channel)
            if channel_data is not None:
                processed_data[channel] = self.downsample_signal(channel_data)
        
        return processed_data
    
    def downsample_data(self, df: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
        """
        Downsample EEG data.
        
        Args:
            df: Input DataFrame
            output_path: Optional path to save processed data
            
        Returns:
            DataFrame with downsampled data
        """
        try:
            logger.info(f"Starting downsampling with factor {self.factor}")
            start_time = time.time()
            
            processed_windows = []
            total_windows = len(df)
            
            # Process each window
            for idx, row in df.iterrows():
                if idx % 100 == 0:
                    logger.info(f"Processing window {idx+1}/{total_windows}")
                
                processed_window = self.process_window(row.to_dict())
                processed_windows.append(processed_window)
            
            # Create output DataFrame
            result_df = pd.DataFrame(processed_windows)
            
            # Log processing statistics
            self._log_statistics(df, result_df)
            
            # Save results if output path provided
            if output_path:
                self._save_data(result_df, output_path)
            
            processing_time = time.time() - start_time
            logger.info(f"Downsampling completed in {processing_time:.2f} seconds")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error in downsampling: {str(e)}")
            raise
    
    def _log_statistics(self, input_df: pd.DataFrame, output_df: pd.DataFrame) -> None:
        """
        Log processing statistics to MLflow.
        
        Args:
            input_df: Input DataFrame
            output_df: Output DataFrame
        """
        # Log basic processing info
        mlflow.log_param("downsampling_factor", self.factor)
        mlflow.log_param("downsampling_method", self.method)
        mlflow.log_metric("processing_windows", len(output_df))
        
        # Compare signal properties before and after
        for channel in self.channels:
            # Original signal statistics
            original_data = np.vstack(input_df[channel].values)
            downsampled_data = np.vstack(output_df[channel].values)
            
            # Log length ratio
            length_ratio = len(downsampled_data[0]) / len(original_data[0])
            mlflow.log_metric(f"{channel}_length_ratio", length_ratio)
            
            # Compute and log spectral preservation (if using decimate)
            if self.method == 'decimate':
                # Calculate power in frequency bands before and after
                for band_name, (low, high) in {
                    'delta': (0.5, 4),
                    'theta': (4, 8),
                    'alpha': (8, 12),
                    'beta': (12, 30)
                }.items():
                    # Log relative power preservation in each band
                    for prefix, data in [("original", original_data), 
                                       ("downsampled", downsampled_data)]:
                        power = np.mean(np.abs(np.fft.fft(data))**2)
                        mlflow.log_metric(
                            f"{channel}_{band_name}_{prefix}_power",
                            power
                        )
            
            # Log general statistics
            for prefix, data in [("original", original_data), 
                               ("downsampled", downsampled_data)]:
                mlflow.log_metric(f"{channel}_{prefix}_mean", np.mean(data))
                mlflow.log_metric(f"{channel}_{prefix}_std", np.std(data))
                mlflow.log_metric(f"{channel}_{prefix}_min", np.min(data))
                mlflow.log_metric(f"{channel}_{prefix}_max", np.max(data))
    
    def _save_data(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save the downsampled data to files.
        
        Args:
            df: DataFrame containing downsampled data
            output_path: Path to save the files
        """
        if not output_path:
            return
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save separate files for each group
        for group in df['group'].unique():
            group_df = df[df['group'] == group]
            file_path = output_path / f"{group.lower()}.parquet"
            group_df.to_parquet(file_path)
            self.logger.info(f"Saved {group} data to {file_path}")

def downsample_eeg_data(config: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to downsample EEG data.
    
    Args:
        config: Configuration dictionary
        df: Input DataFrame
        
    Returns:
        DataFrame with downsampled data
    """
    downsampler = EEGDownsampler(config)
    return downsampler.downsample_data(
        df,
        config['paths']['interim']['downsampled']
    )