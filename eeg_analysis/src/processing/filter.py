import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, freqz
import mlflow
from pathlib import Path
from typing import Dict, Any, Tuple, List
from ..utils.logger import get_logger
import time

logger = get_logger(__name__)

class EEGFilter:
    """EEG signal filtering class."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the filter with configuration."""
        # Get filter parameters from the filter section of config
        self.filter_type = config['filter']['type']
        self.order = config['filter']['order']
        self.cutoff = config['filter']['cutoff_frequency']
        
        # Get channels from data_loader config
        self.channels = config['data_loader']['channels']
        
        # Calculate sampling rate based on window_slicer rate and upsampling factor
        self.base_sampling_rate = config['filter']['sampling_rate']
        self.upsampling_factor = config['upsampler']['factor']
        self.sampling_rate = self.base_sampling_rate * self.upsampling_factor
        
        self.save_interim = config['filter']['save_interim']
        self.logger = get_logger(__name__)
        
        self.logger.info(f"Initialized {self.filter_type} filter:")
        self.logger.info(f"- Order: {self.order}")
        self.logger.info(f"- Cutoff: {self.cutoff} Hz")
        self.logger.info(f"- Sampling rate: {self.sampling_rate} Hz")
        self.logger.info(f"- Channels: {self.channels}")
        
        # Design filter
        self.b, self.a = self._design_filter()
        
        # Calculate and store frequency response
        self.w, self.h = freqz(self.b, self.a)
        
    def _design_filter(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Design the filter.
        
        Returns:
            Tuple of filter coefficients (b, a)
        """
        nyquist = 0.5 * self.sampling_rate
        normalized_cutoff = self.cutoff / nyquist
        
        if self.filter_type == 'butterworth':
            return butter(self.order, normalized_cutoff, btype='low')
        else:
            raise ValueError(f"Unsupported filter type: {self.filter_type}")
    
    def _log_filter_design(self) -> None:
        """Log filter design parameters and frequency response."""
        # Log filter parameters
        mlflow.log_params({
            "filter_type": self.filter_type,
            "filter_order": self.order,
            "cutoff_frequency": self.cutoff,
            "sampling_rate": self.base_sampling_rate
        })
        
        # Calculate and log frequency response metrics
        freqs = self.w * self.sampling_rate / (2 * np.pi)
        magnitude_db = 20 * np.log10(np.abs(self.h))
        
        # Find -3dB point
        cutoff_idx = np.argmin(np.abs(magnitude_db + 3))
        actual_cutoff = freqs[cutoff_idx]
        
        mlflow.log_metric("actual_cutoff_frequency", actual_cutoff)
        mlflow.log_metric("stopband_attenuation", np.min(magnitude_db))
    
    def apply_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply filter to a single signal.
        
        Args:
            signal: Input signal array
            
        Returns:
            Filtered signal array
        """
        try:
            # Ensure signal is numpy array
            if isinstance(signal, (list, pd.Series)):
                signal = np.array(signal)
            
            # Apply zero-phase filtering
            filtered_signal = filtfilt(self.b, self.a, signal)
            
            return filtered_signal
            
        except Exception as e:
            logger.error(f"Error filtering signal: {str(e)}")
            return np.zeros_like(signal)  # Return zero array as fallback
    
    def process_window(self, window_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single window of data.
        
        Args:
            window_data: Dictionary containing window data
            
        Returns:
            Dictionary with filtered data
        """
        try:
            filtered_data = {}
            
            # Copy non-channel data
            for key, value in window_data.items():
                if key not in self.channels:
                    filtered_data[key] = value
                    continue
                
                # Filter channel data
                signal = np.array(value)
                filtered_signal = self.apply_filter(signal)
                filtered_data[key] = filtered_signal
            
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"Error processing window: {str(e)}")
            raise
    
    def filter_data(self, df: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
        """
        Filter EEG data.
        
        Args:
            df: Input DataFrame
            output_path: Optional path to save processed data
            
        Returns:
            DataFrame with filtered data
        """
        try:
            logger.info("Starting filtering process")
            start_time = time.time()
            
            # Log filter design
            self._log_filter_design()
            
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
            logger.info(f"Filtering completed in {processing_time:.2f} seconds")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error in filtering: {str(e)}")
            raise
    
    def _log_statistics(self, input_df: pd.DataFrame, output_df: pd.DataFrame) -> None:
        """
        Log processing statistics to MLflow.
        
        Args:
            input_df: Input DataFrame
            output_df: Output DataFrame
        """
        # Log basic processing info
        mlflow.log_metric("processed_windows", len(output_df))
        
        # Compare signal properties before and after
        for channel in self.channels:
            # Calculate signal power before and after
            original_data = np.vstack(input_df[channel].values)
            filtered_data = np.vstack(output_df[channel].values)
            
            original_power = np.mean(np.square(original_data))
            filtered_power = np.mean(np.square(filtered_data))
            
            # Log power reduction
            power_reduction_db = 10 * np.log10(filtered_power / original_power)
            mlflow.log_metric(f"{channel}_power_reduction_db", power_reduction_db)
            
            # Log statistics for both signals
            for prefix, data in [("original", original_data), ("filtered", filtered_data)]:
                mlflow.log_metric(f"{channel}_{prefix}_mean", np.mean(data))
                mlflow.log_metric(f"{channel}_{prefix}_std", np.std(data))
                mlflow.log_metric(f"{channel}_{prefix}_min", np.min(data))
                mlflow.log_metric(f"{channel}_{prefix}_max", np.max(data))
    
    def _save_data(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save the filtered data to files.
        
        Args:
            df: DataFrame containing filtered data
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

def filter_eeg_data(config: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter EEG data according to configuration.
    
    Args:
        config: Configuration dictionary
        df: Input DataFrame
        
    Returns:
        DataFrame with filtered EEG data
    """
    # Pass the entire config
    filter_obj = EEGFilter(config)
    return filter_obj.filter_data(
        df,
        output_path=config['paths']['interim']['filtered']
    )

if __name__ == '__main__':
    # Example usage
    import yaml
    
    with open('configs/processing_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load input data
    input_df = pd.read_parquet(config['paths']['interim']['upsampled'])
    
    # Apply filtering
    filtered_df = filter_eeg_data(config, input_df)