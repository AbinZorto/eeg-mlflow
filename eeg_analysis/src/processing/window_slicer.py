import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
from ..utils.logger import get_logger
import mlflow
import time

logger = get_logger(__name__)

class EEGWindowSlicer:
    """EEG signal window slicing class."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize window slicer.
        
        Args:
            config: Configuration dictionary containing window parameters
        """
        self.config = config
        self.window_seconds = config['window_slicer']['window_seconds']
        self.sampling_rate = config['window_slicer']['sampling_rate']
        self.overlap_seconds = config['window_slicer']['overlap_seconds']
        self.min_windows = config['window_slicer']['min_windows']
        self.channels = config['data_loader']['channels']
        
        # Calculate window parameters
        self.window_length = int(self.window_seconds * self.sampling_rate)
        self.overlap_length = int(self.overlap_seconds * self.sampling_rate)
        self.step_size = self.window_length - self.overlap_length
        self.logger = get_logger(__name__)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.window_length <= 0:
            raise ValueError("Window length must be positive")
        
        if self.overlap_length < 0:
            raise ValueError("Overlap length must be non-negative")
        
        if self.overlap_length >= self.window_length:
            raise ValueError("Overlap length must be less than window length")
    
    def slice_signal(self, signal: np.ndarray) -> List[np.ndarray]:
        """
        Slice a single signal into windows.
        
        Args:
            signal: Input signal array
            
        Returns:
            List of windowed arrays
        """
        try:
            windows = []
            start = 0
            
            while start + self.window_length <= len(signal):
                window = signal[start:start + self.window_length]
                windows.append(window)
                start += self.step_size
            
            return windows
            
        except Exception as e:
            logger.error(f"Error slicing signal: {str(e)}")
            return []
    
    def process_window(self, window_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a single window of data.
        
        Args:
            window_data: Dictionary containing window data
            
        Returns:
            List of processed window dictionaries
        """
        try:
            # Get metadata columns that should be preserved
            metadata = {
                'participant_id': window_data.get('participant_id', window_data.get('Participant', 'unknown')),
                'group': window_data.get('group', 'unknown')
            }
            
            processed_windows = []
            window_length = self.window_seconds * self.sampling_rate
            
            # Get all channel data first
            channel_data = {}
            for channel in self.channels:
                if channel not in window_data:
                    continue
                channel_data[channel] = window_data[channel]
            
            # Calculate number of windows based on first channel
            first_channel = next(iter(channel_data.values()))
            num_samples = len(first_channel)
            num_windows = (num_samples - window_length) // self.step_size + 1
            
            # Create windows preserving all channels
            for i in range(num_windows):
                start_idx = i * self.step_size
                end_idx = start_idx + window_length
                
                window = {
                    'participant_id': metadata['participant_id'],
                    'group': metadata['group'],
                    'window_start': start_idx,
                    'window_end': end_idx,
                }
                
                # Add all channels to the same window
                for channel in self.channels:
                    if channel in channel_data:
                        window[channel] = channel_data[channel][start_idx:end_idx]
                
                processed_windows.append(window)
            
            return processed_windows
            
        except Exception as e:
            self.logger.error(f"Error processing window: {str(e)}")
            raise
    
    def slice_data(self, df: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
        """
        Slice EEG data into windows.
        
        Args:
            df: Input DataFrame
            output_path: Optional path to save processed data
            
        Returns:
            DataFrame with sliced data
        """
        try:
            logger.info(f"Starting window slicing with {self.window_seconds}s windows")
            start_time = time.time()
            
            all_windows = []
            total_windows = len(df)
            
            # Process each original window
            for idx, row in df.iterrows():
                if idx % 100 == 0:
                    logger.info(f"Processing window {idx+1}/{total_windows}")
                
                processed_windows = self.process_window(row.to_dict())
                all_windows.extend(processed_windows)
            
            # Create output DataFrame
            result_df = pd.DataFrame(all_windows)
            
            # Log processing statistics
            self._log_statistics(df, result_df)
            
            # Save results if output path provided
            if output_path:
                self._save_data(result_df, output_path)
            
            processing_time = time.time() - start_time
            logger.info(f"Window slicing completed in {processing_time:.2f} seconds")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error in window slicing: {str(e)}")
            raise
    
    def _log_statistics(self, input_df: pd.DataFrame, output_df: pd.DataFrame) -> None:
        """
        Log statistics about the windowing process.
        
        Args:
            input_df: Input DataFrame
            output_df: Output DataFrame with windowed data
        """
        try:
            # Log metrics instead of parameters for values that might change
            mlflow.log_metrics({
                'window_sampling_rate': self.sampling_rate,
                'input_windows': len(input_df),
                'output_windows': len(output_df),
                'window_length_seconds': self.window_seconds,
                'window_overlap_seconds': self.overlap_seconds
            })
            
            # Log fixed parameters that won't change
            mlflow.log_params({
                'window_slicer_method': 'fixed_length',
                'min_windows': self.min_windows
            })
            
            # Log participant-specific metrics
            windows_per_participant = output_df.groupby('participant_id').size()
            mlflow.log_metrics({
                f'participant_{pid}_windows': count 
                for pid, count in windows_per_participant.items()
            })
            
            # Log group-specific metrics
            for group in output_df['group'].unique():
                group_windows = len(output_df[output_df['group'] == group])
                mlflow.log_metric(f'{group}_windows', group_windows)
                
            self.logger.info(f"Logged windowing statistics to MLflow")
            
        except Exception as e:
            self.logger.warning(f"Failed to log statistics: {str(e)}")
            # Continue processing even if logging fails
    
    def _save_data(self, df: pd.DataFrame, output_path: str) -> None:
        """Save the windowed data to files."""
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

def slice_eeg_windows(config: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to slice EEG data into windows.
    
    Args:
        config: Configuration dictionary
        df: Input DataFrame
        
    Returns:
        DataFrame with windowed data
    """
    slicer = EEGWindowSlicer(config)
    return slicer.slice_data(
        df,
        config['paths']['interim']['windowed']
    )

if __name__ == '__main__':
    # Example usage
    import yaml
    
    with open('configs/processing_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load input data
    input_df = pd.read_parquet(config['paths']['interim']['downsampled'])
    
    # Apply window slicing
    windowed_df = slice_eeg_windows(config, input_df)