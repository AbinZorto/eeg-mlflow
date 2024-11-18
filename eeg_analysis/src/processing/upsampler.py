import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import mlflow
from pathlib import Path
from typing import Dict, Any, Union, List
from ..utils.logger import get_logger
import time

logger = get_logger(__name__)

class EEGUpsampler:
    """EEG signal upsampling class."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the upsampler with configuration.
        
        Args:
            config: Configuration dictionary containing upsampling parameters
        """
        self.factor = config['upsampler']['factor']
        self.method = config['upsampler']['method']
        self.save_interim = config['upsampler']['save_interim']
        self.channels = config['data_loader']['channels']
        self.logger = get_logger(__name__)
        
        # Log initialization parameters
        self.logger.info(f"Initialized EEGUpsampler with factor={self.factor}")
        self.logger.info(f"Channels to process: {self.channels}")
        self.logger.info(f"Save interim results: {self.save_interim}")
        
        # Validate configuration
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.factor < 1:
            raise ValueError("Upsampling factor must be greater than 1")
            
        valid_methods = ['linear', 'cubic', 'quadratic']
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
    
    def _upsample_array(self, data: np.ndarray, target_length: int) -> np.ndarray:
        """
        Upsample data to target length using linear interpolation.
        
        Args:
            data: Input array to upsample
            target_length: Target length for upsampled data
            
        Returns:
            Upsampled array
        """
        x = np.arange(len(data))
        interpolator = interp1d(x, data, kind='linear')
        x_new = np.linspace(0, len(data) - 1, target_length)
        return interpolator(x_new)
    
    def upsample_data(self, df: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
        """
        Upsample EEG data.
        
        Args:
            df: Input DataFrame containing EEG data
            output_path: Optional path to save interim results
            
        Returns:
            DataFrame with upsampled EEG data
        """
        try:
            start_time = time.time()
            #max_length = max(len(df[channel].iloc[0]) for channel in self.channels)
            # Calculate target length based on maximum window size
            target_length = 2560 * self.factor
            
            self.logger.info(f"Starting upsampling with factor {self.factor}")
            self.logger.info(f"Target length: {target_length} samples")
            
            # Process each window
            processed_data = []
            total_windows = len(df)
            
            for idx, row in df.iterrows():
                if idx % 100 == 0:
                    self.logger.info(f"Processing window {idx+1}/{total_windows}")
                
                # Initialize data dictionary with non-channel columns
                data_dict = {col: row[col] for col in df.columns if col not in self.channels}
                
                # Process each channel
                for channel in self.channels:
                    channel_data = np.array(row[channel])
                    upsampled_data = self._upsample_array(channel_data, target_length)
                    data_dict[channel] = upsampled_data
                
                processed_data.append(data_dict)
            
            # Create result DataFrame
            result_df = pd.DataFrame(processed_data)
            
            # Save interim results if requested
            if self.save_interim and output_path:
                self._save_data(result_df, output_path)
            
            process_time = time.time() - start_time
            self.logger.info(f"Upsampling completed in {process_time:.2f} seconds")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error in upsampling: {str(e)}")
            raise
    
    def _save_data(self, df: pd.DataFrame, output_path: str) -> None:
        """Save the processed data to files."""
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

def upsample_eeg_data(config: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """
    Upsample EEG data according to configuration.
    
    Args:
        config: Configuration dictionary
        df: Input DataFrame
        
    Returns:
        DataFrame with upsampled EEG data
    """
    upsampler = EEGUpsampler(config)
    return upsampler.upsample_data(
        df,
        output_path=config['paths']['interim']['upsampled']
    )