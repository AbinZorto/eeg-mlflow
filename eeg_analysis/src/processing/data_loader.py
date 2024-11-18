import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import scipy.io
import mlflow
from pathlib import Path
from ..utils.logger import get_logger

logger = get_logger(__name__)

class DataLoader:
    def __init__(self, config: Dict[str, Any]):
        self.channels = config['channels']
        self.validate_data = config['validate_data']
        self.save_interim = config['save_interim']
        self.data_keys = config['data_keys']
        self.logger = logging.getLogger(__name__)
        self.config = config

    def process_group_data(self, data: np.ndarray, group: str) -> pd.DataFrame:
        """Process EEG data into DataFrame format."""
        try:
            self.logger.info(f"Processing {group} data with shape: {data.shape}")
            processed_frames = []
            
            for j in range(data.shape[0]):
                participant = data[j, 1][0][:4]  # Get participant ID
                
                # Process each window
                for k in range(data[j, 0].shape[1]):
                    data_dict = {
                        'participant_id': participant,
                        'window_id': k,
                        'group': group
                    }
                    
                    # Extract channel data
                    for i, channel in enumerate(self.channels):
                        data_dict[channel] = data[j, 0][0, k][:, i]
                    
                    processed_frames.append(pd.DataFrame([data_dict]))
            
            df = pd.concat(processed_frames, ignore_index=True)
            
            # Log processing results
            self.logger.info(f"Processed DataFrame shape for {group}: {df.shape}")
            self.logger.info(f"Number of participants: {len(df['participant_id'].unique())}")
            self.logger.info(f"Number of windows: {len(df['window_id'].unique())}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing {group} data: {str(e)}")
            raise

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and process EEG data from .mat file."""
        try:
            self.logger.info(f"Loading MATLAB file: {file_path}")
            mat_data = scipy.io.loadmat(file_path)
            
            # Log available keys
            self.logger.info(f"Available keys in .mat file: {list(mat_data.keys())}")
            
            # Process non-remission group
            self.logger.info("Processing non-remission group...")
            non_remission_df = self.process_group_data(
                mat_data[self.data_keys['non_remission']],
                'non_remission'
            )
            
            # Process remission group
            self.logger.info("Processing remission group...")
            remission_df = self.process_group_data(
                mat_data[self.data_keys['remission']],
                'remission'
            )
            
            # Combine datasets
            df = pd.concat([non_remission_df, remission_df], ignore_index=True)
            
            # Basic validation
            if self.validate_data:
                self._validate_data(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate the loaded DataFrame."""
        try:
            # Check for required columns
            required_columns = ['participant_id', 'window_id', 'group'] + self.channels
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Check for null values
            null_counts = df.isnull().sum()
            if null_counts.any():
                self.logger.warning(f"Found null values:\n{null_counts[null_counts > 0]}")
            
            # Check data types
            for channel in self.channels:
                if not all(isinstance(x, np.ndarray) for x in df[channel]):
                    raise ValueError(f"Invalid data type in channel {channel}")
            
            self.logger.info("Data validation completed successfully")
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            raise

def load_eeg_data(config: Dict) -> pd.DataFrame:
    """Load EEG data from file."""
    try:
        # Get data loader config
        data_loader_config = config.get('data_loader', {})
        
        # Initialize data loader with config
        loader = DataLoader(
            config=data_loader_config
        )
        
        # Load data
        raw_data_path = config.get('paths', {}).get('raw_data')
        if not raw_data_path:
            raise ValueError("Raw data path not specified in config")
            
        return loader.load_data(raw_data_path)
        
    except Exception as e:
        logging.error(f"Error loading EEG data: {str(e)}")
        raise