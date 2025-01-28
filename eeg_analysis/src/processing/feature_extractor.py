import cupy as cp
import numpy as np
import pandas as pd
import dask_cudf
import mlflow
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import yaml
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import scipy.signal
from scipy.stats import entropy
import nolds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUEEGFeatureExtractor:
    """GPU-accelerated EEG signal feature extraction class."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config['feature_extractor']
        self.sampling_rate = config['window_slicer']['sampling_rate']
        self.channels = config['data_loader']['channels']
        self.frequency_bands = self.config['frequency_bands']
        
        # Initialize GPU cluster
        self.cluster = LocalCUDACluster()
        self.client = Client(self.cluster)
        logger.info(f"Dask dashboard available at: {self.client.dashboard_link}")
        
    def __del__(self):
        """Cleanup GPU resources."""
        if hasattr(self, 'client'):
            self.client.close()
        if hasattr(self, 'cluster'):
            self.cluster.close()

    def compute_band_power(self, signal: cp.ndarray, band: Tuple[float, float]) -> float:
        """Calculate power in a specific frequency band using GPU."""
        try:
            # Convert to CPU for welch
            signal_cpu = cp.asnumpy(signal)
            freqs, psd = scipy.signal.welch(signal_cpu, fs=self.sampling_rate, 
                                          nperseg=min(256, len(signal_cpu)))
            
            # Convert back to GPU for further processing
            freqs_gpu = cp.array(freqs)
            psd_gpu = cp.array(psd)
            
            idx_band = cp.logical_and(freqs_gpu >= band[0], freqs_gpu <= band[1])
            band_power = float(cp.trapz(psd_gpu[idx_band], freqs_gpu[idx_band]))
            
            return band_power if cp.isfinite(band_power) else float('nan')
            
        except Exception as e:
            logger.warning(f"Error computing band power: {str(e)}")
            return float('nan')

    def compute_entropy_features(self, signal: cp.ndarray) -> Dict[str, float]:
        """Compute entropy-based features."""
        try:
            features = {}
            # Convert to CPU
            signal_cpu = cp.asnumpy(signal)
            signal_cpu = np.array(signal_cpu, dtype=np.float64)
            
            # Sample entropy (CPU-based)
            try:
                features['sample_entropy'] = nolds.sampen(signal_cpu)
            except Exception as e:
                logger.warning(f"Sample entropy calculation failed: {str(e)}")
                features['sample_entropy'] = float('nan')
            
            # Spectral entropy
            try:
                freqs, psd = scipy.signal.welch(signal_cpu, fs=self.sampling_rate,
                                              nperseg=min(len(signal_cpu), 256))
                psd_norm = psd / np.sum(psd)
                features['spectral_entropy'] = float(entropy(psd_norm))
            except Exception as e:
                logger.warning(f"Spectral entropy calculation failed: {str(e)}")
                features['spectral_entropy'] = float('nan')
            
            return features
            
        except Exception as e:
            logger.warning(f"Error computing entropy features: {str(e)}")
            return {k: float('nan') for k in ['sample_entropy', 'spectral_entropy']}

    def hfd(self, signal: np.ndarray, Kmax: int = 10) -> float:
        """Compute Higuchi Fractal Dimension."""
        try:
            N = len(signal)
            L = []
            x = []
            
            for k in range(1, Kmax + 1):
                Lk = []
                
                for m in range(k):
                    indices = np.arange(m, N-k, k)
                    Lmk = np.mean(np.abs(signal[indices + k] - signal[indices]))
                    Lmk = Lmk * (N - 1) / (((N - m) // k) * k)
                    Lk.append(Lmk)
                
                L.append(np.mean(Lk))
                x.append([np.log(1.0 / k), np.log(np.mean(Lk))])
            
            x = np.array(x)
            slope, _ = np.polyfit(x[:, 0], x[:, 1], 1)
            
            return -slope if np.isfinite(slope) else float('nan')
            
        except Exception as e:
            logger.warning(f"Error computing HFD: {str(e)}")
            return float('nan')

    def prepare_for_lyap(self, signal: np.ndarray) -> np.ndarray:
        """Prepare signal for Lyapunov exponent calculation."""
        # Normalize
        signal = signal - np.mean(signal)
        if np.std(signal) > 0:
            signal = signal / np.std(signal)
        
        # Ensure correct type and memory layout
        signal = np.asarray(signal, dtype=np.float64, order='C')
        return signal

    def compute_complexity_measures(self, signal: cp.ndarray) -> Dict[str, float]:
        """Compute complexity measures."""
        try:
            # Convert to CPU and prepare signal
            signal_cpu = cp.asnumpy(signal)
            signal_cpu = np.asarray(signal_cpu, dtype=np.float64, order='C')
            
            if len(signal_cpu) < 100:
                logger.warning("Signal too short for complexity calculation")
                return {k: float('nan') for k in ['hfd', 'correlation_dim', 'hurst', 'lyap_r', 'dfa']}
            
            features = {}
            
            # Higuchi Fractal Dimension
            try:
                features['hfd'] = self.hfd(signal_cpu)
            except Exception as e:
                logger.warning(f"HFD calculation failed: {str(e)}")
                features['hfd'] = float('nan')
            
            # Correlation Dimension
            try:
                features['correlation_dim'] = nolds.corr_dim(signal_cpu, emb_dim=10)
            except Exception as e:
                logger.warning(f"Correlation dimension calculation failed: {str(e)}")
                features['correlation_dim'] = float('nan')
            
            # Hurst Exponent
            try:
                features['hurst'] = nolds.hurst_rs(signal_cpu)
            except Exception as e:
                logger.warning(f"Hurst exponent calculation failed: {str(e)}")
                features['hurst'] = float('nan')
            
            # Largest Lyapunov Exponent
            try:
                # Normalize and prepare data
                data_norm = (signal_cpu - np.mean(signal_cpu)) / (np.std(signal_cpu) + 1e-10)
                data_norm = np.ascontiguousarray(data_norm, dtype=np.float64)
            
                # Calculate embedding parameters
                emb_dim = 10
                lag = max(1, int(len(data_norm) // 20))  # Use integer division
            
                # Ensure minimum data length
                min_length = (emb_dim - 1) * lag + 1
                if len(data_norm) < min_length:
                    logger.warning(f"Data length {len(data_norm)} insufficient for lyap_r calculation")
                    features['lyap_r'] = float('nan')
                else:
                    features['lyap_r'] = nolds.lyap_r(data_norm, emb_dim=emb_dim, lag=lag, min_tsep=lag)
                
                if not np.isfinite(features['lyap_r']):
                    logger.warning(f"Computed lyap_r is not finite: {features['lyap_r']}")
                    features['lyap_r'] = float('nan')
            except Exception as e:
                logger.warning(f"Lyapunov exponent calculation failed: {str(e)}")
                features['lyap_r'] = float('nan')
            
            # Detrended Fluctuation Analysis
            try:
                features['dfa'] = nolds.dfa(signal_cpu)
            except Exception as e:
                logger.warning(f"DFA calculation failed: {str(e)}")
                features['dfa'] = float('nan')
                
            return features
            
        except Exception as e:
            logger.warning(f"Error computing complexity measures: {str(e)}")
            return {k: float('nan') for k in ['hfd', 'correlation_dim', 'hurst', 'lyap_r', 'dfa']}

    def compute_features(self, signal_gpu: cp.ndarray) -> Dict[str, float]:
        """Compute all features for a signal."""
        features = {}
        
        # Calculate band powers
        for band_name, band_range in self.frequency_bands.items():
            features[f'bp_{band_name}'] = self.compute_band_power(signal_gpu, band_range)
        
        # Add entropy features
        features.update(self.compute_entropy_features(signal_gpu))
        
        # Add complexity measures
        features.update(self.compute_complexity_measures(signal_gpu))
        
        return features

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from all channels using GPU acceleration."""
        all_features = []
        total_windows = len(df)
        
        logger.info(f"Input DataFrame columns: {df.columns.tolist()}")
        logger.info(f"Processing {total_windows} windows")
        
        for index, row in df.iterrows():
            if index % 100 == 0:
                logger.info(f"Processing window {index+1}/{total_windows}")
            
            features = {
                'window_start': row.get('window_start', 'unknown'),
                'window_end': row.get('window_end', 'unknown'),
                'group': row.get('group', 'unknown'),
                'Participant': row.get('participant_id', 'unknown')
            }
            
            try:
                for channel in self.channels:
                    if channel not in row:
                        logger.warning(f"Channel {channel} not found in data")
                        continue
                    
                    channel_data = cp.array(row[channel], dtype=cp.float64)
                    channel_features = self.compute_features(channel_data)
                    
                    for feature_name, value in channel_features.items():
                        features[f'{channel}_{feature_name}'] = value
                
                all_features.append(features)
                
            except Exception as e:
                logger.error(f"Error processing window {index}: {str(e)}")
                continue
        
        if not all_features:
            raise ValueError("No features were successfully extracted")
        
        result_df = pd.DataFrame(all_features)
        
        # Map group values to binary for modeling
        group_mapping = {
            'remission': 1,
            'non_remission': 0
        }
        if 'group' in result_df.columns:
            result_df['Remission'] = result_df['group'].map(group_mapping)
        result_df = result_df.drop(columns=['group'])
        
        return result_df

def run_feature_extraction(config: Dict[str, Any], df: pd.DataFrame = None) -> pd.DataFrame:
    """Run GPU-accelerated feature extraction pipeline."""
    try:
        # Initialize feature extractor
        extractor = GPUEEGFeatureExtractor(config)
        
        # Extract features
        features_df = extractor.extract_features(df)
        
        # Save features if output path provided
        output_path = Path(config['paths']['features']['window'])
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / "features.parquet"
            features_df.to_parquet(output_file)
            logger.info(f"Saved features to {output_file}")
            
        return features_df
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    with open(sys.argv[1], 'r') as f:
        config = yaml.safe_load(f)
    with mlflow.start_run(run_name=config['mlflow']['run_name']):
        run_feature_extraction(config)