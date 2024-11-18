import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis, entropy
import nolds
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from ..utils.logger import get_logger
import mlflow
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = get_logger(__name__)

class EEGFeatureExtractor:
    """EEG signal feature extraction class."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the feature extractor with configuration."""
        # Get feature extraction configuration
        feature_config = config['feature_extractor']
        
        # Feature computation flags with defaults
        self.compute_spectral = feature_config.get('compute_spectral', True)
        self.compute_temporal = feature_config.get('compute_temporal', True)
        self.compute_entropy = feature_config.get('compute_entropy', True)
        self.compute_complexity = feature_config.get('compute_complexity', True)
        self.compute_connectivity = feature_config.get('compute_connectivity', False)
        
        # Get frequency bands
        self.frequency_bands = feature_config['frequency_bands']
        
        # Get feature lists with defaults
        self.statistical_features = feature_config.get('statistical_features', [
            'mean', 'std', 'variance', 'skewness', 'kurtosis', 
            'rms', 'zero_crossings', 'peak_to_peak', 'mean_abs_deviation'
        ])
        
        self.entropy_features = feature_config.get('entropy_features', [
            'sample_entropy', 'spectral_entropy'
        ])
        
        self.complexity_features = feature_config.get('complexity_features', [
            'hfd', 'correlation_dimension', 'hurst_exponent',
            'lyapunov_exponent', 'dfa'
        ])
        
        # Get validation settings with defaults
        validation_config = feature_config.get('validation', {})
        self.check_nan = validation_config.get('check_nan', True)
        self.check_infinite = validation_config.get('check_infinite', True)
        self.remove_invalid = validation_config.get('remove_invalid', False)
        
        # Get channels from data_loader config
        self.channels = config['data_loader']['channels']
        
        # Get sampling rate
        self.sampling_rate = config['window_slicer']['sampling_rate']
        
        # Save interim results flag
        self.save_interim = feature_config.get('save_interim', True)
        
        self.logger = get_logger(__name__)
        
        # Log configuration
        self.logger.info("Initialized EEG Feature Extractor:")
        self.logger.info(f"- Channels: {self.channels}")
        self.logger.info(f"- Frequency bands: {self.frequency_bands}")
        self.logger.info(f"- Statistical features: {self.statistical_features}")
        self.logger.info(f"- Entropy features: {self.entropy_features}")
        self.logger.info(f"- Complexity features: {self.complexity_features}")
    
    def validate_signal(self, signal: np.ndarray) -> Tuple[bool, str]:
        """
        Validate signal data.
        
        Args:
            signal: Input signal array
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            if not isinstance(signal, np.ndarray):
                return False, "Data is not a numpy array"
            
            if len(signal) == 0:
                return False, "Empty signal"
            
            if self.validation['check_nan'] and np.any(np.isnan(signal)):
                return False, "Signal contains NaN values"
            
            if self.validation['check_infinite'] and np.any(np.isinf(signal)):
                return False, "Signal contains infinite values"
            
            return True, "Validation passed"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def compute_band_power(self, signal: np.ndarray, band: Tuple[float, float]) -> float:
        """
        Compute power in specific frequency band.
        
        Args:
            signal: Input signal array
            band: Tuple of (low_freq, high_freq)
            
        Returns:
            Band power value
        """
        try:
            # Compute PSD using Welch's method
            freqs, psd = welch(signal, fs=self.sampling_rate, nperseg=min(256, len(signal)))
            
            # Find frequency band indices
            idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
            
            # Calculate band power
            band_power = np.trapz(psd[idx_band], freqs[idx_band])
            
            return band_power
            
        except Exception as e:
            logger.warning(f"Error computing band power: {str(e)}")
            return np.nan
    
    def compute_spectral_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Compute spectral features.
        
        Args:
            signal: Input signal array
            
        Returns:
            Dictionary of spectral features
        """
        features = {}
        
        try:
            # Compute band powers
            for band_name, band_range in self.frequency_bands.items():
                features[f'bp_{band_name}'] = self.compute_band_power(signal, band_range)
            
            # Compute total power
            features['total_power'] = sum(features.values())
            
            # Compute relative band powers
            for band_name in self.frequency_bands.keys():
                features[f'rbp_{band_name}'] = (
                    features[f'bp_{band_name}'] / features['total_power']
                    if features['total_power'] > 0 else np.nan
                )
            
            # Compute spectral entropy
            freqs, psd = welch(signal, fs=self.sampling_rate)
            psd_norm = psd / np.sum(psd)
            features['spectral_entropy'] = entropy(psd_norm)
            
            # Compute spectral edge frequency
            def find_spectral_edge(psd_curve: np.ndarray, edge_percent: float) -> float:
                cumsum = np.cumsum(psd_curve)
                return np.interp(edge_percent * cumsum[-1], cumsum, freqs)
            
            features['sef_90'] = find_spectral_edge(psd, 0.9)
            features['sef_95'] = find_spectral_edge(psd, 0.95)
            
        except Exception as e:
            logger.warning(f"Error computing spectral features: {str(e)}")
            features = {name: np.nan for name in [
                'total_power', 'spectral_entropy', 'sef_90', 'sef_95'
            ] + [f'bp_{band}' for band in self.frequency_bands.keys()]
              + [f'rbp_{band}' for band in self.frequency_bands.keys()]}
        
        return features
    
    def compute_temporal_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Compute temporal features.
        
        Args:
            signal: Input signal array
            
        Returns:
            Dictionary of temporal features
        """
        features = {}
        
        try:
            # Basic statistics
            features.update({
                'mean': np.mean(signal),
                'std': np.std(signal),
                'var': np.var(signal),
                'skew': skew(signal),
                'kurtosis': kurtosis(signal),
                'rms': np.sqrt(np.mean(np.square(signal))),
                'peak_to_peak': np.ptp(signal),
                'zero_crossings': np.sum(np.diff(np.signbit(signal))),
                'mean_abs_deviation': np.mean(np.abs(signal - np.mean(signal)))
            })
            
            # Hjorth parameters
            diff_signal = np.diff(signal)
            diff_diff_signal = np.diff(diff_signal)
            
            activity = np.var(signal)
            mobility = np.sqrt(np.var(diff_signal) / activity) if activity > 0 else np.nan
            complexity = (
                np.sqrt(np.var(diff_diff_signal) * activity) / 
                np.var(diff_signal) if np.var(diff_signal) > 0 else np.nan
            )
            
            features.update({
                'hjorth_activity': activity,
                'hjorth_mobility': mobility,
                'hjorth_complexity': complexity
            })
            
        except Exception as e:
            logger.warning(f"Error computing temporal features: {str(e)}")
            features = {name: np.nan for name in [
                'mean', 'std', 'var', 'skew', 'kurtosis', 'rms',
                'peak_to_peak', 'zero_crossings', 'mean_abs_deviation',
                'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity'
            ]}
        
        return features
    
    def compute_complexity_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Compute complexity features.
        
        Args:
            signal: Input signal array
            
        Returns:
            Dictionary of complexity features
        """
        features = {}
        
        try:
            # Prepare signal
            if len(signal) > 1000:  # Subsample for computational efficiency
                signal = signal[::2]
            
            # Sample entropy
            features['sample_entropy'] = nolds.sampen(signal)
            
            # Correlation dimension
            features['correlation_dim'] = nolds.corr_dim(signal, emb_dim=5)
            
            # Hurst exponent
            features['hurst'] = nolds.hurst_rs(signal)
            
            # Lyapunov exponent
            features['lyapunov'] = nolds.lyap_r(signal)
            
            # Detrended Fluctuation Analysis
            features['dfa'] = nolds.dfa(signal)
            
        except Exception as e:
            logger.warning(f"Error computing complexity features: {str(e)}")
            features = {name: np.nan for name in [
                'sample_entropy', 'correlation_dim', 'hurst',
                'lyapunov', 'dfa'
            ]}
        
        return features
    
    def compute_connectivity_features(self, signals: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute connectivity features between channels.
        
        Args:
            signals: Dictionary of channel signals
            
        Returns:
            Dictionary of connectivity features
        """
        features = {}
        
        try:
            # Compute cross-correlation between channel pairs
            for i, ch1 in enumerate(self.channels):
                for j, ch2 in enumerate(self.channels[i+1:], i+1):
                    xcorr = np.correlate(signals[ch1], signals[ch2], mode='full')
                    max_xcorr = np.max(np.abs(xcorr))
                    features[f'xcorr_{ch1}_{ch2}'] = max_xcorr
            
        except Exception as e:
            logger.warning(f"Error computing connectivity features: {str(e)}")
            features = {
                f'xcorr_{ch1}_{ch2}': np.nan
                for i, ch1 in enumerate(self.channels)
                for j, ch2 in enumerate(self.channels[i+1:], i+1)
            }
        
        return features
    
    def extract_features(self, window_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract all features from a single window.
        
        Args:
            window_data: Dictionary containing window data
            
        Returns:
            Dictionary of features with proper column structure
        """
        try:
            features = {}
            
            # Preserve metadata columns
            metadata_cols = ['participant_id', 'group', 'window_start', 'window_end']
            for col in metadata_cols:
                if col in window_data:
                    features[col] = window_data[col]
            
            # Debug log
            self.logger.debug(f"Window data keys: {window_data.keys()}")
            self.logger.debug(f"Configured channels: {self.channels}")
            
            # Extract features for each channel
            for channel in self.channels:
                # Debug log
                self.logger.debug(f"Processing channel: {channel}")
                
                # Get signal data
                signal_key = 'data' if channel == window_data.get('channel') else channel
                if signal_key not in window_data:
                    #self.logger.warning(f"Missing signal data for channel {channel}")
                    continue
                
                signal = np.array(window_data[signal_key])
                if len(signal) == 0:
                    self.logger.warning(f"Empty signal for channel {channel}")
                    continue
                
                # Spectral features
                if self.compute_spectral:
                    for band_name, (low, high) in self.frequency_bands.items():
                        power = self.compute_band_power(signal, (low, high))
                        features[f"{channel}_power_{band_name}"] = power
                        self.logger.debug(f"Computed {channel}_power_{band_name}: {power}")
                
                # Temporal features
                if self.compute_temporal:
                    features.update({
                        f"{channel}_mean": np.mean(signal),
                        f"{channel}_std": np.std(signal),
                        f"{channel}_var": np.var(signal),
                        f"{channel}_skew": skew(signal),
                        f"{channel}_kurtosis": kurtosis(signal),
                        f"{channel}_rms": np.sqrt(np.mean(np.square(signal))),
                        f"{channel}_peak_to_peak": np.ptp(signal),
                        f"{channel}_zero_crossings": np.sum(np.diff(np.signbit(signal)))
                    })
                
                # Entropy features
                if self.compute_entropy:
                    features[f"{channel}_sample_entropy"] = nolds.sampen(signal)
                    freqs, psd = welch(signal, fs=self.sampling_rate)
                    psd_norm = psd / np.sum(psd)
                    features[f"{channel}_spectral_entropy"] = entropy(psd_norm)
                
                # Log feature count for debugging
                self.logger.debug(f"Extracted {len(features)} features for {channel}")
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            raise
    
    def extract_all_features(self, df: pd.DataFrame, output_path: Optional[str] = None) -> pd.DataFrame:
        """Extract features from all windows."""
        try:
            logger.info("Starting feature extraction")
            start_time = time.time()
            
            # Process windows in parallel
            all_features = []
            total_windows = len(df)
            
            # Debug first window
            first_window = df.iloc[0].to_dict()
            self.logger.debug(f"First window data: {first_window.keys()}")
            
            with ThreadPoolExecutor() as executor:
                future_to_window = {
                    executor.submit(self.extract_features, row.to_dict()): idx 
                    for idx, row in df.iterrows()
                }
                
                for future in as_completed(future_to_window):
                    idx = future_to_window[future]
                    if idx % 100 == 0:
                        logger.info(f"Processed window {idx+1}/{total_windows}")
                    
                    try:
                        features = future.result()
                        all_features.append(features)
                    except Exception as e:
                        logger.error(f"Error processing window {idx}: {str(e)}")
            
            # Create output DataFrame
            result_df = pd.DataFrame(all_features)
            
            # Debug column names
            self.logger.info(f"Extracted features columns: {result_df.columns.tolist()}")
            
            # Save features if output path provided
            if output_path:
                self._save_features(result_df, output_path)
            
            processing_time = time.time() - start_time
            logger.info(f"Feature extraction completed in {processing_time:.2f} seconds")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error in feature extraction: {str(e)}")
            raise
    
    def _log_statistics(self, df: pd.DataFrame) -> None:
        """
        Log feature extraction statistics to MLflow.
        
        Args:
            df: DataFrame with extracted features
        """
        # Log basic counts
        feature_cols = df.columns.difference(['Participant', 'Remission'])
        mlflow.log_metrics({
            "total_features": len(feature_cols),
            "total_windows": len(df),
            "features_per_window": len(feature_cols)
        })
        
        # Log feature statistics by type
        feature_types = {
            'spectral': ['bp_', 'rbp_', 'spectral_entropy', 'sef_'],
            'temporal': ['mean', 'std', 'var', 'skew', 'hjorth_'],
            'complexity': ['sample_entropy', 'correlation_dim', 'hurst', 'lyapunov', 'dfa'],
            'connectivity': ['xcorr_']
        }
        
        for ftype, patterns in feature_types.items():
            type_features = [col for col in feature_cols 
                           if any(pat in col for pat in patterns)]
            mlflow.log_metric(f"{ftype}_features", len(type_features))
        
        # Log missing value statistics
        for col in feature_cols:
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            if missing_pct > 0:
                mlflow.log_metric(f"{col}_missing_pct", missing_pct)
    
    def _save_features(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save the extracted features to files.
        
        Args:
            df: DataFrame containing features
            output_path: Path to save the features
        """
        try:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the complete feature set
            df.to_parquet(output_path)
            self.logger.info(f"Saved features to {output_path}")
            
            # Optionally save group-specific features if group column exists
            if self.save_interim and 'group' in df.columns:
                for group in df['group'].unique():
                    group_df = df[df['group'] == group]
                    group_path = output_dir / f"{group.lower()}_features.parquet"
                    group_df.to_parquet(group_path)
                    self.logger.info(f"Saved {group} features to {group_path}")
                    
        except Exception as e:
            self.logger.error(f"Error saving features: {str(e)}")
            raise
    
    def _create_feature_documentation(self, df: pd.DataFrame) -> str:
        """
        Create markdown documentation of extracted features.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Markdown string documenting features
        """
        feature_cols = df.columns.difference(['Participant', 'Remission'])
        
        docs = ["# EEG Feature Documentation\n\n"]
        
        # Feature counts
        docs.append("## Feature Summary\n")
        docs.append(f"- Total number of features: {len(feature_cols)}\n")
        docs.append(f"- Number of windows: {len(df)}\n")
        docs.append(f"- Number of participants: {df['Participant'].nunique()}\n\n")
        
        # Feature categories
        feature_categories = {
            'Spectral Features': {
                'prefix': ['bp_', 'rbp_'],
                'description': 'Features related to frequency domain analysis'
            },
            'Temporal Features': {
                'prefix': ['mean', 'std', 'var', 'skew', 'hjorth_'],
                'description': 'Time domain statistical features'
            },
            'Complexity Features': {
                'prefix': ['sample_entropy', 'correlation_dim', 'hurst', 'lyapunov', 'dfa'],
                'description': 'Nonlinear dynamics and complexity measures'
            },
            'Connectivity Features': {
                'prefix': ['xcorr_'],
                'description': 'Inter-channel connectivity measures'
            }
        }
        
        for category, info in feature_categories.items():
            category_features = [col for col in feature_cols 
                               if any(p in col for p in info['prefix'])]
            if category_features:
                docs.append(f"## {category}\n")
                docs.append(f"{info['description']}\n\n")
                docs.append("| Feature | Description | Statistics |\n")
                docs.append("|---------|-------------|------------|\n")
                
                for feature in category_features:
                    stats = df[feature].describe()
                    stats_str = f"Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}"
                    docs.append(f"| {feature} | | {stats_str} |\n")
                
                docs.append("\n")
        
        # Data quality
        docs.append("## Data Quality\n")
        docs.append("| Feature | Missing Values (%) | Zero Values (%) |\n")
        docs.append("|---------|-------------------|-----------------|n")
        
        for col in feature_cols:
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            zero_pct = (df[col] == 0).sum() / len(df) * 100
            docs.append(f"| {col} | {missing_pct:.2f} | {zero_pct:.2f} |\n")
        
        return "".join(docs)


def extract_eeg_features(config: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from EEG data.
    
    Args:
        config: Configuration dictionary
        df: Input DataFrame
        
    Returns:
        DataFrame with extracted features
    """
    extractor = EEGFeatureExtractor(config)
    
    # Get the correct output path from config
    output_path = config['paths']['features']['window']
    
    return extractor.extract_all_features(
        df,
        output_path=output_path
    )


if __name__ == '__main__':
    # Example usage
    import yaml
    import mlflow
    
    # Load configuration
    with open('configs/processing_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Start MLflow run
    with mlflow.start_run(run_name="feature_extraction"):
        try:
            # Load windowed data
            input_df = pd.read_parquet(config['paths']['interim']['windowed'])
            
            # Extract features
            features_df = extract_eeg_features(config, input_df)
            
            # Log success
            mlflow.log_metric("feature_extraction_success", 1)
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            mlflow.log_metric("feature_extraction_success", 0)
            raise