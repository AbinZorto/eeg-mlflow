import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis, entropy
import nolds
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from ..utils.logger import get_logger
import mlflow
import mlflow.data
from mlflow.data.pandas_dataset import PandasDataset
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

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
        self.logger.info(f"- Channels: {self.channels} ({len(self.channels)} channels)")
        self.logger.info(f"- Frequency bands: {self.frequency_bands}")
        self.logger.info(f"- Statistical features: {self.statistical_features}")
        self.logger.info(f"- Entropy features: {self.entropy_features}")
        self.logger.info(f"- Complexity features: {self.complexity_features}")
        
        # Log asymmetry and cross-hemispheric feature availability
        can_compute_frontal = 'af7' in self.channels and 'af8' in self.channels
        can_compute_temporal = 'tp9' in self.channels and 'tp10' in self.channels
        can_compute_hemispheric = all(ch in self.channels for ch in ['af7', 'af8', 'tp9', 'tp10'])
        can_compute_left_regional = 'af7' in self.channels and 'tp9' in self.channels
        can_compute_right_regional = 'af8' in self.channels and 'tp10' in self.channels
        can_compute_diagonal = ('af7' in self.channels and 'tp10' in self.channels) or ('af8' in self.channels and 'tp9' in self.channels)
        
        self.logger.info(f"- Frontal asymmetry features: {'Available' if can_compute_frontal else 'Not available (requires af7 and af8)'}")
        self.logger.info(f"- Temporal asymmetry features: {'Available' if can_compute_temporal else 'Not available (requires tp9 and tp10)'}")
        self.logger.info(f"- Hemispheric asymmetry features: {'Available' if can_compute_hemispheric else 'Not available (requires all 4 channels)'}")
        self.logger.info(f"- Cross-regional features: {'Available' if can_compute_left_regional or can_compute_right_regional else 'Not available'}")
        self.logger.info(f"- Diagonal cross-hemispheric features: {'Available' if can_compute_diagonal else 'Not available'}")
        self.logger.info(f"- Cross-channel coherence features: {'Available' if len(self.channels) >= 2 else 'Not available (requires at least 2 channels)'}")
        self.logger.info("- Computation optimization: PSD and band powers cached to avoid redundancy")
        self.logger.info("- New features added: relative band powers, PSD statistics, Hjorth parameters, and all complexity features per channel")
    
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
    
    def compute_band_power(self, signal: np.ndarray = None, band: Tuple[float, float] = None, 
                          freqs: np.ndarray = None, psd: np.ndarray = None) -> float:
        """
        Compute power in specific frequency band.
        
        Args:
            signal: Input signal array (if freqs/psd not provided)
            band: Tuple of (low_freq, high_freq)
            freqs: Pre-computed frequency array (optional, for efficiency)
            psd: Pre-computed PSD array (optional, for efficiency)
            
        Returns:
            Band power value
        """
        try:
            # Use pre-computed PSD if available, otherwise compute it
            if freqs is not None and psd is not None:
                freqs_use, psd_use = freqs, psd
            elif signal is not None:
                freqs_use, psd_use = welch(signal, fs=self.sampling_rate, nperseg=min(256, len(signal)))
            else:
                raise ValueError("Either signal or (freqs, psd) must be provided")
            
            # Find frequency band indices
            idx_band = np.logical_and(freqs_use >= band[0], freqs_use <= band[1])
            
            # Calculate band power
            band_power = np.trapz(psd_use[idx_band], freqs_use[idx_band])
            
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
            
            # Compute frequency-weighted power
            freqs, psd = welch(signal, fs=self.sampling_rate)
            features['freq_weighted_power'] = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else np.nan
            
            # Compute spectral entropy
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
                'total_power', 'spectral_entropy', 'sef_90', 'sef_95', 'freq_weighted_power'
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
        
        This method is optimized to avoid redundant computations:
        - PSD (Power Spectral Density) is computed once per channel and cached
        - Band powers are computed once per channel and cached for reuse
        - All asymmetry and cross-hemispheric features use cached band powers
        
        Args:
            window_data: Dictionary containing window data
            
        Returns:
            Dictionary of features with proper column structure
        """
        try:
            features = {}
            
            # Update metadata columns mapping
            metadata_mapping = {
                'participant_id': 'Participant',
                'group': 'Remission',
                'window_start': 'window_start',
                'window_end': 'window_end'
            }
            
            # Preserve and rename metadata columns
            for old_col, new_col in metadata_mapping.items():
                if old_col in window_data:
                    # Convert group to binary Remission
                    if old_col == 'group':
                        features[new_col] = 1 if window_data[old_col].lower() == 'remission' else 0
                    else:
                        features[new_col] = window_data[old_col]
            
            # Extract features for each channel and cache computations
            channel_signals = {}
            channel_band_powers = {}  # Cache band power calculations
            channel_psd_data = {}    # Cache PSD calculations
            
            for channel in self.channels:
                # Get signal data
                signal_key = 'data' if channel == window_data.get('channel') else channel
                if signal_key not in window_data:
                    continue
                
                signal = np.array(window_data[signal_key])
                if len(signal) == 0:
                    self.logger.warning(f"Empty signal for channel {channel}")
                    continue
                
                # Store signal for later asymmetry calculations
                channel_signals[channel] = signal
                
                # Compute PSD once per channel (used by multiple features)
                freqs, psd = welch(signal, fs=self.sampling_rate, nperseg=min(256, len(signal)))
                channel_psd_data[channel] = (freqs, psd)
                
                # Compute and cache band powers once per channel using cached PSD
                band_powers = {}
                total_power = 0
                for band_name, (low, high) in self.frequency_bands.items():
                    power = self.compute_band_power(band=(low, high), freqs=freqs, psd=psd)
                    band_powers[band_name] = power
                    total_power += power
                channel_band_powers[channel] = band_powers
                
                # === SPECTRAL FEATURES ===
                if self.compute_spectral:
                    # Band powers
                    for band_name, power in band_powers.items():
                        features[f"{channel}_power_{band_name}"] = power
                    
                    # Total power
                    features[f"{channel}_total_power"] = total_power
                    
                    # Relative band powers (missing feature!)
                    for band_name, power in band_powers.items():
                        features[f"{channel}_relative_power_{band_name}"] = (
                            power / total_power if total_power > 0 else np.nan
                        )
                    
                    # Frequency-weighted power (using cached PSD)
                    features[f"{channel}_freq_weighted_power"] = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else np.nan
                    
                    # Spectral edge frequencies (missing features!)
                    def find_spectral_edge(psd_curve: np.ndarray, edge_percent: float) -> float:
                        cumsum = np.cumsum(psd_curve)
                        return np.interp(edge_percent * cumsum[-1], cumsum, freqs)
                    
                    features[f"{channel}_sef_90"] = find_spectral_edge(psd, 0.9)
                    features[f"{channel}_sef_95"] = find_spectral_edge(psd, 0.95)
                    
                    # PSD summary statistics (missing features!)
                    features[f"{channel}_psd_mean"] = np.mean(psd)
                    features[f"{channel}_psd_std"] = np.std(psd)
                    features[f"{channel}_psd_max"] = np.max(psd)
                    features[f"{channel}_psd_min"] = np.min(psd)
                    features[f"{channel}_psd_median"] = np.median(psd)
                    features[f"{channel}_psd_skewness"] = skew(psd)
                    features[f"{channel}_psd_kurtosis"] = kurtosis(psd)
                
                # === TEMPORAL FEATURES ===
                if self.compute_temporal:
                    # Basic temporal statistics
                    features.update({
                        f"{channel}_mean": np.mean(signal),
                        f"{channel}_std": np.std(signal),
                        f"{channel}_var": np.var(signal),
                        f"{channel}_skew": skew(signal),
                        f"{channel}_kurtosis": kurtosis(signal),
                        f"{channel}_rms": np.sqrt(np.mean(np.square(signal))),
                        f"{channel}_peak_to_peak": np.ptp(signal),
                        f"{channel}_zero_crossings": np.sum(np.diff(np.signbit(signal))),
                        f"{channel}_mean_abs_deviation": np.mean(np.abs(signal - np.mean(signal)))
                    })
                    
                    # Hjorth parameters (missing features!)
                    diff_signal = np.diff(signal)
                    diff_diff_signal = np.diff(diff_signal)
                    
                    activity = np.var(signal)
                    mobility = np.sqrt(np.var(diff_signal) / activity) if activity > 0 else np.nan
                    complexity = (
                        np.sqrt(np.var(diff_diff_signal) * activity) / 
                        np.var(diff_signal) if np.var(diff_signal) > 0 else np.nan
                    )
                    
                    features.update({
                        f"{channel}_hjorth_activity": activity,
                        f"{channel}_hjorth_mobility": mobility,
                        f"{channel}_hjorth_complexity": complexity
                    })
                
                # === ENTROPY FEATURES ===
                if self.compute_entropy:
                    # Sample entropy (already computed correctly)
                    features[f"{channel}_sample_entropy"] = nolds.sampen(signal)
                    
                    # Spectral entropy (using cached PSD)
                    psd_norm = psd / np.sum(psd)
                    features[f"{channel}_spectral_entropy"] = entropy(psd_norm)
                
                # === COMPLEXITY FEATURES (MISSING!) ===
                if self.compute_complexity:
                    # Prepare signal for complexity analysis
                    complexity_signal = signal.copy()
                    if len(complexity_signal) > 1000:  # Subsample for computational efficiency
                        complexity_signal = complexity_signal[::2]
                    
                    try:
                        # Correlation dimension
                        features[f"{channel}_correlation_dim"] = nolds.corr_dim(complexity_signal, emb_dim=5)
                    except Exception as e:
                        self.logger.warning(f"Error computing correlation dimension for {channel}: {str(e)}")
                        features[f"{channel}_correlation_dim"] = np.nan
                    
                    try:
                        # Hurst exponent
                        features[f"{channel}_hurst"] = nolds.hurst_rs(complexity_signal)
                    except Exception as e:
                        self.logger.warning(f"Error computing Hurst exponent for {channel}: {str(e)}")
                        features[f"{channel}_hurst"] = np.nan
                    
                    try:
                        # Lyapunov exponent
                        features[f"{channel}_lyapunov"] = nolds.lyap_r(complexity_signal)
                    except Exception as e:
                        self.logger.warning(f"Error computing Lyapunov exponent for {channel}: {str(e)}")
                        features[f"{channel}_lyapunov"] = np.nan
                    
                    try:
                        # Detrended Fluctuation Analysis
                        features[f"{channel}_dfa"] = nolds.dfa(complexity_signal)
                    except Exception as e:
                        self.logger.warning(f"Error computing DFA for {channel}: {str(e)}")
                        features[f"{channel}_dfa"] = np.nan
            
            # === CONNECTIVITY FEATURES ===
            if self.compute_connectivity and len(channel_signals) >= 2:
                # Compute cross-correlation between all channel pairs
                for i, ch1 in enumerate(self.channels):
                    for j, ch2 in enumerate(self.channels[i+1:], i+1):
                        if ch1 in channel_signals and ch2 in channel_signals:
                            try:
                                xcorr = np.correlate(channel_signals[ch1], channel_signals[ch2], mode='full')
                                max_xcorr = np.max(np.abs(xcorr))
                                features[f'xcorr_{ch1}_{ch2}'] = max_xcorr
                            except Exception as e:
                                self.logger.warning(f"Error computing cross-correlation for {ch1}-{ch2}: {str(e)}")
                                features[f'xcorr_{ch1}_{ch2}'] = np.nan
            
            # Compute asymmetry features using cached band powers
            # Frontal asymmetry (AF7/AF8)
            if 'af7' in channel_band_powers and 'af8' in channel_band_powers:
                for band_name in self.frequency_bands.keys():
                    left_power = channel_band_powers['af7'][band_name]
                    right_power = channel_band_powers['af8'][band_name]
                    
                    if left_power > 0 and right_power > 0:
                        asym_index = np.log(right_power / left_power)
                        features[f"frontal_asymmetry_{band_name}"] = asym_index

                        asym_ratio = (right_power - left_power) / (right_power + left_power)
                        features[f"frontal_asymmetry_ratio_{band_name}"] = asym_ratio
            
            # Temporal asymmetry (TP9/TP10)
            if 'tp9' in channel_band_powers and 'tp10' in channel_band_powers:
                for band_name in self.frequency_bands.keys():
                    left_power = channel_band_powers['tp9'][band_name]
                    right_power = channel_band_powers['tp10'][band_name]
                    
                    if left_power > 0 and right_power > 0:
                        asym_index = np.log(right_power / left_power)
                        features[f"temporal_asymmetry_{band_name}"] = asym_index

                        asym_ratio = (right_power - left_power) / (right_power + left_power)
                        features[f"temporal_asymmetry_ratio_{band_name}"] = asym_ratio
            
            # Cross-hemispheric features using cached band powers
            # Left hemisphere (AF7/TP9) vs Right hemisphere (AF8/TP10)
            if all(ch in channel_band_powers for ch in ['af7', 'af8', 'tp9', 'tp10']):
                for band_name in self.frequency_bands.keys():
                    # Left hemisphere: average of AF7 and TP9 (using cached powers)
                    af7_power = channel_band_powers['af7'][band_name]
                    tp9_power = channel_band_powers['tp9'][band_name]
                    left_hemisphere_power = (af7_power + tp9_power) / 2
                    
                    # Right hemisphere: average of AF8 and TP10 (using cached powers)
                    af8_power = channel_band_powers['af8'][band_name]
                    tp10_power = channel_band_powers['tp10'][band_name]
                    right_hemisphere_power = (af8_power + tp10_power) / 2
                    
                    if left_hemisphere_power > 0 and right_hemisphere_power > 0:
                        # Hemispheric asymmetry
                        hemispheric_asym = np.log(right_hemisphere_power / left_hemisphere_power)
                        features[f"hemispheric_asymmetry_{band_name}"] = hemispheric_asym
                        
                        hemispheric_ratio = (right_hemisphere_power - left_hemisphere_power) / (right_hemisphere_power + left_hemisphere_power)
                        features[f"hemispheric_asymmetry_ratio_{band_name}"] = hemispheric_ratio
            
            # Cross-regional features using cached band powers (Frontal-Temporal interactions)
            # AF7-TP9 (Left side) vs AF8-TP10 (Right side)
            if 'af7' in channel_band_powers and 'tp9' in channel_band_powers:
                for band_name in self.frequency_bands.keys():
                    af7_power = channel_band_powers['af7'][band_name]
                    tp9_power = channel_band_powers['tp9'][band_name]
                    
                    if af7_power > 0 and tp9_power > 0:
                        # Frontal-temporal ratio on left side
                        left_ft_ratio = af7_power / tp9_power
                        features[f"left_frontal_temporal_ratio_{band_name}"] = left_ft_ratio
                        
                        # Frontal-temporal difference on left side
                        left_ft_diff = (af7_power - tp9_power) / (af7_power + tp9_power)
                        features[f"left_frontal_temporal_diff_{band_name}"] = left_ft_diff
            
            if 'af8' in channel_band_powers and 'tp10' in channel_band_powers:
                for band_name in self.frequency_bands.keys():
                    af8_power = channel_band_powers['af8'][band_name]
                    tp10_power = channel_band_powers['tp10'][band_name]
                    
                    if af8_power > 0 and tp10_power > 0:
                        # Frontal-temporal ratio on right side
                        right_ft_ratio = af8_power / tp10_power
                        features[f"right_frontal_temporal_ratio_{band_name}"] = right_ft_ratio
                        
                        # Frontal-temporal difference on right side
                        right_ft_diff = (af8_power - tp10_power) / (af8_power + tp10_power)
                        features[f"right_frontal_temporal_diff_{band_name}"] = right_ft_diff
            
            # Diagonal cross-hemispheric features using cached band powers
            # AF7-TP10 and AF8-TP9 (cross-hemisphere, cross-region)
            if 'af7' in channel_band_powers and 'tp10' in channel_band_powers:
                for band_name in self.frequency_bands.keys():
                    af7_power = channel_band_powers['af7'][band_name]
                    tp10_power = channel_band_powers['tp10'][band_name]
                    
                    if af7_power > 0 and tp10_power > 0:
                        # Left frontal to right temporal
                        af7_tp10_ratio = af7_power / tp10_power
                        features[f"af7_tp10_ratio_{band_name}"] = af7_tp10_ratio
                        
                        af7_tp10_diff = (af7_power - tp10_power) / (af7_power + tp10_power)
                        features[f"af7_tp10_diff_{band_name}"] = af7_tp10_diff
            
            if 'af8' in channel_band_powers and 'tp9' in channel_band_powers:
                for band_name in self.frequency_bands.keys():
                    af8_power = channel_band_powers['af8'][band_name]
                    tp9_power = channel_band_powers['tp9'][band_name]
                    
                    if af8_power > 0 and tp9_power > 0:
                        # Right frontal to left temporal
                        af8_tp9_ratio = af8_power / tp9_power
                        features[f"af8_tp9_ratio_{band_name}"] = af8_tp9_ratio
                        
                        af8_tp9_diff = (af8_power - tp9_power) / (af8_power + tp9_power)
                        features[f"af8_tp9_diff_{band_name}"] = af8_tp9_diff
            
            # Cross-channel coherence and correlation features
            # Compute coherence between all channel pairs
            coherence_pairs = [
                ('af7', 'af8', 'frontal_coherence'),
                ('tp9', 'tp10', 'temporal_coherence'),
                ('af7', 'tp9', 'left_hemisphere_coherence'),
                ('af8', 'tp10', 'right_hemisphere_coherence'),
                ('af7', 'tp10', 'af7_tp10_coherence'),
                ('af8', 'tp9', 'af8_tp9_coherence')
            ]
            
            for ch1, ch2, feature_name in coherence_pairs:
                if ch1 in channel_signals and ch2 in channel_signals:
                    # Compute cross-correlation as a proxy for coherence
                    signal1 = channel_signals[ch1]
                    signal2 = channel_signals[ch2]
                    
                    # Normalize signals
                    signal1_norm = (signal1 - np.mean(signal1)) / (np.std(signal1) + 1e-8)
                    signal2_norm = (signal2 - np.mean(signal2)) / (np.std(signal2) + 1e-8)
                    
                    # Compute cross-correlation
                    xcorr = np.correlate(signal1_norm, signal2_norm, mode='full')
                    max_xcorr = np.max(np.abs(xcorr))
                    features[f"{feature_name}_max"] = max_xcorr
                    
                    # Compute Pearson correlation
                    correlation = np.corrcoef(signal1, signal2)[0, 1]
                    features[f"{feature_name}_pearson"] = correlation if not np.isnan(correlation) else 0.0
            
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
            
            self.logger.info(f"Extracted features columns: {result_df.columns.tolist()}")
            
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
            'spectral': ['_power_', '_relative_power_', '_total_power', '_freq_weighted_power', '_sef_', '_psd_'],
            'temporal': ['_mean', '_std', '_var', '_skew', '_kurtosis', '_rms', '_peak_to_peak', '_zero_crossings', '_mean_abs_deviation', '_hjorth_'],
            'complexity': ['_correlation_dim', '_hurst', '_lyapunov', '_dfa'],
            'entropy': ['_sample_entropy', '_spectral_entropy'],
            'connectivity': ['xcorr_'],
            'asymmetry': ['asymmetry_', '_coherence_', '_pearson']
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
    
    def _save_features(self, df: pd.DataFrame, output_path: str, config: Dict[str, Any]) -> Tuple[str, PandasDataset]:
        """
        Save the extracted features to files and log as MLflow dataset.
        
        Args:
            df: DataFrame containing features
            output_path: Path to save the features
            config: Configuration dictionary containing window and channel info
            
        Returns:
            Tuple of (output_path, mlflow_dataset)
        """
        try:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the complete feature set
            df.to_parquet(output_path)
            self.logger.info(f"Saved features to {output_path}")
            
            # Create MLflow dataset from the saved parquet file
            try:
                # Create dataset with descriptive name including window size and channels
                window_seconds = config['window_slicer']['window_seconds']
                channels = config['data_loader']['channels']
                channels_str = "-".join(channels)
                dataset_name = f"EEG_{window_seconds}s_{channels_str}_{len(df)}windows"
                
                dataset = mlflow.data.from_pandas(
                    df=df,
                    source=output_path,
                    targets="Remission",
                    name=dataset_name
                )
                self.logger.info(f"Created MLflow dataset: {dataset.name}")
                self.logger.info(f"Dataset digest: {dataset.digest}")
                self.logger.info(f"Dataset schema: {dataset.schema}")
                self.logger.info(f"Dataset profile: {dataset.profile}")
                
                # Log the dataset to MLflow
                mlflow.log_input(dataset, context="training")
                self.logger.info(f"Logged dataset to MLflow with name: {dataset.name}")
                
                # Set tags to help training runs discover this dataset
                mlflow.set_tag("mlflow.dataset.logged", "true")
                mlflow.set_tag("mlflow.dataset.context", "training")
                
                # Also log the parquet file as an artifact for backup
                mlflow.log_artifact(output_path, "features")
                self.logger.info(f"Logged features parquet file as MLflow artifact: {output_path}")
                
                return output_path, dataset
                
            except Exception as mlflow_e:
                self.logger.warning(f"Failed to create/log MLflow dataset: {str(mlflow_e)}")
                # Return None dataset if MLflow logging fails
                return output_path, None
            
            # Optionally save group-specific features if group column exists
            if self.save_interim and 'group' in df.columns:
                for group in df['group'].unique():
                    group_df = df[df['group'] == group]
                    group_path = output_dir / f"{group.lower()}_features.parquet"
                    group_df.to_parquet(group_path)
                    self.logger.info(f"Saved {group} features to {group_path}")
                    
                    # Log group-specific files as artifacts too
                    try:
                        mlflow.log_artifact(str(group_path), "features")
                        self.logger.info(f"Logged {group} features as MLflow artifact: {group_path}")
                    except Exception as mlflow_e:
                        self.logger.warning(f"Failed to log {group} features as MLflow artifact: {str(mlflow_e)}")
                    
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
                'prefix': ['_power_', '_relative_power_', '_total_power', '_freq_weighted_power', '_sef_', '_psd_'],
                'description': 'Frequency domain features including band powers, relative powers, PSD statistics, and spectral edge frequencies'
            },
            'Temporal Features': {
                'prefix': ['_mean', '_std', '_var', '_skew', '_kurtosis', '_rms', '_peak_to_peak', '_zero_crossings', '_mean_abs_deviation', '_hjorth_'],
                'description': 'Time domain statistical features and Hjorth parameters'
            },
            'Complexity Features': {
                'prefix': ['_sample_entropy', '_correlation_dim', '_hurst', '_lyapunov', '_dfa'],
                'description': 'Nonlinear dynamics and complexity measures computed per channel'
            },
            'Entropy Features': {
                'prefix': ['_sample_entropy', '_spectral_entropy'],
                'description': 'Information-theoretic measures of signal complexity'
            },
            'Asymmetry Features': {
                'prefix': ['frontal_asymmetry', 'temporal_asymmetry'],
                'description': 'Traditional asymmetry features between homologous electrode pairs'
            },
            'Cross-Hemispheric Features': {
                'prefix': ['hemispheric_asymmetry', 'left_frontal_temporal', 'right_frontal_temporal'],
                'description': 'Asymmetry features between left and right hemispheric regions'
            },
            'Diagonal Cross-Hemispheric Features': {
                'prefix': ['af7_tp10', 'af8_tp9'],
                'description': 'Cross-hemisphere, cross-region connectivity features'
            },
            'Coherence and Correlation Features': {
                'prefix': ['coherence', '_pearson'],
                'description': 'Inter-channel coherence and correlation measures'
            },
            'Connectivity Features': {
                'prefix': ['xcorr_'],
                'description': 'Legacy inter-channel connectivity measures'
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


def extract_eeg_features(config: Dict[str, Any], df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[PandasDataset]]:
    """
    Extract features from EEG data and log as MLflow dataset.
    
    Args:
        config: Configuration dictionary
        df: Input DataFrame
        
    Returns:
        Tuple of (DataFrame with extracted features, MLflow dataset)
    """
    extractor = EEGFeatureExtractor(config)
    
    # Get the window size from config
    window_seconds = config['window_slicer']['window_seconds']
    
    # Create a filename based on window size
    base_path = config['paths']['features']['window']
    base_dir = os.path.dirname(base_path)
    base_filename = os.path.basename(base_path)
    
    # Replace the filename with one that includes window size
    filename_without_ext = os.path.splitext(base_filename)[0]
    new_filename = f"{window_seconds}s_{config['data_loader']['channels']}_window_features.parquet"
    output_path = os.path.join(base_dir, new_filename)
    
    # Log the output path
    logger.info(f"Saving features to {output_path} (window size: {window_seconds}s)")
    
    # Create directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Also update the config to reflect the new path
    config['paths']['features']['window'] = output_path
    
    # Extract features
    result_df = extractor.extract_all_features(
        df,
        output_path=output_path
    )
    
    # Save features and get MLflow dataset
    saved_path, mlflow_dataset = extractor._save_features(result_df, output_path, config)
    
    # Log feature extraction metadata to MLflow
    try:
        mlflow.log_params({
            "window_size_seconds": window_seconds,
            "feature_file_path": output_path,
            "total_features": len(result_df.columns) - 2,  # Exclude Participant and Remission
            "total_windows": len(result_df),
            "total_participants": result_df['Participant'].nunique() if 'Participant' in result_df.columns else 0
        })
        
        # Log dataset information if available
        if mlflow_dataset:
            mlflow.log_params({
                "dataset_name": mlflow_dataset.name,
                "dataset_digest": mlflow_dataset.digest,
                "dataset_num_rows": mlflow_dataset.profile.get('num_rows', 0) if mlflow_dataset.profile else 0,
                "dataset_num_elements": mlflow_dataset.profile.get('num_elements', 0) if mlflow_dataset.profile else 0
            })
        
        logger.info("Logged feature extraction metadata to MLflow")
    except Exception as e:
        logger.warning(f"Failed to log feature extraction metadata to MLflow: {str(e)}")
    
    return result_df, mlflow_dataset


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
            features_df, mlflow_dataset = extract_eeg_features(config, input_df)
            
            # Log success
            mlflow.log_metric("feature_extraction_success", 1)
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            mlflow.log_metric("feature_extraction_success", 0)
            raise