import pandas as pd
from typing import List, Dict, Any, Optional, Set
import logging

logger = logging.getLogger(__name__)

class FeatureFilter:
    """Feature filtering utility for EEG datasets."""
    
    # Define feature categories and their patterns
    FEATURE_CATEGORIES = {
        'spectral_features': {
            'description': 'All spectral features including band powers, relative powers, PSD statistics, etc.',
            'patterns': ['_power_', '_relative_power_', '_total_power', '_freq_weighted_power', '_sef_', '_psd_'],
            'subcategories': {
                'band_powers': ['_power_'],
                'relative_powers': ['_relative_power_'],
                'total_power': ['_total_power'],
                'freq_weighted_power': ['_freq_weighted_power'],
                'spectral_edge_freq': ['_sef_'],
                'psd_statistics': ['_psd_']
            }
        },
        'psd_statistics': {
            'description': 'PSD (Power Spectral Density) statistics including mean, std, max, min, median, skewness, kurtosis',
            'patterns': ['_psd_'],
            'subcategories': {
                'psd_mean': ['_psd_mean'],
                'psd_std': ['_psd_std'],
                'psd_max': ['_psd_max'],
                'psd_min': ['_psd_min'],
                'psd_median': ['_psd_median'],
                'psd_skewness': ['_psd_skewness'],
                'psd_kurtosis': ['_psd_kurtosis']
            }
        },
        'temporal_features': {
            'description': 'All temporal features including basic statistics and Hjorth parameters (excludes PSD statistics)',
            'patterns': ['_mean', '_std', '_var', '_skew', '_kurtosis', '_rms', '_peak_to_peak', '_zero_crossings', '_mean_abs_deviation', '_hjorth_'],
            'subcategories': {
                'basic_statistics': ['_mean', '_std', '_var', '_skew', '_kurtosis', '_rms', '_peak_to_peak', '_zero_crossings', '_mean_abs_deviation'],
                'hjorth_parameters': ['_hjorth_']
            }
        },
        'entropy_features': {
            'description': 'Entropy-based features including sample entropy and spectral entropy',
            'patterns': ['_sample_entropy', '_spectral_entropy'],
            'subcategories': {
                'sample_entropy': ['_sample_entropy'],
                'spectral_entropy': ['_spectral_entropy']
            }
        },
        'complexity_features': {
            'description': 'Complexity measures including correlation dimension, Hurst exponent, etc.',
            'patterns': ['_correlation_dim', '_hurst', '_lyapunov', '_dfa'],
            'subcategories': {
                'correlation_dimension': ['_correlation_dim'],
                'hurst_exponent': ['_hurst'],
                'lyapunov_exponent': ['_lyapunov'],
                'dfa': ['_dfa']
            }
        },
        'connectivity_features': {
            'description': 'Inter-channel connectivity features',
            'patterns': ['xcorr_'],
            'subcategories': {
                'cross_correlation': ['xcorr_']
            }
        },
        'asymmetry_features': {
            'description': 'Traditional asymmetry features between homologous electrode pairs',
            'patterns': ['frontal_asymmetry_', 'temporal_asymmetry_'],
            'subcategories': {
                'frontal_asymmetry': ['frontal_asymmetry_'],
                'temporal_asymmetry': ['temporal_asymmetry_']
            }
        },
        'cross_hemispheric_features': {
            'description': 'Cross-hemispheric and cross-regional features',
            'patterns': ['hemispheric_asymmetry_', 'left_frontal_temporal_', 'right_frontal_temporal_'],
            'subcategories': {
                'hemispheric_asymmetry': ['hemispheric_asymmetry_'],
                'frontal_temporal_ratios': ['left_frontal_temporal_', 'right_frontal_temporal_']
            }
        },
        'diagonal_features': {
            'description': 'Diagonal cross-hemispheric features',
            'patterns': ['af7_tp10', 'af8_tp9'],
            'subcategories': {
                'diagonal_cross_hemispheric': ['af7_tp10', 'af8_tp9']
            }
        },
        'coherence_features': {
            'description': 'Inter-channel coherence and correlation measures',
            'patterns': ['_coherence_', '_pearson'],
            'subcategories': {
                'max_coherence': ['_coherence_max'],
                'pearson_correlation': ['_coherence_pearson']
            }
        }
    }
    
    def __init__(self, channels: List[str], feature_categories: Optional[List[str]] = None):
        """
        Initialize feature filter.
        
        Args:
            channels: List of selected channels (e.g., ['af7', 'tp10'])
            feature_categories: List of feature categories to include (e.g., ['spectral_features', 'psd_statistics'])
        """
        self.channels = channels
        self.feature_categories = feature_categories or list(self.FEATURE_CATEGORIES.keys())
        
        # Validate feature categories
        self._validate_feature_categories()
        
        logger.info(f"FeatureFilter initialized with channels: {channels}")
        logger.info(f"FeatureFilter initialized with categories: {self.feature_categories}")
    
    def _validate_feature_categories(self):
        """Validate that all specified feature categories exist."""
        invalid_categories = []
        for category in self.feature_categories:
            if category not in self.FEATURE_CATEGORIES:
                invalid_categories.append(category)
        
        if invalid_categories:
            raise ValueError(f"Invalid feature categories: {invalid_categories}. "
                           f"Valid categories: {list(self.FEATURE_CATEGORIES.keys())}")
    
    def get_feature_patterns(self, categories: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Get feature patterns for specified categories, organized by category.
        
        Args:
            categories: List of feature categories (if None, uses all initialized categories)
            
        Returns:
            Dictionary mapping category names to their patterns
        """
        if categories is None:
            categories = self.feature_categories
        
        category_patterns = {}
        for category in categories:
            if category in self.FEATURE_CATEGORIES:
                category_patterns[category] = self.FEATURE_CATEGORIES[category]['patterns']
        
        return category_patterns
    
    def get_available_features(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Get available features in the dataset grouped by category.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dictionary mapping category names to lists of available features
        """
        available_features = {}
        
        for category, info in self.FEATURE_CATEGORIES.items():
            if category in self.feature_categories:
                category_features = []
                for pattern in info['patterns']:
                    matching_features = [col for col in df.columns if pattern in col]
                    category_features.extend(matching_features)
                
                if category_features:
                    available_features[category] = sorted(list(set(category_features)))
        
        return available_features
    
    def filter_features(self, df: pd.DataFrame, categories: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Filter DataFrame to include only specified feature categories.
        
        Args:
            df: Input DataFrame
            categories: List of feature categories to include (if None, uses all initialized categories)
            
        Returns:
            Filtered DataFrame
        """
        if categories is None:
            categories = self.feature_categories
        
        # Always include metadata columns
        metadata_columns = ['Participant', 'Remission', 'window_start', 'window_end', 'parent_window_id', 'sub_window_id']
        columns_to_keep = [col for col in metadata_columns if col in df.columns]
        
        # Get patterns for specified categories
        category_patterns = self.get_feature_patterns(categories)
        
        # Track which columns are matched by which category to handle conflicts
        column_matches = {}
        
        # Add features that match patterns and are available for selected channels
        for category, patterns in category_patterns.items():
            for pattern in patterns:
                for col in df.columns:
                    if pattern in col:
                        # Check if this feature is for a selected channel
                        if self._is_feature_for_selected_channels(col):
                            if col not in column_matches:
                                column_matches[col] = []
                            column_matches[col].append(category)
                            columns_to_keep.append(col)
        
        # Handle conflicts: if a column matches multiple categories, prioritize based on specificity
        resolved_columns = []
        for col in columns_to_keep:
            if col in metadata_columns:
                resolved_columns.append(col)
            elif col in column_matches:
                # If column matches multiple categories, use the most specific one
                matching_categories = column_matches[col]
                if len(matching_categories) == 1:
                    resolved_columns.append(col)
                else:
                    # Resolve conflicts based on category priority
                    # PSD statistics should take precedence over temporal features for PSD-related columns
                    if 'psd_statistics' in matching_categories and any('psd' in col for cat in matching_categories):
                        resolved_columns.append(col)
                    elif 'temporal_features' in matching_categories and not any('psd' in col for cat in matching_categories):
                        resolved_columns.append(col)
                    else:
                        # Default: include the column (first category wins)
                        resolved_columns.append(col)
        
        # Remove duplicates and ensure columns exist
        resolved_columns = list(set(resolved_columns))
        available_columns = [col for col in resolved_columns if col in df.columns]
        
        if len(available_columns) < 3:  # At least Participant, Remission, and some features
            raise ValueError(f"Insufficient columns after filtering. Found: {len(available_columns)} columns")
        
        filtered_df = df[available_columns].copy()
        
        logger.info(f"Filtered dataset: {len(filtered_df)} rows, {len(filtered_df.columns)} columns")
        logger.info(f"Features: {len(filtered_df.columns) - len([col for col in metadata_columns if col in df.columns])}")
        
        return filtered_df
    
    def _is_feature_for_selected_channels(self, column_name: str) -> bool:
        """
        Check if a feature column is for the selected channels.
        
        Args:
            column_name: Name of the feature column
            
        Returns:
            True if the feature is for selected channels, False otherwise
        """
        # Individual channel features (e.g., af7_power_alpha)
        for channel in self.channels:
            if column_name.startswith(f'{channel}_'):
                return True
        
        # Cross-channel features that require specific channel combinations
        if column_name.startswith('frontal_') and 'af7' in self.channels and 'af8' in self.channels:
            return True
        
        if column_name.startswith('temporal_') and 'tp9' in self.channels and 'tp10' in self.channels:
            return True
        
        if column_name.startswith('hemispheric_') and all(ch in self.channels for ch in ['af7', 'af8', 'tp9', 'tp10']):
            return True
        
        if column_name.startswith('left_') and 'af7' in self.channels and 'tp9' in self.channels:
            return True
        
        if column_name.startswith('right_') and 'af8' in self.channels and 'tp10' in self.channels:
            return True
        
        if column_name.startswith('af7_tp10') and 'af7' in self.channels and 'tp10' in self.channels:
            return True
        
        if column_name.startswith('af8_tp9') and 'af8' in self.channels and 'tp9' in self.channels:
            return True
        
        # Cross-correlation features
        if column_name.startswith('xcorr_'):
            parts = column_name.split('_')
            if len(parts) >= 3:
                ch1, ch2 = parts[1], parts[2]
                if ch1 in self.channels and ch2 in self.channels:
                    return True
        
        # Coherence features
        coherence_pairs = [
            ('frontal_coherence', ['af7', 'af8']),
            ('temporal_coherence', ['tp9', 'tp10']),
            ('left_hemisphere_coherence', ['af7', 'tp9']),
            ('right_hemisphere_coherence', ['af8', 'tp10']),
            ('af7_tp10_coherence', ['af7', 'tp10']),
            ('af8_tp9_coherence', ['af8', 'tp9'])
        ]
        
        for coherence_type, required_channels in coherence_pairs:
            if column_name.startswith(coherence_type) and all(ch in self.channels for ch in required_channels):
                return True
        
        return False
    
    def get_feature_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a summary of features in the dataset.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dictionary with feature summary information
        """
        available_features = self.get_available_features(df)
        
        summary = {
            'total_features': len(df.columns) - len([col for col in ['Participant', 'Remission', 'window_start', 'window_end', 'parent_window_id', 'sub_window_id'] if col in df.columns]),
            'total_windows': len(df),
            'total_participants': df['Participant'].nunique() if 'Participant' in df.columns else 0,
            'feature_categories': {},
            'channel_features': {}
        }
        
        # Count features by category
        for category, features in available_features.items():
            summary['feature_categories'][category] = len(features)
        
        # Count features by channel
        for channel in self.channels:
            channel_features = [col for col in df.columns if col.startswith(f'{channel}_')]
            summary['channel_features'][channel] = len(channel_features)
        
        return summary
    
    @classmethod
    def list_available_categories(cls) -> Dict[str, str]:
        """
        List all available feature categories with descriptions.
        
        Returns:
            Dictionary mapping category names to descriptions
        """
        return {category: info['description'] for category, info in cls.FEATURE_CATEGORIES.items()}
    
    @classmethod
    def list_available_subcategories(cls) -> Dict[str, Dict[str, str]]:
        """
        List all available feature subcategories with descriptions.
        
        Returns:
            Dictionary mapping category names to subcategory dictionaries
        """
        subcategories = {}
        for category, info in cls.FEATURE_CATEGORIES.items():
            if 'subcategories' in info:
                subcategories[category] = {
                    subcat: f"Subcategory of {category}"
                    for subcat in info['subcategories'].keys()
                }
        return subcategories


def filter_dataset_features(df: pd.DataFrame, channels: List[str], 
                          feature_categories: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convenience function to filter dataset features.
    
    Args:
        df: Input DataFrame
        channels: List of selected channels
        feature_categories: List of feature categories to include
        
    Returns:
        Filtered DataFrame
    """
    feature_filter = FeatureFilter(channels, feature_categories)
    return feature_filter.filter_features(df)


def get_feature_filter_summary(df: pd.DataFrame, channels: List[str], 
                             feature_categories: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Get a summary of features that would be available after filtering.
    
    Args:
        df: Input DataFrame
        channels: List of selected channels
        feature_categories: List of feature categories to include
        
    Returns:
        Feature summary dictionary
    """
    feature_filter = FeatureFilter(channels, feature_categories)
    return feature_filter.get_feature_summary(df) 