#!/usr/bin/env python3

"""
Test script to demonstrate feature filtering functionality.
This script shows how to use the feature filtering system with different channel and feature category combinations.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add the eeg_analysis directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'eeg_analysis'))

from src.utils.feature_filter import FeatureFilter

def create_sample_dataset():
    """Create a sample dataset with all feature types for testing."""
    # Create sample data
    n_samples = 100
    participants = [f"P{i:02d}" for i in range(1, 21)]  # 20 participants
    remission = np.random.choice([0, 1], size=n_samples)
    
    data = {
        'Participant': np.random.choice(participants, size=n_samples),
        'Remission': remission,
        'window_start': np.arange(n_samples),
        'window_end': np.arange(n_samples) + 1000,
        'parent_window_id': np.arange(n_samples),
        'sub_window_id': np.arange(n_samples)
    }
    
    # Add individual channel features for all 4 channels
    channels = ['af7', 'af8', 'tp9', 'tp10']
    frequency_bands = ['alpha', 'beta', 'delta', 'gamma', 'theta']
    
    for channel in channels:
        # Spectral features
        for band in frequency_bands:
            data[f'{channel}_power_{band}'] = np.random.randn(n_samples)
            data[f'{channel}_relative_power_{band}'] = np.random.rand(n_samples)
        
        data[f'{channel}_total_power'] = np.random.randn(n_samples)
        data[f'{channel}_freq_weighted_power'] = np.random.randn(n_samples)
        data[f'{channel}_sef_90'] = np.random.uniform(10, 50, n_samples)
        data[f'{channel}_sef_95'] = np.random.uniform(15, 60, n_samples)
        
        # PSD statistics
        data[f'{channel}_psd_mean'] = np.random.randn(n_samples)
        data[f'{channel}_psd_std'] = np.random.randn(n_samples)
        data[f'{channel}_psd_max'] = np.random.randn(n_samples)
        data[f'{channel}_psd_min'] = np.random.randn(n_samples)
        data[f'{channel}_psd_median'] = np.random.randn(n_samples)
        data[f'{channel}_psd_skewness'] = np.random.randn(n_samples)
        data[f'{channel}_psd_kurtosis'] = np.random.randn(n_samples)
        
        # Temporal features
        data[f'{channel}_mean'] = np.random.randn(n_samples)
        data[f'{channel}_std'] = np.random.randn(n_samples)
        data[f'{channel}_var'] = np.random.randn(n_samples)
        data[f'{channel}_skew'] = np.random.randn(n_samples)
        data[f'{channel}_kurtosis'] = np.random.randn(n_samples)
        data[f'{channel}_rms'] = np.random.randn(n_samples)
        data[f'{channel}_peak_to_peak'] = np.random.randn(n_samples)
        data[f'{channel}_zero_crossings'] = np.random.randint(0, 100, n_samples)
        data[f'{channel}_mean_abs_deviation'] = np.random.randn(n_samples)
        
        # Hjorth parameters
        data[f'{channel}_hjorth_activity'] = np.random.randn(n_samples)
        data[f'{channel}_hjorth_mobility'] = np.random.randn(n_samples)
        data[f'{channel}_hjorth_complexity'] = np.random.randn(n_samples)
        
        # Entropy features
        data[f'{channel}_sample_entropy'] = np.random.randn(n_samples)
        data[f'{channel}_spectral_entropy'] = np.random.randn(n_samples)
        
        # Complexity features
        data[f'{channel}_correlation_dim'] = np.random.randn(n_samples)
        data[f'{channel}_hurst'] = np.random.randn(n_samples)
        data[f'{channel}_lyapunov'] = np.random.randn(n_samples)
        data[f'{channel}_dfa'] = np.random.randn(n_samples)
    
    # Add cross-channel features
    # Cross-correlation features
    data['xcorr_af7_af8'] = np.random.randn(n_samples)
    data['xcorr_af7_tp9'] = np.random.randn(n_samples)
    data['xcorr_af7_tp10'] = np.random.randn(n_samples)
    data['xcorr_af8_tp9'] = np.random.randn(n_samples)
    data['xcorr_af8_tp10'] = np.random.randn(n_samples)
    data['xcorr_tp9_tp10'] = np.random.randn(n_samples)
    
    # Asymmetry features
    for band in frequency_bands:
        data[f'frontal_asymmetry_{band}'] = np.random.randn(n_samples)
        data[f'frontal_asymmetry_ratio_{band}'] = np.random.randn(n_samples)
        data[f'temporal_asymmetry_{band}'] = np.random.randn(n_samples)
        data[f'temporal_asymmetry_ratio_{band}'] = np.random.randn(n_samples)
    
    # Cross-hemispheric features
    for band in frequency_bands:
        data[f'hemispheric_asymmetry_{band}'] = np.random.randn(n_samples)
        data[f'hemispheric_asymmetry_ratio_{band}'] = np.random.randn(n_samples)
        data[f'left_frontal_temporal_ratio_{band}'] = np.random.randn(n_samples)
        data[f'left_frontal_temporal_diff_{band}'] = np.random.randn(n_samples)
        data[f'right_frontal_temporal_ratio_{band}'] = np.random.randn(n_samples)
        data[f'right_frontal_temporal_diff_{band}'] = np.random.randn(n_samples)
    
    # Diagonal features
    for band in frequency_bands:
        data[f'af7_tp10_ratio_{band}'] = np.random.randn(n_samples)
        data[f'af7_tp10_diff_{band}'] = np.random.randn(n_samples)
        data[f'af8_tp9_ratio_{band}'] = np.random.randn(n_samples)
        data[f'af8_tp9_diff_{band}'] = np.random.randn(n_samples)
    
    # Coherence features
    data['frontal_coherence_max'] = np.random.randn(n_samples)
    data['frontal_coherence_pearson'] = np.random.randn(n_samples)
    data['temporal_coherence_max'] = np.random.randn(n_samples)
    data['temporal_coherence_pearson'] = np.random.randn(n_samples)
    data['left_hemisphere_coherence_max'] = np.random.randn(n_samples)
    data['left_hemisphere_coherence_pearson'] = np.random.randn(n_samples)
    data['right_hemisphere_coherence_max'] = np.random.randn(n_samples)
    data['right_hemisphere_coherence_pearson'] = np.random.randn(n_samples)
    data['af7_tp10_coherence_max'] = np.random.randn(n_samples)
    data['af7_tp10_coherence_pearson'] = np.random.randn(n_samples)
    data['af8_tp9_coherence_max'] = np.random.randn(n_samples)
    data['af8_tp9_coherence_pearson'] = np.random.randn(n_samples)
    
    return pd.DataFrame(data)

def test_feature_filtering():
    """Test the feature filtering functionality with different scenarios."""
    print("=== Feature Filtering Test ===\n")
    
    # Create sample dataset
    print("Creating sample dataset...")
    df = create_sample_dataset()
    print(f"Original dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Features: {len(df.columns) - 6}")  # Exclude metadata columns
    print()
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Single channel (tp10) with spectral features only',
            'channels': ['tp10'],
            'categories': ['spectral_features']
        },
        {
            'name': 'Single channel (tp10) with PSD statistics only',
            'channels': ['tp10'],
            'categories': ['psd_statistics']
        },
        {
            'name': 'Frontal channels (af7, af8) with temporal features',
            'channels': ['af7', 'af8'],
            'categories': ['temporal_features']
        },
        {
            'name': 'All channels with entropy and complexity features',
            'channels': ['af7', 'af8', 'tp9', 'tp10'],
            'categories': ['entropy_features', 'complexity_features']
        },
        {
            'name': 'Frontal channels (af7, af8) with asymmetry features',
            'channels': ['af7', 'af8'],
            'categories': ['asymmetry_features']
        },
        {
            'name': 'Temporal channels (tp9, tp10) with asymmetry features',
            'channels': ['tp9', 'tp10'],
            'categories': ['asymmetry_features']
        },
        {
            'name': 'All channels with cross-hemispheric features',
            'channels': ['af7', 'af8', 'tp9', 'tp10'],
            'categories': ['cross_hemispheric_features']
        },
        {
            'name': 'Diagonal channels (af7, tp10) with diagonal features',
            'channels': ['af7', 'tp10'],
            'categories': ['diagonal_features']
        },
        {
            'name': 'All channels with coherence features',
            'channels': ['af7', 'af8', 'tp9', 'tp10'],
            'categories': ['coherence_features']
        },
        {
            'name': 'Multiple feature categories',
            'channels': ['af7', 'tp10'],
            'categories': ['spectral_features', 'temporal_features', 'entropy_features']
        },
        {
            'name': 'Conflict test: temporal_features vs psd_statistics',
            'channels': ['tp10'],
            'categories': ['temporal_features', 'psd_statistics']
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"Test {i}: {scenario['name']}")
        print(f"  Channels: {scenario['channels']}")
        print(f"  Categories: {scenario['categories']}")
        
        try:
            # Create feature filter
            feature_filter = FeatureFilter(scenario['channels'], scenario['categories'])
            
            # Apply filtering
            filtered_df = feature_filter.filter_features(df)
            
            # Get summary
            summary = feature_filter.get_feature_summary(filtered_df)
            
            print(f"  Result: {len(filtered_df)} rows, {len(filtered_df.columns)} columns")
            print(f"  Features: {summary['total_features']}")
            print(f"  Feature breakdown:")
            for category, count in summary['feature_categories'].items():
                print(f"    - {category}: {count} features")
            
            # Debug output for conflict test
            if scenario['name'] == 'Conflict test: temporal_features vs psd_statistics':
                print(f"  Debug - Selected features:")
                feature_cols = [col for col in filtered_df.columns if col not in ['Participant', 'Remission', 'window_start', 'window_end', 'parent_window_id', 'sub_window_id']]
                for col in sorted(feature_cols):
                    if 'psd_' in col:
                        print(f"    - {col} (PSD)")
                    elif any(stat in col for stat in ['_mean', '_std', '_var', '_skew', '_kurtosis', '_rms', '_peak_to_peak', '_zero_crossings', '_mean_abs_deviation', '_hjorth_']):
                        print(f"    - {col} (Temporal)")
                    else:
                        print(f"    - {col} (Other)")
            
            print()
            
        except Exception as e:
            print(f"  Error: {e}")
            print()

def show_available_categories():
    """Show all available feature categories."""
    print("=== Available Feature Categories ===\n")
    
    categories = FeatureFilter.list_available_categories()
    subcategories = FeatureFilter.list_available_subcategories()
    
    for category, description in categories.items():
        print(f"{category}: {description}")
        if category in subcategories:
            print("  Subcategories:")
            for subcat, subdesc in subcategories[category].items():
                print(f"    - {subcat}: {subdesc}")
        print()

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'list':
        show_available_categories()
    else:
        test_feature_filtering() 