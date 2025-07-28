# Feature Filtering System

This document describes the feature filtering system that allows you to selectively train models on specific feature categories and channels.

## Overview

The feature filtering system enables you to:
- Filter features by category (e.g., spectral, temporal, entropy, complexity)
- Filter features by channel (e.g., af7, af8, tp9, tp10)
- Combine both filters to get exactly the features you want
- Work seamlessly with existing channel filtering and feature selection

## Available Feature Categories

### Main Categories

1. **spectral_features** - All spectral features including band powers, relative powers, PSD statistics, etc.
   - Subcategories:
     - `band_powers` - Individual frequency band powers per channel
     - `relative_powers` - Relative band powers (normalized by total power)
     - `total_power` - Total power across all bands
     - `freq_weighted_power` - Frequency-weighted power
     - `spectral_edge_freq` - SEF 90% and 95%
     - `psd_statistics` - PSD mean, std, max, min, median, skewness, kurtosis

2. **temporal_features** - All temporal features including basic statistics and Hjorth parameters
   - Subcategories:
     - `basic_statistics` - mean, std, var, skew, kurtosis, rms, peak_to_peak, etc.
     - `hjorth_parameters` - Activity, mobility, complexity

3. **entropy_features** - Entropy-based features including sample entropy and spectral entropy
   - Subcategories:
     - `sample_entropy` - Sample entropy per channel
     - `spectral_entropy` - Spectral entropy per channel

4. **complexity_features** - Complexity measures including correlation dimension, Hurst exponent, etc.
   - Subcategories:
     - `correlation_dimension` - Correlation dimension
     - `hurst_exponent` - Hurst exponent
     - `lyapunov_exponent` - Lyapunov exponent
     - `dfa` - Detrended Fluctuation Analysis

5. **connectivity_features** - Inter-channel connectivity features
   - Subcategories:
     - `cross_correlation` - Cross-correlation between channel pairs

6. **asymmetry_features** - Traditional asymmetry features between homologous electrode pairs
   - Subcategories:
     - `frontal_asymmetry` - AF7/AF8 asymmetry (requires both channels)
     - `temporal_asymmetry` - TP9/TP10 asymmetry (requires both channels)

7. **cross_hemispheric_features** - Cross-hemispheric and cross-regional features
   - Subcategories:
     - `hemispheric_asymmetry` - Left vs right hemisphere (requires all 4 channels)
     - `frontal_temporal_ratios` - Within-hemisphere frontal-temporal ratios

8. **diagonal_features** - Diagonal cross-hemispheric features
   - Subcategories:
     - `diagonal_cross_hemispheric` - AF7-TP10 and AF8-TP9 interactions

9. **coherence_features** - Inter-channel coherence and correlation measures
   - Subcategories:
     - `max_coherence` - Maximum coherence between channel pairs
     - `pearson_correlation` - Pearson correlation between channel pairs

## Usage

### Command Line Interface

The feature filtering is integrated into the existing training pipeline:

```bash
# List available feature categories
python eeg_analysis/run_pipeline.py --config eeg_analysis/configs/window_model_config_ultra_extreme.yaml train --level window --model-type random_forest --feature-categories list

# Train with specific feature categories
python eeg_analysis/run_pipeline.py --config eeg_analysis/configs/window_model_config_ultra_extreme.yaml train --level window --model-type random_forest --feature-categories "spectral_features,psd_statistics"

# Train with multiple feature categories
python eeg_analysis/run_pipeline.py --config eeg_analysis/configs/window_model_config_ultra_extreme.yaml train --level window --model-type random_forest --feature-categories "temporal_features,entropy_features,complexity_features"

# Combine with feature selection
python eeg_analysis/run_pipeline.py --config eeg_analysis/configs/window_model_config_ultra_extreme.yaml train --level window --model-type random_forest --feature-categories "spectral_features" --enable-feature-selection --n-features-select 10
```

### Examples

#### Example 1: Single Channel with PSD Statistics
```bash
# Configure channels in processing_config.yaml: channels: ['tp10']
# Then run:
python eeg_analysis/run_pipeline.py --config eeg_analysis/configs/window_model_config_ultra_extreme.yaml train --level window --model-type random_forest --feature-categories "psd_statistics"
```
This will give you only the PSD statistics features (mean, std, max, min, median, skewness, kurtosis) for the tp10 channel.

#### Example 2: Frontal Channels with Temporal Features
```bash
# Configure channels in processing_config.yaml: channels: ['af7', 'af8']
# Then run:
python eeg_analysis/run_pipeline.py --config eeg_analysis/configs/window_model_config_ultra_extreme.yaml train --level window --model-type random_forest --feature-categories "temporal_features"
```
This will give you temporal features (mean, std, var, skew, kurtosis, rms, etc.) for both af7 and af8 channels, plus frontal asymmetry features.

#### Example 3: All Channels with Multiple Feature Types
```bash
# Configure channels in processing_config.yaml: channels: ['af7', 'af8', 'tp9', 'tp10']
# Then run:
python eeg_analysis/run_pipeline.py --config eeg_analysis/configs/window_model_config_ultra_extreme.yaml train --level window --model-type random_forest --feature-categories "spectral_features,entropy_features,complexity_features"
```
This will give you spectral, entropy, and complexity features for all channels, plus all cross-channel features that are available.

## How It Works

### Channel Dependencies

The system automatically handles channel dependencies:

- **Individual channel features**: Available for any selected channel
- **Frontal asymmetry**: Requires both `af7` and `af8`
- **Temporal asymmetry**: Requires both `tp9` and `tp10`
- **Hemispheric asymmetry**: Requires all 4 channels (`af7`, `af8`, `tp9`, `tp10`)
- **Cross-regional features**: Requires appropriate channel pairs
- **Diagonal features**: Requires specific diagonal channel pairs
- **Coherence features**: Requires appropriate channel pairs

### Integration with Existing Systems

The feature filtering system integrates seamlessly with:

1. **Channel filtering**: Works with the existing channel selection in `processing_config.yaml`
2. **Feature selection**: Can be combined with sklearn feature selection methods
3. **MLflow tracking**: All filtering operations are logged to MLflow
4. **Model training**: Works with all existing model types

### Implementation Details

The feature filtering is implemented in:

1. **`src/utils/feature_filter.py`**: Core filtering logic
2. **`eeg_analysis/run_pipeline.py`**: CLI integration
3. **`src/models/base_trainer.py`**: Integration with training pipeline

## Testing

You can test the feature filtering system using the provided test script:

```bash
# Run comprehensive tests
python test_feature_filtering.py

# List available categories
python test_feature_filtering.py list
```

## Best Practices

1. **Start with specific categories**: Begin with a few specific feature categories rather than all features
2. **Consider channel dependencies**: Some features require specific channel combinations
3. **Combine with feature selection**: Use feature filtering to reduce the feature space, then apply feature selection
4. **Monitor performance**: Track how different feature combinations affect model performance
5. **Document experiments**: Use descriptive run names and log feature filtering parameters

## Troubleshooting

### Common Issues

1. **"Insufficient columns after filtering"**: This usually means the selected channels don't support the requested feature categories
2. **"Invalid feature categories"**: Check the spelling and use `--feature-categories list` to see valid options
3. **No cross-channel features**: Some feature categories require specific channel combinations

### Debugging

1. Check the logs for detailed information about which features are being included/excluded
2. Use the test script to verify your feature category and channel combinations
3. Monitor MLflow parameters to see exactly what filtering was applied

## Future Enhancements

Potential future improvements:

1. **Subcategory filtering**: Allow filtering by specific subcategories
2. **Custom feature patterns**: Allow users to define custom feature patterns
3. **Feature importance integration**: Integrate with feature importance analysis
4. **Automated feature selection**: Suggest optimal feature combinations based on data analysis 