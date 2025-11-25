#!/usr/bin/env python3

import sys
import os
import pandas as pd
import mlflow
import numpy as np
from pathlib import Path

def filter_dataset_columns(run_id, selected_channels, window_seconds):
    """Filter dataset columns based on selected channels."""
    try:
        # Set up MLflow
        mlflow.set_tracking_uri('file:./mlruns')
        
        # Load original 4-channel dataset
        run = mlflow.get_run(run_id)
        if not (hasattr(run, 'inputs') and run.inputs.dataset_inputs):
            print('ERROR:No dataset in run')
            return
        
        dataset_input = run.inputs.dataset_inputs[0]
        dataset_source = mlflow.data.get_source(dataset_input.dataset)
        data_path = dataset_source.load()
        df = pd.read_parquet(data_path)
        
        print(f'Loaded original dataset with {len(df)} rows and {len(df.columns)} columns', file=sys.stderr)
        
        # Get selected channels
        selected_channels_list = selected_channels.split()
        all_channels = ['af7', 'af8', 'tp9', 'tp10']
        
        # Start with metadata columns
        columns_to_keep = ['Participant', 'Remission']
        
        # Add individual channel features
        for channel in selected_channels_list:
            for col in df.columns:
                if col.startswith(f'{channel}_'):
                    columns_to_keep.append(col)
        
        # Add cross-channel features based on available channel combinations
        
        # Frontal features (requires af7 and af8)
        if 'af7' in selected_channels_list and 'af8' in selected_channels_list:
            for col in df.columns:
                if col.startswith('frontal'):
                    columns_to_keep.append(col)
        
        # Temporal features (requires tp9 and tp10)
        if 'tp9' in selected_channels_list and 'tp10' in selected_channels_list:
            for col in df.columns:
                if col.startswith('temporal'):
                    columns_to_keep.append(col)
        
        # Hemispheric features (requires all 4 channels)
        if all(ch in selected_channels_list for ch in all_channels):
            for col in df.columns:
                if col.startswith('hemispheric'):
                    columns_to_keep.append(col)
        
        # Left hemisphere features (requires af7 and tp9)
        if 'af7' in selected_channels_list and 'tp9' in selected_channels_list:
            for col in df.columns:
                if col.startswith('left_'):
                    columns_to_keep.append(col)
        
        # Right hemisphere features (requires af8 and tp10)
        if 'af8' in selected_channels_list and 'tp10' in selected_channels_list:
            for col in df.columns:
                if col.startswith('right_'):
                    columns_to_keep.append(col)
        
        # Cross-channel features by specific channel pairs
        if 'af7' in selected_channels_list and 'tp10' in selected_channels_list:
            for col in df.columns:
                if col.startswith('af7_tp10'):
                    columns_to_keep.append(col)
        
        if 'af8' in selected_channels_list and 'tp9' in selected_channels_list:
            for col in df.columns:
                if col.startswith('af8_tp9'):
                    columns_to_keep.append(col)
        
        # Cross-correlation features between selected channel pairs
        for col in df.columns:
            if col.startswith('xcorr_'):
                parts = col.split('_')
                if len(parts) >= 3:
                    ch1, ch2 = parts[1], parts[2]
                    if ch1 in selected_channels_list and ch2 in selected_channels_list:
                        columns_to_keep.append(col)
        
        # Remove duplicates and ensure we have the columns
        columns_to_keep = list(set(columns_to_keep))
        available_columns = [col for col in columns_to_keep if col in df.columns]
        
        if len(available_columns) < 3:  # At least Participant, Remission, and some features
            print('ERROR:Insufficient columns after filtering')
            return
        
        # Create filtered dataset
        filtered_df = df[available_columns].copy()
        
        print(f'Filtered dataset: {len(filtered_df)} rows, {len(filtered_df.columns)} columns', file=sys.stderr)
        print(f'Features: {len(filtered_df.columns) - 2}', file=sys.stderr)
        
        # Create new dataset name
        channels_str = "-".join(selected_channels_list)
        new_dataset_name = f"EEG_{window_seconds}s_{channels_str}_filtered"
        new_dataset_path = f"eeg_analysis/data/processed/features/{new_dataset_name}.parquet"
        
        # Save filtered dataset
        output_path = Path(new_dataset_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        filtered_df.to_parquet(output_path)
        
        # Create and log MLflow dataset
        processing_experiment = mlflow.get_experiment_by_name('eeg_processing')
        with mlflow.start_run(experiment_id=processing_experiment.experiment_id) as run:
            dataset = mlflow.data.from_pandas(
                df=filtered_df,
                source=str(output_path),
                targets='Remission',
                name=new_dataset_name
            )
            
            mlflow.log_input(dataset, context='training')
            mlflow.set_tag('mlflow.dataset.logged', 'true')
            mlflow.set_tag('mlflow.dataset.context', 'training')
            mlflow.log_param('dataset_type', 'filtered')
            mlflow.log_param('source_run_id', run_id)
            mlflow.log_param('selected_channels', selected_channels)
            mlflow.log_param('window_seconds', window_seconds)
            mlflow.log_param('total_features', len(filtered_df.columns) - 2)
            
            print(f'SUCCESS:{run.info.run_id}:{dataset.name}')

    except Exception as e:
        print(f'ERROR:Exception occurred: {e}')
        import traceback
        traceback.print_exc(file=sys.stderr)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('ERROR:Usage: filter_dataset.py <run_id> <selected_channels> <window_seconds>')
        sys.exit(1)
    
    run_id = sys.argv[1]
    selected_channels = sys.argv[2]
    window_seconds = sys.argv[3]
    
    filter_dataset_columns(run_id, selected_channels, window_seconds) 