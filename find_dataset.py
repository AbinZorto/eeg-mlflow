import sys
import os
sys.path.append('.')
import mlflow
import pandas as pd
from pathlib import Path

# Set up MLflow - use root mlruns from current directory 
mlflow.set_tracking_uri('file:./mlruns')
print(f'MLflow tracking URI set to: ./mlruns', file=sys.stderr)

if len(sys.argv) < 2 or len(sys.argv) > 3:
    print('ERROR:Usage: find_dataset.py <window_seconds> [ordering_method]')
    print('ERROR:ordering_method can be "sequential" or "completion"')
    sys.exit(1)

window_seconds = sys.argv[1]
ordering_method = sys.argv[2] if len(sys.argv) == 3 else None

# Validate ordering method if provided
if ordering_method and ordering_method not in ['sequential', 'completion']:
    print('ERROR:ordering_method must be "sequential" or "completion"')
    sys.exit(1)

# Determine suffix based on ordering method
ordering_suffix = ""
if ordering_method == 'sequential':
    ordering_suffix = "_seq"
elif ordering_method == 'completion':
    ordering_suffix = "_comp"

print(f'Searching for datasets with window size {window_seconds}s', file=sys.stderr)
if ordering_method:
    print(f'Filtering for ordering method: {ordering_method} (suffix: {ordering_suffix})', file=sys.stderr)

# Search for any 4-channel dataset with this window size (the exact channel order may vary)
dataset_patterns = [
    f'EEG_{window_seconds}s_af7-af8-tp9-tp10_',
    f'EEG_{window_seconds}s_af7-af8-tp10-tp9_',
    f'EEG_{window_seconds}s_af8-af7-tp9-tp10_',
    f'EEG_{window_seconds}s_af8-af7-tp10-tp9_',
    f'EEG_{window_seconds}s_tp9-tp10-af7-af8_',
    f'EEG_{window_seconds}s_tp10-tp9-af7-af8_'
]

try:
    # Search in processing experiment
    processing_experiment = None
    try:
        processing_experiment = mlflow.get_experiment_by_name('eeg_processing')
        print(f'Found eeg_processing experiment: {processing_experiment.experiment_id if processing_experiment else None}', file=sys.stderr)
    except Exception as e:
        print(f'Could not find eeg_processing experiment: {e}', file=sys.stderr)
    
    search_experiment_ids = []
    if processing_experiment:
        search_experiment_ids.append(processing_experiment.experiment_id)
    
    print(f'Will search in {len(search_experiment_ids)} experiments: {search_experiment_ids}', file=sys.stderr)
    
    if search_experiment_ids:
        # Search for all runs, then filter for those with datasets in Python
        all_runs = mlflow.search_runs(
            experiment_ids=search_experiment_ids,
            order_by=['start_time DESC'],
            max_results=10
        )
        
        print(f'Found {len(all_runs)} total runs', file=sys.stderr)
        
        for idx, run in all_runs.iterrows():
            run_id = run['run_id']
            print(f'Checking run: {run_id}', file=sys.stderr)
            
            try:
                mlflow_run = mlflow.get_run(run_id)
                
                # Check if this run has both the required tags and dataset inputs
                if (hasattr(mlflow_run.data, 'tags') and 
                    mlflow_run.data.tags.get('mlflow.dataset.logged') == 'true' and
                    mlflow_run.data.tags.get('mlflow.dataset.context') == 'training' and
                    hasattr(mlflow_run, 'inputs') and mlflow_run.inputs.dataset_inputs):
                    
                    dataset_input = mlflow_run.inputs.dataset_inputs[0]
                    dataset_name = dataset_input.dataset.name
                    print(f'Found dataset run {run_id}: {dataset_name}', file=sys.stderr)
                    
                    # Check if this is a 4-channel dataset with the correct window size
                    for pattern in dataset_patterns:
                        if dataset_name.startswith(pattern):
                            # If ordering method is specified, check for the correct suffix
                            if ordering_method:
                                if ordering_suffix and dataset_name.endswith(ordering_suffix):
                                    print(f'Found matching dataset with {ordering_method} ordering: {dataset_name}', file=sys.stderr)
                                    print(f'SUCCESS:{run_id}:{dataset_name}')
                                    sys.exit(0)
                                elif not ordering_suffix:
                                    # This shouldn't happen due to validation above, but just in case
                                    print(f'Found dataset but no suffix determined: {dataset_name}', file=sys.stderr)
                                    continue
                                else:
                                    print(f'Found dataset but wrong ordering: {dataset_name} (wanted {ordering_method})', file=sys.stderr)
                                    continue
                            else:
                                # No ordering preference - take any matching dataset
                                print(f'Found matching dataset: {dataset_name}', file=sys.stderr)
                                print(f'SUCCESS:{run_id}:{dataset_name}')
                                sys.exit(0)
                            
            except Exception as e:
                print(f'Error checking run {run_id}: {e}', file=sys.stderr)
                continue
    
    if ordering_method:
        print(f'ERROR:No 4-channel {ordering_method} dataset found for window size {window_seconds}s')
    else:
        print(f'ERROR:No 4-channel dataset found for window size {window_seconds}s')
    
except Exception as e:
    print(f'ERROR:Exception occurred: {e}')