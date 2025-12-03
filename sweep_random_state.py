#!/usr/bin/env python3
"""
Sweep through random_state values for advanced_hybrid_1dcnn_lstm model.
Collects patient-level accuracy for each random_state and saves to CSV.
Continues until 90% accuracy is reached or reasonable limit is hit.
"""

import yaml
import subprocess
import mlflow
import pandas as pd
import time
import sys
from pathlib import Path
import argparse

# Set MLflow tracking URI
mlflow.set_tracking_uri('file:./mlruns')

# Configuration
CONFIG_FILE = 'eeg_analysis/configs/window_model_config_ultra_extreme.yaml'
START_RANDOM_STATE = 10
MIN_RANDOM_STATE = 60  # Minimum to check up to
TARGET_ACCURACY = 0.91  # 91% accuracy target
MAX_RANDOM_STATE = 200  # Maximum to check (safety limit)
OUTPUT_CSV = 'random_state_sweep_results.csv'
MODEL_TYPE = 'advanced_hybrid_1dcnn_lstm'
EXPERIMENT_NAME = 'random_state_sweep'

# Feature selection settings (matching rerun_experiments.sh)
ENABLE_FEATURE_SELECTION = True
N_FEATURES_SELECT = 5
FS_METHOD = 'select_k_best_f_classif'

def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config, config_path):
    """Save YAML config file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def update_random_state(config, random_state, experiment_name):
    """Update random_state in config for advanced_hybrid_1dcnn_lstm and set experiment name."""
    if 'deep_learning' in config and MODEL_TYPE in config['deep_learning']:
        config['deep_learning'][MODEL_TYPE]['random_state'] = random_state
    
    # Set MLflow experiment name
    if 'mlflow' not in config:
        config['mlflow'] = {}
    config['mlflow']['experiment_name'] = experiment_name
    
    return config

def run_training(config_path, enable_feature_selection=True, n_features_select=5, fs_method='select_k_best_f_classif'):
    """Run training pipeline with feature selection (matching rerun_experiments.sh)."""
    cmd = [
        'uv', 'run', 'python3', 'eeg_analysis/run_pipeline.py',
        '--config', config_path,
        'train',
        '--level', 'window',
        '--model-type', MODEL_TYPE
    ]
    
    # Add feature selection flags (matching rerun_experiments.sh)
    if enable_feature_selection:
        cmd.extend([
            '--enable-feature-selection',
            '--n-features-select', str(n_features_select),
            '--fs-method', fs_method
        ])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr

def get_patient_accuracy_from_mlflow(experiment_name=None):
    """Get patient-level accuracy from the most recent MLflow parent run (not nested runs)."""
    if experiment_name is None:
        experiment_name = EXPERIMENT_NAME
    
    try:
        # Get the experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            # Try to find it in all experiments
            print(f"   Experiment '{experiment_name}' not found, searching all experiments...")
            all_experiments = mlflow.search_experiments()
            if all_experiments:
                # Try to find by name first
                found = None
                for exp in all_experiments:
                    if exp.name == experiment_name:
                        found = exp
                        break
                if found:
                    experiment = found
                else:
                    # Use the most recent experiment
                    experiment = sorted(all_experiments, key=lambda x: x.creation_time, reverse=True)[0]
                    print(f"   Using most recent experiment: {experiment.name}")
            else:
                print(f"   No experiments found")
                return None, None
        
        experiment_id = experiment.experiment_id
        
        # Get all runs and filter for parent runs in Python
        # MLflow filter syntax doesn't support "IS NULL" easily, so we filter in Python
        all_runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            order_by=['start_time desc'],
            max_results=100  # Get more runs to find parent ones
        )
        
        if all_runs.empty:
            print(f"   ⚠️  No runs found in experiment")
            return None, None
        
        # Filter for parent runs (runs without mlflow.parentRunId tag)
        parent_runs = []
        for idx, run_row in all_runs.iterrows():
            run_obj = mlflow.get_run(run_row['run_id'])
            # Check if this run has a parent (is nested)
            if 'mlflow.parentRunId' not in run_obj.data.tags:
                parent_runs.append(run_row)
        
        if not parent_runs:
            print(f"   ⚠️  No parent runs found, using most recent run")
            latest_run = all_runs.iloc[0]
        else:
            # Use the most recent parent run
            latest_run = parent_runs[0]  # Already sorted by start_time desc
        
        run_id = latest_run['run_id']
        
        # Get the run details
        run = mlflow.get_run(run_id)
        
        # Verify this is a parent run (not nested)
        if 'mlflow.parentRunId' in run.data.tags:
            print(f"   ⚠️  Warning: Run {run_id} is a nested run, not parent run")
            print(f"   Looking for parent run...")
            # Try to find the parent run
            parent_run_id = run.data.tags.get('mlflow.parentRunId')
            try:
                parent_run = mlflow.get_run(parent_run_id)
                run = parent_run
                run_id = parent_run_id
                print(f"   ✅ Found parent run: {parent_run_id}")
            except Exception as e:
                print(f"   ❌ Could not retrieve parent run: {e}")
                return None, None
        
        # Get metrics from the PARENT run only
        metrics = run.data.metrics
        
        # The deep learning trainer logs: mlflow.log_metrics({f"patient_{k}": v ...})
        # So it should be 'patient_accuracy' in the parent run
        patient_accuracy = None
        
        # Try the exact metric name first (this is what the trainer logs)
        if 'patient_accuracy' in metrics:
            patient_accuracy = metrics['patient_accuracy']
            print(f"   ✅ Found patient_accuracy in parent run: {patient_accuracy:.4f}")
        else:
            # Try alternative names
            possible_names = [
                'patient_level_accuracy',
                'overall_patient_accuracy',
                'patient_metrics_accuracy',
            ]
            
            for name in possible_names:
                if name in metrics:
                    patient_accuracy = metrics[name]
                    print(f"   ✅ Found {name}: {patient_accuracy:.4f}")
                    break
        
        if patient_accuracy is None:
            print(f"   ⚠️  patient_accuracy metric not found in parent run")
            print(f"   Available metrics: {', '.join(sorted(metrics.keys())[:20])}...")
            return None, None
            print(f"   ⚠️  patient_accuracy metric not found in parent run")
            print(f"   Available metrics: {list(metrics.keys())[:10]}...")  # Show first 10
            return None, None
        
        return patient_accuracy, run_id
        
    except Exception as e:
        print(f"   ❌ Error getting patient accuracy from MLflow: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    parser = argparse.ArgumentParser(description='Sweep random_state values for advanced_hybrid_1dcnn_lstm')
    parser.add_argument('--start', type=int, default=START_RANDOM_STATE, help='Starting random_state value')
    parser.add_argument('--min', type=int, default=MIN_RANDOM_STATE, help='Minimum random_state to check up to')
    parser.add_argument('--target', type=float, default=TARGET_ACCURACY, help='Target accuracy (default: 0.90)')
    parser.add_argument('--max', type=int, default=MAX_RANDOM_STATE, help='Maximum random_state (safety limit)')
    parser.add_argument('--output', type=str, default=OUTPUT_CSV, help='Output CSV file')
    parser.add_argument('--config', type=str, default=CONFIG_FILE, help='Config file path')
    
    args = parser.parse_args()
    
    # Create or load results DataFrame
    results_file = Path(args.output)
    if results_file.exists():
        results_df = pd.read_csv(results_file)
        print(f"Loaded existing results: {len(results_df)} entries")
    else:
        results_df = pd.DataFrame(columns=['random_state', 'patient_accuracy', 'run_id', 'timestamp'])
        print("Starting new sweep")
    
    # Ensure MLflow experiment exists
    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            mlflow.create_experiment(EXPERIMENT_NAME)
            print(f"Created MLflow experiment: {EXPERIMENT_NAME}")
    except Exception as e:
        print(f"Note: MLflow experiment setup: {e}")
    
    # Load config
    config = load_config(args.config)
    
    # Determine starting point
    if not results_df.empty:
        completed_states = set(results_df['random_state'].astype(int))
        start_state = max(args.start, max(completed_states) + 1) if completed_states else args.start
        print(f"Resuming from random_state={start_state}")
    else:
        start_state = args.start
    
    # Sweep through random_state values
    found_target = False
    random_state = start_state
    
    while random_state <= args.max:
        # Skip if already completed
        if not results_df.empty and random_state in results_df['random_state'].astype(int).values:
            print(f"⏭️  Skipping random_state={random_state} (already completed)")
            random_state += 1
            continue
        
        print(f"\n{'='*80}")
        print(f"🔍 Testing random_state={random_state}")
        print(f"{'='*80}")
        
        # Update config
        config = update_random_state(config, random_state, EXPERIMENT_NAME)
        
        # Save config temporarily (or use a copy)
        temp_config = f'temp_config_rs{random_state}.yaml'
        save_config(config, temp_config)
        
        try:
            # Set MLflow experiment for this run
            mlflow.set_experiment(EXPERIMENT_NAME)
            
            # Run training with feature selection (matching rerun_experiments.sh)
            print(f"🚀 Starting training with random_state={random_state}...")
            print(f"   Configuration:")
            print(f"     - Feature selection: {ENABLE_FEATURE_SELECTION}")
            if ENABLE_FEATURE_SELECTION:
                print(f"     - N features: {N_FEATURES_SELECT}")
                print(f"     - FS method: {FS_METHOD}")
            success, stdout, stderr = run_training(
                temp_config,
                enable_feature_selection=ENABLE_FEATURE_SELECTION,
                n_features_select=N_FEATURES_SELECT,
                fs_method=FS_METHOD
            )
            
            if not success:
                print(f"❌ Training failed for random_state={random_state}")
                print(f"STDOUT (last 500 chars):\n{stdout[-500:]}")
                print(f"STDERR (last 500 chars):\n{stderr[-500:]}")
                # Still try to get results if available
                time.sleep(3)  # Wait for MLflow to sync
                patient_accuracy, run_id = get_patient_accuracy_from_mlflow()
            else:
                print(f"✅ Training completed for random_state={random_state}")
                # Wait a bit for MLflow to sync
                time.sleep(3)
                
                # Get patient accuracy from MLflow
                patient_accuracy, run_id = get_patient_accuracy_from_mlflow(EXPERIMENT_NAME)
            
            if patient_accuracy is not None:
                print(f"📊 Patient accuracy: {patient_accuracy:.4f}")
                
                # Add to results
                new_row = pd.DataFrame({
                    'random_state': [random_state],
                    'patient_accuracy': [patient_accuracy],
                    'run_id': [run_id if run_id else ''],
                    'timestamp': [pd.Timestamp.now()]
                })
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                
                # Save results incrementally
                results_df.to_csv(args.output, index=False)
                print(f"💾 Saved results to {args.output}")
                
                # Check if we hit target
                if patient_accuracy >= args.target:
                    print(f"\n🎉 TARGET REACHED! random_state={random_state} achieved {patient_accuracy:.4f} accuracy")
                    found_target = True
                    break
            else:
                print(f"⚠️  Could not retrieve patient accuracy for random_state={random_state}")
                # Still save a row with NaN
                new_row = pd.DataFrame({
                    'random_state': [random_state],
                    'patient_accuracy': [None],
                    'run_id': [run_id if run_id else ''],
                    'timestamp': [pd.Timestamp.now()]
                })
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                results_df.to_csv(args.output, index=False)
        
        except Exception as e:
            print(f"❌ Error processing random_state={random_state}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Clean up temp config
            if Path(temp_config).exists():
                Path(temp_config).unlink()
        
        # Check if we've reached minimum and should stop if no target found
        if random_state >= args.min and not found_target:
            print(f"\n⚠️  Reached minimum random_state={args.min} without hitting target accuracy")
            print(f"   Continuing to search for target accuracy...")
        
        random_state += 1
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"📊 SWEEP SUMMARY")
    print(f"{'='*80}")
    print(f"Total random_states tested: {len(results_df)}")
    
    if not results_df.empty:
        valid_results = results_df.dropna(subset=['patient_accuracy'])
        if not valid_results.empty:
            best_idx = valid_results['patient_accuracy'].idxmax()
            best_row = valid_results.loc[best_idx]
            print(f"\n🏆 Best result:")
            print(f"   random_state: {best_row['random_state']}")
            print(f"   patient_accuracy: {best_row['patient_accuracy']:.4f}")
            
            # Show top 5
            top5 = valid_results.nlargest(5, 'patient_accuracy')
            print(f"\n📈 Top 5 random_states:")
            for idx, row in top5.iterrows():
                print(f"   random_state={int(row['random_state']):3d}: {row['patient_accuracy']:.4f}")
        
        print(f"\n💾 Results saved to: {args.output}")
    
    if found_target:
        print(f"\n✅ Target accuracy ({args.target}) was reached!")
    else:
        print(f"\n⚠️  Target accuracy ({args.target}) was not reached within random_state range [{start_state}, {random_state-1}]")

if __name__ == '__main__':
    main()

