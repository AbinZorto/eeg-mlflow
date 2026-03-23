#!/usr/bin/env python3
"""
Sweep random_state values for whichever model is selected in config.
Collects patient-level accuracy for each random_state and saves to CSV.
Continues until target accuracy is reached or max random_state is hit.
"""

import argparse
import subprocess
import time
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

# Resolve project root and use stable absolute paths.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
mlflow.set_tracking_uri(f"file:{PROJECT_ROOT / 'mlruns'}")

# Configuration
CONFIG_FILE = str(PROJECT_ROOT / "eeg_analysis" / "configs" / "window_model_config")
START_RANDOM_STATE = 10
MIN_RANDOM_STATE = 60  # Minimum to check up to
TARGET_ACCURACY = 0.91  # 91% accuracy target
MAX_RANDOM_STATE = 200  # Maximum to check (safety limit)
EXPERIMENT_NAME_PREFIX = "random_state_sweep"

# Feature selection settings (matching unified experiment runner defaults)
ENABLE_FEATURE_SELECTION = True
N_FEATURES_SELECT = 5
FS_METHOD = "select_k_best_f_classif"


def ensure_mlflow_experiment_ready(experiment_name):
    """
    Ensure a usable MLflow experiment exists and return its name.

    Behavior:
    - If the experiment exists and is active, keep it.
    - If it does not exist, create it with the requested name.
    - If the requested name exists only in deleted state, create a fresh experiment
      with a unique derived name (do not restore deleted experiments).
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    deleted_match = None

    # Some MLflow versions may not return deleted experiments by name.
    if experiment is None:
        deleted_experiments = client.search_experiments(view_type=ViewType.DELETED_ONLY)
        for exp in deleted_experiments:
            if exp.name == experiment_name:
                deleted_match = exp
                break
    elif experiment.lifecycle_stage == "deleted":
        deleted_match = experiment

    if experiment is not None and experiment.lifecycle_stage == "active":
        return experiment_name

    if deleted_match is not None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        candidate_name = f"{experiment_name}_fresh_{timestamp}"
        suffix = 1

        # Ensure uniqueness across active and deleted experiments.
        while True:
            existing_active = client.get_experiment_by_name(candidate_name)
            existing_deleted = [
                exp for exp in client.search_experiments(view_type=ViewType.DELETED_ONLY)
                if exp.name == candidate_name
            ]
            if existing_active is None and not existing_deleted:
                break
            candidate_name = f"{experiment_name}_fresh_{timestamp}_{suffix}"
            suffix += 1

        client.create_experiment(candidate_name)
        print(
            f"Experiment '{experiment_name}' is deleted "
            f"(id={deleted_match.experiment_id}). "
            f"Created new experiment: {candidate_name}"
        )
        return candidate_name

    client.create_experiment(experiment_name)
    print(f"Created MLflow experiment: {experiment_name}")
    return experiment_name


def resolve_config_path(config_path):
    """
    Resolve config path with extension fallback.

    Supports:
    - eeg_analysis/configs/window_model_config
    - eeg_analysis/configs/window_model_config.yaml
    - eeg_analysis/configs/window_model_config.yml
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    candidates = [path]
    if path.suffix == "":
        candidates.extend([path.with_suffix(".yaml"), path.with_suffix(".yml")])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    candidate_str = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find config file. Tried: {candidate_str}")

def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def save_config(config, config_path):
    """Save YAML config file."""
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def detect_model_type(config):
    """Detect model type from config."""
    model_type = config.get("model_type")
    if isinstance(model_type, str) and model_type.strip():
        return model_type.strip()

    raise ValueError(
        "Config is missing top-level 'model_type'. "
        "Set model_type in config to choose which model to sweep."
    )

def update_random_state(config, model_type, random_state, experiment_name):
    """
    Update random_state in config for the selected model and set experiment name.

    Updates:
    1. model-specific random_state in deep_learning or model.params
    2. data split random_state
    3. root-level random_seed
    4. model_type (kept explicit for run_pipeline)
    """
    model_updated = False

    if "deep_learning" in config and model_type in config["deep_learning"]:
        config["deep_learning"][model_type]["random_state"] = random_state
        model_updated = True
    elif "model" in config and "params" in config["model"] and model_type in config["model"]["params"]:
        config["model"]["params"][model_type]["random_state"] = random_state
        model_updated = True

    if "data" in config and "split" in config["data"]:
        config["data"]["split"]["random_state"] = random_state

    config["random_seed"] = random_state
    config["model_type"] = model_type

    # Set MLflow experiment name
    if "mlflow" not in config:
        config["mlflow"] = {}
    config["mlflow"]["experiment_name"] = experiment_name

    return config, model_updated

def run_training(model_type, config_path, enable_feature_selection=False, n_features_select=5, fs_method="select_k_best_f_classif"):
    """Run training pipeline with optional feature selection."""
    cmd = [
        "uv", "run", "python3", "eeg_analysis/run_pipeline.py",
        "--config", str(config_path),
        "train",
        "--level", "window",
        "--model-type", model_type
    ]

    # Add feature selection flags
    if enable_feature_selection:
        cmd.extend([
            "--enable-feature-selection",
            "--n-features-select", str(n_features_select),
            "--fs-method", fs_method
        ])

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    return result.returncode == 0, result.stdout, result.stderr

def get_patient_accuracy_from_mlflow(experiment_name):
    """Get patient-level accuracy from the most recent MLflow parent run (not nested runs)."""
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
            if "mlflow.parentRunId" not in run_obj.data.tags:
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
        if "mlflow.parentRunId" in run.data.tags:
            print(f"   ⚠️  Warning: Run {run_id} is a nested run, not parent run")
            print(f"   Looking for parent run...")
            # Try to find the parent run
            parent_run_id = run.data.tags.get("mlflow.parentRunId")
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
        if "patient_accuracy" in metrics:
            patient_accuracy = metrics["patient_accuracy"]
            print(f"   ✅ Found patient_accuracy in parent run: {patient_accuracy:.4f}")
        else:
            # Try alternative names
            possible_names = [
                "patient_level_accuracy",
                "overall_patient_accuracy",
                "patient_metrics_accuracy",
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

        return patient_accuracy, run_id

    except Exception as e:
        print(f"   ❌ Error getting patient accuracy from MLflow: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def default_output_path(model_type):
    """Return a model-specific output CSV path."""
    return PROJECT_ROOT / "sweeps" / f"random_state_sweep_results_{model_type}.csv"

def main():
    parser = argparse.ArgumentParser(description="Sweep random_state values for model selected in config")
    parser.add_argument("--start", type=int, default=START_RANDOM_STATE, help="Starting random_state value")
    parser.add_argument("--min", type=int, default=MIN_RANDOM_STATE, help="Minimum random_state to check up to")
    parser.add_argument("--target", type=float, default=TARGET_ACCURACY, help="Target accuracy (default: 0.91)")
    parser.add_argument("--max", type=int, default=MAX_RANDOM_STATE, help="Maximum random_state (safety limit)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file (default: model-specific)")
    parser.add_argument("--config", type=str, default=CONFIG_FILE, help="Config file path")
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="MLflow experiment name (default: random_state_sweep_<model_type>)",
    )
    parser.add_argument(
        "--enable-feature-selection",
        dest="enable_feature_selection",
        action="store_true",
        default=ENABLE_FEATURE_SELECTION,
        help="Enable feature selection during training",
    )
    parser.add_argument(
        "--disable-feature-selection",
        dest="enable_feature_selection",
        action="store_false",
        help="Disable feature selection during training",
    )
    parser.add_argument(
        "--n-features-select",
        type=int,
        default=N_FEATURES_SELECT,
        help="Number of features to select when feature selection is enabled",
    )
    parser.add_argument(
        "--fs-method",
        type=str,
        default=FS_METHOD,
        help="Feature selection method",
    )

    args = parser.parse_args()

    # Resolve and load config
    resolved_config_path = resolve_config_path(args.config)
    config = load_config(resolved_config_path)
    model_type = detect_model_type(config)
    requested_experiment_name = args.experiment or f"{EXPERIMENT_NAME_PREFIX}_{model_type}"
    experiment_name = requested_experiment_name
    output_path = Path(args.output) if args.output else default_output_path(model_type)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure MLflow experiment exists and is active (without restoring deleted).
    try:
        experiment_name = ensure_mlflow_experiment_ready(experiment_name)
    except Exception as e:
        print(f"Note: MLflow experiment setup: {e}")
        experiment_name = requested_experiment_name

    print(f"Using config: {resolved_config_path}")
    print(f"Sweeping model_type: {model_type}")
    if experiment_name != requested_experiment_name:
        print(f"Requested MLflow experiment: {requested_experiment_name}")
        print(f"Using MLflow experiment: {experiment_name}")
    else:
        print(f"MLflow experiment: {experiment_name}")
    print(f"Output CSV: {output_path}")

    # Create or load results DataFrame
    results_file = output_path
    if results_file.exists():
        all_results_df = pd.read_csv(results_file)
        if "model_type" not in all_results_df.columns:
            all_results_df["model_type"] = model_type
        print(f"Loaded existing results: {len(all_results_df)} entries")
    else:
        all_results_df = pd.DataFrame(columns=["model_type", "random_state", "patient_accuracy", "run_id", "timestamp"])
        print("Starting new sweep")
    results_df = all_results_df[all_results_df["model_type"] == model_type].copy()

    # Determine starting point
    if not results_df.empty:
        completed_states = set(results_df["random_state"].astype(int))
        start_state = max(args.start, max(completed_states) + 1) if completed_states else args.start
        print(f"Resuming from random_state={start_state}")
    else:
        start_state = args.start

    # Sweep through random_state values
    found_target = False
    random_state = start_state

    while random_state <= args.max:
        # Skip if already completed
        if not results_df.empty and random_state in results_df["random_state"].astype(int).values:
            print(f"⏭️  Skipping random_state={random_state} (already completed)")
            random_state += 1
            continue

        print(f"\n{'='*80}")
        print(f"🔍 Testing random_state={random_state}")
        print(f"{'='*80}")

        # Update config
        config, model_updated = update_random_state(config, model_type, random_state, experiment_name)
        if not model_updated:
            print(
                f"   ⚠️  random_state field for '{model_type}' not found in config "
                "(deep_learning/model.params). Continuing with root random_seed + split random_state."
            )

        # Save config temporarily (or use a copy)
        temp_config = PROJECT_ROOT / f"temp_config_rs{random_state}_{model_type}.yaml"
        save_config(config, temp_config)

        try:
            # Set MLflow experiment for this run
            try:
                mlflow.set_experiment(experiment_name)
            except MlflowException as e:
                if "deleted experiment" in str(e).lower():
                    previous_experiment_name = experiment_name
                    experiment_name = ensure_mlflow_experiment_ready(experiment_name)
                    if experiment_name != previous_experiment_name:
                        print(f"   Switched to new MLflow experiment: {experiment_name}")
                        if "mlflow" not in config:
                            config["mlflow"] = {}
                        config["mlflow"]["experiment_name"] = experiment_name
                        save_config(config, temp_config)
                    mlflow.set_experiment(experiment_name)
                else:
                    raise

            # Run training with optional feature selection
            print(f"🚀 Starting training with random_state={random_state}...")
            print(f"   Configuration:")
            print(f"     - model_type: {model_type}")
            print(f"     - Feature selection: {args.enable_feature_selection}")
            if args.enable_feature_selection:
                print(f"     - N features: {args.n_features_select}")
                print(f"     - FS method: {args.fs_method}")
            success, stdout, stderr = run_training(
                model_type,
                temp_config,
                enable_feature_selection=args.enable_feature_selection,
                n_features_select=args.n_features_select,
                fs_method=args.fs_method,
            )

            if not success:
                print(f"❌ Training failed for random_state={random_state}")
                print(f"STDOUT (last 500 chars):\n{stdout[-500:]}")
                print(f"STDERR (last 500 chars):\n{stderr[-500:]}")
                # Still try to get results if available
                time.sleep(3)  # Wait for MLflow to sync
                patient_accuracy, run_id = get_patient_accuracy_from_mlflow(experiment_name)
            else:
                print(f"✅ Training completed for random_state={random_state}")
                # Wait a bit for MLflow to sync
                time.sleep(3)

                # Get patient accuracy from MLflow
                patient_accuracy, run_id = get_patient_accuracy_from_mlflow(experiment_name)

            if patient_accuracy is not None:
                print(f"📊 Patient accuracy: {patient_accuracy:.4f}")

                # Add to results
                new_row = pd.DataFrame({
                    "model_type": [model_type],
                    "random_state": [random_state],
                    "patient_accuracy": [patient_accuracy],
                    "run_id": [run_id if run_id else ""],
                    "timestamp": [pd.Timestamp.now()],
                })
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                all_results_df = pd.concat([all_results_df, new_row], ignore_index=True)

                # Save results incrementally
                all_results_df.to_csv(results_file, index=False)
                print(f"💾 Saved results to {results_file}")

                # Check if we hit target
                if patient_accuracy >= args.target:
                    print(f"\n🎉 TARGET REACHED! random_state={random_state} achieved {patient_accuracy:.4f} accuracy")
                    found_target = True
                    break
            else:
                print(f"⚠️  Could not retrieve patient accuracy for random_state={random_state}")
                # Still save a row with NaN
                new_row = pd.DataFrame({
                    "model_type": [model_type],
                    "random_state": [random_state],
                    "patient_accuracy": [None],
                    "run_id": [run_id if run_id else ""],
                    "timestamp": [pd.Timestamp.now()],
                })
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                all_results_df = pd.concat([all_results_df, new_row], ignore_index=True)
                all_results_df.to_csv(results_file, index=False)

        except Exception as e:
            print(f"❌ Error processing random_state={random_state}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Clean up temp config
            if temp_config.exists():
                temp_config.unlink()

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
        valid_results = results_df.dropna(subset=["patient_accuracy"])
        if not valid_results.empty:
            best_idx = valid_results["patient_accuracy"].idxmax()
            best_row = valid_results.loc[best_idx]
            print(f"\n🏆 Best result:")
            print(f"   random_state: {best_row['random_state']}")
            print(f"   patient_accuracy: {best_row['patient_accuracy']:.4f}")

            # Show top 5
            top5 = valid_results.nlargest(5, "patient_accuracy")
            print(f"\n📈 Top 5 random_states:")
            for idx, row in top5.iterrows():
                print(f"   random_state={int(row['random_state']):3d}: {row['patient_accuracy']:.4f}")

        print(f"\n💾 Results saved to: {results_file}")

    if found_target:
        print(f"\n✅ Target accuracy ({args.target}) was reached!")
    else:
        print(f"\n⚠️  Target accuracy ({args.target}) was not reached within random_state range [{start_state}, {random_state-1}]")

if __name__ == "__main__":
    main()
