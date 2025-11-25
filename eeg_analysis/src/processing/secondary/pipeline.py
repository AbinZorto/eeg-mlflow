import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple

import mlflow
import numpy as np

from src.utils.logger import setup_logger
from .loader import load_eeglab_run
from .resampler import resample_raw_to_target
from .saver import save_npz_run

logger = setup_logger(__name__)


def _discover_set_files(source_root: str) -> Dict[str, List[Path]]:
    """
    Recursively discover .set files and group by subject (e.g., sub-001).
    Sort runs per subject by file path name.
    """
    root = Path(source_root)
    if not root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")
    subject_to_files: Dict[str, List[Path]] = defaultdict(list)
    for fp in root.rglob("*.set"):
        subject_id = _infer_subject_from_path(fp)
        subject_to_files[subject_id].append(fp)
    # Sort runs for each subject
    for subject_id, files in subject_to_files.items():
        files.sort(key=lambda p: str(p))
    return subject_to_files


def _infer_subject_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("sub-"):
            return part
    stem = path.stem
    if "sub-" in stem:
        idx = stem.index("sub-")
        candidate = stem[idx:].split("_")[0]
        if candidate:
            return candidate
    return "unknown"


def _setup_mlflow_tracking(config: Dict[str, Any]) -> str:
    """
    Mimic run_pipeline style: environment variable overrides config; default to 'eeg_processing'.
    """
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
    if experiment_name is None:
        experiment_name = config.get("mlflow", {}).get("experiment_name", "eeg_processing")
        logger.info(f"Using experiment name from config: {experiment_name}")
    else:
        logger.info(f"Using experiment name from environment: {experiment_name}")
    try:
        mlflow.set_experiment(experiment_name)
        exp = mlflow.get_experiment_by_name(experiment_name)
        return exp.experiment_id if exp else ""
    except Exception as e:
        logger.warning(f"Failed to set MLflow experiment '{experiment_name}': {e}")
        return ""


def run(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrate the secondary EEG dataset build:
      - discover .set files
      - load each run
      - convert units (Volts→microvolts if needed)
      - resample to target sfreq
      - concatenate per subject
      - concatenate all subjects
      - save as .npz
      - log to MLflow
    """
    t0 = time.time()
    paths_cfg = config.get("paths", {})
    proc_cfg = config.get("processing", {})
    source_root = paths_cfg.get("source_root")
    output_npz = paths_cfg.get("output_npz")
    target_sfreq = float(proc_cfg.get("target_sampling_rate", 256))
    enable_uV = bool(proc_cfg.get("convert_to_microvolts", True))
    register_with_mlflow = bool(proc_cfg.get("register_with_mlflow", True))

    if not source_root or not output_npz:
        raise ValueError("Config must include paths.source_root and paths.output_npz")

    # MLflow setup similar to existing pipeline
    tracking_uri = config.get("mlflow", {}).get("tracking_uri", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = _setup_mlflow_tracking(config)

    # Start MLflow run at the beginning so it shows up immediately
    with mlflow.start_run(run_name="secondary_dataset_build"):
        # Log initial configuration parameters early
        mlflow.log_params(
            {
                "secondary.source_root": source_root,
                "secondary.output_npz": output_npz,
                "secondary.target_sampling_rate": target_sfreq,
                "secondary.convert_to_microvolts": enable_uV,
                "secondary.keep_all_channels": True,
                "secondary.register_with_mlflow": register_with_mlflow,
            }
        )
        mlflow.set_tag("dataset.context", "pretraining")
        mlflow.set_tag("dataset.type", "secondary_eeg_npz")
        mlflow.set_tag("dataset.version", "v1")

        # Discover files
        logger.info(f"Discovering .set files under: {source_root}")
        subject_to_files = _discover_set_files(source_root)
        num_subjects = len(subject_to_files)
        num_runs = sum(len(v) for v in subject_to_files.values())
        logger.info(f"Discovered subjects={num_subjects}, total runs={num_runs}")
        mlflow.log_param("secondary.num_subjects", num_subjects)
        mlflow.log_param("secondary.num_runs", num_runs)

        # Decide unit conversion for the entire dataset:
        # If enabled in config, force conversion of ALL runs to microvolts.
        global_convert_to_uV = bool(enable_uV)
        mlflow.log_param("secondary.global_convert_to_microvolts", bool(global_convert_to_uV))

        # Prepare output directory (use parent of configured output_npz for per-run files)
        output_base_dir = str(Path(output_npz).parent)
        mlflow.log_param("secondary.output_dir", output_base_dir)

        # Process each run
        run_process_start = time.time()
        runs_saved = 0
        for subject_id, files in subject_to_files.items():
            for fp in files:
                meta = load_eeglab_run(str(fp))

                # Extract numpy data
                raw = meta["raw"]
                data = raw.get_data()  # (n_channels, n_samples)

                # Unit conversion: apply global decision consistently across all runs
                if enable_uV and global_convert_to_uV:
                    raw._data = data * 1e6  # Volts → microvolts

                # Resample
                resampled = resample_raw_to_target(raw, target_sfreq)
                resampled_data = resampled.get_data().astype(np.float32, copy=False)

                # Save per run
                units = "uV" if (enable_uV and global_convert_to_uV) else "V"
                saved_file = save_npz_run(
                    output_dir=output_base_dir,
                    subject_id=subject_id,
                    run_name=meta["run_name"],
                    data=resampled_data,
                    channel_names=list(resampled.ch_names),
                    sampling_rate=target_sfreq,
                    original_sampling_rate=meta["original_sampling_rate"],
                    units=units,
                    metadata_version="v1",
                )
                runs_saved += 1
                # Log artifact for this run immediately
                try:
                    mlflow.log_artifact(saved_file, artifact_path=f"secondary_dataset/{subject_id}")
                except Exception as e:
                    logger.warning(f"Failed to log artifact for {saved_file}: {e}")

        # Final metrics
        duration_s = time.time() - t0
        build_duration_s = time.time() - run_process_start
        mlflow.log_metric("secondary.build_duration_s", build_duration_s)
        mlflow.log_metric("secondary.total_duration_s", duration_s)
        mlflow.set_tag("mlflow.dataset.logged", "true")
        mlflow.set_tag("mlflow.dataset.context", "pretraining")
        logger.info(
            f"Secondary per-run save complete: runs_saved={runs_saved}, "
            f"subjects={num_subjects}, output_dir={output_base_dir}, "
            f"duration_s={duration_s:.2f}"
        )

    return {
        "output_dir": output_base_dir,
        "num_subjects": num_subjects,
        "num_runs": num_runs,
        "sampling_rate": target_sfreq,
        "runs_saved": runs_saved,
        "duration_s": duration_s,
    }


