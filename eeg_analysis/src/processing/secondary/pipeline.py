import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple

import mlflow
import numpy as np
import pandas as pd
import yaml
import mne

from src.utils.logger import setup_logger
from .loader import load_run
from .resampler import resample_raw_to_target
from .saver import (
    build_run_windows_records,
    save_parquet_table,
    save_parquet_streaming,
    log_parquet_as_mlflow_dataset,
    save_parquet_run,
)

logger = setup_logger(__name__)


def _discover_eeg_files(source_root: str) -> Dict[str, List[Path]]:
    """
    Recursively discover .set/.vhdr files under ds* folders and group by subject (e.g., sub-001).
    Sort runs per subject by file path name.
    """
    root = Path(source_root)
    if not root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")
    ds_dirs = sorted(p for p in root.glob("ds*") if p.is_dir())
    if not ds_dirs:
        logger.warning(f"No ds* folders found under: {source_root}")
    subject_to_files: Dict[str, List[Path]] = defaultdict(list)
    for ds_dir in ds_dirs:
        for fp in ds_dir.rglob("*.set"):
            subject_id = _infer_subject_from_path(fp)
            subject_to_files[subject_id].append(fp)
        for fp in ds_dir.rglob("*.vhdr"):
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
      - discover .set/.vhdr files
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
    output_dir = paths_cfg.get("output_dir")
    target_sfreq = float(proc_cfg.get("target_sampling_rate", 256))
    enable_uV = bool(proc_cfg.get("convert_to_microvolts", True))
    register_with_mlflow = bool(proc_cfg.get("register_with_mlflow", True))

    if not source_root or not output_dir:
        raise ValueError("Config must include paths.source_root and paths.output_dir")

    # Load windowing parameters from processing_config.yaml (shared with primary pipeline)
    try:
        processing_cfg_path = Path("/home/abin/eeg-mlflow/eeg_analysis/configs/processing_config.yaml")
        with open(processing_cfg_path, "r") as f:
            processing_cfg = yaml.safe_load(f)
        window_seconds = float(processing_cfg["window_slicer"]["window_seconds"])
        overlap_seconds = float(processing_cfg["window_slicer"].get("overlap_seconds", 0))
    except Exception as e:
        logger.warning(f"Failed to load window settings from processing_config.yaml: {e}. Falling back to 10s, 0 overlap.")
        window_seconds = 10.0
        overlap_seconds = 0.0

    def _format_rate(rate: float) -> str:
        if float(rate).is_integer():
            return str(int(rate))
        return str(rate).rstrip("0").rstrip(".")

    def _format_window(seconds: float) -> str:
        if float(seconds).is_integer():
            return str(int(seconds))
        return str(seconds).rstrip("0").rstrip(".")

    window_length = int(window_seconds * target_sfreq)
    overlap_length = int(overlap_seconds * target_sfreq)
    step_size = max(1, window_length - overlap_length)  # guard

    output_base_dir = Path(output_dir)
    window_tag = f"sr{_format_rate(target_sfreq)}_ws{_format_window(window_seconds)}s"
    output_dir = str(output_base_dir / window_tag)

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
                "secondary.output_base_dir": str(output_base_dir),
                "secondary.output_dir": output_dir,
                "secondary.target_sampling_rate": target_sfreq,
                "secondary.convert_to_microvolts": enable_uV,
                "secondary.keep_all_channels": True,
                "secondary.register_with_mlflow": register_with_mlflow,
            }
        )
        mlflow.set_tag("dataset.context", "pretraining")
        mlflow.set_tag("dataset.type", "secondary_eeg_parquet_runs")
        mlflow.set_tag("dataset.version", "v1")

        # Discover files
        logger.info(f"Discovering .set/.vhdr files under ds* folders in: {source_root}")
        subject_to_files = _discover_eeg_files(source_root)
        num_subjects = len(subject_to_files)
        num_runs = sum(len(v) for v in subject_to_files.values())
        logger.info(f"Discovered subjects={num_subjects}, total runs={num_runs}")
        mlflow.log_param("secondary.num_subjects", num_subjects)
        mlflow.log_param("secondary.num_runs", num_runs)

        # Decide unit conversion for the entire dataset:
        # If enabled in config, force conversion of ALL runs to microvolts.
        global_convert_to_uV = bool(enable_uV)
        mlflow.log_param("secondary.global_convert_to_microvolts", bool(global_convert_to_uV))

        mlflow.log_param("secondary.output_base_dir", str(output_base_dir))
        mlflow.log_param("secondary.output_dir", output_dir)

        # Process each run into its own parquet (avoid giant single file)
        run_process_start = time.time()
        total_windows = 0
        runs_saved = 0
        runs_skipped = 0
        for subject_id, files in subject_to_files.items():
            for fp in files:
                run_name = Path(fp).stem
                expected_path = Path(output_dir) / subject_id / f"{run_name}.parquet"
                if expected_path.exists():
                    runs_skipped += 1
                    logger.info(
                        f"Skipping existing run parquet: subject={subject_id}, run={run_name}, path={expected_path}"
                    )
                    continue
                meta = load_run(str(fp))

                # Extract numpy data
                raw = meta["raw"]
                data = raw.get_data()  # (n_channels, n_samples)

                # Unit conversion: apply global decision consistently across all runs
                if enable_uV and global_convert_to_uV:
                    raw._data = data * 1e6  # Volts → microvolts

                # Resample
                resampled = resample_raw_to_target(raw, target_sfreq)
                resampled_data = resampled.get_data().astype(np.float32, copy=False)
                resampled_ch_names = [nm.upper() for nm in list(resampled.ch_names)]

                # Window and build DataFrame
                records = build_run_windows_records(
                    subject_id=subject_id,
                    run_name=meta["run_name"],
                    data=resampled_data,
                    channel_names=resampled_ch_names,
                    window_length=window_length,
                    step_size=step_size,
                )
                total_windows += len(records)
                run_df = pd.DataFrame(records)
                # Sort within run for consistency
                if not run_df.empty:
                    run_df = run_df.sort_values(
                        ["participant", "run", "parent_window_id", "sub_window_id"]
                    ).reset_index(drop=True)

                # Save per-run parquet under output_dir/<subject>/<run>.parquet
                abs_run_path = save_parquet_run(
                    output_dir=output_dir,
                    subject_id=subject_id,
                    run_name=meta["run_name"],
                    df=run_df,
                    channel_names=resampled_ch_names,
                    window_length=window_length,
                )
                runs_saved += 1
                # Log artifact for this run
                try:
                    mlflow.log_artifact(abs_run_path, artifact_path=f"secondary_runs/{subject_id}")
                except Exception as e:
                    logger.warning(f"Failed to log artifact for {abs_run_path}: {e}")

        # Final metrics
        duration_s = time.time() - t0
        build_duration_s = time.time() - run_process_start
        mlflow.log_metric("secondary.build_duration_s", build_duration_s)
        mlflow.log_metric("secondary.total_duration_s", duration_s)
        mlflow.log_metric("secondary.total_windows", total_windows)
        mlflow.log_metric("secondary.runs_skipped", runs_skipped)
        mlflow.log_param("secondary.window_seconds", window_seconds)
        mlflow.log_param("secondary.overlap_seconds", overlap_seconds)
        mlflow.log_param("secondary.window_length_samples", window_length)
        mlflow.set_tag("mlflow.dataset.logged", "true")
        mlflow.set_tag("mlflow.dataset.context", "pretraining")
        logger.info(
            f"Secondary per-run parquet save complete: runs_saved={runs_saved}, windows={total_windows}, "
            f"subjects={num_subjects}, output_dir={output_dir}, duration_s={duration_s:.2f}, "
            f"runs_skipped={runs_skipped}"
        )

    return {
        "output_dir": output_dir,
        "num_subjects": num_subjects,
        "num_runs": num_runs,
        "sampling_rate": target_sfreq,
        "runs_saved": runs_saved,
        "runs_skipped": runs_skipped,
        "total_windows": total_windows,
        "duration_s": duration_s,
    }
