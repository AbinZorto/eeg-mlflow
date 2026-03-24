#!/usr/bin/env python3
"""
Unified EEG experiment runner.

Features:
- Model-set based selection (traditional / boosting / deep_learning / all)
- Optional explicit model list override
- Optional feature-selection sweeps
- inner_k / outer_k forwarding to run_pipeline train
- Dataset auto-discovery per window size with optional channel filtering
- Auto-generated checkpoint/results files derived from normalized args
- Append-only JSONL results ledger with per-attempt metadata and MLflow snapshots
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_PIPELINE = REPO_ROOT / "eeg_analysis" / "run_pipeline.py"
FIND_DATASET_SCRIPT = REPO_ROOT / "scripts" / "find_dataset.py"
FILTER_DATASET_SCRIPT = REPO_ROOT / "scripts" / "filter_dataset.py"

EEG_ANALYSIS_ROOT = REPO_ROOT / "eeg_analysis"
if str(EEG_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(EEG_ANALYSIS_ROOT))

from src.utils.model_registry import get_model_metadata  # noqa: E402

ALL_FOUR_CHANNELS = {"af7", "af8", "tp9", "tp10"}
CANONICAL_MODEL_SETS = ("traditional", "boosting", "deep_learning")
CHECKPOINT_SCHEMA_VERSION = 1
RESULTS_SCHEMA_VERSION = 2
DEFAULT_ARTIFACTS_DIR = "sweeps/artifacts"
TRAIN_RUN_ID_PATTERN = re.compile(r"SUCCESS:TRAIN_RUN_ID:([A-Za-z0-9_-]+)")


@dataclass
class Job:
    model: str
    fs_enabled: bool
    fs_method: Optional[str] = None
    n_features: Optional[int] = None


@dataclass
class JobExecutionResult:
    return_code: int
    train_run_id: Optional[str]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def build_run_payload(
    *,
    config_path: Path,
    processing_config_path: Path,
    channels: Sequence[str],
    window_sizes: Sequence[int],
    model_sets: Sequence[str],
    models_override: Sequence[str],
    models: Sequence[str],
    mode: str,
    fs_methods: Sequence[str],
    feature_counts: Sequence[int],
    dataset_run_id: Optional[str],
    ordering: Optional[str],
    inner_k: Optional[int],
    outer_k: Optional[int],
    equalize_lopo_groups: Optional[str],
    use_smote: Optional[str],
    python_cmd: Sequence[str],
    jobs: Sequence[Job],
) -> Dict[str, Any]:
    normalized_jobs = sorted(
        [
            {
                "model": str(job.model),
                "fs_enabled": bool(job.fs_enabled),
                "fs_method": str(job.fs_method).lower() if job.fs_method is not None else None,
                "n_features": int(job.n_features) if job.n_features is not None else None,
            }
            for job in jobs
        ],
        key=stable_json,
    )

    return {
        "script": "scripts/run_experiments.py",
        "script_version": 2,
        "config": str(config_path.resolve()),
        "processing_config": str(processing_config_path.resolve()),
        "channels": sorted({str(ch).lower() for ch in channels}),
        "window_sizes": sorted({int(ws) for ws in window_sizes}),
        "model_sets": sorted({str(m).strip().lower() for m in model_sets}),
        "models_override": sorted({str(m).strip() for m in models_override}),
        "models_resolved": sorted({str(m).strip() for m in models}),
        "mode": str(mode).lower(),
        "fs_methods": sorted({str(method).strip().lower() for method in fs_methods}),
        "feature_counts": sorted({int(n) for n in feature_counts}),
        "dataset_run_id": dataset_run_id if dataset_run_id else "auto_dataset",
        "ordering": str(ordering).lower() if ordering else "any",
        "inner_k": int(inner_k) if inner_k is not None else None,
        "outer_k": int(outer_k) if outer_k is not None else None,
        "equalize_lopo_groups": str(equalize_lopo_groups).lower() if equalize_lopo_groups is not None else None,
        "use_smote": str(use_smote).lower() if use_smote is not None else None,
        "python_cmd": list(python_cmd),
        "jobs": normalized_jobs,
    }


def compute_run_signature(payload: Dict[str, Any]) -> str:
    digest = hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()
    return digest


def resolve_output_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def slugify(value: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower())
    slug = slug.strip("-_.")
    if not slug:
        slug = "na"
    return slug[:max_len]


def list_token(prefix: str, values: Sequence[Any], max_items: int = 4) -> str:
    if not values:
        return f"{prefix}-none"
    shown = [slugify(str(v), 24) for v in values[:max_items]]
    token = "-".join(shown)
    extra = len(values) - len(shown)
    if extra > 0:
        token = f"{token}-plus{extra}"
    return f"{prefix}-{token}"


def build_artifact_stem(payload: Dict[str, Any]) -> str:
    components = [
        "runexp",
        f"mode-{slugify(str(payload.get('mode', 'na')), 20)}",
        list_token("sets", payload.get("model_sets", []), max_items=3),
        list_token("models", payload.get("models_resolved", []), max_items=3),
        list_token("ws", payload.get("window_sizes", []), max_items=5),
        f"ord-{slugify(str(payload.get('ordering', 'any')), 20)}",
        f"ik-{payload.get('inner_k') if payload.get('inner_k') is not None else 'na'}",
        f"ok-{payload.get('outer_k') if payload.get('outer_k') is not None else 'na'}",
        f"eq-{payload.get('equalize_lopo_groups') if payload.get('equalize_lopo_groups') is not None else 'cfg'}",
        f"sm-{payload.get('use_smote') if payload.get('use_smote') is not None else 'cfg'}",
        f"ds-{'pinned' if payload.get('dataset_run_id') not in (None, 'auto_dataset') else 'auto'}",
    ]
    mode = str(payload.get("mode", "")).lower()
    if mode in {"fs", "both"}:
        components.append(list_token("fsm", payload.get("fs_methods", []), max_items=2))
        components.append(list_token("fn", payload.get("feature_counts", []), max_items=4))
    stem = "-".join(components)
    return slugify(stem, max_len=180)


def derive_artifact_paths(
    *,
    payload: Dict[str, Any],
    run_signature: str,
    artifacts_dir_raw: str,
) -> Tuple[Path, Path]:
    artifacts_dir = resolve_output_path(artifacts_dir_raw)
    stem = build_artifact_stem(payload)
    suffix = run_signature[:12]
    checkpoint_path = artifacts_dir / f"{stem}__{suffix}.checkpoint.json"
    results_path = artifacts_dir / f"{stem}__{suffix}.results.jsonl"
    return checkpoint_path, results_path


def load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "updated_at": utc_now_iso(),
            "runs": {},
        }

    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        eprint(f"[checkpoint] failed to parse {path}: {exc}; starting with empty checkpoint")
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "updated_at": utc_now_iso(),
            "runs": {},
        }

    if data.get("schema_version") != CHECKPOINT_SCHEMA_VERSION or not isinstance(data.get("runs"), dict):
        eprint(
            f"[checkpoint] incompatible schema in {path}; expected schema_version={CHECKPOINT_SCHEMA_VERSION}. "
            "Starting with empty checkpoint state."
        )
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "updated_at": utc_now_iso(),
            "runs": {},
        }

    return data


def save_checkpoint(path: Path, checkpoint: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint["schema_version"] = CHECKPOINT_SCHEMA_VERSION
    checkpoint["updated_at"] = utc_now_iso()
    path.write_text(stable_json(checkpoint) + "\n")


def append_jsonl_record(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(stable_json(record) + "\n")


def load_attempt_counters(path: Path, run_signature: str) -> Dict[str, int]:
    counters: Dict[str, int] = {}
    if not path.exists():
        return counters

    with path.open("r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue

            if record.get("record_type") != "job_attempt":
                continue
            if record.get("run_signature") != run_signature:
                continue

            job_key = record.get("job_key")
            attempt_index = record.get("attempt_index")
            if isinstance(job_key, str) and isinstance(attempt_index, int):
                counters[job_key] = max(counters.get(job_key, 0), int(attempt_index))

    return counters


def ensure_run_state(
    checkpoint: Dict[str, Any],
    run_signature: str,
    payload: Dict[str, Any],
    expected_jobs: int,
) -> Dict[str, Any]:
    runs = checkpoint.setdefault("runs", {})
    now_iso = utc_now_iso()
    state = runs.get(run_signature)
    if not isinstance(state, dict):
        state = {
            "created_at": now_iso,
            "updated_at": now_iso,
            "payload": payload,
            "expected_jobs": int(expected_jobs),
            "completed_jobs": [],
            "failed_jobs": {},
            "status": "running",
        }
        runs[run_signature] = state

    state.setdefault("payload", payload)
    state.setdefault("expected_jobs", int(expected_jobs))
    state.setdefault("completed_jobs", [])
    state.setdefault("failed_jobs", {})
    state.setdefault("status", "running")
    state["updated_at"] = now_iso
    return state


def reset_run_state(run_state: Dict[str, Any], expected_jobs: int) -> None:
    now_iso = utc_now_iso()
    run_state["updated_at"] = now_iso
    run_state["expected_jobs"] = int(expected_jobs)
    run_state["completed_jobs"] = []
    run_state["failed_jobs"] = {}
    run_state["status"] = "running"


def make_job_key(window_seconds: int, dataset_run_id: Optional[str], job: Job) -> str:
    job_payload = {
        "window_seconds": int(window_seconds),
        "dataset_run_id": dataset_run_id,
        "model": job.model,
        "fs_enabled": bool(job.fs_enabled),
        "fs_method": job.fs_method,
        "n_features": job.n_features,
    }
    return stable_json(job_payload)


def eprint(message: str) -> None:
    print(message, file=sys.stderr)


def parse_csv_list(raw: Optional[str], cast=str) -> List:
    if raw is None:
        return []
    items = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(cast(part))
    return items


def dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def parse_window_sizes(processing_cfg: dict, override: Optional[str]) -> List[int]:
    if override:
        values = parse_csv_list(override, int)
        if not values:
            raise ValueError("window_sizes override is empty after parsing.")
        return values

    window_cfg = processing_cfg["window_slicer"]["window_seconds"]
    if isinstance(window_cfg, list):
        return [int(x) for x in window_cfg]
    return [int(window_cfg)]


def get_channels(processing_cfg: dict) -> List[str]:
    channels = processing_cfg["data_loader"]["channels"]
    return [str(ch).lower() for ch in channels]


def has_all_four_channels(channels: Sequence[str]) -> bool:
    return set(channels) == ALL_FOUR_CHANNELS


def resolve_models(
    model_sets_raw: str,
    explicit_models_raw: Optional[str],
    model_metadata: Dict[str, Dict[str, object]],
) -> List[str]:
    available_models = [
        model_name
        for model_name, info in model_metadata.items()
        if bool(info.get("enabled", True))
    ]

    if explicit_models_raw:
        explicit_models = parse_csv_list(explicit_models_raw, str)
        if not explicit_models:
            raise ValueError("--models was provided but no valid model names were parsed.")
        explicit_models = dedupe_preserve_order(explicit_models)
        unknown = [model_name for model_name in explicit_models if model_name not in available_models]
        if unknown:
            raise ValueError(
                f"Unknown or unsupported model(s): {unknown}. "
                f"Available models: {', '.join(available_models)}"
            )
        return explicit_models

    requested_sets = parse_csv_list(model_sets_raw, str)
    if not requested_sets:
        requested_sets = ["all"]
    requested_sets = [set_name.lower() for set_name in requested_sets]

    valid_sets = set(CANONICAL_MODEL_SETS) | {"all"}
    unknown = [s for s in requested_sets if s not in valid_sets]
    if unknown:
        valid = ", ".join(sorted(valid_sets))
        raise ValueError(f"Unknown model set(s): {unknown}. Valid model sets: {valid}")

    if "all" in requested_sets:
        if not available_models:
            raise ValueError("No enabled models found in training config.")
        return available_models

    models: List[str] = []
    for set_name in requested_sets:
        set_models = [
            model_name
            for model_name in available_models
            if str(model_metadata.get(model_name, {}).get("family", "")).lower() == set_name
        ]
        models.extend(set_models)

    models = dedupe_preserve_order(models)
    if not models:
        raise ValueError(
            f"No models resolved for --model-sets '{model_sets_raw}'. "
            "Check model_families/model_registry in config."
        )
    return models


def experiment_name_for_model(
    model: str, fs_enabled: bool, model_metadata: Dict[str, Dict[str, object]]
) -> str:
    suffix = "feature_selection" if fs_enabled else "baseline"
    model_info = model_metadata.get(model, {})
    base_experiment = str(model_info.get("experiment_group", "eeg_other_models"))
    if base_experiment.endswith(f"_{suffix}"):
        return base_experiment
    return f"{base_experiment}_{suffix}"


def run_and_capture(cmd: Sequence[str], cwd: Path) -> Tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def parse_success_line(output: str) -> Optional[List[str]]:
    for line in output.splitlines():
        if line.startswith("SUCCESS:"):
            return line.split(":", 2)
    return None


def discover_base_dataset_run_id(
    python_cmd: Sequence[str], window_seconds: int, ordering: Optional[str]
) -> Optional[str]:
    cmd = list(python_cmd) + [str(FIND_DATASET_SCRIPT), str(window_seconds)]
    if ordering:
        cmd.append(ordering)

    code, stdout, stderr = run_and_capture(cmd, REPO_ROOT)
    if code != 0:
        eprint(f"[dataset] find_dataset.py failed for {window_seconds}s")
        if stderr.strip():
            eprint(stderr.strip())
        if stdout.strip():
            eprint(stdout.strip())
        return None

    parsed = parse_success_line(stdout)
    if not parsed:
        eprint(f"[dataset] no matching 4-channel dataset found for {window_seconds}s")
        if stdout.strip():
            eprint(stdout.strip())
        return None

    # SUCCESS:<run_id>:<dataset_name>
    return parsed[1]


def filter_dataset_for_channels(
    python_cmd: Sequence[str],
    base_run_id: str,
    channels: Sequence[str],
    window_seconds: int,
) -> Optional[str]:
    channels_arg = " ".join(channels)
    cmd = list(python_cmd) + [
        str(FILTER_DATASET_SCRIPT),
        base_run_id,
        channels_arg,
        str(window_seconds),
    ]
    code, stdout, stderr = run_and_capture(cmd, REPO_ROOT)
    if code != 0:
        eprint(f"[dataset] filter_dataset.py failed for {window_seconds}s channels={channels_arg}")
        if stderr.strip():
            eprint(stderr.strip())
        if stdout.strip():
            eprint(stdout.strip())
        return None

    parsed = parse_success_line(stdout)
    if not parsed:
        eprint(f"[dataset] filter_dataset.py returned no SUCCESS line for {window_seconds}s")
        if stdout.strip():
            eprint(stdout.strip())
        if stderr.strip():
            eprint(stderr.strip())
        return None
    return parsed[1]


def resolve_dataset_run_id_for_window(
    args: argparse.Namespace,
    python_cmd: Sequence[str],
    channels: Sequence[str],
    window_seconds: int,
) -> Optional[str]:
    if args.dry_run and not args.dataset_run_id:
        return f"<auto_dataset_for_{window_seconds}s>"

    if args.dataset_run_id:
        return args.dataset_run_id

    base_run = discover_base_dataset_run_id(
        python_cmd=python_cmd,
        window_seconds=window_seconds,
        ordering=args.ordering,
    )
    if base_run is None:
        return None

    if has_all_four_channels(channels):
        return base_run

    return filter_dataset_for_channels(
        python_cmd=python_cmd,
        base_run_id=base_run,
        channels=channels,
        window_seconds=window_seconds,
    )


def build_jobs(
    models: Sequence[str],
    mode: str,
    fs_methods: Sequence[str],
    feature_counts: Sequence[int],
) -> List[Job]:
    jobs: List[Job] = []
    if mode in {"baseline", "both"}:
        for model in models:
            jobs.append(Job(model=model, fs_enabled=False))
    if mode in {"fs", "both"}:
        for model in models:
            for fs_method in fs_methods:
                for n_features in feature_counts:
                    jobs.append(
                        Job(
                            model=model,
                            fs_enabled=True,
                            fs_method=fs_method,
                            n_features=n_features,
                        )
                    )
    return jobs


def build_train_cmd(
    python_cmd: Sequence[str],
    args: argparse.Namespace,
    job: Job,
    window_seconds: int,
    dataset_run_id: Optional[str],
) -> List[str]:
    cmd = list(python_cmd) + [
        str(RUN_PIPELINE),
        "--config",
        args.config,
        "train",
        "--window-size",
        str(window_seconds),
        "--model-type",
        job.model,
    ]

    if dataset_run_id:
        cmd.extend(["--use-dataset-from-run", dataset_run_id])

    if job.fs_enabled:
        cmd.extend(
            [
                "--enable-feature-selection",
                "--n-features-select",
                str(job.n_features),
                "--fs-method",
                str(job.fs_method),
            ]
        )

    if args.inner_k is not None:
        cmd.extend(["--inner-k", str(args.inner_k)])
    if args.outer_k is not None:
        cmd.extend(["--outer-k", str(args.outer_k)])
    if args.equalize_lopo_groups is not None:
        cmd.extend(["--equalize-lopo-groups", str(args.equalize_lopo_groups)])
    if args.use_smote is not None:
        cmd.extend(["--use-smote", str(args.use_smote)])

    return cmd


def run_job(
    cmd: Sequence[str], env: Dict[str, str], dry_run: bool
) -> JobExecutionResult:
    cmd_str = " ".join(shlex.quote(part) for part in cmd)
    prefix = f"MLFLOW_EXPERIMENT_NAME={env.get('MLFLOW_EXPERIMENT_NAME', '')} "
    print(f"$ {prefix}{cmd_str}")
    if dry_run:
        return JobExecutionResult(return_code=0, train_run_id=None)

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    train_run_id: Optional[str] = None
    if proc.stdout is not None:
        for line in proc.stdout:
            print(line, end="")
            marker_match = TRAIN_RUN_ID_PATTERN.search(line)
            if marker_match:
                train_run_id = marker_match.group(1)

    return_code = proc.wait()
    return JobExecutionResult(return_code=return_code, train_run_id=train_run_id)


def fetch_mlflow_run_snapshot(
    run_id: Optional[str],
    tracking_uri: Optional[str],
) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "mlflow_run_id": run_id,
        "mlflow_run_name": None,
        "mlflow_experiment_id": None,
        "mlflow_metrics": {},
        "mlflow_params": {},
        "mlflow_tags": {},
    }
    if not run_id:
        return snapshot

    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        client = MlflowClient()
        run = client.get_run(run_id)
        snapshot["mlflow_experiment_id"] = run.info.experiment_id
        snapshot["mlflow_run_name"] = run.data.tags.get("mlflow.runName")
        snapshot["mlflow_metrics"] = {str(k): float(v) for k, v in run.data.metrics.items()}
        snapshot["mlflow_params"] = {str(k): str(v) for k, v in run.data.params.items()}
        snapshot["mlflow_tags"] = {str(k): str(v) for k, v in run.data.tags.items()}
    except Exception as exc:
        snapshot["mlflow_fetch_error"] = str(exc)
        eprint(f"[results] failed to fetch MLflow run snapshot for run_id={run_id}: {exc}")
    return snapshot


def _safe_metric_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric == numeric and numeric not in (float("inf"), float("-inf")):
            return numeric
    return None


def _metric_value(metrics: Dict[str, Any], *names: str) -> Optional[float]:
    for name in names:
        value = _safe_metric_number(metrics.get(name))
        if value is not None:
            return value
    return None


def _parse_bool_param(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def _parse_int_param(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _parse_float_param(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(str(value))
    except (TypeError, ValueError):
        return None
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        return None
    return numeric


def build_normalized_metrics_snapshot(
    mlflow_metrics: Dict[str, Any],
    mlflow_params: Dict[str, Any],
) -> Dict[str, Any]:
    if not mlflow_metrics and not mlflow_params:
        return {}

    primary_metric_name = str(
        mlflow_params.get("primary_metric_name", "balanced_accuracy")
    )
    patient = {
        "role": "primary",
        "primary_metric_name": primary_metric_name,
        "accuracy": _metric_value(mlflow_metrics, "patient_accuracy"),
        "balanced_accuracy": _metric_value(mlflow_metrics, "patient_balanced_accuracy"),
        "precision": _metric_value(mlflow_metrics, "patient_precision"),
        "recall": _metric_value(mlflow_metrics, "patient_recall"),
        "sensitivity": _metric_value(mlflow_metrics, "patient_sensitivity", "patient_recall"),
        "specificity": _metric_value(mlflow_metrics, "patient_specificity"),
        "f1": _metric_value(mlflow_metrics, "patient_f1"),
        "roc_auc": _metric_value(mlflow_metrics, "patient_roc_auc"),
        "pr_auc": _metric_value(mlflow_metrics, "patient_pr_auc"),
        "npv": _metric_value(mlflow_metrics, "patient_npv"),
        "mcc": _metric_value(mlflow_metrics, "patient_mcc"),
        "true_positives": _metric_value(mlflow_metrics, "patient_true_positives"),
        "true_negatives": _metric_value(mlflow_metrics, "patient_true_negatives"),
        "false_positives": _metric_value(mlflow_metrics, "patient_false_positives"),
        "false_negatives": _metric_value(mlflow_metrics, "patient_false_negatives"),
        "n_patients": _metric_value(mlflow_metrics, "patient_n_patients"),
    }
    patient["primary_metric_value"] = patient.get(primary_metric_name)

    window = {
        "role": str(mlflow_params.get("window_metrics_role", "supporting")),
        "accuracy": _metric_value(mlflow_metrics, "window_accuracy", "avg_window_accuracy"),
        "f1": _metric_value(mlflow_metrics, "window_f1"),
        "roc_auc": _metric_value(mlflow_metrics, "window_roc_auc"),
        "n_windows": _metric_value(mlflow_metrics, "window_n_windows"),
    }

    confidence_intervals = {}
    for metric_name in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "roc_auc"):
        confidence_intervals[metric_name] = {
            "low": _metric_value(mlflow_metrics, f"patient_{metric_name}_ci_low"),
            "high": _metric_value(mlflow_metrics, f"patient_{metric_name}_ci_high"),
            "std": _metric_value(mlflow_metrics, f"patient_{metric_name}_ci_std"),
        }

    permutation_metric = str(
        mlflow_params.get("metrics_reporting_permutation_metric", primary_metric_name)
    )
    stats = {
        "bootstrap_ci_enabled": _parse_bool_param(mlflow_params.get("metrics_reporting_bootstrap_ci_enabled")),
        "bootstrap_ci_level": _parse_float_param(mlflow_params.get("metrics_reporting_bootstrap_ci_level")),
        "bootstrap_iterations": _parse_int_param(mlflow_params.get("metrics_reporting_bootstrap_iterations")),
        "permutation_test_enabled": _parse_bool_param(mlflow_params.get("metrics_reporting_permutation_test_enabled")),
        "permutation_metric": permutation_metric,
        "permutation_iterations": _parse_int_param(mlflow_params.get("metrics_reporting_permutation_iterations")),
        "confidence_intervals": confidence_intervals,
        "permutation_test": {
            "p_value": _metric_value(mlflow_metrics, f"patient_{permutation_metric}_permutation_pvalue"),
            "observed": _metric_value(mlflow_metrics, f"patient_{permutation_metric}_permutation_observed"),
            "null_mean": _metric_value(mlflow_metrics, f"patient_{permutation_metric}_permutation_null_mean"),
            "null_std": _metric_value(mlflow_metrics, f"patient_{permutation_metric}_permutation_null_std"),
        },
        "fold_summary": {
            "patient_accuracy_mean": _metric_value(mlflow_metrics, "patient_fold_accuracy_mean", "patient_accuracy"),
            "patient_accuracy_std": _metric_value(mlflow_metrics, "patient_fold_accuracy_std"),
            "patient_fold_count": _metric_value(mlflow_metrics, "patient_fold_count"),
            "window_accuracy_mean": _metric_value(mlflow_metrics, "window_fold_accuracy_mean", "window_accuracy", "avg_window_accuracy"),
            "window_accuracy_std": _metric_value(mlflow_metrics, "window_fold_accuracy_std"),
            "window_fold_count": _metric_value(mlflow_metrics, "window_fold_count"),
        },
    }

    feature_selection = {
        "feature_set_count": _metric_value(mlflow_metrics, "feature_selection_feature_set_count"),
        "average_features_per_fold": _metric_value(mlflow_metrics, "feature_selection_average_features_per_fold"),
        "mean_pairwise_jaccard": _metric_value(mlflow_metrics, "feature_selection_mean_pairwise_jaccard"),
        "median_pairwise_jaccard": _metric_value(mlflow_metrics, "feature_selection_median_pairwise_jaccard"),
        "unique_feature_count": _metric_value(mlflow_metrics, "feature_selection_unique_feature_count"),
        "top_feature_frequency": _metric_value(mlflow_metrics, "feature_selection_top_feature_frequency"),
        "top_feature_share": _metric_value(mlflow_metrics, "feature_selection_top_feature_share"),
    }

    return {
        "patient": patient,
        "window": window,
        "stats": stats,
        "feature_selection": feature_selection,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified experiment runner with model-set support plus per-fold and consensus feature-count overrides."
    )
    parser.add_argument(
        "--config",
        default="eeg_analysis/configs/model_config.yaml",
        help="Path to training config passed to run_pipeline.py",
    )
    parser.add_argument(
        "--processing-config",
        default="eeg_analysis/configs/processing_config.yaml",
        help="Path to processing config for channels and window sizes",
    )
    parser.add_argument(
        "--model-sets",
        default="all",
        help="Comma-separated model sets (traditional,boosting,deep_learning,all), resolved from training config metadata. Ignored if --models is set.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated explicit models to run (overrides --model-sets, validated against training config).",
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=["baseline", "fs", "both"],
        help="Run baseline only, feature-selection only, or both.",
    )
    parser.add_argument(
        "--fs-methods",
        default="select_k_best_f_classif,select_k_best_mutual_info",
        help="Comma-separated FS methods for --mode fs/both.",
    )
    parser.add_argument(
        "--feature-counts",
        default="10,15,20",
        help="Comma-separated feature counts for --mode fs/both.",
    )
    parser.add_argument(
        "--window-sizes",
        default=None,
        help="Comma-separated window sizes. If omitted, read from processing config.",
    )
    parser.add_argument(
        "--dataset-run-id",
        default=None,
        help="Use this dataset run id for all windows (skip auto-discovery/filtering).",
    )
    parser.add_argument(
        "--ordering",
        default=None,
        choices=["sequential", "completion"],
        help="Dataset ordering to require during auto-discovery.",
    )
    parser.add_argument(
        "--inner-k",
        type=int,
        default=None,
        help="Forwarded to run_pipeline train --inner-k (features selected within each outer LOPO training fold).",
    )
    parser.add_argument(
        "--outer-k",
        type=int,
        default=None,
        help="Forwarded to run_pipeline train --outer-k (final consensus feature count from correct outer folds).",
    )
    parser.add_argument(
        "--equalize-lopo-groups",
        choices=["true", "false"],
        default=None,
        help="Forwarded to run_pipeline train --equalize-lopo-groups to override non-held-out group balancing.",
    )
    parser.add_argument(
        "--use-smote",
        choices=["true", "false"],
        default=None,
        help="Forwarded to run_pipeline train --use-smote to override SMOTE usage.",
    )
    parser.add_argument(
        "--python-cmd",
        default="uv run python3",
        help="Python command prefix for helper scripts and run_pipeline invocations.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=DEFAULT_ARTIFACTS_DIR,
        help="Base directory used for auto-generated checkpoint/results files.",
    )
    parser.add_argument(
        "--checkpoint-file",
        default=None,
        help="Checkpoint ledger path. If omitted, an argument-derived path is auto-generated.",
    )
    parser.add_argument(
        "--disable-checkpoint",
        action="store_true",
        help="Disable checkpoint read/write and always run every expanded job.",
    )
    parser.add_argument(
        "--results-file",
        default=None,
        help="Results ledger JSONL path. If omitted, an argument-derived path is auto-generated.",
    )
    parser.add_argument(
        "--disable-results-ledger",
        action="store_true",
        help="Disable results ledger writes.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore completed jobs in checkpoint for this invocation (still updates checkpoint with new results).",
    )
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Clear checkpoint state for this exact command signature before running.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if one experiment fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)

    if args.inner_k is not None and args.inner_k < 1:
        raise ValueError("--inner-k must be >= 1 when provided.")
    if args.outer_k is not None and args.outer_k < 1:
        raise ValueError("--outer-k must be >= 1 when provided.")

    processing_cfg_path = resolve_output_path(args.processing_config)
    if not processing_cfg_path.exists():
        raise FileNotFoundError(f"Processing config not found: {processing_cfg_path}")
    train_cfg_path = resolve_output_path(args.config)
    if not train_cfg_path.exists():
        raise FileNotFoundError(f"Training config not found: {train_cfg_path}")

    processing_cfg = load_yaml(processing_cfg_path)
    train_cfg = load_yaml(train_cfg_path)
    channels = get_channels(processing_cfg)
    window_sizes = parse_window_sizes(processing_cfg, args.window_sizes)
    model_metadata = get_model_metadata(train_cfg)
    models = resolve_models(args.model_sets, args.models, model_metadata)

    fs_methods = parse_csv_list(args.fs_methods, str)
    feature_counts = parse_csv_list(args.feature_counts, int)
    if args.mode in {"fs", "both"}:
        if not fs_methods:
            raise ValueError("--fs-methods produced an empty list.")
        if not feature_counts:
            raise ValueError("--feature-counts produced an empty list.")

    jobs = build_jobs(models, args.mode, fs_methods, feature_counts)
    python_cmd = shlex.split(args.python_cmd)

    model_sets_for_payload = parse_csv_list(args.model_sets, str)
    models_override_for_payload = parse_csv_list(args.models, str)
    effective_model_sets_for_payload = [] if models_override_for_payload else model_sets_for_payload
    effective_fs_methods_for_payload = fs_methods if args.mode in {"fs", "both"} else []
    effective_feature_counts_for_payload = feature_counts if args.mode in {"fs", "both"} else []
    effective_ordering_for_payload = args.ordering if not args.dataset_run_id else None

    run_payload = build_run_payload(
        config_path=train_cfg_path,
        processing_config_path=processing_cfg_path,
        channels=channels,
        window_sizes=window_sizes,
        model_sets=effective_model_sets_for_payload,
        models_override=models_override_for_payload,
        models=models,
        mode=args.mode,
        fs_methods=effective_fs_methods_for_payload,
        feature_counts=effective_feature_counts_for_payload,
        dataset_run_id=args.dataset_run_id,
        ordering=effective_ordering_for_payload,
        inner_k=args.inner_k,
        outer_k=args.outer_k,
        equalize_lopo_groups=args.equalize_lopo_groups,
        use_smote=args.use_smote,
        python_cmd=python_cmd,
        jobs=jobs,
    )
    run_signature = compute_run_signature(run_payload)
    expected_jobs = len(jobs) * len(window_sizes)

    auto_checkpoint_path, auto_results_path = derive_artifact_paths(
        payload=run_payload,
        run_signature=run_signature,
        artifacts_dir_raw=args.artifacts_dir,
    )

    checkpoint_enabled = not args.disable_checkpoint
    checkpoint_path = resolve_output_path(args.checkpoint_file) if args.checkpoint_file else auto_checkpoint_path
    results_enabled = not args.disable_results_ledger
    results_path = resolve_output_path(args.results_file) if args.results_file else auto_results_path
    results_write_enabled = results_enabled and not args.dry_run
    tracking_uri = train_cfg.get("mlflow", {}).get("tracking_uri")

    checkpoint: Optional[Dict[str, Any]] = None
    run_state: Optional[Dict[str, Any]] = None
    completed_job_keys: set[str] = set()
    failed_jobs: Dict[str, Any] = {}
    attempt_counters: Dict[str, int] = {}

    if results_write_enabled:
        attempt_counters = load_attempt_counters(results_path, run_signature)

    if checkpoint_enabled:
        checkpoint = load_checkpoint(checkpoint_path)
        run_state = ensure_run_state(
            checkpoint=checkpoint,
            run_signature=run_signature,
            payload=run_payload,
            expected_jobs=expected_jobs,
        )
        if args.reset_checkpoint:
            reset_run_state(run_state, expected_jobs=expected_jobs)
            save_checkpoint(checkpoint_path, checkpoint)

        completed_job_keys = set(run_state.get("completed_jobs", []))
        failed_jobs = dict(run_state.get("failed_jobs", {}))

    print("=== Unified Experiment Runner ===")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Training config: {args.config}")
    print(f"Processing config: {args.processing_config}")
    print(f"Channels: {channels}")
    print(f"Window sizes: {window_sizes}")
    print(f"Models ({len(models)}): {models}")
    print(f"Mode: {args.mode}")
    if args.mode in {"fs", "both"}:
        print(f"FS methods: {fs_methods}")
        print(f"Feature counts: {feature_counts}")
    print(f"inner_k: {args.inner_k}")
    print(f"outer_k: {args.outer_k}")
    print(f"equalize_lopo_groups: {args.equalize_lopo_groups if args.equalize_lopo_groups is not None else '<config>'}")
    print(f"use_smote: {args.use_smote if args.use_smote is not None else '<config>'}")
    print(f"Dry run: {args.dry_run}")
    print(f"Total jobs: {expected_jobs}")
    print(f"Run signature: {run_signature[:12]}")
    print(f"Artifacts dir: {resolve_output_path(args.artifacts_dir)}")
    if checkpoint_enabled:
        print(f"Checkpoint file: {checkpoint_path}")
        print(f"Completed in checkpoint: {len(completed_job_keys)}")
        if args.no_resume:
            print("Resume mode: disabled (--no-resume)")
    else:
        print("Checkpoint file: disabled (--disable-checkpoint)")
    if results_enabled:
        print(f"Results file: {results_path}")
    else:
        print("Results file: disabled (--disable-results-ledger)")
    print()

    success = 0
    failed = 0
    skipped = 0
    resumed = 0
    active_job_context: Optional[Dict[str, Any]] = None

    def append_job_attempt_record(
        *,
        job: Job,
        job_key: str,
        status: str,
        window_seconds: int,
        dataset_run_id: Optional[str],
        experiment_name: str,
        cmd: Optional[Sequence[str]],
        return_code: Optional[int],
        train_run_id: Optional[str],
    ) -> None:
        if not results_write_enabled:
            return

        attempt_index = attempt_counters.get(job_key, 0) + 1
        attempt_counters[job_key] = attempt_index
        record: Dict[str, Any] = {
            "schema_version": RESULTS_SCHEMA_VERSION,
            "record_type": "job_attempt",
            "created_at": utc_now_iso(),
            "run_signature": run_signature,
            "attempt_index": attempt_index,
            "job_key": job_key,
            "status": status,
            "return_code": int(return_code) if return_code is not None else None,
            "window_seconds": int(window_seconds),
            "dataset_run_id": dataset_run_id,
            "model": job.model,
            "fs_enabled": bool(job.fs_enabled),
            "fs_method": job.fs_method,
            "n_features": job.n_features,
            "inner_k": args.inner_k,
            "outer_k": args.outer_k,
            "equalize_lopo_groups": args.equalize_lopo_groups,
            "use_smote": args.use_smote,
            "ordering": args.ordering,
            "command": list(cmd) if cmd is not None else None,
            "mlflow_experiment_name": experiment_name,
        }
        snapshot = fetch_mlflow_run_snapshot(train_run_id, tracking_uri)
        record.update(snapshot)
        record["metrics"] = build_normalized_metrics_snapshot(
            snapshot.get("mlflow_metrics", {}),
            snapshot.get("mlflow_params", {}),
        )
        append_jsonl_record(results_path, record)

    try:
        for window_seconds in window_sizes:
            print(f"=== Window {window_seconds}s ===")
            dataset_run_id = resolve_dataset_run_id_for_window(
                args=args,
                python_cmd=python_cmd,
                channels=channels,
                window_seconds=window_seconds,
            )
            if dataset_run_id is None:
                eprint(f"[skip] No dataset resolved for {window_seconds}s; skipping window.")
                skipped += len(jobs)
                for job in jobs:
                    exp_name = experiment_name_for_model(job.model, job.fs_enabled, model_metadata)
                    unresolved_job_key = make_job_key(
                        window_seconds=window_seconds,
                        dataset_run_id=None,
                        job=job,
                    )
                    append_job_attempt_record(
                        job=job,
                        job_key=unresolved_job_key,
                        status="dataset_unresolved",
                        window_seconds=window_seconds,
                        dataset_run_id=None,
                        experiment_name=exp_name,
                        cmd=None,
                        return_code=None,
                        train_run_id=None,
                    )
                continue

            print(f"Dataset run id: {dataset_run_id}")

            for idx, job in enumerate(jobs, start=1):
                exp_name = experiment_name_for_model(job.model, job.fs_enabled, model_metadata)
                env = dict(os.environ)
                env["MLFLOW_EXPERIMENT_NAME"] = exp_name

                cmd = build_train_cmd(
                    python_cmd=python_cmd,
                    args=args,
                    job=job,
                    window_seconds=window_seconds,
                    dataset_run_id=dataset_run_id,
                )
                if job.fs_enabled:
                    print(
                        f"[{idx}/{len(jobs)}] model={job.model} fs={job.fs_method} n={job.n_features} exp={exp_name}"
                    )
                else:
                    print(f"[{idx}/{len(jobs)}] model={job.model} baseline exp={exp_name}")

                job_key = make_job_key(
                    window_seconds=window_seconds,
                    dataset_run_id=dataset_run_id,
                    job=job,
                )
                if checkpoint_enabled and not args.no_resume and job_key in completed_job_keys:
                    print("[resume-skip] already completed in checkpoint")
                    resumed += 1
                    append_job_attempt_record(
                        job=job,
                        job_key=job_key,
                        status="resume_skipped",
                        window_seconds=window_seconds,
                        dataset_run_id=dataset_run_id,
                        experiment_name=exp_name,
                        cmd=cmd,
                        return_code=0,
                        train_run_id=None,
                    )
                    continue

                active_job_context = {
                    "job": job,
                    "job_key": job_key,
                    "window_seconds": window_seconds,
                    "dataset_run_id": dataset_run_id,
                    "experiment_name": exp_name,
                    "cmd": list(cmd),
                }

                execution = run_job(cmd=cmd, env=env, dry_run=args.dry_run)
                active_job_context = None

                if execution.return_code == 0:
                    success += 1
                    append_job_attempt_record(
                        job=job,
                        job_key=job_key,
                        status="success",
                        window_seconds=window_seconds,
                        dataset_run_id=dataset_run_id,
                        experiment_name=exp_name,
                        cmd=cmd,
                        return_code=execution.return_code,
                        train_run_id=execution.train_run_id,
                    )
                    if checkpoint_enabled and not args.dry_run and run_state is not None and checkpoint is not None:
                        completed_job_keys.add(job_key)
                        run_state["completed_jobs"] = sorted(completed_job_keys)
                        failed_jobs.pop(job_key, None)
                        run_state["failed_jobs"] = failed_jobs
                        run_state["status"] = "running"
                        run_state["updated_at"] = utc_now_iso()
                        save_checkpoint(checkpoint_path, checkpoint)
                else:
                    failed += 1
                    append_job_attempt_record(
                        job=job,
                        job_key=job_key,
                        status="failed",
                        window_seconds=window_seconds,
                        dataset_run_id=dataset_run_id,
                        experiment_name=exp_name,
                        cmd=cmd,
                        return_code=execution.return_code,
                        train_run_id=execution.train_run_id,
                    )
                    if checkpoint_enabled and not args.dry_run and run_state is not None and checkpoint is not None:
                        failed_jobs[job_key] = {
                            "return_code": int(execution.return_code),
                            "at": utc_now_iso(),
                            "window_seconds": int(window_seconds),
                            "dataset_run_id": dataset_run_id,
                            "model": job.model,
                            "fs_enabled": bool(job.fs_enabled),
                            "fs_method": job.fs_method,
                            "n_features": job.n_features,
                        }
                        run_state["failed_jobs"] = failed_jobs
                        run_state["status"] = "running"
                        run_state["updated_at"] = utc_now_iso()
                        save_checkpoint(checkpoint_path, checkpoint)

                    if args.stop_on_error:
                        eprint("[stop] stopping due to --stop-on-error")
                        if checkpoint_enabled and not args.dry_run and run_state is not None and checkpoint is not None:
                            run_state["status"] = "stopped_on_error"
                            run_state["updated_at"] = utc_now_iso()
                            save_checkpoint(checkpoint_path, checkpoint)
                        print(f"\nSummary: success={success} failed={failed} skipped={skipped} resumed={resumed}")
                        return 1
            print()
    except KeyboardInterrupt:
        eprint("[interrupt] interrupted by user; checkpoint state saved")
        if active_job_context is not None:
            append_job_attempt_record(
                job=active_job_context["job"],
                job_key=active_job_context["job_key"],
                status="interrupted",
                window_seconds=int(active_job_context["window_seconds"]),
                dataset_run_id=active_job_context["dataset_run_id"],
                experiment_name=active_job_context["experiment_name"],
                cmd=active_job_context["cmd"],
                return_code=130,
                train_run_id=None,
            )
        if checkpoint_enabled and not args.dry_run and run_state is not None and checkpoint is not None:
            run_state["status"] = "interrupted"
            run_state["updated_at"] = utc_now_iso()
            save_checkpoint(checkpoint_path, checkpoint)
        print(f"\nSummary: success={success} failed={failed} skipped={skipped} resumed={resumed}")
        return 130

    if checkpoint_enabled and not args.dry_run and run_state is not None and checkpoint is not None:
        run_state["expected_jobs"] = expected_jobs
        run_state["completed_jobs"] = sorted(completed_job_keys)
        run_state["failed_jobs"] = failed_jobs
        if failed > 0:
            run_state["status"] = "failed_partial"
        elif len(completed_job_keys) >= expected_jobs:
            run_state["status"] = "completed"
            run_state["completed_at"] = utc_now_iso()
        else:
            run_state["status"] = "partial"
        run_state["updated_at"] = utc_now_iso()
        save_checkpoint(checkpoint_path, checkpoint)

    print(f"Summary: success={success} failed={failed} skipped={skipped} resumed={resumed}")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
