#!/usr/bin/env python3
"""
Unified EEG experiment runner.

Features:
- Model-set based selection (traditional / boosting / deep_learning / all)
- Optional explicit model list override
- Optional feature-selection sweeps
- inner_k / outer_k forwarding to run_pipeline train
- Dataset auto-discovery per window size with optional channel filtering
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_PIPELINE = REPO_ROOT / "eeg_analysis" / "run_pipeline.py"
FIND_DATASET_SCRIPT = REPO_ROOT / "scripts" / "find_dataset.py"
FILTER_DATASET_SCRIPT = REPO_ROOT / "scripts" / "filter_dataset.py"

ALL_FOUR_CHANNELS = {"af7", "af8", "tp9", "tp10"}

MODEL_SETS: Dict[str, List[str]] = {
    "traditional": [
        "random_forest",
        "gradient_boosting",
        "logistic_regression",
        "svm_rbf",
    ],
    "boosting": [
        "xgboost_gpu",
        "catboost_gpu",
        "lightgbm_gpu",
    ],
    "deep_learning": [
        "pytorch_mlp",
        "keras_mlp",
        "hybrid_1dcnn_lstm",
        "advanced_hybrid_1dcnn_lstm",
        "efficient_tabular_mlp",
        "advanced_lstm",
    ],
}
MODEL_SETS["all"] = (
    MODEL_SETS["traditional"] + MODEL_SETS["boosting"] + MODEL_SETS["deep_learning"]
)


@dataclass
class Job:
    model: str
    fs_enabled: bool
    fs_method: Optional[str] = None
    n_features: Optional[int] = None


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


def resolve_models(model_sets_raw: str, explicit_models_raw: Optional[str]) -> List[str]:
    if explicit_models_raw:
        explicit_models = parse_csv_list(explicit_models_raw, str)
        if not explicit_models:
            raise ValueError("--models was provided but no valid model names were parsed.")
        return dedupe_preserve_order(explicit_models)

    requested_sets = parse_csv_list(model_sets_raw, str)
    if not requested_sets:
        requested_sets = ["all"]

    unknown = [s for s in requested_sets if s not in MODEL_SETS]
    if unknown:
        valid = ", ".join(sorted(MODEL_SETS.keys()))
        raise ValueError(f"Unknown model set(s): {unknown}. Valid model sets: {valid}")

    if "all" in requested_sets:
        return dedupe_preserve_order(MODEL_SETS["all"])

    models: List[str] = []
    for set_name in requested_sets:
        models.extend(MODEL_SETS[set_name])
    return dedupe_preserve_order(models)


def experiment_name_for_model(model: str, fs_enabled: bool) -> str:
    suffix = "feature_selection" if fs_enabled else "baseline"
    if model in {"pytorch_mlp", "keras_mlp", "hybrid_1dcnn_lstm", "advanced_hybrid_1dcnn_lstm", "efficient_tabular_mlp", "advanced_lstm"}:
        return f"eeg_deep_learning_gpu_{suffix}"
    if model in {"xgboost_gpu", "catboost_gpu", "lightgbm_gpu"}:
        return f"eeg_boosting_gpu_{suffix}"
    if model in {"random_forest", "gradient_boosting", "logistic_regression", "svm_rbf"}:
        return f"eeg_traditional_models_{suffix}"
    return f"eeg_other_models_{suffix}"


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
        "--level",
        args.level,
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

    return cmd


def run_job(
    cmd: Sequence[str], env: Dict[str, str], dry_run: bool
) -> int:
    cmd_str = " ".join(shlex.quote(part) for part in cmd)
    prefix = f"MLFLOW_EXPERIMENT_NAME={env.get('MLFLOW_EXPERIMENT_NAME', '')} "
    print(f"$ {prefix}{cmd_str}")
    if dry_run:
        return 0

    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    return proc.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified experiment runner with model-set and inner/outer-k support."
    )
    parser.add_argument(
        "--config",
        default="eeg_analysis/configs/window_model_config.yaml",
        help="Path to training config passed to run_pipeline.py",
    )
    parser.add_argument(
        "--processing-config",
        default="eeg_analysis/configs/processing_config.yaml",
        help="Path to processing config for channels and window sizes",
    )
    parser.add_argument(
        "--level",
        default="window",
        choices=["window", "patient"],
        help="Training level passed to run_pipeline train",
    )
    parser.add_argument(
        "--model-sets",
        default="all",
        help="Comma-separated model sets (traditional,boosting,deep_learning,all). Ignored if --models is set.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated explicit models to run (overrides --model-sets).",
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
        help="Forwarded to run_pipeline train --inner-k",
    )
    parser.add_argument(
        "--outer-k",
        type=int,
        default=None,
        help="Forwarded to run_pipeline train --outer-k",
    )
    parser.add_argument(
        "--python-cmd",
        default="uv run python3",
        help="Python command prefix for helper scripts and run_pipeline invocations.",
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

    processing_cfg_path = Path(args.processing_config)
    if not processing_cfg_path.exists():
        raise FileNotFoundError(f"Processing config not found: {processing_cfg_path}")
    train_cfg_path = Path(args.config)
    if not train_cfg_path.exists():
        raise FileNotFoundError(f"Training config not found: {train_cfg_path}")

    processing_cfg = load_yaml(processing_cfg_path)
    channels = get_channels(processing_cfg)
    window_sizes = parse_window_sizes(processing_cfg, args.window_sizes)
    models = resolve_models(args.model_sets, args.models)

    fs_methods = parse_csv_list(args.fs_methods, str)
    feature_counts = parse_csv_list(args.feature_counts, int)
    if args.mode in {"fs", "both"}:
        if not fs_methods:
            raise ValueError("--fs-methods produced an empty list.")
        if not feature_counts:
            raise ValueError("--feature-counts produced an empty list.")

    jobs = build_jobs(models, args.mode, fs_methods, feature_counts)
    python_cmd = shlex.split(args.python_cmd)

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
    print(f"Dry run: {args.dry_run}")
    print(f"Total jobs: {len(jobs) * len(window_sizes)}")
    print()

    success = 0
    failed = 0
    skipped = 0

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
            continue

        print(f"Dataset run id: {dataset_run_id}")

        for idx, job in enumerate(jobs, start=1):
            exp_name = experiment_name_for_model(job.model, job.fs_enabled)
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

            return_code = run_job(cmd=cmd, env=env, dry_run=args.dry_run)
            if return_code == 0:
                success += 1
            else:
                failed += 1
                if args.stop_on_error:
                    eprint("[stop] stopping due to --stop-on-error")
                    print(f"\nSummary: success={success} failed={failed} skipped={skipped}")
                    return 1
        print()

    print(f"Summary: success={success} failed={failed} skipped={skipped}")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
