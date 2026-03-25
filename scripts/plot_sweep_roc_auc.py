#!/usr/bin/env python3
"""Plot sweep metric trajectories from results ledgers."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_ARTIFACTS_DIR = Path("sweeps/artifacts")
DEFAULT_OUTPUT_DIR = Path("sweeps/plots")
DEFAULT_RESULTS_GLOB = "*.results.jsonl"
FIGURE_FACE = "#f4f7fb"
AXES_FACE = "#fbfdff"
DIM_COLOR = "#91a4b7"
DIM_MARKER_EDGE = "#70879b"
AVG_COLOR = "#123c69"
AVG_BAND_COLOR = "#6f9ac6"
GRID_COLOR = "#d8e2ee"
TEXT_COLOR = "#1f2d3d"
SPINE_COLOR = "#8fa1b6"


@dataclass(frozen=True)
class PlotRow:
    source_file: str
    created_at: str
    created_at_sort: str
    attempt_index: int
    run_signature: Optional[str]
    job_key: Optional[str]
    model: Optional[str]
    fs_enabled: bool
    fs_method: Optional[str]
    n_features: Optional[int]
    ordering: Optional[str]
    dataset_run_id: Optional[str]
    window_seconds: Optional[int]
    inner_k: Optional[int]
    outer_k: Optional[int]
    equalize_lopo_groups: Optional[str]
    use_smote: Optional[str]
    metric_key: str
    metric_value: float

    @property
    def patient_roc_auc(self) -> Optional[float]:
        if self.metric_key == "patient_roc_auc":
            return self.metric_value
        return None


def slugify(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "na"


def parse_optional_bool_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "false"}:
        return normalized
    raise ValueError(f"Expected 'true' or 'false', got: {value}")


def parse_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_created_at_sort_key(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        return datetime.fromisoformat(text).isoformat()
    except ValueError:
        return text


def _parse_metric_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def extract_metric_value(record: Dict[str, Any], metric_key: str) -> Optional[float]:
    mlflow_metrics = record.get("mlflow_metrics")
    if isinstance(mlflow_metrics, dict):
        numeric = _parse_metric_number(mlflow_metrics.get(metric_key))
        if numeric is not None:
            return numeric

    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return None

    if metric_key.startswith("patient_"):
        patient = metrics.get("patient")
        if isinstance(patient, dict):
            numeric = _parse_metric_number(patient.get(metric_key[len("patient_") :]))
            if numeric is not None:
                return numeric

    if metric_key.startswith("window_"):
        window = metrics.get("window")
        if isinstance(window, dict):
            numeric = _parse_metric_number(window.get(metric_key[len("window_") :]))
            if numeric is not None:
                return numeric

    return None


def extract_patient_roc_auc(record: Dict[str, Any]) -> Optional[float]:
    return extract_metric_value(record, "patient_roc_auc")


def results_record_to_plot_row(
    record: Dict[str, Any],
    source_file: Path,
    metric_key: str = "patient_roc_auc",
) -> Optional[PlotRow]:
    if record.get("record_type") != "job_attempt":
        return None
    if record.get("status") != "success":
        return None

    metric_value = extract_metric_value(record, metric_key)
    if metric_value is None:
        return None

    return PlotRow(
        source_file=str(source_file),
        created_at=str(record.get("created_at") or ""),
        created_at_sort=parse_created_at_sort_key(record.get("created_at")),
        attempt_index=parse_optional_int(record.get("attempt_index")) or 0,
        run_signature=str(record.get("run_signature")) if record.get("run_signature") is not None else None,
        job_key=str(record.get("job_key")) if record.get("job_key") is not None else None,
        model=str(record.get("model")) if record.get("model") is not None else None,
        fs_enabled=bool(record.get("fs_enabled")),
        fs_method=str(record.get("fs_method")) if record.get("fs_method") is not None else None,
        n_features=parse_optional_int(record.get("n_features")),
        ordering=str(record.get("ordering")) if record.get("ordering") is not None else None,
        dataset_run_id=str(record.get("dataset_run_id")) if record.get("dataset_run_id") is not None else None,
        window_seconds=parse_optional_int(record.get("window_seconds")),
        inner_k=parse_optional_int(record.get("inner_k")),
        outer_k=parse_optional_int(record.get("outer_k")),
        equalize_lopo_groups=parse_optional_bool_text(record.get("equalize_lopo_groups"))
        if record.get("equalize_lopo_groups") is not None
        else None,
        use_smote=parse_optional_bool_text(record.get("use_smote")) if record.get("use_smote") is not None else None,
        metric_key=metric_key,
        metric_value=metric_value,
    )


def discover_results_files(artifacts_dir: Path, pattern: str) -> List[Path]:
    return sorted(path for path in artifacts_dir.glob(pattern) if path.is_file())


def load_plot_rows(paths: Sequence[Path], metric_key: str) -> Tuple[List[PlotRow], Dict[str, int]]:
    rows: List[PlotRow] = []
    stats = {
        "files_loaded": 0,
        "records_loaded": 0,
        "success_records": 0,
        "skipped_missing_metric": 0,
        "skipped_non_success": 0,
        "skipped_non_attempt": 0,
    }

    for path in paths:
        stats["files_loaded"] += 1
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                stats["records_loaded"] += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("record_type") != "job_attempt":
                    stats["skipped_non_attempt"] += 1
                    continue
                if record.get("status") != "success":
                    stats["skipped_non_success"] += 1
                    continue
                stats["success_records"] += 1
                row = results_record_to_plot_row(record, path, metric_key)
                if row is None:
                    stats["skipped_missing_metric"] += 1
                    continue
                rows.append(row)
    return rows, stats


def row_matches_filters(
    row: PlotRow,
    *,
    model: Optional[str],
    fs_method: Optional[str],
    n_features: Optional[int],
    ordering: Optional[str],
    equalize_lopo_groups: Optional[str],
    use_smote: Optional[str],
) -> bool:
    if model is not None and row.model != model:
        return False
    if fs_method is not None and row.fs_method != fs_method:
        return False
    if n_features is not None and row.n_features != n_features:
        return False
    if ordering is not None and row.ordering != ordering:
        return False
    if equalize_lopo_groups is not None and row.equalize_lopo_groups != equalize_lopo_groups:
        return False
    if use_smote is not None and row.use_smote != use_smote:
        return False
    return True


def dedupe_point_key(row: PlotRow) -> Tuple[Any, ...]:
    return (
        row.model,
        row.window_seconds,
        row.inner_k,
        row.outer_k,
        row.fs_enabled,
        row.fs_method,
        row.n_features,
        row.ordering,
        row.equalize_lopo_groups,
        row.use_smote,
        row.dataset_run_id,
    )


def deduplicate_plot_rows(rows: Sequence[PlotRow]) -> List[PlotRow]:
    deduped: Dict[Tuple[Any, ...], PlotRow] = {}
    for row in rows:
        key = dedupe_point_key(row)
        current = deduped.get(key)
        if current is None or (row.attempt_index, row.created_at_sort) > (
            current.attempt_index,
            current.created_at_sort,
        ):
            deduped[key] = row
    return sorted(
        deduped.values(),
        key=lambda row: (
            row.model or "",
            row.window_seconds if row.window_seconds is not None else -1,
            row.inner_k if row.inner_k is not None else -1,
            row.outer_k if row.outer_k is not None else -1,
            row.equalize_lopo_groups or "",
            row.use_smote or "",
            row.fs_method or "",
            row.n_features if row.n_features is not None else -1,
            row.ordering or "",
            row.dataset_run_id or "",
        ),
    )


def build_trajectory_key(row: PlotRow, x_field: str) -> Tuple[Any, ...]:
    if x_field == "window_seconds":
        return (
            row.model,
            row.inner_k,
            row.outer_k,
            row.equalize_lopo_groups,
            row.use_smote,
            row.fs_enabled,
            row.fs_method,
            row.n_features,
            row.ordering,
            row.dataset_run_id,
        )
    if x_field == "inner_k":
        return (
            row.model,
            row.window_seconds,
            row.outer_k,
            row.equalize_lopo_groups,
            row.use_smote,
            row.fs_enabled,
            row.fs_method,
            row.n_features,
            row.ordering,
            row.dataset_run_id,
        )
    raise ValueError(f"Unsupported x_field: {x_field}")


def compute_average_series(rows: Sequence[PlotRow], x_field: str) -> List[Tuple[int, float]]:
    grouped: Dict[int, List[float]] = defaultdict(list)
    for row in rows:
        x_value = getattr(row, x_field)
        if x_value is None:
            continue
        grouped[int(x_value)].append(float(row.metric_value))
    return [(x_value, fmean(values)) for x_value, values in sorted(grouped.items()) if values]


def compute_average_envelope(
    rows: Sequence[PlotRow],
    x_field: str,
    *,
    clamp_min: Optional[float] = None,
    clamp_max: Optional[float] = None,
) -> List[Tuple[int, float, float, float, int]]:
    grouped: Dict[int, List[float]] = defaultdict(list)
    for row in rows:
        x_value = getattr(row, x_field)
        if x_value is None:
            continue
        grouped[int(x_value)].append(float(row.metric_value))

    envelope: List[Tuple[int, float, float, float, int]] = []
    for x_value, values in sorted(grouped.items()):
        if not values:
            continue
        mean_value = fmean(values)
        if len(values) > 1:
            variance = fmean([(value - mean_value) ** 2 for value in values])
            std_value = math.sqrt(variance)
            sem = std_value / math.sqrt(len(values))
            ci_half_width = 1.96 * sem
        else:
            ci_half_width = 0.0
        lower = mean_value - ci_half_width
        upper = mean_value + ci_half_width
        if clamp_min is not None:
            lower = max(clamp_min, lower)
            upper = max(clamp_min, upper)
        if clamp_max is not None:
            lower = min(clamp_max, lower)
            upper = min(clamp_max, upper)
        envelope.append((x_value, mean_value, lower, upper, len(values)))
    return envelope


def apply_scientific_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": FIGURE_FACE,
            "axes.facecolor": AXES_FACE,
            "axes.edgecolor": SPINE_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "axes.titlesize": 15,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.9,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.frameon": True,
            "legend.facecolor": "#ffffff",
            "legend.edgecolor": "#d4dce6",
            "legend.framealpha": 0.95,
            "font.family": "DejaVu Sans",
            "savefig.facecolor": FIGURE_FACE,
            "savefig.bbox": "tight",
        }
    )


def build_plot_records(rows: Sequence[PlotRow], x_field: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in rows:
        x_value = getattr(row, x_field)
        if x_value is None:
            continue
        records.append(
            {
                "trajectory_key": "|".join("" if value is None else str(value) for value in build_trajectory_key(row, x_field)),
                "source_file": row.source_file,
                "created_at": row.created_at,
                "attempt_index": row.attempt_index,
                "run_signature": row.run_signature or "",
                "job_key": row.job_key or "",
                "model": row.model or "",
                "fs_enabled": row.fs_enabled,
                "fs_method": row.fs_method or "",
                "n_features": row.n_features,
                "ordering": row.ordering or "",
                "dataset_run_id": row.dataset_run_id or "",
                "window_seconds": row.window_seconds,
                "inner_k": row.inner_k,
                "outer_k": row.outer_k,
                "equalize_lopo_groups": row.equalize_lopo_groups or "",
                "use_smote": row.use_smote or "",
                "metric_key": row.metric_key,
                "metric_value": row.metric_value,
            }
        )
    return records


def write_plot_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trajectory_key",
        "source_file",
        "created_at",
        "attempt_index",
        "run_signature",
        "job_key",
        "model",
        "fs_enabled",
        "fs_method",
        "n_features",
        "ordering",
        "dataset_run_id",
        "window_seconds",
        "inner_k",
        "outer_k",
        "equalize_lopo_groups",
        "use_smote",
        "metric_key",
        "metric_value",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def render_multiline_plot(
    *,
    rows: Sequence[PlotRow],
    x_field: str,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    min_points_per_line: int,
    include_singletons: bool,
    title_suffix: Optional[str],
    dpi: int,
    show_average_band: bool,
    y_min: Optional[float],
    y_max: Optional[float],
) -> Dict[str, int]:
    apply_scientific_plot_style()
    grouped: Dict[Tuple[Any, ...], List[PlotRow]] = defaultdict(list)
    for row in rows:
        x_value = getattr(row, x_field)
        if x_value is None:
            continue
        grouped[build_trajectory_key(row, x_field)].append(row)

    average_envelope = compute_average_envelope(rows, x_field, clamp_min=y_min, clamp_max=y_max)
    if not average_envelope:
        raise ValueError(f"No valid rows available for {x_field} plot.")

    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    dim_label_used = False
    singleton_label_used = False
    dim_trajectories = 0
    singleton_points = 0

    for trajectory_rows in grouped.values():
        ordered_rows = sorted(trajectory_rows, key=lambda row: getattr(row, x_field) or -1)
        x_values = [int(getattr(row, x_field)) for row in ordered_rows if getattr(row, x_field) is not None]
        y_values = [row.metric_value for row in ordered_rows if getattr(row, x_field) is not None]
        if len(x_values) >= min_points_per_line:
            ax.plot(
                x_values,
                y_values,
                color=DIM_COLOR,
                alpha=0.28,
                linewidth=1.15,
                label="Individual trajectories" if not dim_label_used else "_nolegend_",
                solid_capstyle="round",
            )
            dim_label_used = True
            dim_trajectories += 1
        elif include_singletons and x_values:
            ax.scatter(
                x_values,
                y_values,
                color=DIM_COLOR,
                alpha=0.38,
                s=26,
                edgecolors=DIM_MARKER_EDGE,
                linewidths=0.5,
                label="Individual trajectories" if not dim_label_used and not singleton_label_used else "_nolegend_",
            )
            singleton_label_used = True
            singleton_points += len(x_values)

    avg_x = [point[0] for point in average_envelope]
    avg_y = [point[1] for point in average_envelope]
    avg_lower = [point[2] for point in average_envelope]
    avg_upper = [point[3] for point in average_envelope]
    total_points = sum(point[4] for point in average_envelope)

    if show_average_band:
        ax.fill_between(
            avg_x,
            avg_lower,
            avg_upper,
            color=AVG_BAND_COLOR,
            alpha=0.20,
            linewidth=0,
            label="Mean 95% CI",
            zorder=2,
        )
    ax.plot(
        avg_x,
        avg_y,
        color=AVG_COLOR,
        alpha=1.0,
        linewidth=3.2,
        marker="o",
        markersize=6.5,
        markerfacecolor=AVG_COLOR,
        markeredgecolor="#ffffff",
        markeredgewidth=1.0,
        label="Average",
        zorder=3,
    )

    full_title = title if not title_suffix else f"{title} - {title_suffix}"
    ax.set_title(full_title, loc="left", pad=18)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if y_min is not None or y_max is not None:
        lower = y_min if y_min is not None else ax.get_ylim()[0]
        upper = y_max if y_max is not None else ax.get_ylim()[1]
        ax.set_ylim(lower, upper)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.75)
    ax.set_xticks(sorted(set(avg_x)))
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.0)
        ax.spines[spine].set_color(SPINE_COLOR)
    ax.tick_params(axis="both", which="major", length=0)
    subtitle = (
        f"{len(rows)} deduplicated successful runs, "
        f"{dim_trajectories} multi-point trajectories, "
        f"{total_points} observations"
    )
    ax.text(
        0.0,
        1.02,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.6,
        color="#5b6b7d",
    )
    legend = ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), ncols=3, fontsize=9.5, handlelength=2.6)
    for legend_line in legend.get_lines():
        legend_line.set_linewidth(2.6)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    return {
        "dim_trajectories": dim_trajectories,
        "singleton_points": singleton_points,
        "average_points": len(average_envelope),
    }


KNOWN_METRIC_LABELS = {
    "patient_roc_auc": "Patient ROC-AUC",
    "patient_pr_auc": "Patient PR-AUC",
    "patient_balanced_accuracy": "Patient Balanced Accuracy",
    "patient_accuracy": "Patient Accuracy",
    "patient_f1": "Patient F1",
    "patient_mcc": "Patient MCC",
    "feature_selection_mean_pairwise_jaccard": "Mean Pairwise Jaccard",
    "feature_selection_median_pairwise_jaccard": "Median Pairwise Jaccard",
    "feature_selection_kuncheva_index_mean": "Mean Kuncheva Index",
    "feature_selection_kuncheva_index_median": "Median Kuncheva Index",
    "feature_selection_mean_top_k_overlap": "Mean Top-k Overlap",
    "feature_selection_median_top_k_overlap": "Median Top-k Overlap",
    "feature_selection_effect_mean_sign_consistency": "Mean Effect Sign Consistency",
    "feature_selection_importance_mean_variance": "Mean Importance Variance",
    "feature_selection_class_conditional_selection_available": "Class-Conditional Selection Available",
    "feature_selection_ranking_available": "Ranking Stability Available",
    "feature_selection_importance_available": "Importance Stability Available",
    "feature_selection_resampling_enabled": "Resampling Stability Enabled",
}

KNOWN_UNIT_INTERVAL_METRICS = {
    "patient_roc_auc",
    "patient_pr_auc",
    "patient_balanced_accuracy",
    "patient_accuracy",
    "patient_f1",
    "patient_precision",
    "patient_recall",
    "patient_sensitivity",
    "patient_specificity",
    "patient_npv",
    "window_roc_auc",
    "window_accuracy",
    "window_f1",
    "feature_selection_mean_pairwise_jaccard",
    "feature_selection_median_pairwise_jaccard",
    "feature_selection_mean_top_k_overlap",
    "feature_selection_median_top_k_overlap",
    "feature_selection_effect_mean_sign_consistency",
    "feature_selection_top_feature_share",
    "feature_selection_class_conditional_selection_available",
    "feature_selection_ranking_available",
    "feature_selection_importance_available",
    "feature_selection_resampling_enabled",
}

METRIC_PRESETS = {
    "performance_core": [
        "patient_roc_auc",
        "patient_pr_auc",
        "patient_balanced_accuracy",
        "patient_f1",
        "patient_mcc",
    ],
    "performance_extended": [
        "patient_roc_auc",
        "patient_pr_auc",
        "patient_balanced_accuracy",
        "patient_accuracy",
        "patient_f1",
        "patient_mcc",
        "patient_precision",
        "patient_recall",
        "patient_specificity",
        "patient_npv",
    ],
    "biomarker_core": [
        "feature_selection_mean_pairwise_jaccard",
        "feature_selection_kuncheva_index_mean",
        "feature_selection_mean_top_k_overlap",
        "feature_selection_effect_mean_sign_consistency",
    ],
    "biomarker_extended": [
        "feature_selection_mean_pairwise_jaccard",
        "feature_selection_median_pairwise_jaccard",
        "feature_selection_kuncheva_index_mean",
        "feature_selection_kuncheva_index_median",
        "feature_selection_mean_top_k_overlap",
        "feature_selection_median_top_k_overlap",
        "feature_selection_effect_mean_sign_consistency",
        "feature_selection_importance_mean_variance",
        "feature_selection_top_feature_share",
        "feature_selection_top_feature_frequency",
        "feature_selection_unique_feature_count",
        "feature_selection_feature_set_count",
    ],
    "biomarker_availability": [
        "feature_selection_class_conditional_selection_available",
        "feature_selection_ranking_available",
        "feature_selection_importance_available",
        "feature_selection_resampling_enabled",
        "feature_selection_remission_fold_count",
        "feature_selection_non_remission_fold_count",
        "feature_selection_mixed_fold_count",
        "feature_selection_unknown_fold_count",
    ],
    "paper_main": [
        "patient_roc_auc",
        "patient_pr_auc",
        "patient_balanced_accuracy",
        "feature_selection_mean_pairwise_jaccard",
        "feature_selection_kuncheva_index_mean",
        "feature_selection_mean_top_k_overlap",
        "feature_selection_effect_mean_sign_consistency",
    ],
}


def infer_metric_label(metric_key: str) -> str:
    if metric_key in KNOWN_METRIC_LABELS:
        return KNOWN_METRIC_LABELS[metric_key]

    prefix = ""
    remainder = metric_key
    if metric_key.startswith("patient_"):
        prefix = "Patient "
        remainder = metric_key[len("patient_") :]
    elif metric_key.startswith("window_"):
        prefix = "Window "
        remainder = metric_key[len("window_") :]
    elif metric_key.startswith("feature_selection_"):
        remainder = metric_key[len("feature_selection_") :]

    tokens = remainder.split("_")
    words: List[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        next_token = tokens[i + 1] if i + 1 < len(tokens) else None
        if token == "roc" and next_token == "auc":
            words.append("ROC-AUC")
            i += 2
            continue
        if token == "pr" and next_token == "auc":
            words.append("PR-AUC")
            i += 2
            continue
        if token == "top" and next_token == "k":
            words.append("Top-k")
            i += 2
            continue
        if token == "mcc":
            words.append("MCC")
        elif token == "npv":
            words.append("NPV")
        elif token == "f1":
            words.append("F1")
        elif token == "k":
            words.append("k")
        else:
            words.append(token.capitalize())
        i += 1
    label = " ".join(words)
    return f"{prefix}{label}".strip()


def infer_metric_title(metric_key: str, x_label_prefix: str, metric_label: str) -> str:
    return f"{x_label_prefix} vs {metric_label}"


def is_unit_interval_metric(metric_key: str) -> bool:
    if metric_key in KNOWN_UNIT_INTERVAL_METRICS:
        return True
    if metric_key.endswith("_share") or metric_key.endswith("_available") or metric_key.endswith("_enabled"):
        return True
    if metric_key.endswith("_ci_low") or metric_key.endswith("_ci_high"):
        return True
    return False


def infer_y_bounds(
    rows: Sequence[PlotRow],
    metric_key: str,
    *,
    y_min: Optional[float],
    y_max: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    values = [row.metric_value for row in rows]
    if not values:
        return y_min, y_max

    if y_min is not None or y_max is not None:
        return y_min, y_max

    if is_unit_interval_metric(metric_key):
        return 0.0, 1.0

    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        padding = 0.1 if min_value == 0 else abs(min_value) * 0.1
        return min_value - padding, max_value + padding

    padding = max(0.08, 0.08 * (max_value - min_value))
    return min_value - padding, max_value + padding


def build_filter_suffix(args: argparse.Namespace) -> str:
    parts = []
    for field in ("model", "fs_method", "n_features", "ordering", "equalize_lopo_groups", "use_smote"):
        value = getattr(args, field)
        if value is None:
            continue
        parts.append(f"{slugify(field)}-{slugify(str(value))}")
    return "__".join(parts)


def add_output_suffix(base_name: str, suffix: str, fmt: str) -> str:
    if suffix:
        return f"{base_name}__{suffix}.{fmt}"
    return f"{base_name}.{fmt}"


def print_available_presets() -> None:
    print("Available presets:")
    for preset_name, metric_keys in METRIC_PRESETS.items():
        print(f"- {preset_name}")
        for metric_key in metric_keys:
            print(f"  - {metric_key}")


def resolve_requested_metrics(args: argparse.Namespace) -> List[str]:
    requested: List[str] = []
    seen = set()

    for preset_name in args.preset or []:
        for metric_key in METRIC_PRESETS[preset_name]:
            if metric_key not in seen:
                requested.append(metric_key)
                seen.add(metric_key)

    for metric_key in args.metric or []:
        if metric_key not in seen:
            requested.append(metric_key)
            seen.add(metric_key)

    if requested:
        return requested
    return ["patient_roc_auc"]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot sweep metric trajectories from results ledgers.")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR), help="Directory containing *.results.jsonl files.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to write plots and CSVs.")
    parser.add_argument("--glob", dest="glob_pattern", default=DEFAULT_RESULTS_GLOB, help="Glob pattern under artifacts dir.")
    parser.add_argument(
        "--metric",
        action="append",
        help="Flattened metric key to plot from mlflow_metrics. Repeat to request multiple metrics.",
    )
    parser.add_argument(
        "--preset",
        action="append",
        choices=tuple(METRIC_PRESETS.keys()),
        help="Named bundle of commonly used sweep metrics. Repeat to combine presets.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print available metric presets and exit.",
    )
    parser.add_argument("--metric-label", help="Optional custom y-axis / title metric label.")
    parser.add_argument("--model", help="Optional model filter.")
    parser.add_argument("--fs-method", help="Optional feature-selection method filter.")
    parser.add_argument("--n-features", type=int, help="Optional feature-count filter.")
    parser.add_argument("--ordering", help="Optional ordering filter.")
    parser.add_argument("--equalize-lopo-groups", choices=("true", "false"), help="Optional group-balancing filter.")
    parser.add_argument("--use-smote", choices=("true", "false"), help="Optional SMOTE filter.")
    parser.add_argument("--min-points-per-line", type=int, default=2, help="Minimum points required before drawing a dim line.")
    parser.add_argument(
        "--include-singletons",
        dest="include_singletons",
        action="store_true",
        default=True,
        help="Plot single-point trajectories as faint markers.",
    )
    parser.add_argument(
        "--no-include-singletons",
        dest="include_singletons",
        action="store_false",
        help="Do not plot single-point trajectories.",
    )
    parser.add_argument("--title-suffix", help="Optional text appended to plot titles.")
    parser.add_argument("--y-min", type=float, help="Optional manual y-axis minimum.")
    parser.add_argument("--y-max", type=float, help="Optional manual y-axis maximum.")
    parser.add_argument(
        "--no-average-band",
        dest="show_average_band",
        action="store_false",
        default=True,
        help="Disable the stronger average confidence band.",
    )
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png", help="Figure output format.")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    return parser.parse_args(argv)


def plot_requested_metric(
    *,
    args: argparse.Namespace,
    paths: Sequence[Path],
    output_dir: Path,
    metric_key: str,
    metric_label_override: Optional[str] = None,
) -> Dict[str, Any]:
    rows, load_stats = load_plot_rows(paths, metric_key)
    filtered_rows = [
        row
        for row in rows
        if row_matches_filters(
            row,
            model=args.model,
            fs_method=args.fs_method,
            n_features=args.n_features,
            ordering=args.ordering,
            equalize_lopo_groups=args.equalize_lopo_groups,
            use_smote=args.use_smote,
        )
    ]
    deduped_rows = deduplicate_plot_rows(filtered_rows)

    result: Dict[str, Any] = {
        "metric_key": metric_key,
        "load_stats": load_stats,
        "filtered_count": len(filtered_rows),
        "deduped_count": len(deduped_rows),
    }
    if not deduped_rows:
        result["status"] = "skipped"
        result["reason"] = "no_matching_rows_after_filter_and_deduplication"
        return result

    metric_label = metric_label_override or infer_metric_label(metric_key)
    y_min, y_max = infer_y_bounds(deduped_rows, metric_key, y_min=args.y_min, y_max=args.y_max)

    suffix = build_filter_suffix(args)
    metric_slug = slugify(metric_key).replace("-", "_")
    window_plot_path = output_dir / add_output_suffix(f"window_size_vs_{metric_slug}", suffix, args.format)
    inner_plot_path = output_dir / add_output_suffix(f"inner_k_vs_{metric_slug}", suffix, args.format)
    window_csv_path = output_dir / add_output_suffix(f"plot_data_window_size_vs_{metric_slug}", suffix, "csv")
    inner_csv_path = output_dir / add_output_suffix(f"plot_data_inner_k_vs_{metric_slug}", suffix, "csv")

    window_records = build_plot_records(deduped_rows, "window_seconds")
    inner_records = build_plot_records(deduped_rows, "inner_k")
    write_plot_csv(window_csv_path, window_records)
    write_plot_csv(inner_csv_path, inner_records)

    window_stats = render_multiline_plot(
        rows=deduped_rows,
        x_field="window_seconds",
        title=infer_metric_title(metric_key, "Window Size", metric_label),
        x_label="Window Size (s)",
        y_label=metric_label,
        output_path=window_plot_path,
        min_points_per_line=args.min_points_per_line,
        include_singletons=args.include_singletons,
        title_suffix=args.title_suffix,
        dpi=args.dpi,
        show_average_band=args.show_average_band,
        y_min=y_min,
        y_max=y_max,
    )
    inner_stats = render_multiline_plot(
        rows=deduped_rows,
        x_field="inner_k",
        title=infer_metric_title(metric_key, "Inner-k", metric_label),
        x_label="Inner-k",
        y_label=metric_label,
        output_path=inner_plot_path,
        min_points_per_line=args.min_points_per_line,
        include_singletons=args.include_singletons,
        title_suffix=args.title_suffix,
        dpi=args.dpi,
        show_average_band=args.show_average_band,
        y_min=y_min,
        y_max=y_max,
    )

    result.update(
        {
            "status": "success",
            "metric_label": metric_label,
            "window_plot_path": window_plot_path,
            "inner_plot_path": inner_plot_path,
            "window_csv_path": window_csv_path,
            "inner_csv_path": inner_csv_path,
            "window_stats": window_stats,
            "inner_stats": inner_stats,
        }
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.list_presets:
        print_available_presets()
        return 0

    requested_metrics = resolve_requested_metrics(args)
    if args.metric_label and len(requested_metrics) != 1:
        raise SystemExit("--metric-label can only be used when plotting exactly one metric.")

    artifacts_dir = Path(args.artifacts_dir)
    output_dir = Path(args.output_dir)

    paths = discover_results_files(artifacts_dir, args.glob_pattern)
    if not paths:
        raise SystemExit(f"No results files found in {artifacts_dir} matching {args.glob_pattern}.")

    successful_metrics = 0
    skipped_metrics: List[Tuple[str, str]] = []
    for metric_key in requested_metrics:
        result = plot_requested_metric(
            args=args,
            paths=paths,
            output_dir=output_dir,
            metric_key=metric_key,
            metric_label_override=args.metric_label if len(requested_metrics) == 1 else None,
        )

        print(f"Metric key: {metric_key}")
        print(f"Results files loaded: {result['load_stats']['files_loaded']}")
        print(f"JSONL records loaded: {result['load_stats']['records_loaded']}")
        print(f"Successful job attempts seen: {result['load_stats']['success_records']}")
        print(f"Rows skipped due to missing {metric_key}: {result['load_stats']['skipped_missing_metric']}")
        print(f"Rows after CLI filters: {result['filtered_count']}")
        print(f"Rows after deduplication: {result['deduped_count']}")

        if result["status"] != "success":
            skipped_metrics.append((metric_key, result["reason"]))
            print(f"Skipped metric: {metric_key} ({result['reason']})")
            print("")
            continue

        successful_metrics += 1
        window_stats = result["window_stats"]
        inner_stats = result["inner_stats"]
        print(
            "Window plot: "
            f"{window_stats['dim_trajectories']} dim trajectories, "
            f"{window_stats['singleton_points']} singleton points, "
            f"{window_stats['average_points']} average x-values"
        )
        print(
            "Inner-k plot: "
            f"{inner_stats['dim_trajectories']} dim trajectories, "
            f"{inner_stats['singleton_points']} singleton points, "
            f"{inner_stats['average_points']} average x-values"
        )
        print(f"Saved: {result['window_plot_path']}")
        print(f"Saved: {result['inner_plot_path']}")
        print(f"Saved: {result['window_csv_path']}")
        print(f"Saved: {result['inner_csv_path']}")
        print("")

    print(f"Requested metrics: {len(requested_metrics)}")
    print(f"Successfully plotted metrics: {successful_metrics}")
    print(f"Skipped metrics: {len(skipped_metrics)}")
    for metric_key, reason in skipped_metrics:
        print(f"- {metric_key}: {reason}")

    if successful_metrics == 0:
        print("No requested metrics produced plots.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
