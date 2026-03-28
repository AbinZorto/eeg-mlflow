#!/usr/bin/env python3
"""Generate paper-style standalone figures from one resolved MLflow training run."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_curve


REPO_ROOT = Path(__file__).resolve().parent.parent
EEG_ANALYSIS_ROOT = REPO_ROOT / "eeg_analysis"
if str(EEG_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(EEG_ANALYSIS_ROOT))

from src.utils.clinical_metrics import compute_metric_value, compute_per_fold_metrics  # noqa: E402
from src.utils.paper_plot_style import PALETTE, add_subtitle, apply_paper_style, save_figure, style_axes  # noqa: E402
from src.utils.plot_data_loader import (  # noqa: E402
    RunArtifactBundle,
    SweepResultRow,
    deduplicate_results_rows,
    discover_results_files,
    filter_result_rows,
    load_results_rows,
    load_run_artifact_bundle,
    select_latest_result_row,
)


class PlotSkipped(RuntimeError):
    pass


ALL_FIGURES = (
    "roc",
    "pr",
    "confusion",
    "metric_summary",
    "fold_distribution",
    "calibration",
    "selection_frequency",
    "selection_distribution",
    "jaccard_heatmap",
    "jaccard_distribution",
    "kuncheva_summary",
    "delta_scatter",
    "class_conditional_bars",
    "delta_distribution",
    "effect_size",
    "sign_consistency",
    "effect_frequency",
)


def slugify(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    while "--" in text:
        text = text.replace("--", "-")
    return text.strip("-") or "na"


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        return None
    return numeric


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate standalone paper figures from one resolved MLflow run.")
    parser.add_argument("--artifacts-dir", default="sweeps/artifacts", help="Directory containing results ledgers.")
    parser.add_argument("--output-dir", default="sweeps/paper_figures", help="Base directory for generated figures.")
    parser.add_argument("--mlruns-root", default="mlruns", help="Local MLflow file-backend root.")
    parser.add_argument("--results-glob", default="*.results.jsonl", help="Glob pattern under artifacts-dir.")
    parser.add_argument("--run-signature", help="Optional run signature filter.")
    parser.add_argument("--mlflow-run-id", help="Optional explicit MLflow run id.")
    parser.add_argument("--model", help="Optional model filter.")
    parser.add_argument("--window-size", type=int, help="Optional window-size filter.")
    parser.add_argument("--inner-k", type=int, help="Optional inner-k filter.")
    parser.add_argument("--outer-k", type=int, help="Optional outer-k filter.")
    parser.add_argument("--fs-method", help="Optional FS method filter.")
    parser.add_argument("--n-features", type=int, help="Optional feature-count filter.")
    parser.add_argument("--ordering", help="Optional ordering filter.")
    parser.add_argument("--equalize-lopo-groups", choices=("true", "false"), help="Optional group-balancing filter.")
    parser.add_argument("--use-smote", choices=("true", "false"), help="Optional SMOTE filter.")
    parser.add_argument("--top-features", type=int, default=20, help="Top features to show in feature-level plots.")
    parser.add_argument("--figures", help="Comma-separated subset of figure ids to generate.")
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png", help="Figure output format.")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000, help="Bootstrap iterations for missing CI values.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for bootstrap procedures.")
    return parser.parse_args(argv)


def extract_prediction_arrays(rows: Sequence[Dict[str, Any]]) -> Tuple[List[int], List[int], List[float]]:
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []
    for row in rows:
        true_label = _safe_int(row.get("true_label"))
        pred_label = _safe_int(row.get("predicted_label"))
        probability = _safe_float(row.get("probability"))
        if true_label is None or pred_label is None:
            continue
        y_true.append(true_label)
        y_pred.append(pred_label)
        if probability is not None:
            y_prob.append(probability)
    return y_true, y_pred, y_prob


def bootstrap_metric_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float],
    metric_name: str,
    *,
    iterations: int,
    random_seed: int,
) -> Dict[str, Optional[float]]:
    if not y_true or iterations < 1:
        return {"ci_low": None, "ci_high": None, "std": None}

    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float) if y_prob else None
    rng = np.random.default_rng(random_seed)
    values: List[float] = []

    for _ in range(iterations):
        indices = rng.integers(0, len(y_true_arr), size=len(y_true_arr))
        sampled_true = y_true_arr[indices]
        sampled_pred = y_pred_arr[indices]
        sampled_prob = y_prob_arr[indices] if y_prob_arr is not None and len(y_prob_arr) == len(y_true_arr) else None
        value = compute_metric_value(metric_name, sampled_true, sampled_pred, sampled_prob)
        if value is not None:
            values.append(float(value))

    if not values:
        return {"ci_low": None, "ci_high": None, "std": None}

    arr = np.asarray(values, dtype=float)
    return {
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "std": float(arr.std(ddof=0)),
    }


def resolve_ci(
    report: Dict[str, Any],
    metric_name: str,
    *,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float],
    bootstrap_iterations: int,
    random_seed: int,
) -> Dict[str, Optional[float]]:
    ci = (
        report.get("patient", {})
        .get("confidence_intervals", {})
        .get(metric_name, {})
    )
    if isinstance(ci, dict) and any(ci.get(key) is not None for key in ("ci_low", "ci_high")):
        return {
            "ci_low": _safe_float(ci.get("ci_low")),
            "ci_high": _safe_float(ci.get("ci_high")),
            "std": _safe_float(ci.get("std")),
        }
    return bootstrap_metric_ci(
        y_true,
        y_pred,
        y_prob,
        metric_name,
        iterations=bootstrap_iterations,
        random_seed=random_seed,
    )


def compute_patient_metric_summary(
    bundle: RunArtifactBundle,
    *,
    bootstrap_iterations: int,
    random_seed: int,
) -> List[Dict[str, Any]]:
    y_true, y_pred, y_prob = extract_prediction_arrays(bundle.patient_predictions)
    if not y_true:
        raise PlotSkipped("patient_predictions_unavailable")

    report = bundle.clinical_metrics
    patient_metrics = report.get("patient", {}).get("metrics", {}) if isinstance(report.get("patient"), dict) else {}
    metric_order = (
        ("balanced_accuracy", "Balanced Accuracy"),
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
        ("f1", "F1"),
        ("mcc", "MCC"),
    )
    rows: List[Dict[str, Any]] = []
    for metric_name, label in metric_order:
        value = _safe_float(patient_metrics.get(metric_name))
        if value is None:
            value = compute_metric_value(metric_name, y_true, y_pred, y_prob if len(y_prob) == len(y_true) else None)
        ci = resolve_ci(
            report,
            metric_name,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            bootstrap_iterations=bootstrap_iterations,
            random_seed=random_seed,
        )
        rows.append(
            {
                "metric_name": metric_name,
                "label": label,
                "value": value,
                "ci_low": ci.get("ci_low"),
                "ci_high": ci.get("ci_high"),
                "std": ci.get("std"),
            }
        )
    return rows


def merge_effect_frequency_rows(bundle: RunArtifactBundle) -> List[Dict[str, Any]]:
    feature_selection = bundle.feature_selection_summary or {}
    selection_rows = feature_selection.get("selection_frequency_by_feature") or []
    effect_rows = (
        feature_selection.get("effect_stability", {}).get("features")
        if isinstance(feature_selection.get("effect_stability"), dict)
        else []
    ) or []
    effect_by_feature = {
        str(row.get("feature")): row
        for row in effect_rows
        if isinstance(row, dict) and row.get("feature") is not None
    }
    merged: List[Dict[str, Any]] = []
    for row in selection_rows:
        if not isinstance(row, dict) or row.get("feature") is None:
            continue
        feature = str(row["feature"])
        merged_row = dict(row)
        merged_row.update(effect_by_feature.get(feature, {}))
        merged.append(merged_row)
    return merged


def select_top_feature_rows(rows: Sequence[Dict[str, Any]], top_n: int, key: str = "share") -> List[Dict[str, Any]]:
    usable = [row for row in rows if isinstance(row, dict)]
    usable.sort(key=lambda row: (_safe_float(row.get(key)) or 0.0, str(row.get("feature", ""))), reverse=True)
    return usable[:top_n]


def plot_patient_roc_curve(bundle: RunArtifactBundle, output_path: Path, dpi: int) -> None:
    y_true, _, y_prob = extract_prediction_arrays(bundle.patient_predictions)
    if len(y_true) < 2 or len(y_prob) != len(y_true) or len(set(y_true)) < 2:
        raise PlotSkipped("patient_probabilities_unavailable_for_roc")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = compute_metric_value("roc_auc", y_true, y_true, y_prob)
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.plot(fpr, tpr, color=PALETTE["summary"], linewidth=2.8, label=f"ROC-AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color=PALETTE["neutral_dark"], linewidth=1.2, alpha=0.8)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Patient-Level ROC Curve", loc="left", pad=18)
    add_subtitle(ax, f"{len(y_true)} held-out participants")
    style_axes(ax)
    ax.legend(loc="lower right", fontsize=9.5)
    fig.tight_layout()
    save_figure(fig, output_path, dpi)


def plot_patient_pr_curve(bundle: RunArtifactBundle, output_path: Path, dpi: int) -> None:
    y_true, _, y_prob = extract_prediction_arrays(bundle.patient_predictions)
    if len(y_true) < 2 or len(y_prob) != len(y_true) or len(set(y_true)) < 2:
        raise PlotSkipped("patient_probabilities_unavailable_for_pr")

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    prevalence = sum(y_true) / len(y_true)
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.plot(recall, precision, color=PALETTE["summary"], linewidth=2.8, label=f"PR-AUC = {pr_auc:.3f}")
    ax.axhline(prevalence, linestyle="--", color=PALETTE["neutral_dark"], linewidth=1.2, alpha=0.8, label=f"Prevalence = {prevalence:.3f}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Patient-Level Precision-Recall Curve", loc="left", pad=18)
    add_subtitle(ax, f"{len(y_true)} held-out participants")
    style_axes(ax)
    ax.legend(loc="lower left", fontsize=9.5)
    fig.tight_layout()
    save_figure(fig, output_path, dpi)


def plot_patient_confusion_matrix(bundle: RunArtifactBundle, output_path: Path, dpi: int) -> None:
    y_true, y_pred, _ = extract_prediction_arrays(bundle.patient_predictions)
    if not y_true or not y_pred:
        raise PlotSkipped("patient_predictions_unavailable_for_confusion")

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]}\n({normalized[i, j] * 100:.1f}%)",
                ha="center",
                va="center",
                fontsize=10.2,
                color="#0f2235" if normalized[i, j] < 0.6 else "#ffffff",
            )
    ax.set_xticks([0, 1], labels=["Non-remission", "Remission"])
    ax.set_yticks([0, 1], labels=["Non-remission", "Remission"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Patient-Level Confusion Matrix", loc="left", pad=18)
    add_subtitle(ax, f"{len(y_true)} held-out participants")
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized fraction")
    fig.tight_layout()
    save_figure(fig, output_path, dpi)


def plot_metric_summary(bundle: RunArtifactBundle, output_path: Path, dpi: int, bootstrap_iterations: int, random_seed: int) -> None:
    rows = compute_patient_metric_summary(
        bundle,
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed,
    )
    values = [row["value"] for row in rows]
    if all(value is None for value in values):
        raise PlotSkipped("patient_metric_summary_unavailable")

    labels = [row["label"] for row in rows]
    y_positions = np.arange(len(rows))
    metric_values = [row["value"] if row["value"] is not None else 0.0 for row in rows]
    lower_err = [
        max(0.0, row["value"] - row["ci_low"]) if row["value"] is not None and row.get("ci_low") is not None else 0.0
        for row in rows
    ]
    upper_err = [
        max(0.0, row["ci_high"] - row["value"]) if row["value"] is not None and row.get("ci_high") is not None else 0.0
        for row in rows
    ]

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    bars = ax.bar(
        y_positions,
        metric_values,
        color=PALETTE["summary"],
        alpha=0.9,
        width=0.65,
        edgecolor="#ffffff",
        linewidth=1.0,
    )
    ax.errorbar(
        y_positions,
        metric_values,
        yerr=[lower_err, upper_err],
        fmt="none",
        ecolor=PALETTE["neutral_dark"],
        elinewidth=1.5,
        capsize=4,
        capthick=1.5,
        zorder=3,
    )
    for bar, row in zip(bars, rows):
        if row["value"] is None:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            row["value"] + (0.03 if row["value"] >= 0 else -0.06),
            f"{row['value']:.3f}",
            ha="center",
            va="bottom" if row["value"] >= 0 else "top",
            fontsize=9.5,
            color=PALETTE["text"],
        )
    ax.set_xticks(y_positions, labels)
    bound_candidates = [
        row["value"]
        for row in rows
        if row["value"] is not None
    ]
    bound_candidates.extend(
        row["ci_low"]
        for row in rows
        if row.get("ci_low") is not None
    )
    bound_candidates.extend(
        row["ci_high"]
        for row in rows
        if row.get("ci_high") is not None
    )
    min_bound = min(bound_candidates) if bound_candidates else 0.0
    max_bound = max(bound_candidates) if bound_candidates else 1.0
    if min_bound >= 0.0 and max_bound <= 1.0:
        ax.set_ylim(0.0, 1.0)
    else:
        padding = max(0.08, 0.08 * (max_bound - min_bound or 1.0))
        ax.set_ylim(max(-1.0, min_bound - padding), min(1.0, max_bound + padding))
    ax.set_ylabel("Score")
    ax.set_title("Patient-Level Metric Summary", loc="left", pad=18)
    add_subtitle(ax, "Bars show point estimates; error bars show 95% confidence intervals where available")
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, output_path, dpi)


def plot_fold_distribution(bundle: RunArtifactBundle, output_path: Path, dpi: int) -> None:
    report = bundle.clinical_metrics
    patient_fold_reports = report.get("patient", {}).get("fold_metrics")
    if not patient_fold_reports:
        patient_fold_reports = compute_per_fold_metrics(
            bundle.patient_predictions,
            metric_names=["accuracy"],
            count_field_name="n_patients",
        )
    fold_values = [
        _safe_float(row.get("metrics", {}).get("accuracy"))
        for row in patient_fold_reports
        if isinstance(row, dict)
    ]
    fold_values = [value for value in fold_values if value is not None]
    if not fold_values:
        raise PlotSkipped("patient_fold_accuracy_unavailable")

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    violin = ax.violinplot([fold_values], positions=[1], widths=0.5, showmeans=False, showmedians=False, showextrema=False)
    for body in violin["bodies"]:
        body.set_facecolor(PALETTE["summary_band"])
        body.set_edgecolor(PALETTE["neutral_dark"])
        body.set_alpha(0.35)
    rng = np.random.default_rng(42)
    jitter = rng.normal(loc=1.0, scale=0.03, size=len(fold_values))
    ax.scatter(jitter, fold_values, s=34, color=PALETTE["summary"], alpha=0.82, edgecolors="#ffffff", linewidths=0.6, zorder=3)
    ax.hlines(np.mean(fold_values), 0.8, 1.2, colors=PALETTE["non_remission"], linewidth=2.2, label="Mean")
    ax.set_xticks([1], ["Patient fold accuracy"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Fold-Level Performance Distribution", loc="left", pad=18)
    add_subtitle(ax, f"{len(fold_values)} outer folds")
    style_axes(ax)
    ax.legend(loc="upper right", fontsize=9.2)
    fig.tight_layout()
    save_figure(fig, output_path, dpi)


def plot_calibration(bundle: RunArtifactBundle, output_path: Path, dpi: int) -> None:
    y_true, _, y_prob = extract_prediction_arrays(bundle.patient_predictions)
    if len(y_true) < 6 or len(set(y_true)) < 2 or len(set(round(prob, 6) for prob in y_prob)) < 3:
        raise PlotSkipped("insufficient_patient_probability_variation_for_calibration")

    n_bins = min(8, max(4, len(y_true) // 3))
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.plot([0, 1], [0, 1], linestyle="--", color=PALETTE["neutral_dark"], linewidth=1.2, alpha=0.8)
    ax.plot(mean_pred, frac_pos, marker="o", color=PALETTE["summary"], linewidth=2.4, markersize=5.8)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed remission frequency")
    ax.set_title("Patient-Level Calibration Plot", loc="left", pad=18)
    add_subtitle(ax, f"{len(y_true)} held-out participants, {n_bins} bins")
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, output_path, dpi)


def _selection_rows_or_skip(bundle: RunArtifactBundle) -> List[Dict[str, Any]]:
    rows = bundle.feature_selection_summary.get("selection_frequency_by_feature") or []
    if not rows:
        raise PlotSkipped("feature_selection_summary_unavailable")
    return [row for row in rows if isinstance(row, dict)]


def plot_selection_frequency(bundle: RunArtifactBundle, output_path: Path, dpi: int, top_features: int) -> None:
    rows = select_top_feature_rows(_selection_rows_or_skip(bundle), top_features, key="share")
    if not rows:
        raise PlotSkipped("selection_frequency_rows_unavailable")

    labels = [str(row["feature"]) for row in rows][::-1]
    values = [float(row.get("share", 0.0)) for row in rows][::-1]
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(9.6, max(5.4, 0.34 * len(rows) + 2.0)))
    ax.barh(labels, values, color=PALETTE["summary"], alpha=0.92)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Selection frequency")
    ax.set_title(f"Top {len(rows)} Stable Features by Selection Frequency", loc="left", pad=18)
    add_subtitle(ax, "Computed across outer-fold feature selections")
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, output_path, dpi)


def plot_selection_distribution(bundle: RunArtifactBundle, output_path: Path, dpi: int) -> None:
    rows = _selection_rows_or_skip(bundle)
    values = [_safe_float(row.get("share")) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        raise PlotSkipped("selection_frequency_distribution_unavailable")

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.hist(values, bins=min(12, max(6, len(values) // 2)), color=PALETTE["summary_band"], edgecolor=PALETTE["neutral_dark"], linewidth=0.9, alpha=0.85)
    ax.axvline(np.mean(values), color=PALETTE["summary"], linewidth=2.1, linestyle="-", label=f"Mean = {np.mean(values):.3f}")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Selection frequency")
    ax.set_ylabel("Feature count")
    ax.set_title("Selection Frequency Distribution", loc="left", pad=18)
    add_subtitle(ax, f"{len(values)} features")
    style_axes(ax)
    ax.legend(loc="upper right", fontsize=9.2)
    fig.tight_layout()
    save_figure(fig, output_path, dpi)


def plot_jaccard_heatmap(bundle: RunArtifactBundle, output_path: Path, dpi: int) -> None:
    if len(bundle.jaccard_fold_ids) < 2 or not bundle.jaccard_matrix:
        raise PlotSkipped("insufficient_fold_feature_sets_for_jaccard_heatmap")

    matrix = np.asarray(bundle.jaccard_matrix, dtype=float)
    labels = [str(fold_id) for fold_id in bundle.jaccard_fold_ids]
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Fold")
    ax.set_title("Pairwise Jaccard Similarity Across Folds", loc="left", pad=18)
    add_subtitle(ax, f"{len(labels)} fold feature sets")
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Jaccard similarity")
    fig.tight_layout()
    save_figure(fig, output_path, dpi)


def plot_jaccard_distribution(bundle: RunArtifactBundle, output_path: Path, dpi: int) -> None:
    values = bundle.jaccard_values
    if not values:
        raise PlotSkipped("insufficient_fold_feature_sets_for_jaccard_distribution")

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.hist(values, bins=min(12, max(6, len(values) // 2)), color=PALETTE["summary_band"], edgecolor=PALETTE["neutral_dark"], linewidth=0.9, alpha=0.85)
    ax.axvline(np.mean(values), color=PALETTE["summary"], linewidth=2.1, label=f"Mean = {np.mean(values):.3f}")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Pairwise Jaccard similarity")
    ax.set_ylabel("Pair count")
    ax.set_title("Pairwise Jaccard Distribution", loc="left", pad=18)
    add_subtitle(ax, f"{len(values)} fold pairs")
    style_axes(ax)
    ax.legend(loc="upper left", fontsize=9.2)
    fig.tight_layout()
    save_figure(fig, output_path, dpi)


def plot_kuncheva_summary(bundle: RunArtifactBundle, output_path: Path, dpi: int) -> None:
    summary = bundle.feature_selection_summary
    mean_value = _safe_float(summary.get("kuncheva_index_mean"))
    median_value = _safe_float(summary.get("kuncheva_index_median"))
    if mean_value is None and median_value is None:
        raise PlotSkipped(f"kuncheva_unavailable:{summary.get('kuncheva_reason', 'unknown')}")

    labels = []
    values = []
    if mean_value is not None:
        labels.append("Mean")
        values.append(mean_value)
    if median_value is not None:
        labels.append("Median")
        values.append(median_value)

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    bars = ax.bar(labels, values, color=[PALETTE["summary"], PALETTE["summary_band"][:]] if len(values) > 1 else [PALETTE["summary"]], alpha=0.92)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.03 if value >= 0 else -0.06),
            f"{value:.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=10,
        )
    min_bound = min(values)
    max_bound = max(values)
    if min_bound >= 0.0 and max_bound <= 1.0:
        ax.set_ylim(0.0, 1.0)
    else:
        padding = max(0.08, 0.08 * (max_bound - min_bound or 1.0))
        ax.set_ylim(max(-1.0, min_bound - padding), min(1.0, max_bound + padding))
    ax.set_ylabel("Kuncheva index")
    ax.set_title("Kuncheva Stability Summary", loc="left", pad=18)
    add_subtitle(ax, "Chance-adjusted overlap under fixed-size selection")
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, output_path, dpi)


def _class_conditional_rows(bundle: RunArtifactBundle) -> List[Dict[str, Any]]:
    rows = _selection_rows_or_skip(bundle)
    usable = [
        row
        for row in rows
        if row.get("share_remission") is not None and row.get("share_non_remission") is not None
    ]
    if not usable:
        raise PlotSkipped("class_conditional_selection_unavailable")
    return usable


def plot_delta_vs_frequency(bundle: RunArtifactBundle, output_path: Path, dpi: int, top_features: int) -> None:
    rows = merge_effect_frequency_rows(bundle)
    usable = [
        row
        for row in rows
        if row.get("delta_remission_minus_non_remission") is not None and row.get("share") is not None
    ]
    if not usable:
        raise PlotSkipped("delta_frequency_data_unavailable")

    color_values = [_safe_float(row.get("cohens_d_mean")) or 0.0 for row in usable]
    max_abs = max(abs(value) for value in color_values) or 1.0
    cmap = plt.get_cmap("coolwarm")
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    scatter = ax.scatter(
        [row["delta_remission_minus_non_remission"] for row in usable],
        [row["share"] for row in usable],
        c=color_values,
        cmap=cmap,
        vmin=-max_abs,
        vmax=max_abs,
        s=70,
        alpha=0.88,
        edgecolors="#ffffff",
        linewidths=0.7,
    )
    ax.axvline(0.0, color=PALETTE["neutral_dark"], linewidth=1.2, linestyle="--", alpha=0.8)
    for row in select_top_feature_rows(usable, min(top_features, 8), key="share"):
        ax.annotate(
            str(row["feature"]),
            (row["delta_remission_minus_non_remission"], row["share"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8.8,
            color=PALETTE["text"],
        )
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Delta remission minus non-remission")
    ax.set_ylabel("Selection frequency")
    ax.set_title("Class-Conditional Selection Bias vs Stability", loc="left", pad=18)
    add_subtitle(ax, "Point color encodes mean Cohen's d")
    style_axes(ax)
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Mean Cohen's d")
    fig.tight_layout()
    save_figure(fig, output_path, dpi)


def plot_class_conditional_bars(bundle: RunArtifactBundle, output_path: Path, dpi: int, top_features: int) -> None:
    rows = select_top_feature_rows(_class_conditional_rows(bundle), top_features, key="share")
    labels = [str(row["feature"]) for row in rows][::-1]
    remission = [float(row.get("share_remission", 0.0)) for row in rows][::-1]
    non_remission = [float(row.get("share_non_remission", 0.0)) for row in rows][::-1]
    positions = np.arange(len(rows))

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(10.2, max(5.6, 0.36 * len(rows) + 2.0)))
    ax.barh(positions - 0.18, remission, height=0.34, color=PALETTE["remission"], label="Remission")
    ax.barh(positions + 0.18, non_remission, height=0.34, color=PALETTE["non_remission"], label="Non-remission")
    ax.set_yticks(positions, labels)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Conditional selection frequency")
    ax.set_title("Class-Conditional Selection Frequencies", loc="left", pad=18)
    add_subtitle(ax, f"Top {len(rows)} features by overall selection frequency")
    style_axes(ax)
    ax.legend(loc="lower right", fontsize=9.3)
    fig.tight_layout()
    save_figure(fig, output_path, dpi)


def plot_delta_distribution(bundle: RunArtifactBundle, output_path: Path, dpi: int) -> None:
    rows = _class_conditional_rows(bundle)
    values = [_safe_float(row.get("delta_remission_minus_non_remission")) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        raise PlotSkipped("delta_distribution_unavailable")

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.hist(values, bins=min(14, max(6, len(values) // 2)), color=PALETTE["summary_band"], edgecolor=PALETTE["neutral_dark"], linewidth=0.9, alpha=0.85)
    ax.axvline(0.0, color=PALETTE["neutral_dark"], linewidth=1.2, linestyle="--")
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Delta remission minus non-remission")
    ax.set_ylabel("Feature count")
    ax.set_title("Distribution of Class-Conditional Selection Deltas", loc="left", pad=18)
    add_subtitle(ax, f"{len(values)} features")
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, output_path, dpi)


def _effect_rows_or_skip(bundle: RunArtifactBundle) -> List[Dict[str, Any]]:
    effect_stability = bundle.feature_selection_summary.get("effect_stability", {})
    rows = effect_stability.get("features") if isinstance(effect_stability, dict) else None
    if not rows:
        raise PlotSkipped("effect_stability_unavailable")
    return [row for row in rows if isinstance(row, dict)]


def plot_effect_size(bundle: RunArtifactBundle, output_path: Path, dpi: int, top_features: int) -> None:
    rows = select_top_feature_rows(_effect_rows_or_skip(bundle), top_features, key="selection_share")
    if not rows:
        raise PlotSkipped("effect_size_rows_unavailable")

    labels = [str(row["feature"]) for row in rows][::-1]
    values = [float(row.get("cohens_d_mean", 0.0)) for row in rows][::-1]
    errors = [float(row.get("cohens_d_std", 0.0) or 0.0) for row in rows][::-1]
    colors = [PALETTE["remission"] if value >= 0 else PALETTE["non_remission"] for value in values]

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(10.2, max(5.6, 0.36 * len(rows) + 2.0)))
    positions = np.arange(len(rows))
    ax.barh(positions, values, xerr=errors, color=colors, alpha=0.9, ecolor=PALETTE["neutral_dark"], capsize=3)
    ax.axvline(0.0, color=PALETTE["neutral_dark"], linewidth=1.2, linestyle="--")
    ax.set_yticks(positions, labels)
    ax.set_xlabel("Mean Cohen's d")
    ax.set_title("Effect Size Stability Across Folds", loc="left", pad=18)
    add_subtitle(ax, "Bars show mean effect size; error bars show fold-wise standard deviation")
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, output_path, dpi)


def plot_sign_consistency(bundle: RunArtifactBundle, output_path: Path, dpi: int, top_features: int) -> None:
    rows = select_top_feature_rows(_effect_rows_or_skip(bundle), top_features, key="selection_share")
    labels = [str(row["feature"]) for row in rows][::-1]
    values = [float(row.get("sign_consistency", 0.0)) for row in rows][::-1]
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(10.2, max(5.6, 0.36 * len(rows) + 2.0)))
    positions = np.arange(len(rows))
    ax.barh(positions, values, color=PALETTE["summary"], alpha=0.9)
    ax.set_yticks(positions, labels)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Sign consistency")
    ax.set_title("Sign Consistency of Biomarker Effects", loc="left", pad=18)
    add_subtitle(ax, "1.0 means the effect direction never changed across folds")
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, output_path, dpi)


def plot_effect_vs_frequency(bundle: RunArtifactBundle, output_path: Path, dpi: int, top_features: int) -> None:
    rows = merge_effect_frequency_rows(bundle)
    usable = [
        row
        for row in rows
        if row.get("share") is not None and row.get("cohens_d_mean") is not None
    ]
    if not usable:
        raise PlotSkipped("effect_frequency_data_unavailable")

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    colors = [
        PALETTE["remission"] if float(row["cohens_d_mean"]) >= 0 else PALETTE["non_remission"]
        for row in usable
    ]
    ax.scatter(
        [row["share"] for row in usable],
        [row["cohens_d_mean"] for row in usable],
        c=colors,
        s=70,
        alpha=0.88,
        edgecolors="#ffffff",
        linewidths=0.7,
    )
    ax.axhline(0.0, color=PALETTE["neutral_dark"], linewidth=1.2, linestyle="--", alpha=0.8)
    for row in select_top_feature_rows(usable, min(top_features, 8), key="share"):
        ax.annotate(
            str(row["feature"]),
            (row["share"], row["cohens_d_mean"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8.8,
            color=PALETTE["text"],
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Selection frequency")
    ax.set_ylabel("Mean Cohen's d")
    ax.set_title("Effect Size vs Biomarker Stability", loc="left", pad=18)
    add_subtitle(ax, "Red = positive effect, blue = negative effect")
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, output_path, dpi)


FIGURE_SPECS = {
    "roc": ("performance", "patient_roc_curve", plot_patient_roc_curve),
    "pr": ("performance", "patient_pr_curve", plot_patient_pr_curve),
    "confusion": ("performance", "patient_confusion_matrix", plot_patient_confusion_matrix),
    "metric_summary": ("performance", "patient_metric_summary", plot_metric_summary),
    "fold_distribution": ("fold_behavior", "patient_fold_accuracy_distribution", plot_fold_distribution),
    "calibration": ("fold_behavior", "patient_calibration_plot", plot_calibration),
    "selection_frequency": ("biomarker_stability", "selection_frequency_top_features", plot_selection_frequency),
    "selection_distribution": ("biomarker_stability", "selection_frequency_distribution", plot_selection_distribution),
    "jaccard_heatmap": ("biomarker_stability", "pairwise_jaccard_heatmap", plot_jaccard_heatmap),
    "jaccard_distribution": ("biomarker_stability", "pairwise_jaccard_distribution", plot_jaccard_distribution),
    "kuncheva_summary": ("biomarker_stability", "kuncheva_summary", plot_kuncheva_summary),
    "delta_scatter": ("biomarker_interpretation", "delta_vs_frequency_scatter", plot_delta_vs_frequency),
    "class_conditional_bars": ("biomarker_interpretation", "class_conditional_selection_bars", plot_class_conditional_bars),
    "delta_distribution": ("biomarker_interpretation", "delta_distribution", plot_delta_distribution),
    "effect_size": ("biomarker_interpretation", "effect_size_top_features", plot_effect_size),
    "sign_consistency": ("biomarker_interpretation", "sign_consistency_top_features", plot_sign_consistency),
    "effect_frequency": ("biomarker_interpretation", "effect_size_vs_frequency_scatter", plot_effect_vs_frequency),
}


def resolve_selected_result(args: argparse.Namespace) -> Optional[SweepResultRow]:
    paths = discover_results_files(Path(args.artifacts_dir), args.results_glob)
    rows = deduplicate_results_rows(load_results_rows(paths))
    filtered = filter_result_rows(
        rows,
        run_signature=args.run_signature,
        mlflow_run_id=args.mlflow_run_id,
        model=args.model,
        window_size=args.window_size,
        inner_k=args.inner_k,
        outer_k=args.outer_k,
        fs_method=args.fs_method,
        n_features=args.n_features,
        ordering=args.ordering,
        equalize_lopo_groups=args.equalize_lopo_groups,
        use_smote=args.use_smote,
    )
    return select_latest_result_row(filtered)


def build_selection_slug(selected_row: Optional[SweepResultRow], run_id: str, args: argparse.Namespace) -> str:
    parts = [f"run-{run_id[:8]}"]
    if selected_row is not None:
        if selected_row.model:
            parts.append(f"model-{slugify(selected_row.model)}")
        if selected_row.window_seconds is not None:
            parts.append(f"ws-{selected_row.window_seconds}")
        if selected_row.inner_k is not None:
            parts.append(f"ik-{selected_row.inner_k}")
        if selected_row.outer_k is not None:
            parts.append(f"ok-{selected_row.outer_k}")
        if selected_row.equalize_lopo_groups is not None:
            parts.append(f"eq-{selected_row.equalize_lopo_groups}")
        if selected_row.use_smote is not None:
            parts.append(f"sm-{selected_row.use_smote}")
    else:
        if args.model:
            parts.append(f"model-{slugify(args.model)}")
    return "__".join(parts)


def record_manifest_entry(manifest: Dict[str, Any], *, figure_id: str, path: Optional[Path] = None, reason: Optional[str] = None) -> None:
    if path is not None:
        manifest["generated_figures"].append(
            {
                "figure_id": figure_id,
                "path": str(path),
            }
        )
        return
    manifest["skipped_figures"].append(
        {
            "figure_id": figure_id,
            "reason": reason or "unknown",
        }
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    requested_figures = ALL_FIGURES if not args.figures else tuple(part.strip() for part in args.figures.split(",") if part.strip())
    invalid = [figure_id for figure_id in requested_figures if figure_id not in FIGURE_SPECS]
    if invalid:
        raise SystemExit(f"Unknown figure ids: {', '.join(invalid)}")

    selected_row = resolve_selected_result(args)
    if args.mlflow_run_id:
        run_id = args.mlflow_run_id
        experiment_id = selected_row.mlflow_experiment_id if selected_row is not None else None
    else:
        if selected_row is None:
            raise SystemExit("No successful results row matched the provided filters.")
        run_id = selected_row.mlflow_run_id
        experiment_id = selected_row.mlflow_experiment_id

    bundle = load_run_artifact_bundle(
        mlruns_root=Path(args.mlruns_root),
        run_id=run_id,
        experiment_id=experiment_id,
    )
    selection_slug = build_selection_slug(selected_row, run_id, args)
    output_root = Path(args.output_dir) / selection_slug

    manifest: Dict[str, Any] = {
        "selected_run": {
            "mlflow_run_id": bundle.run_id,
            "mlflow_experiment_id": bundle.experiment_id,
            "run_dir": str(bundle.run_dir),
            "artifacts_dir": str(bundle.artifacts_dir),
            "resolved_from_results_row": selected_row.source_file if selected_row is not None else None,
        },
        "filters": {
            "run_signature": args.run_signature,
            "mlflow_run_id": args.mlflow_run_id,
            "model": args.model,
            "window_size": args.window_size,
            "inner_k": args.inner_k,
            "outer_k": args.outer_k,
            "fs_method": args.fs_method,
            "n_features": args.n_features,
            "ordering": args.ordering,
            "equalize_lopo_groups": args.equalize_lopo_groups,
            "use_smote": args.use_smote,
        },
        "requested_figures": list(requested_figures),
        "data_sources": {
            "results_artifacts_dir": str(Path(args.artifacts_dir)),
            "results_glob": args.results_glob,
            "mlruns_root": str(Path(args.mlruns_root)),
            "nested_fold_runs": len(bundle.nested_fold_runs),
            "patient_prediction_rows": len(bundle.patient_predictions),
            "window_prediction_rows": len(bundle.window_predictions),
        },
        "generated_figures": [],
        "skipped_figures": [],
    }

    for figure_id in requested_figures:
        subdir, base_name, plotter = FIGURE_SPECS[figure_id]
        output_path = output_root / subdir / f"{base_name}.{args.format}"
        try:
            if figure_id in {"metric_summary"}:
                plotter(
                    bundle,
                    output_path,
                    args.dpi,
                    args.bootstrap_iterations,
                    args.random_seed,
                )
            elif figure_id in {"selection_frequency", "delta_scatter", "class_conditional_bars", "effect_size", "sign_consistency", "effect_frequency"}:
                plotter(bundle, output_path, args.dpi, args.top_features)
            else:
                plotter(bundle, output_path, args.dpi)
        except PlotSkipped as exc:
            record_manifest_entry(manifest, figure_id=figure_id, reason=str(exc))
        else:
            record_manifest_entry(manifest, figure_id=figure_id, path=output_path)

    manifest_path = output_root / "figure_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"Resolved MLflow run id: {bundle.run_id}")
    print(f"Artifacts dir: {bundle.artifacts_dir}")
    print(f"Nested fold runs: {len(bundle.nested_fold_runs)}")
    print(f"Generated figures: {len(manifest['generated_figures'])}")
    print(f"Skipped figures: {len(manifest['skipped_figures'])}")
    print(f"Manifest: {manifest_path}")
    for entry in manifest["generated_figures"]:
        print(f"Saved: {entry['path']}")
    for entry in manifest["skipped_figures"]:
        print(f"Skipped: {entry['figure_id']} ({entry['reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
