#!/usr/bin/env python3
"""Build manuscript summary figures and tables from sweep and MLflow artifacts."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent
EEG_ANALYSIS_ROOT = REPO_ROOT / "eeg_analysis"
if str(EEG_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(EEG_ANALYSIS_ROOT))

PLOT_SWEEP_SCRIPT_PATH = Path(__file__).resolve().parent / "plot_sweep_roc_auc.py"
PLOT_SWEEP_SCRIPT_SPEC = importlib.util.spec_from_file_location("plot_sweep_roc_auc_module", PLOT_SWEEP_SCRIPT_PATH)
sweep_plots = importlib.util.module_from_spec(PLOT_SWEEP_SCRIPT_SPEC)
assert PLOT_SWEEP_SCRIPT_SPEC is not None and PLOT_SWEEP_SCRIPT_SPEC.loader is not None
sys.modules[PLOT_SWEEP_SCRIPT_SPEC.name] = sweep_plots
PLOT_SWEEP_SCRIPT_SPEC.loader.exec_module(sweep_plots)

from src.utils.paper_plot_style import PALETTE, apply_paper_style, save_figure, style_axes  # noqa: E402
from src.utils.plot_data_loader import (  # noqa: E402
    SweepResultRow,
    deduplicate_results_rows,
    discover_results_files,
    load_results_rows,
    load_run_artifact_bundle,
)


MODEL_ORDER = ("advanced_hybrid_1dcnn_lstm", "svm_linear")
MODEL_LABELS = {
    "advanced_hybrid_1dcnn_lstm": "Hybrid CNN-LSTM",
    "svm_linear": "Linear SVM",
}
MODEL_COLORS = {
    "advanced_hybrid_1dcnn_lstm": PALETTE["summary"],
    "svm_linear": PALETTE["non_remission"],
}
METRIC_PANELS = (
    {
        "panel_label": "A",
        "metric_key": "patient_roc_auc",
        "x_field": "window_seconds",
        "title": "ROC-AUC vs Window Size",
        "y_label": "Patient ROC-AUC",
        "clamp_min": 0.0,
        "clamp_max": 1.0,
        "highlight_best": True,
    },
    {
        "panel_label": "B",
        "metric_key": "patient_roc_auc",
        "x_field": "inner_k",
        "title": "ROC-AUC vs Inner-k",
        "y_label": "Patient ROC-AUC",
        "clamp_min": 0.0,
        "clamp_max": 1.0,
        "highlight_best": True,
    },
    {
        "panel_label": "C",
        "metric_key": "feature_selection_mean_pairwise_jaccard",
        "x_field": "inner_k",
        "title": "Mean Pairwise Jaccard vs Inner-k",
        "y_label": "Mean pairwise Jaccard",
        "clamp_min": 0.0,
        "clamp_max": 1.0,
        "highlight_best": False,
        "group_mode": "shared",
    },
    {
        "panel_label": "D",
        "metric_key": "feature_selection_unique_feature_count",
        "x_field": "inner_k",
        "title": "Unique Selected Features vs Inner-k",
        "y_label": "Unique selected features",
        "clamp_min": 0.0,
        "clamp_max": None,
        "highlight_best": False,
        "group_mode": "shared",
    },
)

COHORT_METADATA = {
    "parent_trial_description": "Fully remote, multisite, sham-controlled home-based tDCS trial for MDD",
    "parent_trial_citation_key": "woodham2025home",
    "female_count": 18,
    "male_count": 3,
    "age_mean_years": 37.1,
    "age_sd_years": 9.7,
    "rest_condition": "Eyes-closed resting state",
    "montage_channels": ("AF7", "AF8", "TP9", "TP10"),
    "channel_coverage": "Frontal (AF7, AF8), temporal (TP9, TP10)",
    "sampling_rate_hz": 256,
    "recording_duration_minutes": 10,
    "recording_file_structure": "single file or two 5 min files",
    "available_export_segment_seconds": 10,
    "samples_per_segment": 2560,
    "total_windows": 1203,
    "remission_windows": 393,
    "non_remission_windows": 810,
    "class_ratio": "2.06:1 (non-remission:remission)",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manuscript summary figures and tables.")
    parser.add_argument("--artifacts-dir", default="sweeps/artifacts", help="Directory containing sweep results ledgers.")
    parser.add_argument("--results-glob", default="*.results.jsonl", help="Glob for sweep results ledgers.")
    parser.add_argument("--mlruns-root", default="mlruns", help="Local MLflow file-backend root.")
    parser.add_argument("--output-root", default="paper", help="Paper workspace root.")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    return parser.parse_args(argv)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _load_metric_rows(results_paths: Sequence[Path], metric_key: str) -> List[Any]:
    rows, _ = sweep_plots.load_plot_rows(results_paths, metric_key)
    return sweep_plots.deduplicate_plot_rows(rows)


def _select_best_row(rows: Sequence[SweepResultRow], model: str) -> SweepResultRow:
    model_rows = [row for row in rows if row.model == model and row.patient_roc_auc is not None]
    if not model_rows:
        raise RuntimeError(f"No deduplicated results rows found for model={model}")
    return max(model_rows, key=lambda row: (row.patient_roc_auc or float("-inf"), row.created_at_sort, row.attempt_index))


def _compute_model_envelopes(
    rows: Sequence[Any],
    x_field: str,
    *,
    clamp_min: Optional[float],
    clamp_max: Optional[float],
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.model not in MODEL_ORDER:
            continue
        x_value = getattr(row, x_field)
        if x_value is None:
            continue
        grouped[str(row.model)][int(x_value)].append(float(row.metric_value))

    envelopes: Dict[str, List[Dict[str, Any]]] = {}
    for model, by_x in grouped.items():
        model_rows: List[Dict[str, Any]] = []
        for x_value, values in sorted(by_x.items()):
            if not values:
                continue
            mean_value = sum(values) / len(values)
            if len(values) > 1:
                variance = sum((value - mean_value) ** 2 for value in values) / len(values)
                std_value = math.sqrt(variance)
                ci_half_width = 1.96 * std_value / math.sqrt(len(values))
            else:
                ci_half_width = 0.0
            ci_low = mean_value - ci_half_width
            ci_high = mean_value + ci_half_width
            if clamp_min is not None:
                ci_low = max(clamp_min, ci_low)
                ci_high = max(clamp_min, ci_high)
            if clamp_max is not None:
                ci_low = min(clamp_max, ci_low)
                ci_high = min(clamp_max, ci_high)
            model_rows.append(
                {
                    "model": model,
                    "x_value": x_value,
                    "mean_value": mean_value,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n": len(values),
                }
            )
        envelopes[model] = model_rows
    return envelopes


def _compute_shared_envelope(
    rows: Sequence[Any],
    x_field: str,
    *,
    clamp_min: Optional[float],
    clamp_max: Optional[float],
) -> List[Dict[str, Any]]:
    per_setting: Dict[tuple[Any, ...], List[float]] = defaultdict(list)
    grouped: Dict[int, List[float]] = defaultdict(list)
    for row in rows:
        x_value = getattr(row, x_field)
        if x_value is None:
            continue
        setting_key = (
            row.metric_key,
            row.window_seconds,
            row.inner_k,
            row.outer_k,
            row.fs_enabled,
            row.fs_method,
            row.n_features,
            row.ordering,
            row.dataset_run_id,
            row.equalize_lopo_groups,
            row.use_smote,
        )
        per_setting[setting_key].append(float(row.metric_value))

    for setting_key, values in per_setting.items():
        x_value = setting_key[1] if x_field == "window_seconds" else setting_key[2]
        if x_value is None:
            continue
        grouped[int(x_value)].append(sum(values) / len(values))

    envelope: List[Dict[str, Any]] = []
    for x_value, values in sorted(grouped.items()):
        if not values:
            continue
        mean_value = sum(values) / len(values)
        if len(values) > 1:
            variance = sum((value - mean_value) ** 2 for value in values) / len(values)
            std_value = math.sqrt(variance)
            ci_half_width = 1.96 * std_value / math.sqrt(len(values))
        else:
            ci_half_width = 0.0
        ci_low = mean_value - ci_half_width
        ci_high = mean_value + ci_half_width
        if clamp_min is not None:
            ci_low = max(clamp_min, ci_low)
            ci_high = max(clamp_min, ci_high)
        if clamp_max is not None:
            ci_low = min(clamp_max, ci_low)
            ci_high = min(clamp_max, ci_high)
        envelope.append(
            {
                "model": "shared_feature_selection",
                "x_value": x_value,
                "mean_value": mean_value,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n": len(values),
            }
        )
    return envelope


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_number(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def _format_int(value: Optional[int]) -> str:
    if value is None:
        return "--"
    return str(int(value))


def _build_best_run_comparison_rows(best_runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in best_runs:
        patient = item["patient_metrics"]
        feature = item["feature_selection"]
        rows.append(
            {
                "model": item["model_label"],
                "window_seconds": item["window_seconds"],
                "inner_k": item["inner_k"],
                "outer_k": item["outer_k"],
                "patient_roc_auc": patient.get("roc_auc"),
                "patient_pr_auc": patient.get("pr_auc"),
                "patient_balanced_accuracy": patient.get("balanced_accuracy"),
                "patient_f1": patient.get("f1"),
                "patient_mcc": patient.get("mcc"),
                "average_features_per_fold": feature.get("average_features_per_fold"),
                "unique_feature_count": feature.get("unique_feature_count"),
                "mean_pairwise_jaccard": feature.get("mean_pairwise_jaccard"),
                "kuncheva_index_mean": feature.get("kuncheva_index_mean"),
            }
        )
    return rows


def _comparison_table_tex(rows: Sequence[Dict[str, Any]]) -> str:
    body_lines = [
        r"\begin{tabular}{lcccccccccc}",
        r"\toprule",
        r"Model & Window & Inner-k & ROC-AUC & PR-AUC & Bal.\ Acc. & F1 & MCC & Feat./fold & Unique feat. & Mean Jaccard \\",
        r"\midrule",
    ]
    for row in rows:
        body_lines.append(
            " & ".join(
                [
                    row["model"],
                    _format_int(row["window_seconds"]),
                    _format_int(row["inner_k"]),
                    _format_number(_safe_float(row["patient_roc_auc"])),
                    _format_number(_safe_float(row["patient_pr_auc"])),
                    _format_number(_safe_float(row["patient_balanced_accuracy"])),
                    _format_number(_safe_float(row["patient_f1"])),
                    _format_number(_safe_float(row["patient_mcc"])),
                    _format_number(_safe_float(row["average_features_per_fold"]), digits=1),
                    _format_int(row["unique_feature_count"]),
                    _format_number(_safe_float(row["mean_pairwise_jaccard"])),
                ]
            )
            + r" \\"
        )
    body_lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(body_lines) + "\n"


def _build_hybrid_biomarker_rows(bundle: Any) -> List[Dict[str, Any]]:
    selection_rows = bundle.feature_selection_summary.get("selection_frequency_by_feature") or []
    effect_rows = (
        bundle.feature_selection_summary.get("effect_stability", {}).get("features")
        if isinstance(bundle.feature_selection_summary.get("effect_stability"), dict)
        else []
    ) or []
    effect_by_feature = {
        row.get("feature"): row
        for row in effect_rows
        if isinstance(row, dict) and row.get("feature")
    }

    rows: List[Dict[str, Any]] = []
    for selection_row in selection_rows:
        feature = selection_row.get("feature")
        effect_row = effect_by_feature.get(feature, {})
        mean_d = _safe_float(effect_row.get("cohens_d_mean"))
        if mean_d is None:
            direction = "NA"
        elif mean_d > 0:
            direction = "Remission > non-remission"
        elif mean_d < 0:
            direction = "Non-remission > remission"
        else:
            direction = "No direction"
        rows.append(
            {
                "feature": feature,
                "fold_count": selection_row.get("count"),
                "selection_share": selection_row.get("share"),
                "delta_remission_minus_non_remission": selection_row.get("delta_remission_minus_non_remission"),
                "sign_consistency": effect_row.get("sign_consistency"),
                "cohens_d_mean": mean_d,
                "direction": direction,
            }
        )
    return rows


def _biomarker_table_tex(rows: Sequence[Dict[str, Any]]) -> str:
    body_lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Feature & Folds & Share & $\Delta$ sel. & Sign cons. & Mean $d$ & Effect direction \\",
        r"\midrule",
    ]
    for row in rows:
        feature_name = str(row["feature"]).replace("_", r"\_")
        body_lines.append(
            " & ".join(
                [
                    feature_name,
                    _format_int(row["fold_count"]),
                    _format_number(_safe_float(row["selection_share"])),
                    _format_number(_safe_float(row["delta_remission_minus_non_remission"])),
                    _format_number(_safe_float(row["sign_consistency"])),
                    _format_number(_safe_float(row["cohens_d_mean"])),
                    str(row["direction"]),
                ]
            )
            + r" \\"
        )
    body_lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(body_lines) + "\n"


def _cohort_summary_facts(hybrid_bundle: Any, window_sizes: Sequence[int], inner_ks: Sequence[int]) -> Dict[str, Any]:
    patient_rows = hybrid_bundle.patient_predictions
    n_patients = len(patient_rows)
    n_remission = sum(1 for row in patient_rows if row.get("true_label") == 1)
    n_non_remission = sum(1 for row in patient_rows if row.get("true_label") == 0)
    return {
        "participants": n_patients,
        "remission_patients": n_remission,
        "non_remission_patients": n_non_remission,
        "female_count": COHORT_METADATA["female_count"],
        "male_count": COHORT_METADATA["male_count"],
        "age_mean_years": COHORT_METADATA["age_mean_years"],
        "age_sd_years": COHORT_METADATA["age_sd_years"],
        "rest_condition": COHORT_METADATA["rest_condition"],
        "montage_channels": list(COHORT_METADATA["montage_channels"]),
        "channel_coverage": COHORT_METADATA["channel_coverage"],
        "sampling_rate_hz": COHORT_METADATA["sampling_rate_hz"],
        "recording_duration_minutes": COHORT_METADATA["recording_duration_minutes"],
        "recording_file_structure": COHORT_METADATA["recording_file_structure"],
        "available_export_segment_seconds": COHORT_METADATA["available_export_segment_seconds"],
        "samples_per_segment": COHORT_METADATA["samples_per_segment"],
        "total_windows": COHORT_METADATA["total_windows"],
        "remission_windows": COHORT_METADATA["remission_windows"],
        "non_remission_windows": COHORT_METADATA["non_remission_windows"],
        "class_ratio": COHORT_METADATA["class_ratio"],
        "window_sizes_seconds": list(window_sizes),
        "inner_k_values": list(inner_ks),
        "outer_k": 10,
        "feature_selector": "SelectKBest (f-classif)",
        "validation": "Leave-one-participant-out",
        "evaluation_level": "Patient-level aggregation",
        "training_adjustments": "LOPO-group equalization + SMOTE",
    }


def _cohort_summary_tex(hybrid_bundle: Any, window_sizes: Sequence[int], inner_ks: Sequence[int]) -> str:
    summary = _cohort_summary_facts(hybrid_bundle, window_sizes, inner_ks)
    remission_window_share = 100.0 * summary["remission_windows"] / summary["total_windows"]
    non_remission_window_share = 100.0 * summary["non_remission_windows"] / summary["total_windows"]
    lines = [
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.42\linewidth}>{\raggedright\arraybackslash}p{0.50\linewidth}}",
        r"\toprule",
        r"Characteristic & Value \\",
        r"\midrule",
        r"\multicolumn{2}{l}{\textbf{Parent trial and cohort descriptors}} \\",
        r"\addlinespace[0.2em]",
        rf"Parent trial & {COHORT_METADATA['parent_trial_description']} \citep{{{COHORT_METADATA['parent_trial_citation_key']}}} \\",
        rf"Participants & {summary['participants']} \\",
        rf"Outcome groups & {summary['remission_patients']} remission, {summary['non_remission_patients']} non-remission \\",
        rf"Sex (female/male) & {summary['female_count']}/{summary['male_count']} \\",
        rf"Age (years, mean $\pm$ SD) & {summary['age_mean_years']:.1f} $\pm$ {summary['age_sd_years']:.1f} \\",
        r"\midrule",
        r"\multicolumn{2}{l}{\textbf{EEG acquisition and export descriptors}} \\",
        r"\addlinespace[0.2em]",
        rf"Rest condition & {summary['rest_condition']} \\",
        rf"EEG montage & {len(summary['montage_channels'])} channels ({', '.join(summary['montage_channels'])}) \\",
        rf"Channel coverage & {summary['channel_coverage']} \\",
        rf"Sampling rate & {summary['sampling_rate_hz']} Hz \\",
        rf"Recording structure & {summary['recording_duration_minutes']} min total; {summary['recording_file_structure']} \\",
        rf"Available export segment length & {summary['available_export_segment_seconds']} s ({summary['samples_per_segment']:,} samples/channel) \\",
        rf"Total exported windows & {summary['total_windows']:,} \\",
        rf"Remission windows & {summary['remission_windows']:,} ({remission_window_share:.1f}\%) \\",
        rf"Non-remission windows & {summary['non_remission_windows']:,} ({non_remission_window_share:.1f}\%) \\",
        rf"Class ratio & {summary['class_ratio']} \\",
        r"\midrule",
        r"\multicolumn{2}{l}{\textbf{Current sweep methodology settings}} \\",
        r"\addlinespace[0.2em]",
        rf"Sweep window sizes & {', '.join(str(value) for value in summary['window_sizes_seconds'])} s \\",
        rf"Inner-k sweep & {', '.join(str(value) for value in summary['inner_k_values'])} \\",
        r"Outer-k & 10 \\",
        r"Feature selector & SelectKBest ($f$-classif) \\",
        r"Training adjustments & LOPO-group equalization + SMOTE \\",
        r"Validation & Leave-one-participant-out \\",
        r"Evaluation level & Patient-level aggregation \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def _facts_for_best_run(best_row: SweepResultRow, bundle: Any) -> Dict[str, Any]:
    patient_metrics = bundle.clinical_metrics.get("patient", {}).get("metrics", {})
    roc_ci = bundle.clinical_metrics.get("patient", {}).get("confidence_intervals", {}).get("roc_auc", {})
    permutation = bundle.clinical_metrics.get("stats", {}).get("permutation_test", {})
    feature_selection = bundle.feature_selection_summary
    return {
        "model": best_row.model,
        "model_label": MODEL_LABELS.get(best_row.model or "", best_row.model or "Unknown model"),
        "mlflow_run_id": best_row.mlflow_run_id,
        "mlflow_experiment_id": best_row.mlflow_experiment_id,
        "window_seconds": best_row.window_seconds,
        "inner_k": best_row.inner_k,
        "outer_k": best_row.outer_k,
        "equalize_lopo_groups": best_row.equalize_lopo_groups,
        "use_smote": best_row.use_smote,
        "patient_metrics": patient_metrics,
        "roc_auc_ci": roc_ci,
        "permutation_test": permutation,
        "feature_selection": {
            "average_features_per_fold": feature_selection.get("average_features_per_fold"),
            "unique_feature_count": feature_selection.get("unique_feature_count"),
            "mean_pairwise_jaccard": feature_selection.get("mean_pairwise_jaccard"),
            "kuncheva_index_mean": feature_selection.get("kuncheva_index_mean"),
            "top_feature": feature_selection.get("top_feature"),
            "top_feature_frequency": feature_selection.get("top_feature_frequency"),
            "top_feature_share": feature_selection.get("top_feature_share"),
            "mean_sign_consistency": (feature_selection.get("effect_stability") or {}).get("mean_sign_consistency")
            if isinstance(feature_selection.get("effect_stability"), dict)
            else None,
        },
    }


def _plot_summary_figure(
    figure_path: Path,
    aggregated_csv_path: Path,
    results_paths: Sequence[Path],
    best_rows: Dict[str, SweepResultRow],
    *,
    dpi: int,
) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.2), facecolor=PALETTE["figure_face"])
    aggregated_rows: List[Dict[str, Any]] = []

    for ax, panel in zip(axes.flat, METRIC_PANELS):
        metric_rows = _load_metric_rows(results_paths, panel["metric_key"])
        if panel.get("group_mode") == "shared":
            shared_series = _compute_shared_envelope(
                metric_rows,
                panel["x_field"],
                clamp_min=panel["clamp_min"],
                clamp_max=panel["clamp_max"],
            )
            if shared_series:
                x_values = [row["x_value"] for row in shared_series]
                means = [row["mean_value"] for row in shared_series]
                lowers = [row["ci_low"] for row in shared_series]
                uppers = [row["ci_high"] for row in shared_series]
                ax.fill_between(x_values, lowers, uppers, color=PALETTE["neutral_dark"], alpha=0.12)
                ax.plot(
                    x_values,
                    means,
                    color=PALETTE["neutral_dark"],
                    linewidth=2.6,
                    marker="o",
                    markersize=5.6,
                )
                for row in shared_series:
                    aggregated_rows.append(
                        {
                            "metric_key": panel["metric_key"],
                            "x_field": panel["x_field"],
                            "model": row["model"],
                            "model_label": "Shared feature selection",
                            "x_value": row["x_value"],
                            "mean_value": row["mean_value"],
                            "ci_low": row["ci_low"],
                            "ci_high": row["ci_high"],
                            "n": row["n"],
                        }
                    )
        else:
            envelopes = _compute_model_envelopes(
                metric_rows,
                panel["x_field"],
                clamp_min=panel["clamp_min"],
                clamp_max=panel["clamp_max"],
            )
            for model in MODEL_ORDER:
                series = envelopes.get(model) or []
                if not series:
                    continue
                x_values = [row["x_value"] for row in series]
                means = [row["mean_value"] for row in series]
                lowers = [row["ci_low"] for row in series]
                uppers = [row["ci_high"] for row in series]
                color = MODEL_COLORS[model]
                ax.fill_between(x_values, lowers, uppers, color=color, alpha=0.12)
                ax.plot(
                    x_values,
                    means,
                    color=color,
                    linewidth=2.6,
                    marker="o",
                    markersize=5.6,
                    label=MODEL_LABELS[model],
                )
                for row in series:
                    aggregated_rows.append(
                        {
                            "metric_key": panel["metric_key"],
                            "x_field": panel["x_field"],
                            "model": model,
                            "model_label": MODEL_LABELS[model],
                            "x_value": row["x_value"],
                            "mean_value": row["mean_value"],
                            "ci_low": row["ci_low"],
                            "ci_high": row["ci_high"],
                            "n": row["n"],
                        }
                    )

                if panel["highlight_best"]:
                    best_row = best_rows[model]
                    x_value = getattr(best_row, panel["x_field"])
                    y_value = best_row.patient_roc_auc
                    if x_value is not None and y_value is not None:
                        ax.scatter(
                            [x_value],
                            [y_value],
                            color=color,
                            edgecolors="#ffffff",
                            linewidths=1.0,
                            marker="*",
                            s=150,
                            zorder=5,
                        )

        ax.set_title(f"{panel['panel_label']}. {panel['title']}", loc="left", pad=10)
        ax.set_xlabel("Window size (s)" if panel["x_field"] == "window_seconds" else "Inner-k")
        ax.set_ylabel(panel["y_label"])
        style_axes(ax, ygrid=True, xgrid=False)
        if panel["clamp_min"] is not None:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(bottom=panel["clamp_min"], top=ymax)
        if panel["clamp_max"] is not None:
            ymin, _ = ax.get_ylim()
            ax.set_ylim(bottom=ymin, top=panel["clamp_max"])

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=True, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("Sweep Overview: Discrimination, Stability, and Parsimony", x=0.03, y=0.975, ha="left", va="top", fontsize=17)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.08, hspace=0.32, wspace=0.20)
    save_figure(fig, figure_path, dpi)
    _write_csv(aggregated_csv_path, aggregated_rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    output_root = Path(args.output_root)
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    data_dir = output_root / "data"

    results_paths = discover_results_files(Path(args.artifacts_dir), args.results_glob)
    if not results_paths:
        raise SystemExit("No sweep results ledgers were found.")

    deduped_result_rows = deduplicate_results_rows(load_results_rows(results_paths))
    if not deduped_result_rows:
        raise SystemExit("No deduplicated successful sweep rows were found.")

    best_rows = {model: _select_best_row(deduped_result_rows, model) for model in MODEL_ORDER}
    best_run_facts: List[Dict[str, Any]] = []
    bundles: Dict[str, Any] = {}
    for model, best_row in best_rows.items():
        bundle = load_run_artifact_bundle(
            mlruns_root=Path(args.mlruns_root),
            run_id=best_row.mlflow_run_id,
            experiment_id=best_row.mlflow_experiment_id,
        )
        bundles[model] = bundle
        best_run_facts.append(_facts_for_best_run(best_row, bundle))

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    summary_figure_path = figures_dir / "figure1_sweep_overview.png"
    aggregated_csv_path = data_dir / "sweep_overview_aggregates.csv"
    _plot_summary_figure(summary_figure_path, aggregated_csv_path, results_paths, best_rows, dpi=args.dpi)

    comparison_rows = _build_best_run_comparison_rows(best_run_facts)
    _write_csv(data_dir / "best_run_comparison.csv", comparison_rows)
    (tables_dir / "best_run_comparison.tex").write_text(_comparison_table_tex(comparison_rows), encoding="utf-8")

    hybrid_biomarker_rows = _build_hybrid_biomarker_rows(bundles["advanced_hybrid_1dcnn_lstm"])
    _write_csv(data_dir / "hybrid_recurrent_biomarkers.csv", hybrid_biomarker_rows)
    (tables_dir / "hybrid_recurrent_biomarkers.tex").write_text(_biomarker_table_tex(hybrid_biomarker_rows), encoding="utf-8")

    cohort_window_sizes = sorted({row.window_seconds for row in deduped_result_rows if row.window_seconds is not None})
    cohort_inner_ks = sorted({row.inner_k for row in deduped_result_rows if row.inner_k is not None})
    cohort_summary = _cohort_summary_facts(bundles["advanced_hybrid_1dcnn_lstm"], cohort_window_sizes, cohort_inner_ks)
    (tables_dir / "cohort_summary.tex").write_text(
        _cohort_summary_tex(bundles["advanced_hybrid_1dcnn_lstm"], cohort_window_sizes, cohort_inner_ks),
        encoding="utf-8",
    )

    facts = {
        "best_runs": best_run_facts,
        "cohort_summary": cohort_summary,
        "generated_files": {
            "summary_figure": str(summary_figure_path),
            "aggregated_csv": str(aggregated_csv_path),
            "best_run_comparison_csv": str(data_dir / "best_run_comparison.csv"),
            "best_run_comparison_tex": str(tables_dir / "best_run_comparison.tex"),
            "hybrid_biomarker_csv": str(data_dir / "hybrid_recurrent_biomarkers.csv"),
            "hybrid_biomarker_tex": str(tables_dir / "hybrid_recurrent_biomarkers.tex"),
            "cohort_summary_tex": str(tables_dir / "cohort_summary.tex"),
        },
    }
    (data_dir / "manuscript_facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")

    print(f"Summary figure: {summary_figure_path}")
    print(f"Best-run comparison table: {tables_dir / 'best_run_comparison.tex'}")
    print(f"Hybrid biomarker table: {tables_dir / 'hybrid_recurrent_biomarkers.tex'}")
    print(f"Facts file: {data_dir / 'manuscript_facts.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
