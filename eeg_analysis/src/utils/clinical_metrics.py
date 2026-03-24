from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_METRICS_REPORTING: Dict[str, Any] = {
    "primary_metric": "balanced_accuracy",
    "bootstrap_ci_enabled": True,
    "bootstrap_ci_level": 0.95,
    "bootstrap_iterations": 1000,
    "permutation_test_enabled": True,
    "permutation_metric": "balanced_accuracy",
    "permutation_iterations": 500,
    "random_seed": 42,
    "window_auc_enabled": True,
}

PATIENT_METRIC_NAMES: List[str] = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "sensitivity",
    "specificity",
    "f1",
    "roc_auc",
    "pr_auc",
    "npv",
    "mcc",
]
WINDOW_METRIC_NAMES: List[str] = ["accuracy", "f1", "roc_auc"]
CI_METRIC_NAMES: List[str] = [
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "f1",
    "roc_auc",
]


def _dedupe(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _to_label_array(values: Sequence[Any]) -> np.ndarray:
    arr = np.asarray(values)
    if arr.size == 0:
        return np.asarray([], dtype=int)
    return arr.astype(int)


def _to_prob_array(values: Optional[Sequence[Any]]) -> Optional[np.ndarray]:
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    return arr.astype(float)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if np.isnan(value) or np.isinf(value):
        return None
    return value


def _safe_divide(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _has_both_classes(y_true: np.ndarray) -> bool:
    return np.unique(y_true).size >= 2


def confusion_counts(y_true: Sequence[Any], y_pred: Sequence[Any]) -> Dict[str, int]:
    y_true_arr = _to_label_array(y_true)
    y_pred_arr = _to_label_array(y_pred)
    if y_true_arr.size == 0:
        return {
            "true_positives": 0,
            "true_negatives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()
    return {
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


def compute_metric_value(
    metric_name: str,
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_prob: Optional[Sequence[Any]] = None,
) -> Optional[float]:
    y_true_arr = _to_label_array(y_true)
    y_pred_arr = _to_label_array(y_pred)
    y_prob_arr = _to_prob_array(y_prob)

    if y_true_arr.size == 0:
        return None

    counts = confusion_counts(y_true_arr, y_pred_arr)
    tp = counts["true_positives"]
    tn = counts["true_negatives"]
    fp = counts["false_positives"]
    fn = counts["false_negatives"]

    if metric_name == "accuracy":
        return _safe_float(accuracy_score(y_true_arr, y_pred_arr))
    if metric_name == "balanced_accuracy":
        if not _has_both_classes(y_true_arr):
            return None
        return _safe_float(balanced_accuracy_score(y_true_arr, y_pred_arr))
    if metric_name == "precision":
        return _safe_float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    if metric_name in {"recall", "sensitivity"}:
        return _safe_float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    if metric_name == "specificity":
        return _safe_divide(tn, tn + fp)
    if metric_name == "f1":
        return _safe_float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    if metric_name == "roc_auc":
        if y_prob_arr is None or not _has_both_classes(y_true_arr):
            return None
        return _safe_float(roc_auc_score(y_true_arr, y_prob_arr))
    if metric_name == "pr_auc":
        if y_prob_arr is None or not _has_both_classes(y_true_arr):
            return None
        return _safe_float(average_precision_score(y_true_arr, y_prob_arr))
    if metric_name == "npv":
        return _safe_divide(tn, tn + fn)
    if metric_name == "mcc":
        return _safe_float(matthews_corrcoef(y_true_arr, y_pred_arr))
    raise ValueError(f"Unsupported metric: {metric_name}")


def compute_binary_classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_prob: Optional[Sequence[Any]] = None,
    *,
    metric_names: Optional[Sequence[str]] = None,
    count_field_name: str,
) -> Dict[str, Optional[float]]:
    y_true_arr = _to_label_array(y_true)
    y_pred_arr = _to_label_array(y_pred)
    y_prob_arr = _to_prob_array(y_prob)

    names = _dedupe(metric_names or PATIENT_METRIC_NAMES)
    metrics = {
        name: compute_metric_value(name, y_true_arr, y_pred_arr, y_prob_arr)
        for name in names
    }
    metrics.update(confusion_counts(y_true_arr, y_pred_arr))
    metrics[count_field_name] = int(y_true_arr.size)
    return metrics


def summarize_fold_metrics(values: Sequence[Any], *, count_field_name: str) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "mean": None,
            "std": None,
            count_field_name: 0,
        }

    arr = np.asarray(values, dtype=float)
    return {
        "mean": _safe_float(arr.mean()),
        "std": _safe_float(arr.std(ddof=0)),
        count_field_name: int(arr.size),
    }


def bootstrap_metric_intervals(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_prob: Optional[Sequence[Any]] = None,
    *,
    metric_names: Optional[Sequence[str]] = None,
    iterations: int = 1000,
    ci_level: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, Dict[str, Optional[float]]]:
    y_true_arr = _to_label_array(y_true)
    y_pred_arr = _to_label_array(y_pred)
    y_prob_arr = _to_prob_array(y_prob)

    names = _dedupe(metric_names or CI_METRIC_NAMES)
    result: Dict[str, Dict[str, Optional[float]]] = {}
    if y_true_arr.size == 0 or iterations < 1:
        for name in names:
            result[name] = {
                "mean": None,
                "std": None,
                "ci_low": None,
                "ci_high": None,
                "valid_samples": 0,
            }
        return result

    rng = np.random.default_rng(random_seed)
    values: Dict[str, List[float]] = {name: [] for name in names}
    alpha = (1.0 - ci_level) / 2.0

    for _ in range(int(iterations)):
        indices = rng.integers(0, y_true_arr.size, size=y_true_arr.size)
        y_true_sample = y_true_arr[indices]
        y_pred_sample = y_pred_arr[indices]
        y_prob_sample = y_prob_arr[indices] if y_prob_arr is not None else None

        for name in names:
            metric_value = compute_metric_value(name, y_true_sample, y_pred_sample, y_prob_sample)
            if metric_value is not None:
                values[name].append(float(metric_value))

    for name in names:
        metric_values = np.asarray(values[name], dtype=float)
        if metric_values.size == 0:
            result[name] = {
                "mean": None,
                "std": None,
                "ci_low": None,
                "ci_high": None,
                "valid_samples": 0,
            }
            continue

        result[name] = {
            "mean": _safe_float(metric_values.mean()),
            "std": _safe_float(metric_values.std(ddof=0)),
            "ci_low": _safe_float(np.quantile(metric_values, alpha)),
            "ci_high": _safe_float(np.quantile(metric_values, 1.0 - alpha)),
            "valid_samples": int(metric_values.size),
        }

    return result


def permutation_test_metric(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_prob: Optional[Sequence[Any]] = None,
    *,
    metric_name: str = "balanced_accuracy",
    iterations: int = 500,
    random_seed: int = 42,
) -> Dict[str, Optional[float]]:
    y_true_arr = _to_label_array(y_true)
    y_pred_arr = _to_label_array(y_pred)
    y_prob_arr = _to_prob_array(y_prob)

    observed = compute_metric_value(metric_name, y_true_arr, y_pred_arr, y_prob_arr)
    if observed is None or y_true_arr.size == 0 or iterations < 1:
        return {
            "metric_name": metric_name,
            "observed": observed,
            "p_value": None,
            "null_mean": None,
            "null_std": None,
            "iterations": 0,
        }

    rng = np.random.default_rng(random_seed)
    null_distribution = []
    for _ in range(int(iterations)):
        permuted_labels = rng.permutation(y_true_arr)
        metric_value = compute_metric_value(metric_name, permuted_labels, y_pred_arr, y_prob_arr)
        if metric_value is not None:
            null_distribution.append(float(metric_value))

    if not null_distribution:
        return {
            "metric_name": metric_name,
            "observed": observed,
            "p_value": None,
            "null_mean": None,
            "null_std": None,
            "iterations": 0,
        }

    null_values = np.asarray(null_distribution, dtype=float)
    p_value = (1.0 + float(np.sum(null_values >= observed))) / (1.0 + float(null_values.size))
    return {
        "metric_name": metric_name,
        "observed": _safe_float(observed),
        "p_value": _safe_float(p_value),
        "null_mean": _safe_float(null_values.mean()),
        "null_std": _safe_float(null_values.std(ddof=0)),
        "iterations": int(null_values.size),
    }


def summarize_feature_selection(feature_sets: Optional[Sequence[Sequence[str]]]) -> Dict[str, Any]:
    normalized_sets = [sorted(set(feature_set)) for feature_set in (feature_sets or []) if feature_set]
    if not normalized_sets:
        return {
            "feature_set_count": 0,
            "average_features_per_fold": None,
            "mean_pairwise_jaccard": None,
            "median_pairwise_jaccard": None,
            "unique_feature_count": 0,
            "top_feature": None,
            "top_feature_frequency": 0,
            "top_feature_share": None,
        }

    feature_counter: Counter[str] = Counter()
    for feature_set in normalized_sets:
        feature_counter.update(feature_set)

    jaccard_scores = []
    for left, right in combinations((set(item) for item in normalized_sets), 2):
        union = len(left | right)
        if union == 0:
            continue
        jaccard_scores.append(len(left & right) / union)

    top_feature, top_frequency = feature_counter.most_common(1)[0]
    feature_lengths = np.asarray([len(item) for item in normalized_sets], dtype=float)
    jaccard_arr = np.asarray(jaccard_scores, dtype=float) if jaccard_scores else np.asarray([1.0], dtype=float)

    return {
        "feature_set_count": int(len(normalized_sets)),
        "average_features_per_fold": _safe_float(feature_lengths.mean()),
        "mean_pairwise_jaccard": _safe_float(jaccard_arr.mean()),
        "median_pairwise_jaccard": _safe_float(np.median(jaccard_arr)),
        "unique_feature_count": int(len(feature_counter)),
        "top_feature": top_feature,
        "top_feature_frequency": int(top_frequency),
        "top_feature_share": _safe_float(top_frequency / len(normalized_sets)),
    }


def build_clinical_metrics_report(
    *,
    patient_true_labels: Sequence[Any],
    patient_pred_labels: Sequence[Any],
    patient_pred_probs: Optional[Sequence[Any]],
    window_true_labels: Sequence[Any],
    window_pred_labels: Sequence[Any],
    window_pred_probs: Optional[Sequence[Any]],
    fold_patient_accuracies: Optional[Sequence[Any]] = None,
    fold_window_accuracies: Optional[Sequence[Any]] = None,
    feature_sets: Optional[Sequence[Sequence[str]]] = None,
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    reporting = dict(DEFAULT_METRICS_REPORTING)
    if settings:
        reporting.update(settings)

    patient_metrics = compute_binary_classification_metrics(
        patient_true_labels,
        patient_pred_labels,
        patient_pred_probs,
        metric_names=PATIENT_METRIC_NAMES,
        count_field_name="n_patients",
    )

    window_metric_names = ["accuracy", "f1"]
    if bool(reporting.get("window_auc_enabled", True)):
        window_metric_names.append("roc_auc")
    window_metrics = compute_binary_classification_metrics(
        window_true_labels,
        window_pred_labels,
        window_pred_probs,
        metric_names=window_metric_names,
        count_field_name="n_windows",
    )

    ci_metrics = bootstrap_metric_intervals(
        patient_true_labels,
        patient_pred_labels,
        patient_pred_probs,
        metric_names=CI_METRIC_NAMES,
        iterations=int(reporting.get("bootstrap_iterations", 1000)),
        ci_level=float(reporting.get("bootstrap_ci_level", 0.95)),
        random_seed=int(reporting.get("random_seed", 42)),
    ) if bool(reporting.get("bootstrap_ci_enabled", True)) else {}

    permutation = permutation_test_metric(
        patient_true_labels,
        patient_pred_labels,
        patient_pred_probs,
        metric_name=str(reporting.get("permutation_metric", reporting.get("primary_metric", "balanced_accuracy"))),
        iterations=int(reporting.get("permutation_iterations", 500)),
        random_seed=int(reporting.get("random_seed", 42)),
    ) if bool(reporting.get("permutation_test_enabled", True)) else {
        "metric_name": str(reporting.get("permutation_metric", reporting.get("primary_metric", "balanced_accuracy"))),
        "observed": None,
        "p_value": None,
        "null_mean": None,
        "null_std": None,
        "iterations": 0,
    }

    primary_metric_name = str(reporting.get("primary_metric", "balanced_accuracy"))
    feature_summary = summarize_feature_selection(feature_sets)

    return {
        "patient": {
            "role": "primary",
            "primary_metric_name": primary_metric_name,
            "primary_metric_value": patient_metrics.get(primary_metric_name),
            "metrics": patient_metrics,
            "confidence_intervals": ci_metrics,
            "fold_summary": summarize_fold_metrics(
                fold_patient_accuracies or [],
                count_field_name="n_folds",
            ),
        },
        "window": {
            "role": "supporting",
            "metrics": window_metrics,
            "fold_summary": summarize_fold_metrics(
                fold_window_accuracies or [],
                count_field_name="n_folds",
            ),
        },
        "feature_selection": feature_summary,
        "stats": {
            "settings": {
                "bootstrap_ci_enabled": bool(reporting.get("bootstrap_ci_enabled", True)),
                "bootstrap_ci_level": float(reporting.get("bootstrap_ci_level", 0.95)),
                "bootstrap_iterations": int(reporting.get("bootstrap_iterations", 1000)),
                "permutation_test_enabled": bool(reporting.get("permutation_test_enabled", True)),
                "permutation_metric": str(reporting.get("permutation_metric", primary_metric_name)),
                "permutation_iterations": int(reporting.get("permutation_iterations", 500)),
                "random_seed": int(reporting.get("random_seed", 42)),
            },
            "permutation_test": permutation,
        },
    }


def flatten_metric_report_for_mlflow(report: Dict[str, Any]) -> Dict[str, float]:
    flattened: Dict[str, float] = {}

    def add_metric(name: str, value: Any) -> None:
        numeric = _safe_float(value)
        if numeric is not None:
            flattened[name] = numeric

    for metric_name, value in report.get("patient", {}).get("metrics", {}).items():
        add_metric(f"patient_{metric_name}", value)

    for metric_name, summary in report.get("patient", {}).get("confidence_intervals", {}).items():
        if not isinstance(summary, dict):
            continue
        add_metric(f"patient_{metric_name}_ci_low", summary.get("ci_low"))
        add_metric(f"patient_{metric_name}_ci_high", summary.get("ci_high"))
        add_metric(f"patient_{metric_name}_ci_std", summary.get("std"))

    patient_fold_summary = report.get("patient", {}).get("fold_summary", {})
    if isinstance(patient_fold_summary, dict):
        add_metric("patient_fold_accuracy_mean", patient_fold_summary.get("mean"))
        add_metric("patient_fold_accuracy_std", patient_fold_summary.get("std"))
        add_metric("patient_fold_count", patient_fold_summary.get("n_folds"))

    for metric_name, value in report.get("window", {}).get("metrics", {}).items():
        add_metric(f"window_{metric_name}", value)

    window_fold_summary = report.get("window", {}).get("fold_summary", {})
    if isinstance(window_fold_summary, dict):
        add_metric("window_fold_accuracy_mean", window_fold_summary.get("mean"))
        add_metric("window_fold_accuracy_std", window_fold_summary.get("std"))
        add_metric("window_fold_count", window_fold_summary.get("n_folds"))
        add_metric("avg_window_accuracy", window_fold_summary.get("mean"))

    permutation = report.get("stats", {}).get("permutation_test", {})
    if isinstance(permutation, dict):
        metric_name = permutation.get("metric_name")
        if metric_name:
            add_metric(f"patient_{metric_name}_permutation_pvalue", permutation.get("p_value"))
            add_metric(f"patient_{metric_name}_permutation_observed", permutation.get("observed"))
            add_metric(f"patient_{metric_name}_permutation_null_mean", permutation.get("null_mean"))
            add_metric(f"patient_{metric_name}_permutation_null_std", permutation.get("null_std"))

    for metric_name, value in report.get("feature_selection", {}).items():
        if isinstance(value, str):
            continue
        add_metric(f"feature_selection_{metric_name}", value)

    return flattened
