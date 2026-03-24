from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    "primary_metric": "roc_auc",
    "bootstrap_ci_enabled": True,
    "bootstrap_ci_level": 0.95,
    "bootstrap_iterations": 1000,
    "permutation_test_enabled": True,
    "permutation_metric": "roc_auc",
    "permutation_iterations": 500,
    "random_seed": 42,
    "window_auc_enabled": True,
    "patient_probability_aggregation": "mean_window_probability",
    "ranking_stability_top_k": 10,
    "resampling_stability_enabled": False,
    "resampling_method": "bootstrap",
    "resampling_iterations": 100,
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


def compute_per_fold_metrics(
    prediction_rows: Optional[Sequence[Dict[str, Any]]],
    *,
    metric_names: Sequence[str],
    count_field_name: str,
) -> List[Dict[str, Any]]:
    if not prediction_rows:
        return []

    grouped_rows: Dict[int, List[Dict[str, Any]]] = {}
    for row in prediction_rows:
        if not isinstance(row, dict):
            continue
        raw_fold = row.get("fold", row.get("fold_idx"))
        try:
            fold_idx = int(raw_fold)
        except (TypeError, ValueError):
            continue
        grouped_rows.setdefault(fold_idx, []).append(row)

    fold_reports: List[Dict[str, Any]] = []
    for fold_idx in sorted(grouped_rows):
        rows = grouped_rows[fold_idx]
        y_true = [row["true_label"] for row in rows if "true_label" in row]
        y_pred = [row["predicted_label"] for row in rows if "predicted_label" in row]
        y_prob = [
            row["probability"]
            for row in rows
            if row.get("probability") is not None
        ]
        metrics = compute_binary_classification_metrics(
            y_true,
            y_pred,
            y_prob if len(y_prob) == len(y_true) else None,
            metric_names=metric_names,
            count_field_name=count_field_name,
        )
        fold_reports.append({
            "fold_idx": int(fold_idx),
            "metrics": metrics,
        })
    return fold_reports


def summarize_per_fold_metric_reports(
    fold_reports: Sequence[Dict[str, Any]],
    *,
    metric_names: Sequence[str],
) -> Dict[str, Dict[str, Optional[float]]]:
    summary: Dict[str, Dict[str, Optional[float]]] = {}
    for metric_name in metric_names:
        values = []
        for fold_report in fold_reports:
            if not isinstance(fold_report, dict):
                continue
            metrics = fold_report.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            metric_value = _safe_float(metrics.get(metric_name))
            if metric_value is not None:
                values.append(metric_value)
        summary[metric_name] = summarize_fold_metrics(values, count_field_name="n_folds")
    return summary


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


def _normalize_feature_selection_rows(
    feature_selection_rows: Optional[Sequence[Dict[str, Any]]] = None,
    feature_sets: Optional[Sequence[Sequence[str]]] = None,
) -> List[Dict[str, Any]]:
    normalized_rows: List[Dict[str, Any]] = []
    if feature_selection_rows:
        for row in feature_selection_rows:
            if not isinstance(row, dict):
                continue

            features = sorted({str(feature) for feature in row.get("features", []) if feature is not None})

            ranking_scores = None
            raw_scores = row.get("ranking_scores")
            if isinstance(raw_scores, dict):
                ranking_scores = {}
                for feature_name, score in raw_scores.items():
                    numeric_score = _safe_float(score)
                    if numeric_score is not None:
                        ranking_scores[str(feature_name)] = numeric_score
                if not ranking_scores:
                    ranking_scores = None

            effect_stats: Dict[str, Dict[str, Any]] = {}
            raw_effect_stats = row.get("effect_stats")
            if isinstance(raw_effect_stats, dict):
                for feature_name, stats in raw_effect_stats.items():
                    if not isinstance(stats, dict):
                        continue
                    cohens_d = _safe_float(stats.get("cohens_d"))
                    sign = stats.get("sign")
                    effect_stats[str(feature_name)] = {
                        "cohens_d": cohens_d,
                        "sign": sign if sign in {"positive", "negative", "zero"} else None,
                    }

            candidate_feature_count = row.get("candidate_feature_count")
            try:
                candidate_feature_count = int(candidate_feature_count) if candidate_feature_count is not None else None
            except (TypeError, ValueError):
                candidate_feature_count = None

            fold_target_class = str(row.get("fold_target_class", "unknown") or "unknown").strip().lower()
            if fold_target_class not in {"remission", "non_remission", "mixed", "unknown"}:
                fold_target_class = "unknown"

            normalized_rows.append(
                {
                    "fold_idx": row.get("fold_idx"),
                    "fold_target_class": fold_target_class,
                    "features": features,
                    "ranking_scores": ranking_scores,
                    "ranking_source": row.get("ranking_source"),
                    "effect_stats": effect_stats,
                    "candidate_feature_count": candidate_feature_count,
                }
            )

    if normalized_rows:
        return normalized_rows

    for feature_set in feature_sets or []:
        normalized_rows.append(
            {
                "fold_idx": None,
                "fold_target_class": "unknown",
                "features": sorted({str(feature) for feature in feature_set if feature is not None}),
                "ranking_scores": None,
                "ranking_source": None,
                "effect_stats": {},
                "candidate_feature_count": None,
            }
        )
    return normalized_rows


def _compute_pairwise_jaccard_scores(feature_sets: Sequence[set[str]]) -> List[float]:
    scores: List[float] = []
    for left, right in combinations(feature_sets, 2):
        union = len(left | right)
        if union == 0:
            continue
        scores.append(len(left & right) / union)
    return scores


def _compute_pairwise_kuncheva_scores(
    feature_sets: Sequence[set[str]],
    candidate_feature_count: Optional[int],
) -> Tuple[Optional[List[float]], Optional[str]]:
    if len(feature_sets) < 2:
        return None, "insufficient_folds"
    if candidate_feature_count is None:
        return None, "candidate_feature_count_unavailable"

    selected_sizes = {len(feature_set) for feature_set in feature_sets}
    if len(selected_sizes) != 1:
        return None, "variable_selected_feature_count"

    k = selected_sizes.pop()
    p = int(candidate_feature_count)
    if k < 1:
        return None, "no_features_selected"
    if p <= k:
        return None, "candidate_feature_count_incompatible"

    denominator = k - ((k * k) / p)
    if abs(denominator) < 1e-12:
        return None, "kuncheva_denominator_zero"

    scores = []
    expected_overlap = (k * k) / p
    for left, right in combinations(feature_sets, 2):
        overlap = len(left & right)
        scores.append((overlap - expected_overlap) / denominator)
    return scores, None


def _select_top_k_features(ranking_scores: Dict[str, float], top_k: int) -> List[str]:
    if top_k < 1:
        return []
    ranked_items = sorted(ranking_scores.items(), key=lambda item: (-item[1], item[0]))
    return [feature for feature, _ in ranked_items[:top_k]]


def _compute_pairwise_top_k_overlap(
    ranking_rows: Sequence[Dict[str, Any]],
    top_k: int,
) -> Tuple[Optional[List[float]], Optional[str]]:
    if top_k < 1:
        return None, "invalid_top_k"

    top_k_sets: List[set[str]] = []
    for row in ranking_rows:
        ranking_scores = row.get("ranking_scores")
        if not isinstance(ranking_scores, dict) or not ranking_scores:
            continue
        top_features = _select_top_k_features(ranking_scores, top_k)
        if not top_features:
            continue
        top_k_sets.append(set(top_features))

    if len(top_k_sets) < 2:
        return None, "insufficient_ranked_folds"

    overlaps = []
    for left, right in combinations(top_k_sets, 2):
        denominator = min(top_k, len(left), len(right))
        if denominator == 0:
            continue
        overlaps.append(len(left & right) / denominator)
    if not overlaps:
        return None, "insufficient_ranked_folds"
    return overlaps, None


def _summarize_effect_stability(
    normalized_rows: Sequence[Dict[str, Any]],
    selection_frequency: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    records = []
    total_folds = max(len(normalized_rows), 1)
    selected_features = sorted(selection_frequency)

    for feature in selected_features:
        effect_values: List[float] = []
        sign_counter: Counter[str] = Counter()
        for row in normalized_rows:
            effect_stats = row.get("effect_stats") or {}
            feature_stats = effect_stats.get(feature)
            if not isinstance(feature_stats, dict):
                continue
            cohens_d = _safe_float(feature_stats.get("cohens_d"))
            sign = feature_stats.get("sign")
            if cohens_d is None or sign not in {"positive", "negative", "zero"}:
                continue
            effect_values.append(cohens_d)
            sign_counter.update([sign])

        if not effect_values:
            continue

        arr = np.asarray(effect_values, dtype=float)
        fold_count = int(arr.size)
        dominant_share = max(sign_counter.values()) / fold_count if fold_count else None
        selection_share = selection_frequency[feature]["share"]

        records.append(
            {
                "feature": feature,
                "fold_count": fold_count,
                "selection_share": selection_share,
                "sign_consistency": _safe_float(dominant_share),
                "positive_effect_fraction": _safe_float(sign_counter.get("positive", 0) / fold_count),
                "negative_effect_fraction": _safe_float(sign_counter.get("negative", 0) / fold_count),
                "zero_effect_fraction": _safe_float(sign_counter.get("zero", 0) / fold_count),
                "cohens_d_mean": _safe_float(arr.mean()),
                "cohens_d_variance": _safe_float(arr.var(ddof=0)),
                "cohens_d_std": _safe_float(arr.std(ddof=0)),
            }
        )

    if not records:
        return {
            "available": False,
            "reason": "effect_stats_unavailable",
            "features": [],
            "feature_count": 0,
        }

    records.sort(key=lambda item: (-float(item["selection_share"] or 0.0), item["feature"]))
    return {
        "available": True,
        "reason": None,
        "features": records,
        "feature_count": int(len(records)),
        "mean_sign_consistency": _safe_float(
            np.mean([record["sign_consistency"] for record in records if record["sign_consistency"] is not None])
        ),
    }


def _summarize_importance_stability(
    normalized_rows: Sequence[Dict[str, Any]],
    selection_frequency: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    ranked_rows = [
        row for row in normalized_rows if isinstance(row.get("ranking_scores"), dict) and row.get("ranking_scores")
    ]
    if not ranked_rows:
        return {
            "available": False,
            "reason": "ranking_scores_unavailable",
            "features": [],
            "feature_count": 0,
        }

    records = []
    for feature in sorted(selection_frequency):
        values = [
            row["ranking_scores"].get(feature)
            for row in ranked_rows
            if feature in row["ranking_scores"]
        ]
        values = [value for value in values if _safe_float(value) is not None]
        if not values:
            continue

        arr = np.asarray(values, dtype=float)
        records.append(
            {
                "feature": feature,
                "fold_count": int(arr.size),
                "importance_mean": _safe_float(arr.mean()),
                "importance_variance": _safe_float(arr.var(ddof=0)),
                "importance_std": _safe_float(arr.std(ddof=0)),
            }
        )

    if not records:
        return {
            "available": False,
            "reason": "ranking_scores_unavailable",
            "features": [],
            "feature_count": 0,
        }

    records.sort(key=lambda item: item["feature"])
    return {
        "available": True,
        "reason": None,
        "features": records,
        "feature_count": int(len(records)),
        "mean_importance_variance": _safe_float(
            np.mean([record["importance_variance"] for record in records if record["importance_variance"] is not None])
        ),
    }


def summarize_feature_selection(
    feature_sets: Optional[Sequence[Sequence[str]]] = None,
    *,
    feature_selection_rows: Optional[Sequence[Dict[str, Any]]] = None,
    top_k: int = 10,
    resampling_enabled: bool = False,
    resampling_method: str = "bootstrap",
    resampling_iterations: int = 100,
) -> Dict[str, Any]:
    ratio_epsilon = 1e-9
    normalized_rows = _normalize_feature_selection_rows(
        feature_selection_rows=feature_selection_rows,
        feature_sets=feature_sets,
    )
    if not normalized_rows:
        return {
            "feature_set_count": 0,
            "average_features_per_fold": None,
            "mean_pairwise_jaccard": None,
            "median_pairwise_jaccard": None,
            "unique_feature_count": 0,
            "top_feature": None,
            "top_feature_frequency": 0,
            "top_feature_share": None,
            "selection_frequency_by_feature": [],
            "selected_features_per_fold": [],
            "fixed_k_detected": False,
            "kuncheva_index_mean": None,
            "kuncheva_index_median": None,
            "kuncheva_reason": "no_feature_sets",
            "fold_class_counts": {
                "remission_folds": 0,
                "non_remission_folds": 0,
                "mixed_folds": 0,
                "unknown_folds": 0,
            },
            "class_conditional_selection_available": False,
            "class_conditional_selection_reason": "no_feature_sets",
            "ranking_stability": {
                "available": False,
                "reason": "ranking_scores_unavailable",
                "top_k": int(top_k),
                "mean_top_k_overlap": None,
                "median_top_k_overlap": None,
                "n_ranked_folds": 0,
            },
            "effect_stability": {
                "available": False,
                "reason": "effect_stats_unavailable",
                "features": [],
                "feature_count": 0,
            },
            "importance_stability": {
                "available": False,
                "reason": "ranking_scores_unavailable",
                "features": [],
                "feature_count": 0,
            },
            "resampling_stability": {
                "enabled": bool(resampling_enabled),
                "method": str(resampling_method),
                "iterations": int(resampling_iterations),
                "reason": "disabled_by_config" if not resampling_enabled else "not_implemented",
                "results": None,
            },
        }

    feature_counter: Counter[str] = Counter()
    feature_sets_as_sets: List[set[str]] = []
    feature_lengths = []
    candidate_feature_counts = set()
    for row in normalized_rows:
        feature_set = set(row["features"])
        feature_sets_as_sets.append(feature_set)
        feature_lengths.append(len(feature_set))
        feature_counter.update(sorted(feature_set))
        candidate_feature_count = row.get("candidate_feature_count")
        if candidate_feature_count is not None:
            candidate_feature_counts.add(int(candidate_feature_count))

    jaccard_scores = _compute_pairwise_jaccard_scores(feature_sets_as_sets)
    jaccard_arr = np.asarray(jaccard_scores, dtype=float) if jaccard_scores else np.asarray([], dtype=float)
    feature_lengths_arr = np.asarray(feature_lengths, dtype=float)

    top_feature = None
    top_frequency = 0
    if feature_counter:
        top_feature, top_frequency = feature_counter.most_common(1)[0]

    total_folds = len(normalized_rows)
    fold_class_counts = Counter(row.get("fold_target_class", "unknown") for row in normalized_rows)
    remission_fold_count = int(fold_class_counts.get("remission", 0))
    non_remission_fold_count = int(fold_class_counts.get("non_remission", 0))
    mixed_fold_count = int(fold_class_counts.get("mixed", 0))
    unknown_fold_count = int(fold_class_counts.get("unknown", 0))
    class_conditional_available = remission_fold_count > 0 and non_remission_fold_count > 0
    class_conditional_reason = None if class_conditional_available else "insufficient_class_fold_coverage"

    selection_counter_by_class: Dict[str, Counter[str]] = {
        "remission": Counter(),
        "non_remission": Counter(),
    }
    for row in normalized_rows:
        fold_target_class = row.get("fold_target_class")
        if fold_target_class not in selection_counter_by_class:
            continue
        selection_counter_by_class[fold_target_class].update(set(row["features"]))

    selection_frequency = {
        feature: {
            "feature": feature,
            "count": int(count),
            "share": _safe_float(count / total_folds),
            "count_remission": int(selection_counter_by_class["remission"].get(feature, 0)),
            "share_remission": (
                _safe_float(selection_counter_by_class["remission"].get(feature, 0) / remission_fold_count)
                if remission_fold_count
                else None
            ),
            "count_non_remission": int(selection_counter_by_class["non_remission"].get(feature, 0)),
            "share_non_remission": (
                _safe_float(selection_counter_by_class["non_remission"].get(feature, 0) / non_remission_fold_count)
                if non_remission_fold_count
                else None
            ),
            "delta_remission_minus_non_remission": None,
            "ratio_remission_to_non_remission": None,
        }
        for feature, count in feature_counter.most_common()
    }
    for feature_summary in selection_frequency.values():
        share_remission = feature_summary["share_remission"]
        share_non_remission = feature_summary["share_non_remission"]
        if share_remission is not None and share_non_remission is not None:
            feature_summary["delta_remission_minus_non_remission"] = _safe_float(
                share_remission - share_non_remission
            )
            feature_summary["ratio_remission_to_non_remission"] = _safe_float(
                share_remission / (share_non_remission + ratio_epsilon)
            )

    selected_features_per_fold = [int(length) for length in feature_lengths]
    fixed_k_detected = bool(feature_lengths and len(set(feature_lengths)) == 1)

    candidate_feature_count = None
    if len(candidate_feature_counts) == 1:
        candidate_feature_count = next(iter(candidate_feature_counts))

    kuncheva_scores, kuncheva_reason = _compute_pairwise_kuncheva_scores(
        feature_sets_as_sets,
        candidate_feature_count,
    )
    kuncheva_arr = np.asarray(kuncheva_scores, dtype=float) if kuncheva_scores else np.asarray([], dtype=float)

    ranked_rows = [
        row for row in normalized_rows if isinstance(row.get("ranking_scores"), dict) and row.get("ranking_scores")
    ]
    top_k_overlap_scores, top_k_reason = _compute_pairwise_top_k_overlap(ranked_rows, int(top_k))
    top_k_arr = np.asarray(top_k_overlap_scores, dtype=float) if top_k_overlap_scores else np.asarray([], dtype=float)

    effect_stability = _summarize_effect_stability(normalized_rows, selection_frequency)
    importance_stability = _summarize_importance_stability(normalized_rows, selection_frequency)

    return {
        "feature_set_count": int(total_folds),
        "average_features_per_fold": _safe_float(feature_lengths_arr.mean()),
        "mean_pairwise_jaccard": _safe_float(jaccard_arr.mean()) if jaccard_arr.size else None,
        "median_pairwise_jaccard": _safe_float(np.median(jaccard_arr)) if jaccard_arr.size else None,
        "unique_feature_count": int(len(feature_counter)),
        "top_feature": top_feature,
        "top_feature_frequency": int(top_frequency),
        "top_feature_share": _safe_float(top_frequency / total_folds) if total_folds else None,
        "selection_frequency_by_feature": list(selection_frequency.values()),
        "selected_features_per_fold": selected_features_per_fold,
        "fixed_k_detected": fixed_k_detected,
        "kuncheva_index_mean": _safe_float(kuncheva_arr.mean()) if kuncheva_arr.size else None,
        "kuncheva_index_median": _safe_float(np.median(kuncheva_arr)) if kuncheva_arr.size else None,
        "kuncheva_reason": None if kuncheva_arr.size else kuncheva_reason,
        "fold_class_counts": {
            "remission_folds": remission_fold_count,
            "non_remission_folds": non_remission_fold_count,
            "mixed_folds": mixed_fold_count,
            "unknown_folds": unknown_fold_count,
        },
        "class_conditional_selection_available": class_conditional_available,
        "class_conditional_selection_reason": class_conditional_reason,
        "ranking_stability": {
            "available": bool(top_k_arr.size),
            "reason": None if top_k_arr.size else top_k_reason,
            "top_k": int(top_k),
            "mean_top_k_overlap": _safe_float(top_k_arr.mean()) if top_k_arr.size else None,
            "median_top_k_overlap": _safe_float(np.median(top_k_arr)) if top_k_arr.size else None,
            "n_ranked_folds": int(len(ranked_rows)),
        },
        "effect_stability": effect_stability,
        "importance_stability": importance_stability,
        "resampling_stability": {
            "enabled": bool(resampling_enabled),
            "method": str(resampling_method),
            "iterations": int(resampling_iterations),
            "reason": "disabled_by_config" if not resampling_enabled else "not_implemented",
            "results": None,
        },
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
    feature_selection_rows: Optional[Sequence[Dict[str, Any]]] = None,
    patient_prediction_rows: Optional[Sequence[Dict[str, Any]]] = None,
    window_prediction_rows: Optional[Sequence[Dict[str, Any]]] = None,
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
    patient_fold_reports = compute_per_fold_metrics(
        patient_prediction_rows,
        metric_names=PATIENT_METRIC_NAMES,
        count_field_name="n_patients",
    )
    window_fold_reports = compute_per_fold_metrics(
        window_prediction_rows,
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
    feature_summary = summarize_feature_selection(
        feature_sets,
        feature_selection_rows=feature_selection_rows,
        top_k=int(reporting.get("ranking_stability_top_k", 10)),
        resampling_enabled=bool(reporting.get("resampling_stability_enabled", False)),
        resampling_method=str(reporting.get("resampling_method", "bootstrap")),
        resampling_iterations=int(reporting.get("resampling_iterations", 100)),
    )

    return {
        "patient": {
            "role": "primary",
            "primary_metric_name": primary_metric_name,
            "primary_metric_value": patient_metrics.get(primary_metric_name),
            "probability_aggregation": {
                "method": str(reporting.get("patient_probability_aggregation", "mean_window_probability")),
            },
            "metrics": patient_metrics,
            "confidence_intervals": ci_metrics,
            "fold_summary": summarize_fold_metrics(
                fold_patient_accuracies or [],
                count_field_name="n_folds",
            ),
            "fold_metrics": patient_fold_reports,
            "fold_metric_summary": summarize_per_fold_metric_reports(
                patient_fold_reports,
                metric_names=PATIENT_METRIC_NAMES,
            ),
        },
        "window": {
            "role": "supporting",
            "metrics": window_metrics,
            "fold_summary": summarize_fold_metrics(
                fold_window_accuracies or [],
                count_field_name="n_folds",
            ),
            "fold_metrics": window_fold_reports,
            "fold_metric_summary": summarize_per_fold_metric_reports(
                window_fold_reports,
                metric_names=window_metric_names,
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
                "patient_probability_aggregation": str(
                    reporting.get("patient_probability_aggregation", "mean_window_probability")
                ),
                "ranking_stability_top_k": int(reporting.get("ranking_stability_top_k", 10)),
                "resampling_stability_enabled": bool(reporting.get("resampling_stability_enabled", False)),
                "resampling_method": str(reporting.get("resampling_method", "bootstrap")),
                "resampling_iterations": int(reporting.get("resampling_iterations", 100)),
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
    patient_fold_metric_summary = report.get("patient", {}).get("fold_metric_summary", {})
    if isinstance(patient_fold_metric_summary, dict):
        for metric_name, summary in patient_fold_metric_summary.items():
            if not isinstance(summary, dict):
                continue
            add_metric(f"patient_fold_{metric_name}_mean", summary.get("mean"))
            add_metric(f"patient_fold_{metric_name}_std", summary.get("std"))

    for metric_name, value in report.get("window", {}).get("metrics", {}).items():
        add_metric(f"window_{metric_name}", value)

    window_fold_summary = report.get("window", {}).get("fold_summary", {})
    if isinstance(window_fold_summary, dict):
        add_metric("window_fold_accuracy_mean", window_fold_summary.get("mean"))
        add_metric("window_fold_accuracy_std", window_fold_summary.get("std"))
        add_metric("window_fold_count", window_fold_summary.get("n_folds"))
        add_metric("avg_window_accuracy", window_fold_summary.get("mean"))
    window_fold_metric_summary = report.get("window", {}).get("fold_metric_summary", {})
    if isinstance(window_fold_metric_summary, dict):
        for metric_name, summary in window_fold_metric_summary.items():
            if not isinstance(summary, dict):
                continue
            add_metric(f"window_fold_{metric_name}_mean", summary.get("mean"))
            add_metric(f"window_fold_{metric_name}_std", summary.get("std"))

    permutation = report.get("stats", {}).get("permutation_test", {})
    if isinstance(permutation, dict):
        metric_name = permutation.get("metric_name")
        if metric_name:
            add_metric(f"patient_{metric_name}_permutation_pvalue", permutation.get("p_value"))
            add_metric(f"patient_{metric_name}_permutation_observed", permutation.get("observed"))
            add_metric(f"patient_{metric_name}_permutation_null_mean", permutation.get("null_mean"))
            add_metric(f"patient_{metric_name}_permutation_null_std", permutation.get("null_std"))

    feature_selection = report.get("feature_selection", {})
    if isinstance(feature_selection, dict):
        for metric_name in (
            "feature_set_count",
            "average_features_per_fold",
            "mean_pairwise_jaccard",
            "median_pairwise_jaccard",
            "unique_feature_count",
            "top_feature_frequency",
            "top_feature_share",
            "fixed_k_detected",
            "kuncheva_index_mean",
            "kuncheva_index_median",
            "class_conditional_selection_available",
        ):
            add_metric(f"feature_selection_{metric_name}", feature_selection.get(metric_name))

        fold_class_counts = feature_selection.get("fold_class_counts", {})
        if isinstance(fold_class_counts, dict):
            add_metric("feature_selection_remission_fold_count", fold_class_counts.get("remission_folds"))
            add_metric("feature_selection_non_remission_fold_count", fold_class_counts.get("non_remission_folds"))
            add_metric("feature_selection_mixed_fold_count", fold_class_counts.get("mixed_folds"))
            add_metric("feature_selection_unknown_fold_count", fold_class_counts.get("unknown_folds"))

        ranking_stability = feature_selection.get("ranking_stability", {})
        if isinstance(ranking_stability, dict):
            add_metric("feature_selection_ranking_available", ranking_stability.get("available"))
            add_metric("feature_selection_ranking_top_k", ranking_stability.get("top_k"))
            add_metric("feature_selection_mean_top_k_overlap", ranking_stability.get("mean_top_k_overlap"))
            add_metric("feature_selection_median_top_k_overlap", ranking_stability.get("median_top_k_overlap"))
            add_metric("feature_selection_ranked_fold_count", ranking_stability.get("n_ranked_folds"))

        effect_stability = feature_selection.get("effect_stability", {})
        if isinstance(effect_stability, dict):
            add_metric("feature_selection_effect_available", effect_stability.get("available"))
            add_metric("feature_selection_effect_feature_count", effect_stability.get("feature_count"))
            add_metric(
                "feature_selection_effect_mean_sign_consistency",
                effect_stability.get("mean_sign_consistency"),
            )

        importance_stability = feature_selection.get("importance_stability", {})
        if isinstance(importance_stability, dict):
            add_metric("feature_selection_importance_available", importance_stability.get("available"))
            add_metric("feature_selection_importance_feature_count", importance_stability.get("feature_count"))
            add_metric(
                "feature_selection_importance_mean_variance",
                importance_stability.get("mean_importance_variance"),
            )

        resampling_stability = feature_selection.get("resampling_stability", {})
        if isinstance(resampling_stability, dict):
            add_metric("feature_selection_resampling_enabled", resampling_stability.get("enabled"))
            add_metric("feature_selection_resampling_iterations", resampling_stability.get("iterations"))

    return flattened
