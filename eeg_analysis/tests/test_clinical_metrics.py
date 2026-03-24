import pytest

from src.utils.clinical_metrics import (
    build_clinical_metrics_report,
    flatten_metric_report_for_mlflow,
    permutation_test_metric,
    summarize_feature_selection,
)


def test_build_clinical_metrics_report_includes_q1_metrics():
    report = build_clinical_metrics_report(
        patient_true_labels=[1, 0, 1, 0],
        patient_pred_labels=[1, 0, 0, 0],
        patient_pred_probs=[0.9, 0.2, 0.4, 0.1],
        window_true_labels=[1, 1, 0, 0, 1, 0],
        window_pred_labels=[1, 0, 0, 0, 1, 1],
        window_pred_probs=[0.8, 0.4, 0.3, 0.2, 0.7, 0.6],
        fold_patient_accuracies=[1.0, 1.0, 0.0, 1.0],
        fold_window_accuracies=[0.5, 1.0, 0.5, 1.0],
        feature_sets=[["f1", "f2"], ["f1", "f3"]],
        settings={
            "bootstrap_iterations": 25,
            "permutation_iterations": 25,
            "random_seed": 7,
        },
    )

    patient_metrics = report["patient"]["metrics"]
    assert report["patient"]["primary_metric_name"] == "balanced_accuracy"
    assert patient_metrics["balanced_accuracy"] == 0.75
    assert patient_metrics["specificity"] == 1.0
    assert patient_metrics["sensitivity"] == 0.5
    assert report["window"]["metrics"]["accuracy"] is not None
    assert report["stats"]["permutation_test"]["p_value"] is not None
    assert report["feature_selection"]["mean_pairwise_jaccard"] == pytest.approx(1.0 / 3.0)


def test_flatten_metric_report_for_mlflow_preserves_expected_names():
    report = build_clinical_metrics_report(
        patient_true_labels=[1, 0, 1, 0],
        patient_pred_labels=[1, 0, 0, 0],
        patient_pred_probs=[0.9, 0.2, 0.4, 0.1],
        window_true_labels=[1, 0, 1, 0],
        window_pred_labels=[1, 0, 1, 1],
        window_pred_probs=[0.8, 0.1, 0.7, 0.6],
        fold_patient_accuracies=[1.0, 1.0, 0.0, 1.0],
        fold_window_accuracies=[0.5, 1.0, 1.0, 0.5],
        settings={
            "bootstrap_iterations": 10,
            "permutation_iterations": 10,
            "random_seed": 5,
        },
    )

    flat = flatten_metric_report_for_mlflow(report)

    assert "patient_balanced_accuracy" in flat
    assert "patient_specificity" in flat
    assert "patient_balanced_accuracy_permutation_pvalue" in flat
    assert "window_accuracy" in flat
    assert "avg_window_accuracy" in flat


def test_permutation_test_metric_is_deterministic():
    first = permutation_test_metric(
        [1, 0, 1, 0],
        [1, 0, 0, 0],
        [0.9, 0.2, 0.4, 0.1],
        iterations=30,
        random_seed=11,
    )
    second = permutation_test_metric(
        [1, 0, 1, 0],
        [1, 0, 0, 0],
        [0.9, 0.2, 0.4, 0.1],
        iterations=30,
        random_seed=11,
    )

    assert first["p_value"] == second["p_value"]
    assert first["null_mean"] == second["null_mean"]


def test_summarize_feature_selection_handles_empty_input():
    summary = summarize_feature_selection([])

    assert summary["feature_set_count"] == 0
    assert summary["mean_pairwise_jaccard"] is None
