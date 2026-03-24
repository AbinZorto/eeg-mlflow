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
    assert report["patient"]["primary_metric_name"] == "roc_auc"
    assert report["patient"]["primary_metric_value"] == pytest.approx(1.0)
    assert patient_metrics["balanced_accuracy"] == 0.75
    assert patient_metrics["specificity"] == 1.0
    assert patient_metrics["sensitivity"] == 0.5
    assert report["patient"]["probability_aggregation"]["method"] == "mean_window_probability"
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
    assert "patient_roc_auc_permutation_pvalue" in flat
    assert "window_accuracy" in flat
    assert "avg_window_accuracy" in flat


def test_build_clinical_metrics_report_extended_feature_stability():
    report = build_clinical_metrics_report(
        patient_true_labels=[1, 0, 1, 0],
        patient_pred_labels=[1, 0, 1, 0],
        patient_pred_probs=[0.9, 0.2, 0.8, 0.1],
        window_true_labels=[1, 1, 0, 0],
        window_pred_labels=[1, 1, 0, 0],
        window_pred_probs=[0.9, 0.8, 0.2, 0.1],
        fold_patient_accuracies=[1.0, 1.0],
        fold_window_accuracies=[1.0, 1.0],
        feature_selection_rows=[
            {
                "fold_idx": 0,
                "fold_target_class": "remission",
                "features": ["f1", "f2"],
                "candidate_feature_count": 4,
                "ranking_scores": {"f1": 3.0, "f2": 2.0, "f3": 1.0, "f4": 0.5},
                "effect_stats": {
                    "f1": {"cohens_d": 0.5, "sign": "positive"},
                    "f2": {"cohens_d": -0.3, "sign": "negative"},
                },
            },
            {
                "fold_idx": 1,
                "fold_target_class": "non_remission",
                "features": ["f1", "f3"],
                "candidate_feature_count": 4,
                "ranking_scores": {"f1": 2.5, "f3": 2.0, "f2": 1.0, "f4": 0.25},
                "effect_stats": {
                    "f1": {"cohens_d": 0.4, "sign": "positive"},
                    "f3": {"cohens_d": 0.2, "sign": "positive"},
                },
            },
        ],
        settings={
            "bootstrap_iterations": 10,
            "permutation_iterations": 10,
            "random_seed": 5,
            "ranking_stability_top_k": 2,
        },
    )

    feature_selection = report["feature_selection"]
    assert feature_selection["fixed_k_detected"] is True
    assert feature_selection["kuncheva_index_mean"] == pytest.approx(0.0)
    assert feature_selection["ranking_stability"]["mean_top_k_overlap"] == pytest.approx(0.5)
    assert feature_selection["effect_stability"]["available"] is True
    assert feature_selection["effect_stability"]["feature_count"] == 3
    assert feature_selection["importance_stability"]["available"] is True
    assert feature_selection["class_conditional_selection_available"] is True
    assert feature_selection["fold_class_counts"] == {
        "remission_folds": 1,
        "non_remission_folds": 1,
        "mixed_folds": 0,
        "unknown_folds": 0,
    }
    assert feature_selection["selection_frequency_by_feature"][0]["feature"] == "f1"
    assert feature_selection["selection_frequency_by_feature"][0]["share"] == pytest.approx(1.0)
    assert feature_selection["selection_frequency_by_feature"][0]["share_remission"] == pytest.approx(1.0)
    assert feature_selection["selection_frequency_by_feature"][0]["share_non_remission"] == pytest.approx(1.0)
    assert feature_selection["selection_frequency_by_feature"][0]["delta_remission_minus_non_remission"] == pytest.approx(0.0)
    assert feature_selection["selection_frequency_by_feature"][1]["feature"] == "f2"
    assert feature_selection["selection_frequency_by_feature"][1]["share_remission"] == pytest.approx(1.0)
    assert feature_selection["selection_frequency_by_feature"][1]["share_non_remission"] == pytest.approx(0.0)
    assert feature_selection["selection_frequency_by_feature"][1]["delta_remission_minus_non_remission"] == pytest.approx(1.0)

    flat = flatten_metric_report_for_mlflow(report)
    assert flat["feature_selection_kuncheva_index_mean"] == pytest.approx(0.0)
    assert flat["feature_selection_mean_top_k_overlap"] == pytest.approx(0.5)
    assert flat["feature_selection_class_conditional_selection_available"] == pytest.approx(1.0)
    assert flat["feature_selection_remission_fold_count"] == pytest.approx(1.0)
    assert flat["feature_selection_non_remission_fold_count"] == pytest.approx(1.0)
    assert "feature_selection_correct_remission_window_share" not in flat


def test_build_clinical_metrics_report_includes_per_fold_metrics():
    report = build_clinical_metrics_report(
        patient_true_labels=[1, 0, 1, 0],
        patient_pred_labels=[1, 0, 0, 0],
        patient_pred_probs=[0.9, 0.2, 0.4, 0.1],
        patient_prediction_rows=[
            {"fold": 0, "true_label": 1, "predicted_label": 1, "probability": 0.9},
            {"fold": 0, "true_label": 0, "predicted_label": 0, "probability": 0.2},
            {"fold": 1, "true_label": 1, "predicted_label": 0, "probability": 0.4},
            {"fold": 1, "true_label": 0, "predicted_label": 0, "probability": 0.1},
        ],
        window_true_labels=[1, 0, 1, 0],
        window_pred_labels=[1, 0, 1, 1],
        window_pred_probs=[0.8, 0.1, 0.7, 0.6],
        window_prediction_rows=[
            {"fold": 0, "true_label": 1, "predicted_label": 1, "probability": 0.8},
            {"fold": 0, "true_label": 0, "predicted_label": 0, "probability": 0.1},
            {"fold": 1, "true_label": 1, "predicted_label": 1, "probability": 0.7},
            {"fold": 1, "true_label": 0, "predicted_label": 1, "probability": 0.6},
        ],
        fold_patient_accuracies=[1.0, 0.5],
        fold_window_accuracies=[1.0, 0.5],
        settings={
            "bootstrap_iterations": 10,
            "permutation_iterations": 10,
            "random_seed": 5,
        },
    )

    patient_fold_metrics = report["patient"]["fold_metrics"]
    window_fold_metrics = report["window"]["fold_metrics"]
    assert len(patient_fold_metrics) == 2
    assert patient_fold_metrics[0]["fold_idx"] == 0
    assert patient_fold_metrics[0]["metrics"]["accuracy"] == pytest.approx(1.0)
    assert patient_fold_metrics[1]["metrics"]["accuracy"] == pytest.approx(0.5)
    assert report["patient"]["fold_metric_summary"]["accuracy"]["mean"] == pytest.approx(0.75)

    assert len(window_fold_metrics) == 2
    assert window_fold_metrics[1]["metrics"]["f1"] == pytest.approx(2.0 / 3.0)
    assert report["window"]["fold_metric_summary"]["accuracy"]["mean"] == pytest.approx(0.75)

    flat = flatten_metric_report_for_mlflow(report)
    assert flat["patient_fold_accuracy_mean"] == pytest.approx(0.75)
    assert flat["window_fold_f1_mean"] == pytest.approx((1.0 + (2.0 / 3.0)) / 2.0)


def test_summarize_feature_selection_handles_single_class_fold_coverage():
    summary = summarize_feature_selection(
        feature_selection_rows=[
            {"fold_idx": 0, "fold_target_class": "remission", "features": ["f1", "f2"]},
            {"fold_idx": 1, "fold_target_class": "remission", "features": ["f1"]},
        ]
    )

    assert summary["class_conditional_selection_available"] is False
    assert summary["class_conditional_selection_reason"] == "insufficient_class_fold_coverage"
    assert summary["fold_class_counts"]["remission_folds"] == 2
    assert summary["selection_frequency_by_feature"][0]["feature"] == "f1"
    assert summary["selection_frequency_by_feature"][0]["share_remission"] == pytest.approx(1.0)
    assert summary["selection_frequency_by_feature"][0]["share_non_remission"] is None
    assert summary["selection_frequency_by_feature"][0]["delta_remission_minus_non_remission"] is None
    assert summary["selection_frequency_by_feature"][0]["ratio_remission_to_non_remission"] is None


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
    assert summary["kuncheva_reason"] == "no_feature_sets"
    assert summary["class_conditional_selection_available"] is False
