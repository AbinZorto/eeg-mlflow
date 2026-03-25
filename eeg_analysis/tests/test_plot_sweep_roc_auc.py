import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "plot_sweep_roc_auc.py"
SPEC = importlib.util.spec_from_file_location("plot_sweep_roc_auc", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_record(**overrides):
    record = {
        "record_type": "job_attempt",
        "status": "success",
        "created_at": "2026-03-24T12:00:00+00:00",
        "attempt_index": 1,
        "run_signature": "sig",
        "job_key": "job",
        "model": "advanced_hybrid_1dcnn_lstm",
        "fs_enabled": True,
        "fs_method": "select_k_best_f_classif",
        "n_features": 10,
        "ordering": "sequential",
        "dataset_run_id": "dataset-a",
        "window_seconds": 2,
        "inner_k": 5,
        "outer_k": 10,
        "equalize_lopo_groups": "false",
        "use_smote": "true",
        "mlflow_metrics": {
            "patient_roc_auc": 0.61,
        },
    }
    record.update(overrides)
    return record


def make_row(metric_key="patient_roc_auc", **overrides):
    row = MODULE.results_record_to_plot_row(make_record(**overrides), Path("fake.results.jsonl"), metric_key)
    assert row is not None
    return row


def test_extract_patient_roc_auc_prefers_mlflow_metrics():
    value = MODULE.extract_patient_roc_auc(
        make_record(
            mlflow_metrics={"patient_roc_auc": 0.72},
            metrics={"patient": {"roc_auc": 0.55}},
        )
    )
    assert value == 0.72


def test_extract_patient_roc_auc_falls_back_to_normalized_metrics():
    value = MODULE.extract_patient_roc_auc(
        make_record(
            mlflow_metrics={},
            metrics={"patient": {"roc_auc": 0.55}},
        )
    )
    assert value == 0.55


def test_extract_metric_value_supports_patient_pr_auc_and_feature_selection_metrics():
    record = make_record(
        mlflow_metrics={
            "patient_pr_auc": 0.67,
            "feature_selection_mean_pairwise_jaccard": 0.42,
        }
    )

    assert MODULE.extract_metric_value(record, "patient_pr_auc") == 0.67
    assert MODULE.extract_metric_value(record, "feature_selection_mean_pairwise_jaccard") == 0.42


def test_deduplicate_plot_rows_keeps_latest_successful_attempt():
    rows = [
        make_row(attempt_index=1, created_at="2026-03-24T12:00:00+00:00"),
        make_row(attempt_index=3, created_at="2026-03-24T12:05:00+00:00", mlflow_metrics={"patient_roc_auc": 0.83}),
        make_row(window_seconds=4, mlflow_metrics={"patient_roc_auc": 0.65}),
    ]

    deduped = MODULE.deduplicate_plot_rows(rows)

    assert len(deduped) == 2
    same_point = [row for row in deduped if row.window_seconds == 2][0]
    assert same_point.attempt_index == 3
    assert same_point.metric_value == 0.83


def test_build_trajectory_key_window_seconds_groups_by_other_settings():
    row = make_row(window_seconds=8, inner_k=7, outer_k=11, equalize_lopo_groups="true", use_smote="false")
    key = MODULE.build_trajectory_key(row, "window_seconds")
    assert key == (
        "advanced_hybrid_1dcnn_lstm",
        7,
        11,
        "true",
        "false",
        True,
        "select_k_best_f_classif",
        10,
        "sequential",
        "dataset-a",
    )


def test_build_trajectory_key_inner_k_groups_by_other_settings():
    row = make_row(window_seconds=8, inner_k=7, outer_k=11, equalize_lopo_groups="true", use_smote="false")
    key = MODULE.build_trajectory_key(row, "inner_k")
    assert key == (
        "advanced_hybrid_1dcnn_lstm",
        8,
        11,
        "true",
        "false",
        True,
        "select_k_best_f_classif",
        10,
        "sequential",
        "dataset-a",
    )


def test_compute_average_series_groups_by_x_value():
    rows = [
        make_row(window_seconds=2, mlflow_metrics={"patient_roc_auc": 0.4}),
        make_row(window_seconds=2, inner_k=7, mlflow_metrics={"patient_roc_auc": 0.8}),
        make_row(window_seconds=4, mlflow_metrics={"patient_roc_auc": 0.5}),
    ]

    averages = MODULE.compute_average_series(rows, "window_seconds")

    assert averages[0][0] == 2
    assert averages[0][1] == pytest.approx(0.6)
    assert averages[1] == (4, 0.5)


def test_infer_metric_label_handles_requested_metric_families():
    assert MODULE.infer_metric_label("patient_pr_auc") == "Patient PR-AUC"
    assert MODULE.infer_metric_label("patient_balanced_accuracy") == "Patient Balanced Accuracy"
    assert MODULE.infer_metric_label("feature_selection_mean_pairwise_jaccard") == "Mean Pairwise Jaccard"
    assert MODULE.infer_metric_label("feature_selection_kuncheva_index_mean") == "Mean Kuncheva Index"


def test_infer_y_bounds_uses_unit_interval_and_signed_autoscale():
    unit_rows = [
        make_row(metric_key="patient_pr_auc", mlflow_metrics={"patient_pr_auc": 0.4}),
        make_row(metric_key="patient_pr_auc", window_seconds=4, mlflow_metrics={"patient_pr_auc": 0.8}),
    ]
    signed_rows = [
        make_row(metric_key="feature_selection_kuncheva_index_mean", mlflow_metrics={"feature_selection_kuncheva_index_mean": -0.2}),
        make_row(
            metric_key="feature_selection_kuncheva_index_mean",
            window_seconds=4,
            mlflow_metrics={"feature_selection_kuncheva_index_mean": 0.3},
        ),
    ]

    assert MODULE.infer_y_bounds(unit_rows, "patient_pr_auc", y_min=None, y_max=None) == (0.0, 1.0)
    lower, upper = MODULE.infer_y_bounds(
        signed_rows,
        "feature_selection_kuncheva_index_mean",
        y_min=None,
        y_max=None,
    )
    assert lower < -0.2
    assert upper > 0.3


def test_resolve_requested_metrics_defaults_to_patient_roc_auc():
    args = MODULE.parse_args([])
    assert MODULE.resolve_requested_metrics(args) == ["patient_roc_auc"]


def test_resolve_requested_metrics_expands_presets_and_deduplicates_explicit_metrics():
    args = MODULE.parse_args(
        [
            "--preset",
            "performance_core",
            "--preset",
            "biomarker_core",
            "--metric",
            "patient_pr_auc",
            "--metric",
            "feature_selection_mean_pairwise_jaccard",
        ]
    )

    assert MODULE.resolve_requested_metrics(args) == [
        "patient_roc_auc",
        "patient_pr_auc",
        "patient_balanced_accuracy",
        "patient_f1",
        "patient_mcc",
        "feature_selection_mean_pairwise_jaccard",
        "feature_selection_kuncheva_index_mean",
        "feature_selection_mean_top_k_overlap",
        "feature_selection_effect_mean_sign_consistency",
    ]


def test_main_lists_presets_and_exits_cleanly(capsys):
    exit_code = MODULE.main(["--list-presets"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Available presets:" in output
    assert "performance_core" in output
    assert "biomarker_core" in output


def test_main_writes_two_plots_and_csvs_for_default_metric(tmp_path):
    results_path = tmp_path / "sample.results.jsonl"
    records = [
        make_record(window_seconds=2, inner_k=5, outer_k=10, mlflow_metrics={"patient_roc_auc": 0.4}),
        make_record(window_seconds=4, inner_k=5, outer_k=10, mlflow_metrics={"patient_roc_auc": 0.6}),
        make_record(window_seconds=2, inner_k=7, outer_k=10, mlflow_metrics={"patient_roc_auc": 0.5}),
        make_record(window_seconds=4, inner_k=7, outer_k=10, mlflow_metrics={"patient_roc_auc": 0.7}),
        make_record(window_seconds=2, inner_k=5, outer_k=15, mlflow_metrics={"patient_roc_auc": 0.45}),
        make_record(window_seconds=4, inner_k=5, outer_k=15, mlflow_metrics={"patient_roc_auc": 0.65}),
        make_record(window_seconds=2, inner_k=7, outer_k=15, mlflow_metrics={"patient_roc_auc": 0.55}),
        make_record(window_seconds=4, inner_k=7, outer_k=15, mlflow_metrics={"patient_roc_auc": 0.75}),
    ]
    with results_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(MODULE.json.dumps(record) + "\n")

    output_dir = tmp_path / "plots"
    exit_code = MODULE.main(
        [
            "--artifacts-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "window_size_vs_patient_roc_auc.png").exists()
    assert (output_dir / "inner_k_vs_patient_roc_auc.png").exists()
    assert (output_dir / "plot_data_window_size_vs_patient_roc_auc.csv").exists()
    assert (output_dir / "plot_data_inner_k_vs_patient_roc_auc.csv").exists()


def test_main_writes_metric_specific_outputs_for_pr_auc(tmp_path):
    results_path = tmp_path / "sample.results.jsonl"
    records = [
        make_record(window_seconds=2, inner_k=5, outer_k=10, mlflow_metrics={"patient_pr_auc": 0.4}),
        make_record(window_seconds=4, inner_k=5, outer_k=10, mlflow_metrics={"patient_pr_auc": 0.6}),
        make_record(window_seconds=2, inner_k=7, outer_k=10, mlflow_metrics={"patient_pr_auc": 0.5}),
        make_record(window_seconds=4, inner_k=7, outer_k=10, mlflow_metrics={"patient_pr_auc": 0.7}),
    ]
    with results_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(MODULE.json.dumps(record) + "\n")

    output_dir = tmp_path / "plots"
    exit_code = MODULE.main(
        [
            "--artifacts-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--metric",
            "patient_pr_auc",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "window_size_vs_patient_pr_auc.png").exists()
    assert (output_dir / "inner_k_vs_patient_pr_auc.png").exists()
    assert (output_dir / "plot_data_window_size_vs_patient_pr_auc.csv").exists()
    assert (output_dir / "plot_data_inner_k_vs_patient_pr_auc.csv").exists()


def test_main_writes_metric_specific_outputs_for_jaccard(tmp_path):
    results_path = tmp_path / "sample.results.jsonl"
    records = [
        make_record(
            window_seconds=2,
            inner_k=5,
            outer_k=10,
            mlflow_metrics={"feature_selection_mean_pairwise_jaccard": 0.2},
        ),
        make_record(
            window_seconds=4,
            inner_k=5,
            outer_k=10,
            mlflow_metrics={"feature_selection_mean_pairwise_jaccard": 0.4},
        ),
        make_record(
            window_seconds=2,
            inner_k=7,
            outer_k=10,
            mlflow_metrics={"feature_selection_mean_pairwise_jaccard": 0.3},
        ),
        make_record(
            window_seconds=4,
            inner_k=7,
            outer_k=10,
            mlflow_metrics={"feature_selection_mean_pairwise_jaccard": 0.5},
        ),
    ]
    with results_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(MODULE.json.dumps(record) + "\n")

    output_dir = tmp_path / "plots"
    exit_code = MODULE.main(
        [
            "--artifacts-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--metric",
            "feature_selection_mean_pairwise_jaccard",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "window_size_vs_feature_selection_mean_pairwise_jaccard.png").exists()
    assert (output_dir / "inner_k_vs_feature_selection_mean_pairwise_jaccard.png").exists()
    assert (output_dir / "plot_data_window_size_vs_feature_selection_mean_pairwise_jaccard.csv").exists()
    assert (output_dir / "plot_data_inner_k_vs_feature_selection_mean_pairwise_jaccard.csv").exists()


def test_main_preset_writes_multiple_metric_outputs(tmp_path):
    results_path = tmp_path / "sample.results.jsonl"
    records = [
        make_record(
            window_seconds=2,
            inner_k=5,
            outer_k=10,
            mlflow_metrics={
                "patient_roc_auc": 0.61,
                "patient_pr_auc": 0.51,
                "patient_balanced_accuracy": 0.55,
                "patient_f1": 0.58,
                "patient_mcc": 0.21,
            },
        ),
        make_record(
            window_seconds=4,
            inner_k=5,
            outer_k=10,
            mlflow_metrics={
                "patient_roc_auc": 0.71,
                "patient_pr_auc": 0.63,
                "patient_balanced_accuracy": 0.66,
                "patient_f1": 0.67,
                "patient_mcc": 0.34,
            },
        ),
        make_record(
            window_seconds=2,
            inner_k=7,
            outer_k=10,
            mlflow_metrics={
                "patient_roc_auc": 0.65,
                "patient_pr_auc": 0.56,
                "patient_balanced_accuracy": 0.6,
                "patient_f1": 0.61,
                "patient_mcc": 0.28,
            },
        ),
        make_record(
            window_seconds=4,
            inner_k=7,
            outer_k=10,
            mlflow_metrics={
                "patient_roc_auc": 0.73,
                "patient_pr_auc": 0.66,
                "patient_balanced_accuracy": 0.68,
                "patient_f1": 0.69,
                "patient_mcc": 0.37,
            },
        ),
    ]
    with results_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(MODULE.json.dumps(record) + "\n")

    output_dir = tmp_path / "plots"
    exit_code = MODULE.main(
        [
            "--artifacts-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--preset",
            "performance_core",
        ]
    )

    assert exit_code == 0
    for metric_key in MODULE.METRIC_PRESETS["performance_core"]:
        metric_slug = MODULE.slugify(metric_key).replace("-", "_")
        assert (output_dir / f"window_size_vs_{metric_slug}.png").exists()
        assert (output_dir / f"inner_k_vs_{metric_slug}.png").exists()


def test_main_preset_continues_when_some_metrics_are_missing(tmp_path, capsys):
    results_path = tmp_path / "sample.results.jsonl"
    records = [
        make_record(
            window_seconds=2,
            inner_k=5,
            outer_k=10,
            mlflow_metrics={
                "feature_selection_mean_pairwise_jaccard": 0.21,
                "feature_selection_kuncheva_index_mean": 0.08,
            },
        ),
        make_record(
            window_seconds=4,
            inner_k=5,
            outer_k=10,
            mlflow_metrics={
                "feature_selection_mean_pairwise_jaccard": 0.34,
                "feature_selection_kuncheva_index_mean": 0.19,
            },
        ),
    ]
    with results_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(MODULE.json.dumps(record) + "\n")

    output_dir = tmp_path / "plots"
    exit_code = MODULE.main(
        [
            "--artifacts-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--preset",
            "biomarker_core",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "window_size_vs_feature_selection_mean_pairwise_jaccard.png").exists()
    assert (output_dir / "window_size_vs_feature_selection_kuncheva_index_mean.png").exists()
    assert not (output_dir / "window_size_vs_feature_selection_mean_top_k_overlap.png").exists()
    output = capsys.readouterr().out
    assert "Skipped metric: feature_selection_mean_top_k_overlap" in output


def test_main_returns_non_zero_when_no_requested_metric_produces_plots(tmp_path):
    results_path = tmp_path / "sample.results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        handle.write(MODULE.json.dumps(make_record(mlflow_metrics={"patient_roc_auc": 0.61})) + "\n")

    output_dir = tmp_path / "plots"
    exit_code = MODULE.main(
        [
            "--artifacts-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--preset",
            "biomarker_core",
        ]
    )

    assert exit_code == 1
