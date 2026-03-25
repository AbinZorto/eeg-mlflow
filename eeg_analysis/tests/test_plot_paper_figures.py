import importlib.util
import json
import sys
from pathlib import Path

import pytest

from src.utils.plot_data_loader import load_run_artifact_bundle


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "plot_paper_figures.py"
SPEC = importlib.util.spec_from_file_location("plot_paper_figures", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_plot_fixture(tmp_path: Path, *, with_nested_runs: bool = True) -> dict:
    artifacts_dir = tmp_path / "artifacts"
    mlruns_root = tmp_path / "mlruns"
    experiment_id = "111222333"
    run_id = "1234567890abcdef1234567890abcdef"
    run_dir = mlruns_root / experiment_id / run_id
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    _write_json(
        artifacts / "clinical_metrics_summary.json",
        {
            "patient": {
                "metrics": {
                    "balanced_accuracy": 0.75,
                    "roc_auc": 1.0,
                    "pr_auc": 1.0,
                    "f1": 0.8,
                    "mcc": 0.5773502691896258,
                },
                "confidence_intervals": {
                    "balanced_accuracy": {"ci_low": 0.5, "ci_high": 1.0, "std": 0.1},
                    "roc_auc": {"ci_low": 0.8, "ci_high": 1.0, "std": 0.05},
                    "pr_auc": {"ci_low": 0.75, "ci_high": 1.0, "std": 0.06},
                    "f1": {"ci_low": 0.5, "ci_high": 1.0, "std": 0.12},
                    "mcc": {"ci_low": 0.2, "ci_high": 0.9, "std": 0.14},
                },
                "fold_metrics": [
                    {"fold_idx": 0, "metrics": {"accuracy": 1.0}},
                    {"fold_idx": 1, "metrics": {"accuracy": 0.5}},
                ],
            },
            "window": {
                "metrics": {
                    "accuracy": 0.75,
                    "f1": 0.8,
                    "roc_auc": 0.9,
                }
            },
        },
    )
    _write_json(
        artifacts / "feature_selection_stability.json",
        {
            "kuncheva_index_mean": 0.0,
            "kuncheva_index_median": 0.0,
            "effect_stability": {
                "available": True,
                "features": [
                    {
                        "feature": "f1",
                        "selection_share": 1.0,
                        "cohens_d_mean": 0.5,
                        "cohens_d_std": 0.1,
                        "sign_consistency": 1.0,
                    },
                    {
                        "feature": "f2",
                        "selection_share": 0.5,
                        "cohens_d_mean": -0.3,
                        "cohens_d_std": 0.05,
                        "sign_consistency": 1.0,
                    },
                    {
                        "feature": "f3",
                        "selection_share": 0.5,
                        "cohens_d_mean": 0.2,
                        "cohens_d_std": 0.08,
                        "sign_consistency": 1.0,
                    },
                ],
            },
        },
    )
    _write_text(
        artifacts / "advanced_hybrid_1dcnn_lstm_patient_predictions.csv",
        "\n".join(
            [
                "fold,participant,true_label,predicted_label,probability,n_windows,n_positive_windows,window_accuracy",
                "0,p1,1,1,0.90,10,9,0.80",
                "1,p2,0,0,0.10,10,1,0.90",
                "2,p3,1,1,0.80,10,8,0.70",
                "3,p4,0,1,0.60,10,6,0.50",
            ]
        )
        + "\n",
    )
    _write_text(
        artifacts / "advanced_hybrid_1dcnn_lstm_window_predictions.csv",
        "\n".join(
            [
                "fold,participant,true_label,predicted_label,probability,correct",
                "0,p1,1,1,0.95,True",
                "0,p1,1,1,0.88,True",
                "1,p2,0,0,0.12,True",
                "1,p2,0,0,0.08,True",
            ]
        )
        + "\n",
    )

    if with_nested_runs:
        nested_payloads = [
            (
                "child-run-a",
                0,
                ["f1", "f2"],
                [{"fold": 0, "participant": "p1", "true_label": 1, "predicted_label": 1, "probability": 0.9}],
                [
                    {"fold": 0, "participant": "p1", "true_label": 1, "predicted_label": 1, "probability": 0.95},
                    {"fold": 0, "participant": "p1", "true_label": 1, "predicted_label": 1, "probability": 0.88},
                ],
            ),
            (
                "child-run-b",
                1,
                ["f1", "f3"],
                [{"fold": 1, "participant": "p2", "true_label": 0, "predicted_label": 0, "probability": 0.1}],
                [
                    {"fold": 1, "participant": "p2", "true_label": 0, "predicted_label": 0, "probability": 0.12},
                    {"fold": 1, "participant": "p2", "true_label": 0, "predicted_label": 0, "probability": 0.08},
                ],
            ),
        ]
        for nested_run_id, fold_idx, features, patient_rows, window_rows in nested_payloads:
            nested_dir = mlruns_root / experiment_id / nested_run_id
            _write_text(nested_dir / "tags" / "mlflow.parentRunId", run_id)
            _write_text(nested_dir / "params" / "fold_index", f"{fold_idx}\n")
            _write_text(nested_dir / "params" / "selected_features_list", repr(features) + "\n")
            _write_json(nested_dir / "artifacts" / "fold_patient_predictions.json", patient_rows)
            _write_json(nested_dir / "artifacts" / "fold_window_predictions.json", window_rows)

    results_record = {
        "record_type": "job_attempt",
        "status": "success",
        "created_at": "2026-03-24T12:00:00+00:00",
        "attempt_index": 1,
        "run_signature": "sig-paper",
        "job_key": "job-paper",
        "mlflow_run_id": run_id,
        "mlflow_experiment_id": experiment_id,
        "model": "advanced_hybrid_1dcnn_lstm",
        "fs_enabled": True,
        "fs_method": "select_k_best_f_classif",
        "n_features": 10,
        "ordering": "sequential",
        "dataset_run_id": "dataset-paper",
        "window_seconds": 2,
        "inner_k": 10,
        "outer_k": 10,
        "equalize_lopo_groups": "false",
        "use_smote": "false",
        "mlflow_metrics": {"patient_roc_auc": 1.0},
    }
    _write_text(artifacts_dir / "sample.results.jsonl", json.dumps(results_record) + "\n")

    return {
        "artifacts_dir": artifacts_dir,
        "mlruns_root": mlruns_root,
        "experiment_id": experiment_id,
        "run_id": run_id,
    }


def test_load_run_artifact_bundle_reconstructs_class_conditional_selection(tmp_path):
    fixture = _build_plot_fixture(tmp_path, with_nested_runs=True)

    bundle = load_run_artifact_bundle(
        mlruns_root=fixture["mlruns_root"],
        run_id=fixture["run_id"],
        experiment_id=fixture["experiment_id"],
    )

    assert len(bundle.nested_fold_runs) == 2
    assert bundle.jaccard_fold_ids == [0, 1]
    assert bundle.jaccard_matrix[0][1] == pytest.approx(1.0 / 3.0)
    assert bundle.jaccard_values == pytest.approx([1.0 / 3.0])

    feature_rows = {
        row["feature"]: row
        for row in bundle.feature_selection_summary["selection_frequency_by_feature"]
    }
    assert bundle.feature_selection_summary["fold_class_counts"] == {
        "remission_folds": 1,
        "non_remission_folds": 1,
        "mixed_folds": 0,
        "unknown_folds": 0,
    }
    assert feature_rows["f1"]["share_remission"] == pytest.approx(1.0)
    assert feature_rows["f1"]["share_non_remission"] == pytest.approx(1.0)
    assert feature_rows["f2"]["delta_remission_minus_non_remission"] == pytest.approx(1.0)
    assert feature_rows["f3"]["delta_remission_minus_non_remission"] == pytest.approx(-1.0)


def test_plot_paper_figures_main_writes_manifest_and_figures(tmp_path):
    fixture = _build_plot_fixture(tmp_path, with_nested_runs=True)
    output_dir = tmp_path / "paper_figures"

    exit_code = MODULE.main(
        [
            "--artifacts-dir",
            str(fixture["artifacts_dir"]),
            "--mlruns-root",
            str(fixture["mlruns_root"]),
            "--output-dir",
            str(output_dir),
            "--model",
            "advanced_hybrid_1dcnn_lstm",
            "--window-size",
            "2",
            "--inner-k",
            "10",
            "--outer-k",
            "10",
            "--fs-method",
            "select_k_best_f_classif",
            "--n-features",
            "10",
            "--equalize-lopo-groups",
            "false",
            "--use-smote",
            "false",
            "--figures",
            "roc,pr,confusion,metric_summary,selection_frequency,jaccard_heatmap,delta_scatter",
        ]
    )

    run_dir = output_dir / "run-12345678__model-advanced-hybrid-1dcnn-lstm__ws-2__ik-10__ok-10__eq-false__sm-false"
    manifest = json.loads((run_dir / "figure_manifest.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert (run_dir / "performance" / "patient_roc_curve.png").exists()
    assert (run_dir / "performance" / "patient_pr_curve.png").exists()
    assert (run_dir / "performance" / "patient_confusion_matrix.png").exists()
    assert (run_dir / "performance" / "patient_metric_summary.png").exists()
    assert (run_dir / "biomarker_stability" / "selection_frequency_top_features.png").exists()
    assert (run_dir / "biomarker_stability" / "pairwise_jaccard_heatmap.png").exists()
    assert (run_dir / "biomarker_interpretation" / "delta_vs_frequency_scatter.png").exists()
    assert len(manifest["generated_figures"]) == 7
    assert manifest["skipped_figures"] == []


def test_plot_paper_figures_records_skipped_figure_when_data_missing(tmp_path):
    fixture = _build_plot_fixture(tmp_path, with_nested_runs=False)
    output_dir = tmp_path / "paper_figures"

    exit_code = MODULE.main(
        [
            "--artifacts-dir",
            str(fixture["artifacts_dir"]),
            "--mlruns-root",
            str(fixture["mlruns_root"]),
            "--output-dir",
            str(output_dir),
            "--model",
            "advanced_hybrid_1dcnn_lstm",
            "--window-size",
            "2",
            "--inner-k",
            "10",
            "--outer-k",
            "10",
            "--fs-method",
            "select_k_best_f_classif",
            "--n-features",
            "10",
            "--equalize-lopo-groups",
            "false",
            "--use-smote",
            "false",
            "--figures",
            "jaccard_heatmap",
        ]
    )

    run_dir = output_dir / "run-12345678__model-advanced-hybrid-1dcnn-lstm__ws-2__ik-10__ok-10__eq-false__sm-false"
    manifest = json.loads((run_dir / "figure_manifest.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert manifest["generated_figures"] == []
    assert manifest["skipped_figures"] == [
        {
            "figure_id": "jaccard_heatmap",
            "reason": "insufficient_fold_feature_sets_for_jaccard_heatmap",
        }
    ]
