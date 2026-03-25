import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "compose_paper_panels.py"
SPEC = importlib.util.spec_from_file_location("compose_paper_panels", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_record(run_id: str, roc_auc: float, **overrides):
    record = {
        "record_type": "job_attempt",
        "status": "success",
        "created_at": "2026-03-24T12:00:00+00:00",
        "attempt_index": 1,
        "run_signature": "sig-default",
        "job_key": f"job-{run_id[:6]}",
        "mlflow_run_id": run_id,
        "mlflow_experiment_id": "exp-default",
        "model": "advanced_hybrid_1dcnn_lstm",
        "fs_enabled": True,
        "fs_method": "select_k_best_f_classif",
        "n_features": 10,
        "ordering": "sequential",
        "dataset_run_id": "dataset-default",
        "window_seconds": 2,
        "inner_k": 10,
        "outer_k": 10,
        "equalize_lopo_groups": "false",
        "use_smote": "false",
        "mlflow_metrics": {"patient_roc_auc": roc_auc},
    }
    record.update(overrides)
    return record


def write_results_file(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def write_fake_panel(path: Path, color: tuple[float, float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.ones((60, 90, 3), dtype=float)
    image[..., 0] *= color[0]
    image[..., 1] *= color[1]
    image[..., 2] *= color[2]
    plt.imsave(path, image)


def selection_args(tmp_path: Path, *extra: str):
    return MODULE.parse_args(
        [
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--paper-figures-dir",
            str(tmp_path / "paper_figures"),
            "--output-dir",
            str(tmp_path / "paper_panels"),
            *extra,
        ]
    )


def create_source_panels(selection, paper_figures_dir: Path, figure_ids):
    colors = (
        (0.2, 0.4, 0.7),
        (0.7, 0.3, 0.3),
        (0.3, 0.6, 0.4),
        (0.6, 0.4, 0.7),
    )
    for idx, figure_id in enumerate(figure_ids):
        path = MODULE.panel_output_path(
            paper_figures_dir=paper_figures_dir,
            selection_slug=selection.selection_slug,
            figure_id=figure_id,
        )
        write_fake_panel(path, colors[idx % len(colors)])


def test_resolve_selection_best_overall_prefers_highest_patient_roc_auc(tmp_path):
    write_results_file(
        tmp_path / "artifacts" / "sample.results.jsonl",
        [
            make_record("run-low", 0.62),
            make_record("run-high", 0.91, created_at="2026-03-24T12:05:00+00:00"),
        ],
    )

    selection = MODULE.resolve_selection(selection_args(tmp_path))

    assert selection.run_id == "run-high"
    assert selection.selected_row is not None
    assert selection.selected_row.patient_roc_auc == 0.91


def test_resolve_selection_run_signature_picks_best_within_signature(tmp_path):
    write_results_file(
        tmp_path / "artifacts" / "sample.results.jsonl",
        [
            make_record("run-a", 0.70, run_signature="sig-a"),
            make_record("run-b", 0.82, run_signature="sig-a", created_at="2026-03-24T12:10:00+00:00"),
            make_record("run-c", 0.95, run_signature="sig-c"),
        ],
    )

    selection = MODULE.resolve_selection(
        selection_args(tmp_path, "--select", "run-signature", "--run-signature", "sig-a")
    )

    assert selection.run_id == "run-b"
    assert selection.selected_row is not None
    assert selection.selected_row.run_signature == "sig-a"


def test_resolve_selection_run_signature_accepts_unique_prefix(tmp_path):
    full_signature = "7275973beaa2ed265738102706bf5d5009ae3762c0d740e5f98ad6b9f26450a4"
    write_results_file(
        tmp_path / "artifacts" / "sample.results.jsonl",
        [make_record("run-prefix", 0.77, run_signature=full_signature)],
    )

    selection = MODULE.resolve_selection(
        selection_args(tmp_path, "--select", "run-signature", "--run-signature", "7275973beaa2")
    )

    assert selection.run_id == "run-prefix"
    assert selection.selected_row is not None
    assert selection.selected_row.run_signature == full_signature


def test_resolve_selection_mlflow_run_id_bypasses_results_rows(tmp_path):
    selection = MODULE.resolve_selection(
        selection_args(tmp_path, "--select", "mlflow-run-id", "--mlflow-run-id", "explicit-run-id")
    )

    assert selection.run_id == "explicit-run-id"
    assert selection.selected_row is None
    assert selection.selection_slug == "run-explicit"


def test_main_writes_three_composites_and_manifest(tmp_path):
    write_results_file(
        tmp_path / "artifacts" / "sample.results.jsonl",
        [make_record("run-composite", 0.88)],
    )
    selection = MODULE.resolve_selection(selection_args(tmp_path))
    create_source_panels(
        selection,
        tmp_path / "paper_figures",
        MODULE.required_figure_ids(MODULE.ALL_COMPOSITES),
    )

    exit_code = MODULE.main(
        [
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--paper-figures-dir",
            str(tmp_path / "paper_figures"),
            "--output-dir",
            str(tmp_path / "paper_panels"),
        ]
    )

    output_root = tmp_path / "paper_panels" / selection.selection_slug
    manifest = json.loads((output_root / "panel_manifest.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert (output_root / "main_results_composite.png").exists()
    assert (output_root / "biomarker_stability_composite.png").exists()
    assert (output_root / "biomarker_interpretation_composite.png").exists()
    assert len(manifest["generated_composites"]) == 3
    assert manifest["missing_panels"] == []


def test_main_uses_placeholder_for_missing_source_panel(tmp_path, monkeypatch):
    write_results_file(
        tmp_path / "artifacts" / "sample.results.jsonl",
        [make_record("run-missing", 0.81)],
    )
    selection = MODULE.resolve_selection(
        selection_args(tmp_path, "--composites", "biomarker_stability")
    )
    figure_ids = MODULE.required_figure_ids(("biomarker_stability",))
    create_source_panels(selection, tmp_path / "paper_figures", figure_ids)
    missing_path = MODULE.panel_output_path(
        paper_figures_dir=tmp_path / "paper_figures",
        selection_slug=selection.selection_slug,
        figure_id="jaccard_heatmap",
    )
    missing_path.unlink()

    monkeypatch.setattr(MODULE.single_figures, "main", lambda argv: 0)

    exit_code = MODULE.main(
        [
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--paper-figures-dir",
            str(tmp_path / "paper_figures"),
            "--output-dir",
            str(tmp_path / "paper_panels"),
            "--composites",
            "biomarker_stability",
        ]
    )

    output_root = tmp_path / "paper_panels" / selection.selection_slug
    manifest = json.loads((output_root / "panel_manifest.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert (output_root / "biomarker_stability_composite.png").exists()
    assert manifest["missing_panels"] == [
        {
            "composite_id": "biomarker_stability",
            "expected_path": str(missing_path),
            "figure_id": "jaccard_heatmap",
            "panel_label": "B",
            "reason": "missing_source_panel",
        }
    ]


def test_force_regenerate_single_panels_invokes_single_figure_generator(tmp_path, monkeypatch):
    write_results_file(
        tmp_path / "artifacts" / "sample.results.jsonl",
        [make_record("run-force", 0.79)],
    )
    selection = MODULE.resolve_selection(
        selection_args(tmp_path, "--composites", "main_results")
    )
    create_source_panels(
        selection,
        tmp_path / "paper_figures",
        MODULE.required_figure_ids(("main_results",)),
    )

    captured = {"argv": None}

    def fake_single_main(argv):
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(MODULE.single_figures, "main", fake_single_main)

    exit_code = MODULE.main(
        [
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--paper-figures-dir",
            str(tmp_path / "paper_figures"),
            "--output-dir",
            str(tmp_path / "paper_panels"),
            "--composites",
            "main_results",
            "--force-regenerate-single-panels",
        ]
    )

    output_root = tmp_path / "paper_panels" / selection.selection_slug
    assert exit_code == 0
    assert (output_root / "main_results_composite.png").exists()
    assert captured["argv"] is not None
    assert "--mlflow-run-id" in captured["argv"]
    assert "run-force" in captured["argv"]
    assert "--figures" in captured["argv"]
