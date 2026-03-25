#!/usr/bin/env python3
"""Compose manuscript-ready multi-panel figures from paper-style single panels."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
EEG_ANALYSIS_ROOT = REPO_ROOT / "eeg_analysis"
if str(EEG_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(EEG_ANALYSIS_ROOT))

PLOT_PAPER_FIGURES_PATH = Path(__file__).resolve().parent / "plot_paper_figures.py"
PLOT_PAPER_FIGURES_SPEC = importlib.util.spec_from_file_location("plot_paper_figures_module", PLOT_PAPER_FIGURES_PATH)
single_figures = importlib.util.module_from_spec(PLOT_PAPER_FIGURES_SPEC)
assert PLOT_PAPER_FIGURES_SPEC is not None and PLOT_PAPER_FIGURES_SPEC.loader is not None
sys.modules[PLOT_PAPER_FIGURES_SPEC.name] = single_figures
PLOT_PAPER_FIGURES_SPEC.loader.exec_module(single_figures)

from src.utils.panel_composer import CompositePanel, compose_composite_figure  # noqa: E402
from src.utils.plot_data_loader import (  # noqa: E402
    SweepResultRow,
    deduplicate_results_rows,
    discover_results_files,
    filter_result_rows,
    load_results_rows,
)


SOURCE_PANEL_FORMAT = "png"
ALL_COMPOSITES = ("main_results", "biomarker_stability", "biomarker_interpretation")

COMPOSITE_SPECS = {
    "main_results": {
        "title": "Main Results",
        "panels": [
            ("A", "roc", "ROC Curve"),
            ("B", "pr", "Precision-Recall Curve"),
            ("C", "confusion", "Confusion Matrix"),
            ("D", "metric_summary", "Metric Summary"),
        ],
    },
    "biomarker_stability": {
        "title": "Biomarker Stability",
        "panels": [
            ("A", "selection_frequency", "Selection Frequency"),
            ("B", "jaccard_heatmap", "Pairwise Jaccard Heatmap"),
            ("C", "jaccard_distribution", "Jaccard Distribution"),
            ("D", "kuncheva_summary", "Kuncheva Summary"),
        ],
    },
    "biomarker_interpretation": {
        "title": "Biomarker Interpretation",
        "panels": [
            ("A", "delta_scatter", "Delta vs Frequency"),
            ("B", "effect_size", "Effect Size"),
            ("C", "sign_consistency", "Sign Consistency"),
            ("D", "class_conditional_bars", "Class-Conditional Selection"),
        ],
    },
}


@dataclass(frozen=True)
class ResolvedSelection:
    mode: str
    run_id: str
    experiment_id: Optional[str]
    selected_row: Optional[SweepResultRow]
    selection_slug: str


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose multi-panel paper figures from single-panel outputs.")
    parser.add_argument("--artifacts-dir", default="sweeps/artifacts", help="Directory containing results ledgers.")
    parser.add_argument("--mlruns-root", default="mlruns", help="Local MLflow file-backend root.")
    parser.add_argument("--paper-figures-dir", default="sweeps/paper_figures", help="Directory containing single-panel paper figures.")
    parser.add_argument("--output-dir", default="sweeps/paper_panels", help="Directory for generated composite figures.")
    parser.add_argument("--results-glob", default="*.results.jsonl", help="Glob pattern under artifacts-dir.")
    parser.add_argument(
        "--select",
        choices=("best-overall", "mlflow-run-id", "run-signature"),
        default="best-overall",
        help="Run-selection mode for panel composition.",
    )
    parser.add_argument("--mlflow-run-id", help="Explicit MLflow run id when --select mlflow-run-id is used.")
    parser.add_argument("--run-signature", help="Explicit run signature when --select run-signature is used.")
    parser.add_argument("--model", help="Optional model filter.")
    parser.add_argument("--window-size", type=int, help="Optional window-size filter.")
    parser.add_argument("--inner-k", type=int, help="Optional inner-k filter.")
    parser.add_argument("--outer-k", type=int, help="Optional outer-k filter.")
    parser.add_argument("--fs-method", help="Optional FS method filter.")
    parser.add_argument("--n-features", type=int, help="Optional feature-count filter.")
    parser.add_argument("--ordering", help="Optional ordering filter.")
    parser.add_argument("--equalize-lopo-groups", choices=("true", "false"), help="Optional group-balancing filter.")
    parser.add_argument("--use-smote", choices=("true", "false"), help="Optional SMOTE filter.")
    parser.add_argument("--composites", help="Comma-separated subset of composites to generate.")
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png", help="Composite output format.")
    parser.add_argument("--dpi", type=int, default=300, help="Composite figure DPI.")
    parser.add_argument(
        "--force-regenerate-single-panels",
        action="store_true",
        help="Regenerate the underlying single-panel figures before composing.",
    )
    return parser.parse_args(argv)


def _row_priority(row: SweepResultRow) -> tuple[float, str, int, str]:
    roc_auc = row.patient_roc_auc if row.patient_roc_auc is not None else float("-inf")
    return (roc_auc, row.created_at_sort, row.attempt_index, row.mlflow_run_id)


def select_best_result_row(rows: Sequence[SweepResultRow]) -> Optional[SweepResultRow]:
    if not rows:
        return None
    return max(rows, key=_row_priority)


def load_filtered_rows(args: argparse.Namespace) -> List[SweepResultRow]:
    results_paths = discover_results_files(Path(args.artifacts_dir), args.results_glob)
    rows = deduplicate_results_rows(load_results_rows(results_paths))
    filtered = filter_result_rows(
        rows,
        mlflow_run_id=args.mlflow_run_id if args.select == "mlflow-run-id" else None,
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
    if args.select == "run-signature" and args.run_signature:
        exact = [row for row in filtered if row.run_signature == args.run_signature]
        if exact:
            return exact
        return [
            row
            for row in filtered
            if row.run_signature is not None and row.run_signature.startswith(args.run_signature)
        ]
    return filtered


def resolve_selection(args: argparse.Namespace) -> ResolvedSelection:
    rows = load_filtered_rows(args)

    if args.select == "mlflow-run-id":
        if not args.mlflow_run_id:
            raise SystemExit("--mlflow-run-id is required when --select mlflow-run-id is used.")
        selected_row = rows[0] if rows else None
        run_id = args.mlflow_run_id
        experiment_id = selected_row.mlflow_experiment_id if selected_row is not None else None
    elif args.select == "run-signature":
        if not args.run_signature:
            raise SystemExit("--run-signature is required when --select run-signature is used.")
        selected_row = select_best_result_row(rows)
        if selected_row is None:
            raise SystemExit("No successful results row matched the requested run signature and filters.")
        run_id = selected_row.mlflow_run_id
        experiment_id = selected_row.mlflow_experiment_id
    else:
        selected_row = select_best_result_row(rows)
        if selected_row is None:
            raise SystemExit("No successful results row matched the provided filters.")
        run_id = selected_row.mlflow_run_id
        experiment_id = selected_row.mlflow_experiment_id

    selection_slug = single_figures.build_selection_slug(
        selected_row,
        run_id,
        SimpleNamespace(model=args.model),
    )
    return ResolvedSelection(
        mode=args.select,
        run_id=run_id,
        experiment_id=experiment_id,
        selected_row=selected_row,
        selection_slug=selection_slug,
    )


def build_subtitle(selection: ResolvedSelection) -> str:
    row = selection.selected_row
    parts = [f"Selection mode: {selection.mode}", f"Run: {selection.run_id[:12]}"]
    if row is not None:
        if row.model:
            parts.append(f"Model: {row.model}")
        if row.window_seconds is not None:
            parts.append(f"Window: {row.window_seconds}s")
        if row.inner_k is not None:
            parts.append(f"Inner-k: {row.inner_k}")
        if row.outer_k is not None:
            parts.append(f"Outer-k: {row.outer_k}")
        if row.patient_roc_auc is not None:
            parts.append(f"Patient ROC-AUC: {row.patient_roc_auc:.3f}")
    return " | ".join(parts)


def panel_output_path(
    *,
    paper_figures_dir: Path,
    selection_slug: str,
    figure_id: str,
) -> Path:
    subdir, base_name, _ = single_figures.FIGURE_SPECS[figure_id]
    return paper_figures_dir / selection_slug / subdir / f"{base_name}.{SOURCE_PANEL_FORMAT}"


def required_figure_ids(composites: Sequence[str]) -> List[str]:
    ordered: List[str] = []
    for composite_id in composites:
        for _, figure_id, _ in COMPOSITE_SPECS[composite_id]["panels"]:
            if figure_id not in ordered:
                ordered.append(figure_id)
    return ordered


def ensure_single_panels(
    *,
    args: argparse.Namespace,
    selection: ResolvedSelection,
    composite_ids: Sequence[str],
) -> Dict[str, Any]:
    paper_figures_dir = Path(args.paper_figures_dir)
    figure_ids = required_figure_ids(composite_ids)
    missing = [
        figure_id
        for figure_id in figure_ids
        if not panel_output_path(
            paper_figures_dir=paper_figures_dir,
            selection_slug=selection.selection_slug,
            figure_id=figure_id,
        ).exists()
    ]
    should_regenerate = args.force_regenerate_single_panels or bool(missing)
    if not should_regenerate:
        return {
            "regenerated": False,
            "requested_figure_ids": figure_ids,
            "missing_before_regeneration": missing,
        }

    figure_arg = ",".join(figure_ids)
    cli_args = [
        "--artifacts-dir",
        str(Path(args.artifacts_dir)),
        "--mlruns-root",
        str(Path(args.mlruns_root)),
        "--output-dir",
        str(Path(args.paper_figures_dir)),
        "--mlflow-run-id",
        selection.run_id,
        "--figures",
        figure_arg,
        "--format",
        SOURCE_PANEL_FORMAT,
    ]
    exit_code = single_figures.main(cli_args)
    if exit_code != 0:
        raise SystemExit(f"Single-panel regeneration failed with exit code {exit_code}.")

    return {
        "regenerated": True,
        "requested_figure_ids": figure_ids,
        "missing_before_regeneration": missing,
    }


def compose_requested_panels(
    *,
    args: argparse.Namespace,
    selection: ResolvedSelection,
    composite_ids: Sequence[str],
) -> Dict[str, Any]:
    paper_figures_dir = Path(args.paper_figures_dir)
    output_root = Path(args.output_dir) / selection.selection_slug
    output_root.mkdir(parents=True, exist_ok=True)
    subtitle = build_subtitle(selection)

    generated: List[Dict[str, str]] = []
    missing_panels: List[Dict[str, str]] = []

    for composite_id in composite_ids:
        spec = COMPOSITE_SPECS[composite_id]
        panels: List[CompositePanel] = []
        for label, figure_id, title in spec["panels"]:
            source_path = panel_output_path(
                paper_figures_dir=paper_figures_dir,
                selection_slug=selection.selection_slug,
                figure_id=figure_id,
            )
            reason = None
            if not source_path.exists():
                reason = "missing_source_panel"
                missing_panels.append(
                    {
                        "composite_id": composite_id,
                        "panel_label": label,
                        "figure_id": figure_id,
                        "expected_path": str(source_path),
                        "reason": reason,
                    }
                )
            panels.append(
                CompositePanel(
                    label=label,
                    figure_id=figure_id,
                    title=title,
                    path=source_path,
                    reason=reason,
                )
            )

        output_path = output_root / f"{composite_id}_composite.{args.format}"
        compose_composite_figure(
            title=spec["title"],
            subtitle=subtitle,
            panels=panels,
            output_path=output_path,
            dpi=args.dpi,
        )
        generated.append({"composite_id": composite_id, "path": str(output_path)})

    return {
        "output_root": str(output_root),
        "generated_composites": generated,
        "missing_panels": missing_panels,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    composite_ids = ALL_COMPOSITES if not args.composites else tuple(part.strip() for part in args.composites.split(",") if part.strip())
    invalid = [composite_id for composite_id in composite_ids if composite_id not in COMPOSITE_SPECS]
    if invalid:
        raise SystemExit(f"Unknown composite ids: {', '.join(invalid)}")

    selection = resolve_selection(args)
    single_panel_summary = ensure_single_panels(args=args, selection=selection, composite_ids=composite_ids)
    composition_summary = compose_requested_panels(args=args, selection=selection, composite_ids=composite_ids)

    manifest = {
        "selection_mode": selection.mode,
        "resolved_run": {
            "mlflow_run_id": selection.run_id,
            "mlflow_experiment_id": selection.experiment_id,
            "selection_slug": selection.selection_slug,
            "results_row_source_file": selection.selected_row.source_file if selection.selected_row is not None else None,
            "run_signature": selection.selected_row.run_signature if selection.selected_row is not None else args.run_signature,
        },
        "filters": {
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
        "requested_composites": list(composite_ids),
        "single_panel_generation": single_panel_summary,
        "generated_composites": composition_summary["generated_composites"],
        "missing_panels": composition_summary["missing_panels"],
    }

    manifest_path = Path(composition_summary["output_root"]) / "panel_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Selection mode: {selection.mode}")
    print(f"Resolved MLflow run id: {selection.run_id}")
    print(f"Selection slug: {selection.selection_slug}")
    print(f"Single panels regenerated: {single_panel_summary['regenerated']}")
    print(f"Generated composites: {len(composition_summary['generated_composites'])}")
    print(f"Missing source panels: {len(composition_summary['missing_panels'])}")
    print(f"Manifest: {manifest_path}")
    for entry in composition_summary["generated_composites"]:
        print(f"Saved: {entry['path']}")
    for entry in composition_summary["missing_panels"]:
        print(f"Placeholder: {entry['composite_id']} {entry['panel_label']} ({entry['figure_id']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
