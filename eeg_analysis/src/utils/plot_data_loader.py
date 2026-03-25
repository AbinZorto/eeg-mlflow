from __future__ import annotations

import ast
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.utils.clinical_metrics import summarize_feature_selection


@dataclass(frozen=True)
class SweepResultRow:
    source_file: str
    created_at: str
    created_at_sort: str
    attempt_index: int
    run_signature: Optional[str]
    job_key: Optional[str]
    mlflow_run_id: str
    mlflow_experiment_id: Optional[str]
    model: Optional[str]
    fs_enabled: bool
    fs_method: Optional[str]
    n_features: Optional[int]
    ordering: Optional[str]
    dataset_run_id: Optional[str]
    window_seconds: Optional[int]
    inner_k: Optional[int]
    outer_k: Optional[int]
    equalize_lopo_groups: Optional[str]
    use_smote: Optional[str]
    patient_roc_auc: Optional[float]


@dataclass(frozen=True)
class NestedFoldRun:
    run_id: str
    fold_idx: Optional[int]
    selected_features: List[str]
    fold_target_class: str
    patient_predictions: List[Dict[str, Any]]
    window_predictions: List[Dict[str, Any]]


@dataclass(frozen=True)
class RunArtifactBundle:
    run_id: str
    experiment_id: Optional[str]
    run_dir: Path
    artifacts_dir: Path
    clinical_metrics: Dict[str, Any]
    feature_selection_summary: Dict[str, Any]
    patient_predictions: List[Dict[str, Any]]
    window_predictions: List[Dict[str, Any]]
    nested_fold_runs: List[NestedFoldRun]
    jaccard_fold_ids: List[int]
    jaccard_matrix: List[List[float]]
    jaccard_values: List[float]


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


def _normalize_bool_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "false"}:
        return normalized
    return None


def _created_at_sort_key(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        return datetime.fromisoformat(text).isoformat()
    except ValueError:
        return text


def _extract_patient_roc_auc(record: Dict[str, Any]) -> Optional[float]:
    mlflow_metrics = record.get("mlflow_metrics")
    if isinstance(mlflow_metrics, dict):
        value = _safe_float(mlflow_metrics.get("patient_roc_auc"))
        if value is not None:
            return value

    metrics = record.get("metrics")
    if isinstance(metrics, dict):
        patient = metrics.get("patient")
        if isinstance(patient, dict):
            value = _safe_float(patient.get("roc_auc"))
            if value is not None:
                return value
    return None


def results_record_to_row(record: Dict[str, Any], source_file: Path) -> Optional[SweepResultRow]:
    if record.get("record_type") != "job_attempt":
        return None
    if record.get("status") != "success":
        return None
    run_id = record.get("mlflow_run_id")
    if not run_id:
        return None

    return SweepResultRow(
        source_file=str(source_file),
        created_at=str(record.get("created_at") or ""),
        created_at_sort=_created_at_sort_key(record.get("created_at")),
        attempt_index=_safe_int(record.get("attempt_index")) or 0,
        run_signature=str(record.get("run_signature")) if record.get("run_signature") is not None else None,
        job_key=str(record.get("job_key")) if record.get("job_key") is not None else None,
        mlflow_run_id=str(run_id),
        mlflow_experiment_id=str(record.get("mlflow_experiment_id")) if record.get("mlflow_experiment_id") is not None else None,
        model=str(record.get("model")) if record.get("model") is not None else None,
        fs_enabled=bool(record.get("fs_enabled")),
        fs_method=str(record.get("fs_method")) if record.get("fs_method") is not None else None,
        n_features=_safe_int(record.get("n_features")),
        ordering=str(record.get("ordering")) if record.get("ordering") is not None else None,
        dataset_run_id=str(record.get("dataset_run_id")) if record.get("dataset_run_id") is not None else None,
        window_seconds=_safe_int(record.get("window_seconds")),
        inner_k=_safe_int(record.get("inner_k")),
        outer_k=_safe_int(record.get("outer_k")),
        equalize_lopo_groups=_normalize_bool_text(record.get("equalize_lopo_groups")),
        use_smote=_normalize_bool_text(record.get("use_smote")),
        patient_roc_auc=_extract_patient_roc_auc(record),
    )


def discover_results_files(artifacts_dir: Path, pattern: str = "*.results.jsonl") -> List[Path]:
    return sorted(path for path in artifacts_dir.glob(pattern) if path.is_file())


def load_results_rows(paths: Sequence[Path]) -> List[SweepResultRow]:
    rows: List[SweepResultRow] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row = results_record_to_row(record, path)
                if row is not None:
                    rows.append(row)
    return rows


def _dedupe_key(row: SweepResultRow) -> Tuple[Any, ...]:
    return (
        row.model,
        row.window_seconds,
        row.inner_k,
        row.outer_k,
        row.fs_enabled,
        row.fs_method,
        row.n_features,
        row.ordering,
        row.equalize_lopo_groups,
        row.use_smote,
        row.dataset_run_id,
    )


def deduplicate_results_rows(rows: Sequence[SweepResultRow]) -> List[SweepResultRow]:
    deduped: Dict[Tuple[Any, ...], SweepResultRow] = {}
    for row in rows:
        key = _dedupe_key(row)
        current = deduped.get(key)
        if current is None or (row.attempt_index, row.created_at_sort) > (
            current.attempt_index,
            current.created_at_sort,
        ):
            deduped[key] = row
    return sorted(
        deduped.values(),
        key=lambda row: (
            row.model or "",
            row.window_seconds if row.window_seconds is not None else -1,
            row.inner_k if row.inner_k is not None else -1,
            row.outer_k if row.outer_k is not None else -1,
            row.equalize_lopo_groups or "",
            row.use_smote or "",
            row.fs_method or "",
            row.n_features if row.n_features is not None else -1,
            row.ordering or "",
            row.dataset_run_id or "",
        ),
    )


def filter_result_rows(
    rows: Sequence[SweepResultRow],
    *,
    run_signature: Optional[str] = None,
    mlflow_run_id: Optional[str] = None,
    model: Optional[str] = None,
    window_size: Optional[int] = None,
    inner_k: Optional[int] = None,
    outer_k: Optional[int] = None,
    fs_method: Optional[str] = None,
    n_features: Optional[int] = None,
    ordering: Optional[str] = None,
    equalize_lopo_groups: Optional[str] = None,
    use_smote: Optional[str] = None,
) -> List[SweepResultRow]:
    filtered: List[SweepResultRow] = []
    for row in rows:
        if run_signature is not None and row.run_signature != run_signature:
            continue
        if mlflow_run_id is not None and row.mlflow_run_id != mlflow_run_id:
            continue
        if model is not None and row.model != model:
            continue
        if window_size is not None and row.window_seconds != window_size:
            continue
        if inner_k is not None and row.inner_k != inner_k:
            continue
        if outer_k is not None and row.outer_k != outer_k:
            continue
        if fs_method is not None and row.fs_method != fs_method:
            continue
        if n_features is not None and row.n_features != n_features:
            continue
        if ordering is not None and row.ordering != ordering:
            continue
        if equalize_lopo_groups is not None and row.equalize_lopo_groups != equalize_lopo_groups:
            continue
        if use_smote is not None and row.use_smote != use_smote:
            continue
        filtered.append(row)
    return filtered


def select_latest_result_row(rows: Sequence[SweepResultRow]) -> Optional[SweepResultRow]:
    if not rows:
        return None
    return max(rows, key=lambda row: (row.created_at_sort, row.attempt_index, row.mlflow_run_id))


def resolve_run_dir(mlruns_root: Path, run_id: str, experiment_id: Optional[str] = None) -> Path:
    if experiment_id is not None:
        candidate = mlruns_root / str(experiment_id) / run_id
        if candidate.exists():
            return candidate
    for candidate in mlruns_root.glob(f"*/{run_id}"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not resolve MLflow run directory for run_id={run_id}")


def load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_csv_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized: Dict[str, Any] = {}
            for key, value in row.items():
                if value is None:
                    normalized[key] = None
                    continue
                stripped = value.strip()
                if stripped == "":
                    normalized[key] = ""
                    continue
                if stripped in {"True", "False"}:
                    normalized[key] = stripped == "True"
                    continue
                maybe_int = _safe_int(stripped)
                if maybe_int is not None and str(maybe_int) == stripped:
                    normalized[key] = maybe_int
                    continue
                maybe_float = _safe_float(stripped)
                if maybe_float is not None and any(ch in stripped for ch in ".eE"):
                    normalized[key] = maybe_float
                    continue
                normalized[key] = stripped
            rows.append(normalized)
    return rows


def _find_first_artifact(artifacts_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(artifacts_dir.glob(pattern))
    return matches[0] if matches else None


def load_top_level_prediction_rows(artifacts_dir: Path, kind: str) -> List[Dict[str, Any]]:
    path = _find_first_artifact(artifacts_dir, f"*_{kind}_predictions.csv")
    if path is None:
        fallback = artifacts_dir / f"{kind}_predictions.csv"
        path = fallback if fallback.exists() else None
    if path is None:
        return []
    return load_csv_rows(path)


def parse_selected_features_list(raw_text: str) -> List[str]:
    text = raw_text.strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        parsed = None

    if isinstance(parsed, (list, tuple, set)):
        return [str(item) for item in parsed if item is not None]

    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]

    return [text]


def infer_fold_target_class(patient_predictions: Sequence[Dict[str, Any]]) -> str:
    labels = sorted({row.get("true_label") for row in patient_predictions if row.get("true_label") is not None})
    labels = [label for label in labels if label in {0, 1}]
    if not labels:
        return "unknown"
    if labels == [1]:
        return "remission"
    if labels == [0]:
        return "non_remission"
    return "mixed"


def discover_nested_fold_runs(mlruns_root: Path, experiment_id: Optional[str], parent_run_id: str) -> List[NestedFoldRun]:
    if experiment_id is None:
        return []

    experiment_dir = mlruns_root / str(experiment_id)
    if not experiment_dir.exists():
        return []

    runs: List[NestedFoldRun] = []
    for run_dir in experiment_dir.iterdir():
        if not run_dir.is_dir() or run_dir.name == parent_run_id:
            continue
        parent_tag = run_dir / "tags" / "mlflow.parentRunId"
        if not parent_tag.exists() or parent_tag.read_text().strip() != parent_run_id:
            continue

        fold_idx_path = run_dir / "params" / "fold_index"
        selected_features_path = run_dir / "params" / "selected_features_list"
        patient_predictions_path = run_dir / "artifacts" / "fold_patient_predictions.json"
        window_predictions_path = run_dir / "artifacts" / "fold_window_predictions.json"

        patient_predictions = []
        if patient_predictions_path.exists():
            try:
                patient_predictions = json.loads(patient_predictions_path.read_text())
            except Exception:
                patient_predictions = []

        window_predictions = []
        if window_predictions_path.exists():
            try:
                window_predictions = json.loads(window_predictions_path.read_text())
            except Exception:
                window_predictions = []

        selected_features = []
        if selected_features_path.exists():
            selected_features = parse_selected_features_list(selected_features_path.read_text())

        runs.append(
            NestedFoldRun(
                run_id=run_dir.name,
                fold_idx=_safe_int(fold_idx_path.read_text().strip()) if fold_idx_path.exists() else None,
                selected_features=selected_features,
                fold_target_class=infer_fold_target_class(patient_predictions),
                patient_predictions=patient_predictions,
                window_predictions=window_predictions,
            )
        )

    return sorted(runs, key=lambda run: (run.fold_idx if run.fold_idx is not None else -1, run.run_id))


def build_feature_selection_rows(nested_runs: Sequence[NestedFoldRun]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in nested_runs:
        if not run.selected_features:
            continue
        rows.append(
            {
                "fold_idx": run.fold_idx,
                "fold_target_class": run.fold_target_class,
                "features": run.selected_features,
            }
        )
    return rows


def merge_feature_selection_summary(
    top_level_summary: Optional[Dict[str, Any]],
    nested_runs: Sequence[NestedFoldRun],
) -> Dict[str, Any]:
    feature_selection_rows = build_feature_selection_rows(nested_runs)
    reconstructed = summarize_feature_selection(feature_selection_rows=feature_selection_rows) if feature_selection_rows else {}
    if not top_level_summary:
        return reconstructed

    merged = dict(top_level_summary)
    for key in (
        "feature_set_count",
        "average_features_per_fold",
        "mean_pairwise_jaccard",
        "median_pairwise_jaccard",
        "unique_feature_count",
        "top_feature",
        "top_feature_frequency",
        "top_feature_share",
        "selection_frequency_by_feature",
        "selected_features_per_fold",
        "fixed_k_detected",
        "fold_class_counts",
        "class_conditional_selection_available",
        "class_conditional_selection_reason",
    ):
        if key in reconstructed:
            merged[key] = reconstructed[key]
    return merged


def compute_pairwise_jaccard(nested_runs: Sequence[NestedFoldRun]) -> Tuple[List[int], List[List[float]], List[float]]:
    usable_runs = [run for run in nested_runs if run.selected_features]
    fold_ids = [run.fold_idx if run.fold_idx is not None else idx for idx, run in enumerate(usable_runs)]
    feature_sets = [set(run.selected_features) for run in usable_runs]

    matrix: List[List[float]] = []
    values: List[float] = []
    for i, left in enumerate(feature_sets):
        row: List[float] = []
        for j, right in enumerate(feature_sets):
            if i == j:
                score = 1.0
            else:
                union = len(left | right)
                score = float(len(left & right) / union) if union else 0.0
                if j > i:
                    values.append(score)
            row.append(score)
        matrix.append(row)
    return fold_ids, matrix, values


def _aggregate_nested_predictions(nested_runs: Sequence[NestedFoldRun], kind: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in nested_runs:
        if kind == "patient":
            rows.extend(run.patient_predictions)
        else:
            rows.extend(run.window_predictions)
    return rows


def load_run_artifact_bundle(
    *,
    mlruns_root: Path,
    run_id: str,
    experiment_id: Optional[str] = None,
) -> RunArtifactBundle:
    run_dir = resolve_run_dir(mlruns_root, run_id, experiment_id)
    artifacts_dir = run_dir / "artifacts"
    clinical_metrics = load_json_if_exists(artifacts_dir / "clinical_metrics_summary.json") or {}
    top_level_feature_selection = (
        load_json_if_exists(artifacts_dir / "feature_selection_stability.json")
        or clinical_metrics.get("feature_selection")
        or {}
    )

    patient_predictions = load_top_level_prediction_rows(artifacts_dir, "patient")
    window_predictions = load_top_level_prediction_rows(artifacts_dir, "window")

    nested_runs = discover_nested_fold_runs(mlruns_root, experiment_id, run_id)
    if not patient_predictions:
        patient_predictions = _aggregate_nested_predictions(nested_runs, "patient")
    if not window_predictions:
        window_predictions = _aggregate_nested_predictions(nested_runs, "window")

    feature_selection_summary = merge_feature_selection_summary(top_level_feature_selection, nested_runs)
    jaccard_fold_ids, jaccard_matrix, jaccard_values = compute_pairwise_jaccard(nested_runs)

    return RunArtifactBundle(
        run_id=run_id,
        experiment_id=experiment_id,
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        clinical_metrics=clinical_metrics,
        feature_selection_summary=feature_selection_summary,
        patient_predictions=patient_predictions,
        window_predictions=window_predictions,
        nested_fold_runs=nested_runs,
        jaccard_fold_ids=jaccard_fold_ids,
        jaccard_matrix=jaccard_matrix,
        jaccard_values=jaccard_values,
    )
