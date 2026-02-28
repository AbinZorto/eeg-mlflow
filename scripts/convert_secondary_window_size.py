#!/usr/bin/env python3
"""
Convert secondary EEG per-run parquet windows to a larger window size by concatenating
fixed-size, non-overlapping windows. This assumes overlap_seconds = 0 in the source data.

Example:
  uv run python3 scripts/convert_secondary_window_size.py \
    --input-root eeg_analysis/secondarydata/raw/sr256_ws4s \
    --output-base eeg_analysis/secondarydata/raw \
    --factor 2
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Resolve project root for consistency with other scripts.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


META_COLS = {
    "participant",
    "run",
    "parent_window_id",
    "sub_window_id",
    "window_start",
    "window_end",
}


def _format_rate(rate: float) -> str:
    if float(rate).is_integer():
        return str(int(rate))
    return str(rate).rstrip("0").rstrip(".")


def _format_window(seconds: float) -> str:
    if float(seconds).is_integer():
        return str(int(seconds))
    return str(seconds).rstrip("0").rstrip(".")


def _parse_window_tag(path: Path) -> Tuple[float, float]:
    match = re.search(r"sr(?P<sr>[0-9.]+)_ws(?P<ws>[0-9.]+)s", str(path))
    if not match:
        raise ValueError(f"Could not infer sampling rate/window seconds from path: {path}")
    sr = float(match.group("sr"))
    ws = float(match.group("ws"))
    return sr, ws


def _infer_schema_columns(schema: pa.Schema) -> Tuple[List[str], List[str]]:
    meta_cols: List[str] = []
    channel_cols: List[str] = []
    for field in schema:
        if pa.types.is_list(field.type) or pa.types.is_large_list(field.type) or pa.types.is_fixed_size_list(field.type):
            channel_cols.append(field.name)
        else:
            meta_cols.append(field.name)
    # Ensure meta columns include expected fields
    if not META_COLS.issubset(set(meta_cols)):
        missing = META_COLS.difference(meta_cols)
        raise ValueError(f"Missing expected meta columns: {sorted(missing)}")
    return meta_cols, channel_cols


def _iter_rows(pf: pq.ParquetFile, columns: List[str]) -> Iterable[Dict[str, object]]:
    for batch in pf.iter_batches(columns=columns):
        cols = {name: batch.column(i).to_pylist() for i, name in enumerate(batch.schema.names)}
        num_rows = batch.num_rows
        for idx in range(num_rows):
            yield {name: cols[name][idx] for name in columns}


def _concat_windows(
    rows: List[Dict[str, object]],
    channel_cols: List[str],
    new_sub_id: int,
    expected_len: int,
    pad_value: float | None,
) -> Dict[str, object]:
    first = rows[0]
    last = rows[-1]
    out: Dict[str, object] = {}
    # Meta columns
    for col in first:
        if col in channel_cols:
            continue
        if col == "sub_window_id":
            out[col] = int(new_sub_id)
        elif col == "window_start":
            out[col] = int(first[col])
        elif col == "window_end":
            out[col] = int(last[col])
        else:
            out[col] = first[col]

    # Channel columns: concatenate arrays
    for ch in channel_cols:
        parts = []
        for row in rows:
            arr = np.asarray(row[ch], dtype=np.float32).ravel()
            parts.append(arr)
        merged = np.concatenate(parts, axis=0)
        if merged.shape[0] != expected_len:
            if pad_value is None:
                raise ValueError(
                    f"Unexpected merged length for {ch}: got {merged.shape[0]}, expected {expected_len}"
                )
            pad_len = expected_len - merged.shape[0]
            if pad_len < 0:
                merged = merged[:expected_len]
            else:
                pad = np.full(pad_len, pad_value, dtype=np.float32)
                merged = np.concatenate([merged, pad], axis=0)
        out[ch] = merged
    return out


def convert_run(
    input_path: Path,
    output_path: Path,
    factor: int,
    overwrite: bool,
    pad_value: float | None,
) -> Tuple[int, int]:
    if output_path.exists() and not overwrite:
        print(f"Skip existing: {output_path}")
        return 0, 0

    pf = pq.ParquetFile(str(input_path))
    meta_cols, channel_cols = _infer_schema_columns(pf.schema_arrow)
    columns = meta_cols + channel_cols

    rows_buffer: List[Dict[str, object]] = []
    out_rows: List[Dict[str, object]] = []
    new_sub_id = 0

    # Infer window length from first row
    first_batch = next(pf.iter_batches(columns=[channel_cols[0]]), None)
    if first_batch is None or first_batch.num_rows == 0:
        return 0, 0
    first_vals = first_batch.column(0).to_pylist()
    old_len = len(first_vals[0])
    new_len = old_len * factor

    for row in _iter_rows(pf, columns):
        rows_buffer.append(row)
        if len(rows_buffer) == factor:
            out_rows.append(_concat_windows(rows_buffer, channel_cols, new_sub_id, new_len, pad_value))
            new_sub_id += 1
            rows_buffer = []
        if len(out_rows) >= 1024:
            df = pa.Table.from_pylist(out_rows)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(df, str(output_path), compression="snappy", append=True)
            out_rows = []

    dropped = 0
    if rows_buffer:
        if pad_value is None:
            dropped = len(rows_buffer)
        else:
            out_rows.append(_concat_windows(rows_buffer, channel_cols, new_sub_id, new_len, pad_value))
            new_sub_id += 1

    if out_rows:
        df = pa.Table.from_pylist(out_rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(df, str(output_path), compression="snappy", append=True)

    return new_sub_id, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert secondary EEG windows to a larger window size.")
    parser.add_argument("--input-root", type=str, required=True, help="Input windowed dataset root (sr*_ws*s)")
    parser.add_argument("--output-base", type=str, required=True, help="Base output directory (window tag appended)")
    parser.add_argument("--factor", type=int, required=True, help="Multiplicative factor for window size (e.g., 2)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output parquet files")
    parser.add_argument(
        "--pad-remainder",
        type=str,
        default=None,
        help="Pad leftover windows instead of dropping them. Use 'nan' or a numeric value.",
    )
    args = parser.parse_args()

    pad_value: float | None = None
    if args.pad_remainder is not None:
        if args.pad_remainder.lower() == "nan":
            pad_value = float("nan")
        else:
            pad_value = float(args.pad_remainder)

    if args.factor < 2:
        raise ValueError("--factor must be >= 2")

    input_root = Path(args.input_root)
    output_base = Path(args.output_base)
    sr, ws = _parse_window_tag(input_root)
    new_ws = ws * args.factor
    output_root = output_base / f"sr{_format_rate(sr)}_ws{_format_window(new_ws)}s"

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    total_runs = 0
    total_written = 0
    total_dropped = 0

    for subject_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        for run_file in sorted(subject_dir.glob("*.parquet")):
            out_path = output_root / subject_dir.name / run_file.name
            print(f"Converting: {run_file} -> {out_path}")
            written, dropped = convert_run(run_file, out_path, args.factor, args.overwrite, pad_value)
            total_runs += 1
            total_written += written
            total_dropped += dropped

    print(f"Input root:  {input_root}")
    print(f"Output root: {output_root}")
    print(f"Factor:      {args.factor}")
    print(f"Runs:        {total_runs}")
    print(f"Windows out: {total_written}")
    print(f"Dropped in-group windows: {total_dropped}")


if __name__ == "__main__":
    main()
