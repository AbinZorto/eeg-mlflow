from typing import Dict, List, Any
from pathlib import Path
import time

import numpy as np
import pandas as pd
import mlflow
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Optional

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def save_npz_dataset(
    output_path: str,
    data: np.ndarray,
    subject_boundaries: Dict[str, List[int]],
    run_boundaries: Dict[str, List[int]],
    channel_names: List[str],
    sampling_rate: float,
    original_sampling_rates: Dict[str, Any],
    subject_ids: List[str],
    metadata_version: str = "v1",
) -> str:
    """
    Save the combined dataset as a single .npz with required metadata.
    
    Returns:
        Absolute path to the saved file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    abs_out = str(Path(output_path).resolve())

    # JAX-compatible numpy arrays are standard ndarrays
    logger.info(
        f"Saving .npz to {abs_out} with shape={data.shape}, sampling_rate={sampling_rate}"
    )
    np.savez_compressed(
        abs_out,
        data=data,
        subject_boundaries=subject_boundaries,
        run_boundaries=run_boundaries,
        channel_names=np.array(channel_names, dtype=object),
        sampling_rate=sampling_rate,
        original_sampling_rates=original_sampling_rates,
        subject_ids=np.array(subject_ids, dtype=object),
        metadata_version=metadata_version,
        created_ts=int(time.time()),
    )
    return abs_out


def build_run_windows_records(
    subject_id: str,
    run_name: str,
    data: np.ndarray,
    channel_names: List[str],
    window_length: int,
    step_size: int,
) -> List[Dict[str, Any]]:
    """
    Slice a single run (n_channels, n_samples) into fixed-length windows and
    return a list of row dicts suitable for a parquet table. Each channel
    column contains a 1D numpy array of length = window_length.
    """
    num_channels, total_samples = int(data.shape[0]), int(data.shape[1])
    records: List[Dict[str, Any]] = []
    sub_window_id = 0

    start = 0
    while start + window_length <= total_samples:
        end = start + window_length
        row: Dict[str, Any] = {
            "participant": subject_id,
            "run": run_name,
            "parent_window_id": 0,  # parent is the run itself
            "sub_window_id": sub_window_id,
            "window_start": start,
            "window_end": end,
        }
        for ch_idx in range(num_channels):
            ch_name = channel_names[ch_idx]
            row[ch_name] = data[ch_idx, start:end]
        records.append(row)
        sub_window_id += 1
        start += step_size

    return records


def save_parquet_table(output_path: str, df: pd.DataFrame) -> str:
    """
    Save a DataFrame to parquet at output_path and return absolute path.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    abs_out = str(out_path.resolve())
    logger.info(f"Saving parquet table with shape={df.shape} to {abs_out}")
    df.to_parquet(abs_out)
    return abs_out


def save_parquet_streaming(
    output_path: str,
    chunk_iter,
    channel_names: List[str],
    window_length: int,
) -> str:
    """
    Stream-write parquet by accepting an iterator of pandas DataFrames (chunks).
    Ensures channel columns are Arrow list arrays (each element is a vector for a window).
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    abs_out = str(out_path.resolve())

    writer = None
    try:
        for chunk_df in chunk_iter:
            if chunk_df is None or len(chunk_df) == 0:
                continue
            # Build a consistent Arrow schema
            fields = [
                pa.field("participant", pa.string()),
                pa.field("run", pa.string()),
                pa.field("parent_window_id", pa.int64()),
                pa.field("sub_window_id", pa.int64()),
                pa.field("window_start", pa.int64()),
                pa.field("window_end", pa.int64()),
            ]
            list_float_type = pa.list_(pa.float32())
            for ch in channel_names:
                fields.append(pa.field(ch, list_float_type))
            schema = pa.schema(fields)

            # Coerce chunk columns to match schema strictly
            arrays: Dict[str, pa.Array] = {}
            # Meta columns
            arrays["participant"] = pa.array(chunk_df["participant"].astype(str).tolist(), type=pa.string())
            arrays["run"] = pa.array(chunk_df["run"].astype(str).tolist(), type=pa.string())
            arrays["parent_window_id"] = pa.array(chunk_df["parent_window_id"].astype("int64").tolist(), type=pa.int64())
            arrays["sub_window_id"] = pa.array(chunk_df["sub_window_id"].astype("int64").tolist(), type=pa.int64())
            arrays["window_start"] = pa.array(chunk_df["window_start"].astype("int64").tolist(), type=pa.int64())
            arrays["window_end"] = pa.array(chunk_df["window_end"].astype("int64").tolist(), type=pa.int64())

            # Channel columns: ensure list<float32> of fixed length (pad/truncate with NaN)
            def _coerce_vector(v):
                if v is None:
                    return [float("nan")] * window_length
                arr = np.asarray(v, dtype=np.float32)
                if arr.ndim != 1:
                    arr = arr.ravel()
                if arr.shape[0] < window_length:
                    pad = np.full(window_length - arr.shape[0], np.nan, dtype=np.float32)
                    arr = np.concatenate([arr, pad], axis=0)
                elif arr.shape[0] > window_length:
                    arr = arr[:window_length]
                return arr.tolist()

            for ch in channel_names:
                if ch in chunk_df.columns:
                    values = chunk_df[ch].tolist()
                else:
                    values = [None] * len(chunk_df)
                coerced = [_coerce_vector(v) for v in values]
                arrays[ch] = pa.array(coerced, type=list_float_type, from_pandas=True)

            table = pa.Table.from_pydict(arrays, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(abs_out, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    logger.info(f"Streaming parquet saved to {abs_out}")
    return abs_out


def save_parquet_run(
    output_dir: str,
    subject_id: str,
    run_name: str,
    df: pd.DataFrame,
    channel_names: List[str],
    window_length: int,
) -> str:
    """
    Save a single run's windowed DataFrame to parquet at:
        output_dir/subject_id/run_name.parquet
    Enforces schema:
      - participant, run: string
      - parent_window_id, sub_window_id, window_start, window_end: int64
      - each channel: list<float32> (length window_length, padded/truncated as needed)
    """
    run_dir = Path(output_dir) / subject_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = str((run_dir / f"{run_name}.parquet").resolve())

    # Build Arrow schema
    fields = [
        pa.field("participant", pa.string()),
        pa.field("run", pa.string()),
        pa.field("parent_window_id", pa.int64()),
        pa.field("sub_window_id", pa.int64()),
        pa.field("window_start", pa.int64()),
        pa.field("window_end", pa.int64()),
    ]
    list_float_type = pa.list_(pa.float32())
    for ch in channel_names:
        fields.append(pa.field(ch, list_float_type))
    schema = pa.schema(fields)

    # Ensure column order and presence
    meta_cols = ["participant", "run", "parent_window_id", "sub_window_id", "window_start", "window_end"]
    for ch in channel_names:
        if ch not in df.columns:
            df[ch] = None
    df = df[meta_cols + channel_names]

    # Coerce to Arrow arrays
    arrays: Dict[str, pa.Array] = {}
    arrays["participant"] = pa.array(df["participant"].astype(str).tolist(), type=pa.string())
    arrays["run"] = pa.array(df["run"].astype(str).tolist(), type=pa.string())
    arrays["parent_window_id"] = pa.array(df["parent_window_id"].astype("int64").tolist(), type=pa.int64())
    arrays["sub_window_id"] = pa.array(df["sub_window_id"].astype("int64").tolist(), type=pa.int64())
    arrays["window_start"] = pa.array(df["window_start"].astype("int64").tolist(), type=pa.int64())
    arrays["window_end"] = pa.array(df["window_end"].astype("int64").tolist(), type=pa.int64())

    def _coerce_vector(v):
        if v is None:
            return [float("nan")] * window_length
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim != 1:
            arr = arr.ravel()
        if arr.shape[0] < window_length:
            pad = np.full(window_length - arr.shape[0], np.nan, dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] > window_length:
            arr = arr[:window_length]
        return arr.tolist()

    for ch in channel_names:
        coerced = [_coerce_vector(v) for v in df[ch].tolist()]
        arrays[ch] = pa.array(coerced, type=list_float_type, from_pandas=True)

    table = pa.Table.from_pydict(arrays, schema=schema)
    pq.write_table(table, out_path, compression="snappy")
    logger.info(f"Saved run parquet: subject={subject_id}, run={run_name}, rows={len(df)}, path={out_path}")
    return out_path


def log_parquet_as_mlflow_dataset(
    df: pd.DataFrame,
    source_path: str,
    dataset_name: str,
    context: str = "pretraining",
) -> None:
    """
    Register the parquet as an MLflow dataset input for the active run.
    """
    try:
        dataset = mlflow.data.from_pandas(
            df=df,
            source=source_path,
            name=dataset_name,
        )
        mlflow.log_input(dataset, context=context)
        mlflow.set_tag("mlflow.dataset.logged", "true")
        mlflow.set_tag("mlflow.dataset.context", context)
        logger.info(f"Logged MLflow dataset: {dataset_name}")
    except Exception as e:
        logger.warning(f"Failed to log MLflow dataset '{dataset_name}': {e}")


def save_npz_run(
    output_dir: str,
    subject_id: str,
    run_name: str,
    data: np.ndarray,
    channel_names: List[str],
    sampling_rate: float,
    original_sampling_rate: float,
    units: str = "uV",
    metadata_version: str = "v1",
) -> str:
    """
    Save a single run as .npz in output_dir/subject_id/run_name.npz.
    
    Returns:
        Absolute path to the saved .npz file.
    """
    run_dir = Path(output_dir) / subject_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"{run_name}.npz"
    abs_out = str(out_path.resolve())
    logger.info(
        f"Saving run .npz: subject={subject_id}, run={run_name}, path={abs_out}, "
        f"shape={data.shape}, sfreq={sampling_rate}, units={units}"
    )
    np.savez_compressed(
        abs_out,
        data=data,
        channel_names=np.array(channel_names, dtype=object),
        sampling_rate=sampling_rate,
        original_sampling_rate=original_sampling_rate,
        subject_id=subject_id,
        run_name=run_name,
        units=units,
        metadata_version=metadata_version,
        created_ts=int(time.time()),
    )
    return abs_out


