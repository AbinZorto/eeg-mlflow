from typing import Dict, List, Any
from pathlib import Path
import time

import numpy as np

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


