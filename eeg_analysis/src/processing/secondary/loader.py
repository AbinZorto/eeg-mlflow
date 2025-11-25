import os
from pathlib import Path
from typing import Dict, Any

import mne
import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def load_eeglab_run(file_path: str) -> Dict[str, Any]:
    """
    Load an EEGLAB .set file (with associated .fdt) using MNE with preload=True.
    Keeps all channels without dropping or standardizing/scaling.
    
    Returns:
        dict with:
          - raw: mne.io.Raw (preloaded)
          - channel_names: list[str]
          - sampling_rate: float
          - original_sampling_rate: float
          - subject_id: str
          - run_name: str
          - file_path: str
    """
    file_path = str(file_path)
    fp = Path(file_path)
    if not fp.exists():
        raise FileNotFoundError(f"SET file not found: {file_path}")
    if fp.suffix.lower() != ".set":
        raise ValueError(f"Expected a .set file, got: {file_path}")

    logger.info(f"Loading EEGLAB file: {file_path}")
    raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)

    channel_names = list(raw.ch_names)
    sampling_rate = float(raw.info["sfreq"])
    original_sampling_rate = sampling_rate

    # Infer subject and run names from path conventions (e.g., sub-001/.../sub-001_task-..._run-..._eeg.set)
    subject_id = _infer_subject_id(fp)
    run_name = fp.stem

    # Validate data is preloaded and has expected shape
    data = raw.get_data()
    if not isinstance(data, np.ndarray):
        raise RuntimeError("Failed to load EEGLAB data into numpy array")

    logger.info(
        f"Loaded run: subject={subject_id}, run={run_name}, channels={len(channel_names)}, "
        f"sfreq={sampling_rate}, samples={data.shape[1]}"
    )

    return {
        "raw": raw,
        "channel_names": channel_names,
        "sampling_rate": sampling_rate,
        "original_sampling_rate": original_sampling_rate,
        "subject_id": subject_id,
        "run_name": run_name,
        "file_path": file_path,
    }


def _infer_subject_id(path: Path) -> str:
    """
    Infer subject ID from path components by picking the first component that matches 'sub-*'.
    """
    for part in path.parts:
        if part.startswith("sub-"):
            return part
    # Fallback: try to derive from filename
    stem = path.stem
    if "sub-" in stem:
        idx = stem.index("sub-")
        candidate = stem[idx:].split("_")[0]
        if candidate:
            return candidate
    return "unknown"


