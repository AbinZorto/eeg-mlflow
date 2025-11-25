from typing import Tuple

import mne
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def resample_raw_to_target(raw: mne.io.BaseRaw, target_sfreq: float) -> mne.io.BaseRaw:
    """
    Resample the raw recording to the target sampling frequency using Raw.resample.
    """
    original_sfreq = float(raw.info["sfreq"])
    if abs(original_sfreq - target_sfreq) < 1e-9:
        logger.info(f"Resampling skipped; already at target {target_sfreq} Hz")
        return raw
    logger.info(f"Resampling: {original_sfreq} Hz → {target_sfreq} Hz")
    # Work on a copy to avoid side effects on caller references
    resampled = raw.copy()
    resampled.resample(target_sfreq)
    return resampled


