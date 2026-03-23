import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import mlflow

from ..utils.logger import get_logger

logger = get_logger(__name__)


class EEGParticipantDemeaner:
    """Demean each EEG channel per participant before signal resampling."""

    def __init__(self, config: Dict[str, Any]):
        demean_cfg = config.get("participant_demean", {})
        self.enabled = bool(demean_cfg.get("enabled", True))
        self.save_interim = bool(demean_cfg.get("save_interim", True))
        self.channels = config.get("data_loader", {}).get("channels", [])
        self.participant_col = "participant_id"
        self.logger = get_logger(__name__)

    def _compute_participant_channel_mean(self, df: pd.DataFrame, participant_id: Any, channel: str) -> float:
        """Compute mean value for one participant and one channel across all windows/samples."""
        participant_rows = df[df[self.participant_col] == participant_id]
        signals = []
        for signal in participant_rows[channel].values:
            arr = np.asarray(signal, dtype=float)
            if arr.size:
                signals.append(arr)

        if not signals:
            return 0.0

        concatenated = np.concatenate(signals)
        mean_val = np.nanmean(concatenated)
        if np.isnan(mean_val):
            return 0.0
        return float(mean_val)

    def _build_mean_lookup(self, df: pd.DataFrame) -> Dict[Tuple[Any, str], float]:
        """Create lookup table for participant-channel means."""
        mean_lookup: Dict[Tuple[Any, str], float] = {}
        participants = df[self.participant_col].dropna().unique()
        for participant_id in participants:
            for channel in self.channels:
                mean_lookup[(participant_id, channel)] = self._compute_participant_channel_mean(
                    df, participant_id, channel
                )
        return mean_lookup

    def _demean_row(self, row: Dict[str, Any], mean_lookup: Dict[Tuple[Any, str], float]) -> Dict[str, Any]:
        participant_id = row.get(self.participant_col)
        processed = {}
        for key, value in row.items():
            if key not in self.channels:
                processed[key] = value
                continue

            arr = np.asarray(value, dtype=float)
            offset = mean_lookup.get((participant_id, key), 0.0)
            processed[key] = arr - offset
        return processed

    def demean(self, df: pd.DataFrame, output_path: Optional[str] = None) -> pd.DataFrame:
        """Apply participant-level channel demeaning to the input dataset."""
        if not self.enabled:
            self.logger.info("Participant/channel demeaning disabled by configuration; skipping step.")
            return df

        if self.participant_col not in df.columns:
            raise ValueError(
                f"Column '{self.participant_col}' is required for participant-level demeaning but was not found."
            )

        self.logger.info(
            "Starting participant/channel demeaning on %d rows across %d participants",
            len(df),
            df[self.participant_col].nunique(),
        )

        mean_lookup = self._build_mean_lookup(df)
        processed_rows = []
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                self.logger.info("Demeaning row %d/%d", idx + 1, len(df))
            processed_rows.append(self._demean_row(row.to_dict(), mean_lookup))

        result_df = pd.DataFrame(processed_rows)

        if mlflow.active_run() is not None:
            mlflow.log_param("participant_demean_enabled", True)
            mlflow.log_metric("participant_demean_rows_processed", len(result_df))
            mlflow.log_metric("participant_demean_participants", result_df[self.participant_col].nunique())

        if output_path and self.save_interim:
            out_dir = Path(output_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            if "group" in result_df.columns and result_df["group"].notna().any():
                for group in result_df["group"].dropna().unique():
                    group_df = result_df[result_df["group"] == group]
                    group_fp = out_dir / f"{str(group).lower()}.parquet"
                    group_df.to_parquet(group_fp)
                    self.logger.info("Saved demeaned %s data to %s", group, group_fp)
            else:
                fp = out_dir / "demeaned.parquet"
                result_df.to_parquet(fp)
                self.logger.info("Saved demeaned data to %s", fp)

        self.logger.info("Participant/channel demeaning completed.")
        return result_df


def demean_eeg_data(config: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper for participant-level channel demeaning."""
    demeaner = EEGParticipantDemeaner(config)
    output_path = None
    if demeaner.save_interim:
        output_path = config.get("paths", {}).get("interim", {}).get("demeaned")
    return demeaner.demean(df, output_path=output_path)
