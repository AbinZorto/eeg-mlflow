import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from ..utils.logger import get_logger
import mlflow
import time

logger = get_logger(__name__)


class EEGDCOffsetRemover:
    """Remove DC offset from EEG windows per channel."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DC offset remover with configuration.
        Expects the following config structure:
          config['dc_offset_removal'] = {
              'enabled': bool,
              'method': 'mean' | 'median',
              'per_channel': bool,
              'save_interim': bool
          }
        """
        self.config = config
        dc_cfg = config.get('dc_offset_removal', {})
        self.enabled = bool(dc_cfg.get('enabled', True))
        self.method = dc_cfg.get('method', 'mean')
        self.per_channel = bool(dc_cfg.get('per_channel', True))
        self.save_interim = bool(dc_cfg.get('save_interim', True))
        self.channels = config.get('data_loader', {}).get('channels', [])
        self.logger = get_logger(__name__)

        valid_methods = {'mean', 'median'}
        if self.method not in valid_methods:
            raise ValueError(f"dc_offset_removal.method must be one of {valid_methods}")

    def _center_signal(self, signal: np.ndarray) -> np.ndarray:
        """Center a 1D signal by subtracting mean/median (ignoring NaNs).
        Additionally, replace exact zeros with the mean of non-zero values before centering.
        """
        if not isinstance(signal, np.ndarray):
            signal = np.asarray(signal)
        if signal.size == 0:
            return signal
        # Replace zeros with mean of non-zero values (ignoring NaNs)
        zeros_mask = (signal == 0) & ~np.isnan(signal)
        nonzero_vals = signal[(signal != 0) & ~np.isnan(signal)]
        if nonzero_vals.size > 0:
            nonzero_mean = float(np.mean(nonzero_vals))
            filled = np.where(zeros_mask, nonzero_mean, signal)
        else:
            # All values are zero or NaN; leave as-is
            filled = signal
        # Compute center on the filled signal
        if self.method == 'mean':
            center = np.nanmean(filled)
        else:
            center = np.nanmedian(filled)
        # Subtract center; preserve dtype semantics via numpy broadcasting
        return filled - center

    def process_window(self, window_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove DC offset for all configured channels in a single window dict.
        Non-channel keys are copied unchanged.
        """
        if not self.enabled:
            # No changes; return as-is
            return dict(window_data)

        processed = {}
        # Copy metadata and other non-channel keys
        for key, value in window_data.items():
            if key not in self.channels:
                processed[key] = value

        # Center each channel independently if present
        for ch in self.channels:
            sig = window_data.get(ch)
            if sig is None:
                continue
            processed[ch] = self._center_signal(np.asarray(sig))

        return processed

    def remove_dc(self, df: pd.DataFrame, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Apply DC offset removal to all windows in a DataFrame.
        """
        try:
            start = time.time()
            if not self.enabled:
                self.logger.info("DC offset removal disabled by configuration; skipping step.")
                return df

            self.logger.info(f"Starting DC offset removal (method={self.method}) on {len(df)} windows")
            processed_rows = []

            for idx, row in df.iterrows():
                if idx % 100 == 0:
                    self.logger.info(f"DC removal: processing window {idx+1}/{len(df)}")
                processed_rows.append(self.process_window(row.to_dict()))

            result_df = pd.DataFrame(processed_rows)

            # Log basic metrics
            mlflow.log_param('dc_offset_method', self.method)
            mlflow.log_param('dc_offset_per_channel', self.per_channel)
            mlflow.log_metric('dc_offset_windows_processed', len(result_df))

            # Save interim results if requested
            if output_path:
                out_dir = Path(output_path)
                out_dir.mkdir(parents=True, exist_ok=True)
                # Save separate parquet per group if available, else single file
                if 'group' in result_df.columns and result_df['group'].notna().any():
                    for group in result_df['group'].dropna().unique():
                        group_df = result_df[result_df['group'] == group]
                        group_fp = out_dir / f"{str(group).lower()}.parquet"
                        group_df.to_parquet(group_fp)
                        self.logger.info(f"Saved DC-removed {group} windows to {group_fp}")
                else:
                    fp = out_dir / "dc_removed.parquet"
                    result_df.to_parquet(fp)
                    self.logger.info(f"Saved DC-removed windows to {fp}")

            duration = time.time() - start
            self.logger.info(f"DC offset removal completed in {duration:.2f}s")
            return result_df
        except Exception as e:
            self.logger.error(f"Error in DC offset removal: {str(e)}")
            raise


def remove_dc_offset_eeg_data(config: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function matching pipeline style to remove DC offset.
    """
    remover = EEGDCOffsetRemover(config)
    # Resolve interim output path if configured
    output_path = None
    try:
        if remover.save_interim:
            output_path = config['paths']['interim'].get('dc_removed')
    except Exception:
        output_path = None
    return remover.remove_dc(df, output_path)


