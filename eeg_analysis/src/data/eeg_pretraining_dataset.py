from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional

from pathlib import Path
import random
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
import math

META_COLS = {
    "participant",
    "run",
    "parent_window_id",
    "sub_window_id",
    "window_start",
    "window_end",
}

EXCLUDED_CHANNELS = {"VEOG", "HEOG"}


def _list_channel_columns_from_schema(pf: pq.ParquetFile) -> List[str]:
    schema = pf.schema_arrow
    ch_cols: List[str] = []
    for i in range(schema.num_fields):
        field = schema.field(i)
        name = field.name
        if name in META_COLS:
            continue
        typ = field.type
        if pa.types.is_list(typ):
            ch_cols.append(name)
    return ch_cols


class EEGPretrainingDataset(Dataset):
    """
    Treat each (file, channel) pair as one sequence sample.
    Each sample returns:
        {
          "windows": FloatTensor [num_windows, window_length],
          "channel_name": str,
          "seq_len": int
        }
    """
    def __init__(self, dataset_root: str, window_length: int = 2048, file_glob: str = "*.parquet") -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.window_length = int(window_length)
        self.files: List[Path] = sorted(self.dataset_root.rglob(file_glob))
        if not self.files:
            raise FileNotFoundError(f"No parquet files found under: {self.dataset_root}")
        # Build index of (file, channel)
        self.items: List[Tuple[Path, str]] = []
        for fp in self.files:
            try:
                pf = pq.ParquetFile(str(fp))
                channels = _list_channel_columns_from_schema(pf)
                for ch in channels:
                    if ch.upper() in EXCLUDED_CHANNELS:
                        continue
                    self.items.append((fp, ch))
            except Exception:
                # Fallback: try pandas read (first row) to infer channels
                try:
                    df = pd.read_parquet(str(fp), engine="pyarrow")
                    for col in df.columns:
                        if col in META_COLS:
                            continue
                        if col.upper() in EXCLUDED_CHANNELS:
                            continue
                        if isinstance(df[col].iloc[0], (list, np.ndarray)):
                            self.items.append((fp, col))
                except Exception:
                    continue
        if not self.items:
            raise RuntimeError(f"No channel sequences found in parquet files under {self.dataset_root}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        fp, channel = self.items[index]
        # Load only the channel column for efficiency
        df = pd.read_parquet(str(fp), engine="pyarrow", columns=[channel])
        # Convert list-of-floats per row to tensor
        seq_list = df[channel].tolist()
        # Each token (window) is already the correct length; enforce strictly
        windows: List[np.ndarray] = []
        for i, v in enumerate(seq_list):
            arr = np.asarray(v, dtype=np.float32).ravel()
            if arr.shape[0] != self.window_length:
                raise ValueError(
                    f"Window length mismatch in {fp.name} channel {channel} at row {i}: "
                    f"got {arr.shape[0]}, expected {self.window_length}"
                )
            windows.append(arr)
        windows_np = np.stack(windows, axis=0)  # (L, W)
        windows_t = torch.from_numpy(windows_np).to(torch.float32)  # (L, W)
        return {
            "windows": windows_t,
            "channel_name": str(channel).upper(),
            "seq_len": int(windows_t.shape[0]),
        }


def collate_eeg_sequences(
    batch: List[Dict[str, Any]],
    mask_ratio: float,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Collate function:
      - Pad sequences to max_len (with zeros)
      - Build mask per sequence (15–20% tokens; use mask_ratio from config)
      - Apply replacements: 80% zeros, 10% Gaussian noise, 10% keep
      - Return both original and masked windows
    """
    if not batch:
        raise ValueError("Empty batch")
    device = device or torch.device("cpu")
    B = len(batch)
    seq_lens = torch.tensor([b["seq_len"] for b in batch], dtype=torch.long)
    max_len = int(seq_lens.max().item())
    window_length = int(batch[0]["windows"].shape[1])
    # Allocate tensors
    orig = torch.zeros((B, max_len, window_length), dtype=torch.float32)
    masked = torch.zeros((B, max_len, window_length), dtype=torch.float32)
    mask_bool = torch.zeros((B, max_len), dtype=torch.bool)
    channel_names = [b["channel_name"] for b in batch]
    # Fill
    for i, b in enumerate(batch):
        L = b["seq_len"]
        orig[i, :L, :] = b["windows"]
        masked[i, :L, :] = b["windows"]
        # Build mask positions
        k = max(1, int(math.ceil(mask_ratio * L)))  # at least 1 token masked
        idxs = np.random.choice(L, size=k, replace=False)
        mask_bool[i, idxs] = True
        # Replacement strategy: per position decide
        for t in idxs:
            r = random.random()
            if r < 0.8:
                # zero out
                masked[i, t, :] = 0.0
            elif r < 0.9:
                # gaussian noise
                masked[i, t, :] = torch.randn(window_length, dtype=torch.float32) * 0.5
            else:
                # keep as is
                pass
    return {
        "windows": orig,                # (B, L, W)
        "windows_masked": masked,       # (B, L, W)
        "mask_bool": mask_bool,         # (B, L)
        "seq_lengths": seq_lens,        # (B,)
        "channel_names": channel_names, # list[str]
    }


