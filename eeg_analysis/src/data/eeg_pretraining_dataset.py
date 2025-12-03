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
    def __init__(self, dataset_root: str, window_length: int = 2048, file_glob: str = "*.parquet", split: str = "train", val_ratio: float = 0.2, seed: int = 42) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.window_length = int(window_length)
        self.split = split
        self.val_ratio = val_ratio
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
        
        # Split into train/val deterministically
        rng = random.Random(seed)
        indices = list(range(len(self.items)))
        rng.shuffle(indices)
        split_idx = int(len(indices) * (1 - val_ratio))
        if split == "train":
            indices = indices[:split_idx]
        elif split == "val":
            indices = indices[split_idx:]
        else:
            raise ValueError(f"Invalid split: {split}")
        self.items = [self.items[i] for i in indices]

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
            "file_path": str(fp),  # Add file path for multi-channel masking
        }


def collate_eeg_sequences(
    batch: List[Dict[str, Any]],
    mask_ratio: float,
    device: Optional[torch.device] = None,
    masking_style: str = "mae",  # "mae", "bert", "within_token", or "multi_channel"
    mask_samples_within_token: bool = False,  # For within_token style
    mask_replacement: str = "zeros",  # "zeros" or "gaussian_noise"
) -> Dict[str, Any]:
    """
    Collate function with configurable masking strategy:
    
    MAE-style (default, for mask_ratio > 0.2):
      - Select mask_ratio * seq_len tokens (windows) independently per channel
      - Replace ALL masked tokens with zeros
      - No information leakage
      - Harder reconstruction task
      
    BERT-style (optional, for mask_ratio <= 0.2):
      - Select mask_ratio * seq_len tokens independently per channel
      - Of masked tokens: 80% zeros, 10% noise, 10% unchanged
      - Some information leakage (easier task)
    
    WITHIN_TOKEN-style:
      - Keep ALL tokens (windows) in sequence
      - Mask mask_ratio * window_length samples WITHIN each token independently per channel
      - Model must learn from unmasked samples within tokens
      - Always has signal content available (no zero tokens)
      - Better for learning signal patterns, less position-only learning
    
    MULTI_CHANNEL-style (NEW - recommended for learning cross-channel relationships):
      - Group channels by file (same participant-run)
      - Mask the SAME temporal positions across all channels from the same file
      - Forces model to learn spatial relationships (use neighboring channels to reconstruct)
      - Can combine with token-level or within-token masking
      - Example: If C3 is masked at position 10, also mask C1, C5, F3, P3 at position 10
      - Encourages learning: "Use spatial context from other channels to reconstruct masked channel"
    
    Loss should be computed ONLY on masked positions/samples.
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
    
    # Mask can be token-level (B, L) or sample-level (B, L, W) depending on masking_style
    # - "within_token": always sample-level
    # - "multi_channel" + mask_samples_within_token=True: sample-level synchronized
    # - "multi_channel" + mask_samples_within_token=False: token-level synchronized
    # - "mae"/"bert": token-level independent
    use_sample_level_mask = (
        masking_style == "within_token" or 
        (masking_style == "multi_channel" and mask_samples_within_token)
    )
    if use_sample_level_mask:
        mask_bool = torch.zeros((B, max_len, window_length), dtype=torch.bool)  # Sample-level mask
    else:
        mask_bool = torch.zeros((B, max_len), dtype=torch.bool)  # Token-level mask
    
    channel_names = [b["channel_name"] for b in batch]
    file_paths = [b.get("file_path", "") for b in batch]  # Get file paths for multi-channel grouping
    
    # Group channels by file for multi-channel masking
    if masking_style == "multi_channel":
        # Group batch indices by file path
        file_to_indices: Dict[str, List[int]] = {}
        for i, fp in enumerate(file_paths):
            if fp not in file_to_indices:
                file_to_indices[fp] = []
            file_to_indices[fp].append(i)
        
        # Generate synchronized masks for each file group
        file_masks: Dict[str, torch.Tensor] = {}  # file_path -> (L,) or (L, W) mask
        for fp, indices in file_to_indices.items():
            # Use the minimum sequence length for this file group (all channels should have same length)
            group_seq_lens = [seq_lens[i].item() for i in indices]
            min_seq_len = min(group_seq_lens) if group_seq_lens else max_len
            
            if mask_samples_within_token:
                # Sample-level synchronized masking
                file_mask = torch.zeros((min_seq_len, window_length), dtype=torch.bool)
                for t in range(min_seq_len):
                    num_samples_to_mask = max(1, int(math.ceil(mask_ratio * window_length)))
                    sample_idxs = np.random.choice(window_length, size=num_samples_to_mask, replace=False)
                    file_mask[t, sample_idxs] = True
            else:
                # Token-level synchronized masking
                file_mask = torch.zeros(min_seq_len, dtype=torch.bool)
                k = max(1, int(math.ceil(mask_ratio * min_seq_len)))
                idxs = np.random.choice(min_seq_len, size=k, replace=False)
                file_mask[idxs] = True
            
            file_masks[fp] = file_mask
    
    # Fill
    for i, b in enumerate(batch):
        L = b["seq_len"]
        orig[i, :L, :] = b["windows"]
        masked[i, :L, :] = b["windows"]  # Start with original
        
        # Get mask for this sample (synchronized if multi_channel, independent otherwise)
        if masking_style == "multi_channel":
            fp = file_paths[i]
            if fp in file_masks:
                file_mask = file_masks[fp]
                # Apply synchronized mask (trim to actual sequence length)
                actual_len = min(L, file_mask.shape[0])
                if mask_samples_within_token:
                    # Sample-level synchronized masking
                    mask_bool[i, :actual_len, :] = file_mask[:actual_len, :]
                    # Replace masked samples
                    masked_slice = masked[i, :actual_len, :]  # (actual_len, window_length)
                    if mask_replacement == "gaussian_noise":
                        # Generate noise matching each token's variance
                        for t in range(actual_len):
                            if file_mask[t, :].any():
                                token_std = masked_slice[t, :].std().item()
                                token_std = max(token_std, 1e-6)  # Prevent zero std
                                masked_slice[t, file_mask[t, :]] = torch.randn(file_mask[t, :].sum().item(), dtype=torch.float32) * token_std
                    else:
                        masked_slice[file_mask[:actual_len, :]] = 0.0
                    masked[i, :actual_len, :] = masked_slice
                else:
                    # Token-level synchronized masking
                    mask_bool[i, :actual_len] = file_mask[:actual_len]
                    # Replace masked tokens
                    if mask_replacement == "gaussian_noise":
                        # Generate noise matching each token's variance
                        for t in range(actual_len):
                            if file_mask[t]:
                                token_std = masked[i, t, :].std().item()
                                token_std = max(token_std, 1e-6)  # Prevent zero std
                                masked[i, t, :] = torch.randn(window_length, dtype=torch.float32) * token_std
                    else:
                        masked[i, :actual_len, :][file_mask[:actual_len], :] = 0.0
            # If file not found in file_masks, skip masking (shouldn't happen)
        elif masking_style == "within_token" or mask_samples_within_token:
            # WITHIN_TOKEN: Mask samples within each token, keep all tokens
            # For each token, mask mask_ratio * window_length samples independently per channel
            num_samples_to_mask = max(1, int(math.ceil(mask_ratio * window_length)))
            
            for t in range(L):
                # Randomly select samples to mask within this token
                sample_idxs = np.random.choice(window_length, size=num_samples_to_mask, replace=False)
                mask_bool[i, t, sample_idxs] = True
                # Replace masked samples
                if mask_replacement == "gaussian_noise":
                    # Generate noise matching this token's variance
                    token_std = masked[i, t, :].std().item()
                    token_std = max(token_std, 1e-6)  # Prevent zero std
                    masked[i, t, sample_idxs] = torch.randn(num_samples_to_mask, dtype=torch.float32) * token_std
                else:
                    masked[i, t, sample_idxs] = 0.0
            
            # Token-level mask: True if ANY sample in token is masked (for compatibility)
            token_mask = mask_bool[i, :L, :].any(dim=1)  # (L,)
            
        else:
            # Token-level masking (original MAE/BERT style)
            # Select mask positions
            k = max(1, int(math.ceil(mask_ratio * L)))  # at least 1 token masked
            idxs = np.random.choice(L, size=k, replace=False)
            mask_bool[i, idxs] = True
            
            if masking_style == "mae":
                # MAE-style: Replace masked tokens
                # No information leakage - model must reconstruct from context only
                if mask_replacement == "gaussian_noise":
                    # Generate noise matching each token's variance
                    for t in idxs:
                        token_std = masked[i, t, :].std().item()
                        token_std = max(token_std, 1e-6)  # Prevent zero std
                        masked[i, t, :] = torch.randn(window_length, dtype=torch.float32) * token_std
                else:
                    masked[i, idxs, :] = 0.0
                
            elif masking_style == "bert":
                # BERT-style: 80% zeros, 10% noise, 10% unchanged
                # Some information leakage - easier reconstruction
                for t in idxs:
                    r = random.random()
                    if r < 0.8:
                        # Zero out
                        masked[i, t, :] = 0.0
                    elif r < 0.9:
                        # Gaussian noise matching token variance
                        token_std = masked[i, t, :].std().item()
                        token_std = max(token_std, 1e-6)  # Prevent zero std
                        masked[i, t, :] = torch.randn(window_length, dtype=torch.float32) * token_std
                    # else: keep as is (10% unchanged)
            else:
                raise ValueError(f"Unknown masking_style: {masking_style}. Use 'mae', 'bert', 'within_token', or 'multi_channel'.")
    
    return {
        "windows": orig,                # (B, L, W) - original unmasked
        "windows_masked": masked,       # (B, L, W) - masked input
        "mask_bool": mask_bool,         # (B, L) or (B, L, W) - True at masked positions/samples
        "seq_lengths": seq_lens,        # (B,)
        "channel_names": channel_names, # list[str]
        "file_paths": file_paths,      # list[str] - file paths for multi-channel analysis
    }


