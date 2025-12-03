"""
EEG Supervised Fine-Tuning Dataset
Loads primary windowed EEG data for classification fine-tuning.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class EEGSFTDataset(Dataset):
    """
    Dataset for supervised fine-tuning of EEG models.
    
    Each sample is a sequence of windows for all channels from one participant.
    Returns:
        {
          "windows": FloatTensor [num_channels, num_windows, window_length],
          "label": LongTensor (0 or 1),
          "participant": str,
          "channel_names": List[str]
        }
    """
    
    def __init__(
        self,
        data_path: str,
        window_length: int = 2048,
        split: str = "train",
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        seed: int = 42,
        channels: Optional[List[str]] = None
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.window_length = int(window_length)
        self.split = split
        self.seed = seed
        
        # Default channels if not provided
        self.channels = channels or ['af7', 'af8', 'tp9', 'tp10']
        self.channels = [ch.upper() for ch in self.channels]
        
        # Load data
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        df = pd.read_parquet(self.data_path)
        
        # Group by participant
        participants = df['Participant'].unique()
        labels = df.groupby('Participant')['Remission'].first().to_dict()
        
        # Split participants into train/val/test
        train_participants, test_participants = train_test_split(
            participants,
            test_size=test_ratio,
            random_state=seed,
            stratify=[labels[p] for p in participants]
        )
        
        train_participants, val_participants = train_test_split(
            train_participants,
            test_size=val_ratio / (1 - test_ratio),
            random_state=seed,
            stratify=[labels[p] for p in train_participants]
        )
        
        # Select participants based on split
        if split == "train":
            selected_participants = train_participants
        elif split == "val":
            selected_participants = val_participants
        elif split == "test":
            selected_participants = test_participants
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train', 'val', or 'test'.")
        
        # Filter data for selected participants
        self.df = df[df['Participant'].isin(selected_participants)].copy()
        self.participants = sorted(selected_participants)
        self.labels = {p: labels[p] for p in self.participants}
        
        # CRITICAL: Verify temporal ordering is preserved in the dataframe
        # Windows should be sorted by Participant -> parent_window_id -> sub_window_id
        if 'parent_window_id' in self.df.columns:
            sort_cols = ['Participant', 'parent_window_id']
            if 'sub_window_id' in self.df.columns:
                sort_cols.append('sub_window_id')
            
            # Check if already sorted
            is_sorted = self.df[sort_cols].equals(self.df[sort_cols].sort_values(sort_cols))
            
            if not is_sorted:
                print(f"[{split.upper()}] Warning: Data not sorted by window IDs, sorting now...")
                self.df = self.df.sort_values(sort_cols).reset_index(drop=True)
                print(f"[{split.upper()}] ✓ Data sorted to preserve temporal window order")
            else:
                print(f"[{split.upper()}] ✓ Data already sorted in temporal order")
        else:
            print(f"[{split.upper()}] Warning: No window IDs found, assuming data is pre-sorted")
        
        print(f"[{split.upper()}] Loaded {len(self.participants)} participants, "
              f"{len(self.df)} windows")
        print(f"  Remission: {sum(self.labels.values())} participants")
        print(f"  Non-remission: {len(self.participants) - sum(self.labels.values())} participants")
    
    def __len__(self) -> int:
        return len(self.participants)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        participant = self.participants[index]
        label = self.labels[participant]
        
        # Get all windows for this participant
        participant_data = self.df[self.df['Participant'] == participant].copy()
        
        # CRITICAL: Sort by window IDs to maintain temporal order
        # Windows must be in sequential order for temporal modeling
        sort_cols = []
        if 'parent_window_id' in participant_data.columns:
            sort_cols.append('parent_window_id')
        if 'sub_window_id' in participant_data.columns:
            sort_cols.append('sub_window_id')
        
        if sort_cols:
            participant_data = participant_data.sort_values(sort_cols)
            
            # Verify ordering is correct (window IDs should be sequential or at least monotonic)
            if 'parent_window_id' in participant_data.columns:
                window_ids = participant_data['parent_window_id'].values
                if len(window_ids) > 1:
                    # Check if mostly monotonic (allowing for some resets between runs)
                    is_ordered = sum(window_ids[i] <= window_ids[i+1] for i in range(len(window_ids)-1)) / (len(window_ids)-1) > 0.9
                    if not is_ordered:
                        print(f"Warning: Window ordering may be incorrect for participant {participant}")
        else:
            # If no window IDs, use dataframe index order (assumes data was pre-sorted)
            print(f"Warning: No window IDs found for participant {participant}, using dataframe order")
        
        # Extract channel data
        channel_windows = []
        for ch in self.channels:
            if ch not in participant_data.columns:
                # Channel not available, use zeros
                num_windows = len(participant_data)
                channel_windows.append(np.zeros((num_windows, self.window_length), dtype=np.float32))
            else:
                # Extract windows for this channel
                windows = []
                for _, row in participant_data.iterrows():
                    window_data = row[ch]
                    if isinstance(window_data, (list, np.ndarray)):
                        window_array = np.array(window_data, dtype=np.float32).ravel()
                        # Ensure correct length
                        if len(window_array) < self.window_length:
                            # Pad if too short
                            window_array = np.pad(
                                window_array,
                                (0, self.window_length - len(window_array)),
                                mode='constant'
                            )
                        elif len(window_array) > self.window_length:
                            # Truncate if too long
                            window_array = window_array[:self.window_length]
                        windows.append(window_array)
                    else:
                        # Invalid data, use zeros
                        windows.append(np.zeros(self.window_length, dtype=np.float32))
                
                channel_windows.append(np.stack(windows, axis=0))
        
        # Stack channels: [num_channels, num_windows, window_length]
        windows_tensor = torch.from_numpy(np.stack(channel_windows, axis=0)).to(torch.float32)
        
        return {
            "windows": windows_tensor,  # [C, W, L]
            "label": torch.tensor(label, dtype=torch.long),
            "participant": participant,
            "channel_names": self.channels
        }


def collate_sft_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for SFT dataset.
    Pads sequences to the same number of windows within the batch.
    
    IMPORTANT: Window order is preserved from the dataset.
    Each participant's windows are in temporal order (sorted by parent_window_id, sub_window_id).
    This is critical for temporal modeling in the Mamba backbone.
    
    Args:
        batch: List of samples from EEGSFTDataset (windows already in temporal order)
        
    Returns:
        Dictionary with batched tensors (temporal order preserved)
    """
    # Find max number of windows in batch
    max_windows = max(sample["windows"].shape[1] for sample in batch)
    
    batch_size = len(batch)
    num_channels = batch[0]["windows"].shape[0]
    window_length = batch[0]["windows"].shape[2]
    
    # Allocate tensors
    windows_batch = torch.zeros((batch_size, num_channels, max_windows, window_length), dtype=torch.float32)
    labels_batch = torch.zeros(batch_size, dtype=torch.long)
    seq_lengths = torch.zeros(batch_size, dtype=torch.long)
    participants = []
    
    # Get channel names (same for all samples in batch)
    channel_names = batch[0]["channel_names"]
    
    for i, sample in enumerate(batch):
        num_windows = sample["windows"].shape[1]
        windows_batch[i, :, :num_windows, :] = sample["windows"]
        labels_batch[i] = sample["label"]
        seq_lengths[i] = num_windows
        participants.append(sample["participant"])
    
    return {
        "windows": windows_batch,  # [B, C, W, L]
        "labels": labels_batch,  # [B]
        "seq_lengths": seq_lengths,  # [B]
        "participants": participants,  # List[str]
        "channel_names": channel_names  # List[str] - channel names for all samples
    }

