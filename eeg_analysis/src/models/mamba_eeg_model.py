from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # Preferred: official Mamba library
    from mamba_ssm import Mamba
except Exception:  # pragma: no cover - fallback when mamba-ssm not available
    Mamba = None  # type: ignore

from src.utils.logger import setup_logger
from src.utils.electrodes import load_asa_file_to_unit_sphere

logger = setup_logger(__name__)


def _make_orthonormal_projection(in_features: int, out_features: int, device: torch.device) -> torch.Tensor:
    """
    Create a fixed orthonormal-like projection matrix of shape (in_features, out_features).
    Uses QR decomposition on a random matrix. Not trainable.
    """
    with torch.no_grad():
        a = torch.randn(in_features, out_features, device=device, dtype=torch.float32)
        # QR on tall/skinny or short/fat
        if in_features >= out_features:
            q, _ = torch.linalg.qr(a, mode="reduced")
            proj = q  # (in, out)
        else:
            q, _ = torch.linalg.qr(a.T, mode="reduced")
            proj = q.T  # (in, out)
    return proj


class SpatialEncoderFixed(nn.Module):
    """
    Fixed spatial encoding from 3D EEG electrode unit coordinates → d_model via fixed 3xD projection.
    Channel coordinates are normalized to unit vectors, then projected by a fixed orthonormal matrix.
    """
    def __init__(self, d_model: int, channel_to_xyz: Dict[str, Tuple[float, float, float]], device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.register_buffer("proj", _make_orthonormal_projection(3, d_model, device or torch.device("cpu")), persistent=False)
        # store coordinates in a module buffer for easy device moves
        # keys kept in Python dict; vectors kept in a tensor map built on demand
        self.channel_to_xyz = {k.upper(): v for k, v in channel_to_xyz.items()}

    def _coords_for(self, channel_name: str) -> torch.Tensor:
        xyz = self.channel_to_xyz.get(channel_name.upper())
        if xyz is None:
            # Unknown channels map to zero vector (no spatial bias)
            return torch.zeros(3, dtype=torch.float32, device=self.proj.device)
        v = torch.tensor(xyz, dtype=torch.float32, device=self.proj.device)
        norm = torch.linalg.vector_norm(v)
        if norm > 0:
            v = v / norm
        return v

    def forward(self, channel_names: List[str]) -> torch.Tensor:
        """
        Args:
            channel_names: list of channel identifiers for batch items; length B
        Returns:
            spatial_embeddings: Tensor of shape (B, d_model)
        """
        coords = torch.stack([self._coords_for(nm) for nm in channel_names], dim=0)  # (B, 3)
        # (B, 3) @ (3, D) -> (B, D)
        return coords @ self.proj


class TemporalEncoder(nn.Module):
    """
    Temporal encoding: scalar t/T → Linear(1→d_model). Trainable.
    """
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.lin = nn.Linear(1, d_model)

    def forward(self, seq_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq_lengths: LongTensor shape (B,) containing valid sequence lengths
        Returns:
            temporal_enc: (B, L, d_model) for max(L) in batch, with t normalized by T for each sequence
        """
        device = seq_lengths.device
        max_len = int(seq_lengths.max().item())
        positions = torch.arange(max_len, device=device, dtype=torch.float32)  # (L,)
        # Build per-batch normalized positions t/T
        norm_pos = []
        for T in seq_lengths.tolist():
            T_val = max(1, int(T))
            p = positions[:T_val] / float(T_val)
            if T_val < max_len:
                pad = torch.zeros(max_len - T_val, device=device, dtype=torch.float32)
                p = torch.cat([p, pad], dim=0)
            norm_pos.append(p)
        norm_positions = torch.stack(norm_pos, dim=0).unsqueeze(-1)  # (B, L, 1)
        return self.lin(norm_positions)  # (B, L, d_model)


class TokenEncoder(nn.Module):
    """
    Input projection for tokens: Linear(window_length → d_model) + LayerNorm for stability.
    """
    def __init__(self, window_length: int, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(window_length, d_model)
        self.norm = nn.LayerNorm(d_model)
        # Initialize projection with smaller weights to avoid explosion
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        nn.init.zeros_(self.proj.bias)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """
        Args:
            windows: (B, L, window_length)
        Returns:
            token_embeddings: (B, L, d_model)
        """
        x = self.proj(windows)
        x = self.norm(x)
        return x


class MambaBackbone(nn.Module):
    """
    Mamba-2 backbone (6 layers, d_model=128) or Transformer fallback if mamba_ssm unavailable.
    Expects input of shape (B, L, D). Returns (B, L, D).
    """
    def __init__(self, d_model: int, num_layers: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        if Mamba is not None:
            self.layers = nn.ModuleList([Mamba(d_model=d_model) for _ in range(num_layers)])
            self.dropout = nn.Dropout(dropout)
            # Add layer norm between Mamba blocks for stability
            self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
            logger.info("Using Mamba backbone")
        else:
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True)
            self.layers = nn.ModuleList([nn.TransformerEncoder(encoder_layer, num_layers=1) for _ in range(num_layers)])
            self.dropout = nn.Dropout(dropout)
            self.layer_norms = None
            logger.warning("mamba_ssm not found; using Transformer fallback")

        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        if self.layer_norms is not None:
            # Mamba with per-layer normalization
            for layer, norm in zip(self.layers, self.layer_norms):
                h = layer(h) + h  # residual connection
                h = norm(h)
                h = self.dropout(h)
        else:
            # Transformer fallback
            for layer in self.layers:
                h = layer(h)
            h = self.dropout(h)
        return self.out_norm(h)


class MambaEEGModel(nn.Module):
    """
    Full EEG encoder with:
      - Token input projection (2048 → d_model)
      - Fixed spatial encoding (3 → d_model)
      - Temporal encoding (t/T → d_model)
      - Mamba-2 backbone (sequence modeling)
    Forward consumes masked windows and returns predicted embeddings for each token position.
    """
    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 6,
        window_length: int = 2048,
        channel_to_xyz: Optional[Dict[str, Tuple[float, float, float]]] = None,
        asa_path: Optional[str] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.window_length = window_length
        # Default partial 10–20 mapping (extend as needed)
        default_xyz = {
            "FP1": (-0.3, 0.9, 0.3), "FPZ": (0.0, 1.0, 0.0), "FP2": (0.3, 0.9, 0.3),
            "AF3": (-0.4, 0.7, 0.5), "AF4": (0.4, 0.7, 0.5),
            "F7": (-0.8, 0.4, 0.4), "F5": (-0.6, 0.5, 0.5), "F3": (-0.4, 0.6, 0.6), "F1": (-0.2, 0.7, 0.6),
            "FZ": (0.0, 0.8, 0.6), "F2": (0.2, 0.7, 0.6), "F4": (0.4, 0.6, 0.6), "F6": (0.6, 0.5, 0.5), "F8": (0.8, 0.4, 0.4),
            "FT7": (-0.9, 0.2, 0.3), "FT8": (0.9, 0.2, 0.3),
            "FC5": (-0.7, 0.2, 0.6), "FC3": (-0.5, 0.3, 0.7), "FC1": (-0.3, 0.4, 0.7),
            "FCZ": (0.0, 0.5, 0.7), "FC2": (0.3, 0.4, 0.7), "FC4": (0.5, 0.3, 0.7), "FC6": (0.7, 0.2, 0.6),
            "T7": (-1.0, 0.0, 0.2), "T8": (1.0, 0.0, 0.2),
            "C5": (-0.7, 0.0, 0.7), "C3": (-0.5, 0.0, 0.8), "C1": (-0.3, 0.0, 0.8),
            "CZ": (0.0, 0.0, 0.9), "C2": (0.3, 0.0, 0.8), "C4": (0.5, 0.0, 0.8), "C6": (0.7, 0.0, 0.7),
            "TP7": (-0.9, -0.2, 0.3), "TP8": (0.9, -0.2, 0.3),
            "CP5": (-0.7, -0.2, 0.6), "CP3": (-0.5, -0.3, 0.7), "CP1": (-0.3, -0.4, 0.7),
            "CPZ": (0.0, -0.5, 0.7), "CP2": (0.3, -0.4, 0.7), "CP4": (0.5, -0.3, 0.7), "CP6": (0.7, -0.2, 0.6),
            "P7": (-0.8, -0.4, 0.4), "P5": (-0.6, -0.5, 0.5), "P3": (-0.4, -0.6, 0.6), "P1": (-0.2, -0.7, 0.6),
            "PZ": (0.0, -0.8, 0.6), "P2": (0.2, -0.7, 0.6), "P4": (0.4, -0.6, 0.6), "P6": (0.6, -0.5, 0.5), "P8": (0.8, -0.4, 0.4),
            "PO7": (-0.6, -0.7, 0.4), "PO5": (-0.5, -0.7, 0.5), "PO3": (-0.3, -0.8, 0.5), "POZ": (0.0, -0.9, 0.5),
            "PO4": (0.3, -0.8, 0.5), "PO6": (0.5, -0.7, 0.5), "PO8": (0.6, -0.7, 0.4),
            "O1": (-0.3, -0.9, 0.3), "OZ": (0.0, -1.0, 0.0), "O2": (0.3, -0.9, 0.3),
            "M1": (-1.0, -0.1, 0.0), "M2": (1.0, -0.1, 0.0),
            "HEOG": (0.0, 0.95, 0.0), "VEOG": (0.0, 0.9, -0.2),
            "CB1": (-0.5, -0.8, 0.1), "CB2": (0.5, -0.8, 0.1),
        }
        if channel_to_xyz is None and asa_path:
            try:
                channel_to_xyz = load_asa_file_to_unit_sphere(asa_path)
                logger.info(f"Loaded {len(channel_to_xyz)} channels from ASA file")
            except Exception as e:
                logger.warning(f"Failed to load ASA file '{asa_path}': {e}. Falling back to default mapping.")
                channel_to_xyz = default_xyz
        else:
            channel_to_xyz = channel_to_xyz or default_xyz

        self.token_encoder = TokenEncoder(window_length=window_length, d_model=d_model)
        self.spatial_encoder = SpatialEncoderFixed(d_model=d_model, channel_to_xyz=channel_to_xyz)
        self.temporal_encoder = TemporalEncoder(d_model=d_model)
        self.backbone = MambaBackbone(d_model=d_model, num_layers=num_layers, dropout=dropout)

    @torch.no_grad()
    def encode_tokens_only(self, windows: torch.Tensor) -> torch.Tensor:
        """
        Token projection only (for targets).
        Args:
            windows: (B, L, window_length)
        Returns:
            token_embeddings: (B, L, d_model)
        """
        return self.token_encoder(windows)

    def forward(
        self,
        windows_masked: torch.Tensor,
        channel_names: List[str],
        seq_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict embeddings for token positions from masked inputs.
        Args:
            windows_masked: (B, L, window_length)
            channel_names: list length B
            seq_lengths: (B,) valid lengths
        Returns:
            predicted_embeddings: (B, L, d_model)
        """
        device = windows_masked.device
        B, L, _ = windows_masked.shape

        token_emb = self.token_encoder(windows_masked)               # (B, L, D)
        temporal = self.temporal_encoder(seq_lengths)                # (B, L, D)
        spatial = self.spatial_encoder(channel_names)                # (B, D)
        spatial = spatial.unsqueeze(1).expand(B, L, self.d_model)    # (B, L, D)

        x = token_emb + temporal + spatial
        y = self.backbone(x)
        return y



