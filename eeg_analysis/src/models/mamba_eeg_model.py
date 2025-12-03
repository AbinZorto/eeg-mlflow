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
    Deep MLP encoder for tokens: window_length → hidden1 → hidden2 → hidden3 → d_model
    
    Uses learnable mask token to prevent encoder collapse when input is zero.
    When masked tokens are zero, they're replaced with a learnable mask token vector.
    """
    def __init__(self, window_length: int, d_model: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        # Reduced capacity for parameter efficiency (500k tokens)
        hidden_dim = hidden_dim or (d_model * 2)  # Reduced: 2x d_model (was 4x)
        hidden_dim2 = d_model * 1.5  # Intermediate layer (was 3x)
        
        # More efficient encoder: 3 layers (was 4)
        self.layers = nn.Sequential(
            # Layer 1: window_length → hidden_dim
            nn.Linear(window_length, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            
            # Layer 2: hidden_dim → hidden_dim2
            nn.Linear(hidden_dim, int(hidden_dim2)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim2)),
            nn.Dropout(0.1),
            
            # Layer 3: hidden_dim2 → d_model
            nn.Linear(int(hidden_dim2), d_model),
            nn.LayerNorm(d_model),
        )
        # Initialize with smaller weights
        for module in self.layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)
        
        # Learnable mask token: base vector for masked inputs
        # Initialize with small random values (not zero!)
        self.mask_token_base = nn.Parameter(torch.randn(window_length) * 0.02)
        
        # Position-dependent mask tokens: prevents encoder collapse
        # Each position gets a slightly different mask token (base + small learnable offset)
        # This ensures masked tokens at different positions produce different embeddings
        # Max sequence length: 1000 (can be increased if needed)
        max_seq_len = 1000
        self.mask_token_offsets = nn.Parameter(torch.randn(max_seq_len, window_length) * 0.01)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """
        Args:
            windows: (B, L, window_length) - may contain zeros for masked tokens
        Returns:
            token_embeddings: (B, L, d_model)
        """
        # Replace zero tokens with position-dependent learnable mask token
        # Check if token is all zeros (masked) - use small threshold to avoid numerical issues
        token_norms = windows.abs().sum(dim=-1)  # (B, L) - sum of absolute values per token
        is_masked = token_norms < 1e-6  # True if token is all zeros (masked)
        
        windows_with_mask = windows.clone()
        # Replace masked tokens with position-dependent mask tokens
        if is_masked.any():
            B, L, W = windows.shape
            # Get position indices for masked tokens
            batch_indices, seq_indices = torch.where(is_masked)  # (N_masked,), (N_masked,)
            
            # Create position-dependent mask tokens: base + position-specific offset
            # Clamp seq_indices to valid range for mask_token_offsets
            seq_indices_clamped = torch.clamp(seq_indices, 0, self.mask_token_offsets.shape[0] - 1)
            position_offsets = self.mask_token_offsets[seq_indices_clamped]  # (N_masked, window_length)
            mask_tokens = self.mask_token_base.unsqueeze(0) + position_offsets  # (N_masked, window_length)
            
            # Replace masked tokens
            windows_with_mask[batch_indices, seq_indices] = mask_tokens
        
        return self.layers(windows_with_mask)
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Handle loading old checkpoints that use 'mask_token' instead of 'mask_token_base' and 'mask_token_offsets'."""
        old_key = prefix + 'mask_token'
        new_base_key = prefix + 'mask_token_base'
        new_offsets_key = prefix + 'mask_token_offsets'
        
        if old_key in state_dict:
            # Old checkpoint format: convert mask_token to new format
            old_mask_token = state_dict.pop(old_key)
            state_dict[new_base_key] = old_mask_token
            # Initialize offsets as zeros (will learn during training)
            window_length = old_mask_token.shape[0]
            max_seq_len = 1000
            state_dict[new_offsets_key] = torch.zeros(max_seq_len, window_length)
        
        # Call parent to handle normal loading
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


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
        
        # FLEXIBLE ARCHITECTURE: Use lightweight projection if window_length != d_model
        # For 8-second windows (2048 samples) with d_model=512, use strided convolution
        if window_length != d_model:
            # Use strided 1D convolution for efficient downsampling
            # This preserves spatial structure better than linear projection
            # Example: 2048 → 512 using stride=4 convolution
            if window_length % d_model == 0:
                # Perfect division: use single strided conv
                stride = window_length // d_model
                kernel_size = stride  # Match stride for non-overlapping downsampling
                self.input_proj = nn.Sequential(
                    nn.Conv1d(1, 1, kernel_size=kernel_size, stride=stride, bias=False),
                    nn.LayerNorm(d_model),
                )
                logger.info(f"Using strided convolution projection: {window_length} → {d_model} (stride={stride})")
            else:
                # Not perfect division: use adaptive pooling + linear
                self.input_proj = nn.Sequential(
                    nn.AdaptiveAvgPool1d(d_model),
                    nn.LayerNorm(d_model),
                )
                logger.info(f"Using adaptive pooling projection: {window_length} → {d_model}")
            
            # Output projection: d_model → window_length (for reconstruction)
            if window_length % d_model == 0:
                # Use transposed convolution for upsampling
                stride = window_length // d_model
                kernel_size = stride
                self.output_proj = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.ConvTranspose1d(1, 1, kernel_size=kernel_size, stride=stride, bias=False),
                )
                logger.info(f"Using transposed convolution: {d_model} → {window_length} (stride={stride})")
            else:
                # Use linear interpolation
                self.output_proj = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, window_length),
                )
                logger.info(f"Using linear projection: {d_model} → {window_length}")
        else:
            # Direct feed: no projection needed
            self.input_proj = None
            self.output_proj = None
            logger.info(f"Direct feed: d_model={d_model} == window_length={window_length} (no projection)")
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

        # FLEXIBLE ARCHITECTURE: Lightweight projection if needed, otherwise direct feed
        self.spatial_encoder = SpatialEncoderFixed(d_model=d_model, channel_to_xyz=channel_to_xyz)
        self.temporal_encoder = TemporalEncoder(d_model=d_model)
        self.backbone = MambaBackbone(d_model=d_model, num_layers=num_layers, dropout=dropout)
        
        logger.info(f"Architecture: window_length={window_length}, d_model={d_model}")
        if self.input_proj is None:
            logger.info("Direct feed - complexity in Mamba backbone only")
        else:
            logger.info("Lightweight projection - complexity in Mamba backbone + minimal projection")

    @torch.no_grad()
    def encode_tokens_only(self, windows: torch.Tensor) -> torch.Tensor:
        """
        DEPRECATED: No encoder in simplified architecture.
        Returns windows directly (since d_model == window_length).
        Args:
            windows: (B, L, window_length)
        Returns:
            windows: (B, L, window_length) - same as input
        """
        return windows  # No encoding needed - direct feed

    def forward(
        self,
        windows_masked: torch.Tensor,
        channel_names: List[str],
        seq_lengths: torch.Tensor,
        disable_temporal: bool = False,  # For control experiments
        disable_spatial: bool = False,   # For control experiments
        decode_to_signal: bool = False,  # NEW: Return reconstructed signal instead of embeddings
        mask_bool: Optional[torch.Tensor] = None,  # (B, L) or (B, L, W) mask - token-level or sample-level
    ) -> torch.Tensor:
        """
        Predict embeddings (or reconstructed signal) for token positions from masked inputs.
        Args:
            windows_masked: (B, L, window_length)
            channel_names: list length B
            seq_lengths: (B,) valid lengths
            disable_temporal: If True, removes temporal encoding (for control)
            disable_spatial: If True, removes spatial encoding (for control)
            decode_to_signal: If True, decodes embeddings back to signal space (proper MAE)
            mask_bool: (B, L) or (B, L, W) boolean mask. If (B, L, W), converted to token-level
                      by checking if any samples in token are masked. Positional encoding is zeroed
                      for tokens with masked samples to prevent position leakage.
        Returns:
            If decode_to_signal=False: predicted_embeddings (B, L, d_model)
            If decode_to_signal=True: reconstructed_signal (B, L, window_length)
        """
        device = windows_masked.device
        B, L, W = windows_masked.shape

        # Normalize windows per-token to prevent numerical issues
        # Raw windows can be very large (e.g., -242221 to 254118), which causes NaN when combined with normalized encodings
        token_emb_mean = windows_masked.mean(dim=-1, keepdim=True)  # (B, L, 1)
        token_emb_std = windows_masked.std(dim=-1, keepdim=True)  # (B, L, 1)
        # Handle zero-variance windows (all samples same value)
        token_emb_std = torch.clamp(token_emb_std, min=1e-6)  # Prevent division by very small numbers
        windows_normalized = (windows_masked - token_emb_mean) / token_emb_std  # (B, L, window_length)
        
        # Project to d_model if needed (using lightweight convolution/pooling)
        if self.input_proj is not None:
            # Reshape for Conv1d: (B, L, W) -> (B*L, 1, W) -> conv -> (B*L, 1, d_model) -> (B, L, d_model)
            windows_reshaped = windows_normalized.view(B * L, 1, W)  # (B*L, 1, window_length)
            token_emb = self.input_proj(windows_reshaped)  # (B*L, 1, d_model)
            token_emb = token_emb.squeeze(1).view(B, L, self.d_model)  # (B, L, d_model)
        else:
            # Direct feed: already correct shape
            token_emb = windows_normalized  # (B, L, window_length) = (B, L, d_model)
        
        # Convert sample-level mask (B, L, W) to token-level mask (B, L) if needed
        token_mask = None
        if mask_bool is not None:
            if mask_bool.dim() == 3:  # Sample-level mask (B, L, W)
                # Token is masked if ANY sample in it is masked
                token_mask = mask_bool.any(dim=2)  # (B, L)
            else:  # Token-level mask (B, L)
                token_mask = mask_bool
        
        # Temporal encoding (can be disabled for control)
        if disable_temporal:
            temporal = torch.zeros(B, L, self.d_model, device=device)  # No position information
        else:
            temporal = self.temporal_encoder(seq_lengths)  # (B, L, d_model)
            # CRITICAL FIX: Zero positional encoding for masked positions to prevent position leakage
            if token_mask is not None:
                temporal = temporal * (~token_mask).unsqueeze(-1)  # Zero encoding for masked tokens
        
        # Spatial encoding (can be disabled for control)
        if disable_spatial:
            spatial = torch.zeros(B, L, self.d_model, device=device)  # No channel information
        else:
            spatial = self.spatial_encoder(channel_names)  # (B, d_model)
            spatial = spatial.unsqueeze(1).expand(B, L, self.d_model)  # (B, L, d_model)
            # CRITICAL FIX: Zero spatial encoding for masked positions
            if token_mask is not None:
                spatial = spatial * (~token_mask).unsqueeze(-1)  # Zero encoding for masked tokens

        # Combine: normalized windows + temporal + spatial encodings
        # All components are now on similar scale (normalized), preventing numerical instability
        x = token_emb + temporal + spatial  # (B, L, d_model)
        
        # DEBUG: Log embedding statistics to detect position leakage
        if not hasattr(self, '_debug_log_counter'):
            self._debug_log_counter = 0
        self._debug_log_counter += 1
        
        # Reduced debug logging (every 10000 forward passes)
        if token_mask is not None and self._debug_log_counter % 10000 == 0:
            import logging
            debug_logger = logging.getLogger(__name__)
            b = 0
            seq_len = seq_lengths[b].item()
            temporal_sliced = temporal[b, :seq_len] if not disable_temporal else torch.zeros(B, seq_len, self.d_model, device=device)
            spatial_sliced = spatial[b, :seq_len] if not disable_spatial else torch.zeros(B, seq_len, self.d_model, device=device)
            token_mask_sliced = token_mask[b, :seq_len]
            if token_mask_sliced.any():
                masked_temporal_norm = temporal_sliced[token_mask_sliced].norm().item() / token_mask_sliced.sum().item()
                masked_spatial_norm = spatial_sliced[token_mask_sliced].norm().item() / token_mask_sliced.sum().item()
                debug_logger.info(f"[DEBUG #{self._debug_log_counter}] Masked norms - Temporal: {masked_temporal_norm:.8f}, Spatial: {masked_spatial_norm:.8f}")
        
        # Mamba backbone processes the combined embeddings
        embeddings = self.backbone(x)  # (B, L, d_model)
        
        # Project back to window_length if needed (for reconstruction)
        if decode_to_signal and self.output_proj is not None:
            # Handle both ConvTranspose1d (Sequential) and Linear (single layer)
            if isinstance(self.output_proj, nn.Sequential):
                # ConvTranspose1d: needs (B*L, 1, d_model) shape
                embeddings_reshaped = embeddings.view(B * L, 1, self.d_model)  # (B*L, 1, d_model)
                reconstructed = self.output_proj(embeddings_reshaped)  # (B*L, 1, window_length)
                reconstructed = reconstructed.squeeze(1).view(B, L, self.window_length)  # (B, L, window_length)
            else:
                # Linear: needs (B*L, d_model) shape
                embeddings_reshaped = embeddings.view(B * L, self.d_model)  # (B*L, d_model)
                reconstructed = self.output_proj(embeddings_reshaped)  # (B*L, window_length)
                reconstructed = reconstructed.view(B, L, self.window_length)  # (B, L, window_length)
            return reconstructed
        elif decode_to_signal and self.output_proj is None:
            # Direct feed: d_model == window_length, output is already correct shape
            return embeddings  # (B, L, d_model) = (B, L, window_length)
        else:
            # Return embeddings (not signal space)
            return embeddings  # (B, L, d_model)



