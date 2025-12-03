from __future__ import annotations

from typing import Any, Dict, List, Optional
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import mlflow
import os
import math
import numpy as np
from pathlib import Path
import sys
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from contextlib import nullcontext

# Ensure 'src' package (under eeg_analysis/) is importable when running this file directly
CURRENT_FILE = Path(__file__).resolve()
EEG_ANALYSIS_ROOT = CURRENT_FILE.parents[2]  # .../eeg_analysis
if str(EEG_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(EEG_ANALYSIS_ROOT))

from src.models.mamba_eeg_model import MambaEEGModel
from src.utils.logger import setup_logger
from src.data.eeg_pretraining_dataset import EEGPretrainingDataset, collate_eeg_sequences

logger = setup_logger(__name__)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def compute_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask_exp: torch.Tensor,
    loss_function: str = "mse",
    variance_matching_weight: float = 0.0,
) -> torch.Tensor:
    """
    Compute reconstruction loss based on the specified loss function.
    
    Args:
        pred: Predicted signal (B, L, W)
        target: Target signal (B, L, W)
        mask_exp: Boolean mask (B, L, W)
        loss_function: Type of loss ("mse", "mae", "cosine", "huber", "combined")
    
    Returns:
        Scalar loss tensor
    """
    # For cosine similarity, we want to compute per-token (per window)
    # So we need to handle the mask differently: extract masked tokens, not flattened samples
    if loss_function in ["cosine", "combined"]:
        # Extract masked tokens: find which tokens are masked
        # mask_exp is (B, L, W), we need to find which (B, L) positions are masked
        if mask_exp.dim() == 3:  # (B, L, W)
            token_mask = mask_exp.any(dim=-1)  # (B, L) - True if any sample in token is masked
        else:  # (B, L) - already token-level
            token_mask = mask_exp
        
        # Extract masked tokens: (B, L, W) -> (N_masked_tokens, W)
        B, L, W = pred.shape
        pred_masked_tokens = pred[token_mask]  # (N_masked_tokens, W)
        target_masked_tokens = target[token_mask]  # (N_masked_tokens, W)
        
        if loss_function == "cosine":
            # Cosine similarity per token, then average
            # pred_masked_tokens: (N_masked_tokens, W), target_masked_tokens: (N_masked_tokens, W)
            
            # Check for NaN/Inf in inputs
            if torch.isnan(pred_masked_tokens).any() or torch.isnan(target_masked_tokens).any():
                # If NaN detected, return a large loss to signal the issue
                return torch.tensor(1e6, device=pred.device, requires_grad=True)
            
            # Compute norms with better numerical stability
            pred_norm = torch.norm(pred_masked_tokens, dim=-1, keepdim=True)  # (N_masked_tokens, 1)
            target_norm = torch.norm(target_masked_tokens, dim=-1, keepdim=True)  # (N_masked_tokens, 1)
            
            # Clamp norms to prevent division by zero (use larger epsilon for stability)
            pred_norm = torch.clamp(pred_norm, min=1e-6)
            target_norm = torch.clamp(target_norm, min=1e-6)
            
            # Compute cosine similarity
            dot_product = (pred_masked_tokens * target_masked_tokens).sum(dim=-1, keepdim=True)  # (N_masked_tokens, 1)
            cosine_sim_per_token = dot_product / (pred_norm * target_norm)  # (N_masked_tokens, 1)
            
            # Clamp cosine similarity to [-1, 1] to prevent numerical errors
            cosine_sim_per_token = torch.clamp(cosine_sim_per_token, min=-1.0, max=1.0)
            
            cosine_sim = cosine_sim_per_token.mean()
            
            # Check for NaN in cosine similarity
            if torch.isnan(cosine_sim):
                return torch.tensor(1e6, device=pred.device, requires_grad=True)
            
            cosine_loss = 1.0 - cosine_sim  # Minimize (1 - cosine) = maximize cosine
            
            # Variance matching penalty: Encourage predictions to match target variance
            if variance_matching_weight > 0.0:
                # Compute variance per token
                pred_var_per_token = pred_masked_tokens.var(dim=-1)  # (N_masked_tokens,)
                target_var_per_token = target_masked_tokens.var(dim=-1)  # (N_masked_tokens,)
                
                # Mean variance across tokens
                pred_mean_var = pred_var_per_token.mean()
                target_mean_var = target_var_per_token.mean()
                
                # Relative variance difference (scale-invariant)
                var_ratio = pred_mean_var / (target_mean_var + 1e-8)
                variance_penalty = torch.abs(var_ratio - 1.0)  # Penalty when ratio ≠ 1.0
                
                # Scale penalty by cosine loss magnitude to keep it proportional
                variance_penalty_scaled = variance_penalty * (cosine_loss.detach() + 1e-8)
                
                loss = cosine_loss + variance_matching_weight * variance_penalty_scaled
            else:
                loss = cosine_loss
        
        elif loss_function == "combined":
            # Cosine component (per token) - with NaN protection
            if torch.isnan(pred_masked_tokens).any() or torch.isnan(target_masked_tokens).any():
                return torch.tensor(1e6, device=pred.device, requires_grad=True)
            
            pred_norm = torch.norm(pred_masked_tokens, dim=-1, keepdim=True)
            target_norm = torch.norm(target_masked_tokens, dim=-1, keepdim=True)
            pred_norm = torch.clamp(pred_norm, min=1e-6)
            target_norm = torch.clamp(target_norm, min=1e-6)
            
            dot_product = (pred_masked_tokens * target_masked_tokens).sum(dim=-1, keepdim=True)
            cosine_sim_per_token = dot_product / (pred_norm * target_norm)
            cosine_sim_per_token = torch.clamp(cosine_sim_per_token, min=-1.0, max=1.0)
            cosine_sim = cosine_sim_per_token.mean()
            
            if torch.isnan(cosine_sim):
                cosine_loss = torch.tensor(1e6, device=pred.device, requires_grad=True)
            else:
                cosine_loss = 1.0 - cosine_sim
            
            # MAE component (per sample, across all masked samples)
            pred_masked_samples = pred[mask_exp]  # Flattened: (N_masked_samples,)
            target_masked_samples = target[mask_exp]  # Flattened: (N_masked_samples,)
            
            # Check for NaN in samples
            if torch.isnan(pred_masked_samples).any() or torch.isnan(target_masked_samples).any():
                mae_loss = torch.tensor(1e6, device=pred.device, requires_grad=True)
            else:
                mae_loss = torch.abs(pred_masked_samples - target_masked_samples).mean()
            
            # Combined base loss
            base_loss = 0.5 * cosine_loss + 0.5 * mae_loss
            
            # Variance matching penalty (if enabled)
            if variance_matching_weight > 0.0:
                pred_var_per_token = pred_masked_tokens.var(dim=-1)
                target_var_per_token = target_masked_tokens.var(dim=-1)
                pred_mean_var = pred_var_per_token.mean()
                target_mean_var = target_var_per_token.mean()
                var_ratio = pred_mean_var / (target_mean_var + 1e-8)
                variance_penalty = torch.abs(var_ratio - 1.0)
                variance_penalty_scaled = variance_penalty * (base_loss.detach() + 1e-8)
                loss = base_loss + variance_matching_weight * variance_penalty_scaled
            else:
                loss = base_loss
        
        return loss
    
    # For MSE, MAE, Huber: compute per-sample (standard approach)
    pred_masked = pred[mask_exp]  # Flattened: (N_masked_samples,)
    target_masked = target[mask_exp]  # Flattened: (N_masked_samples,)
    
    if loss_function == "mse":
        # Mean Squared Error (L2)
        diff = pred_masked - target_masked
        loss = diff.pow(2).mean()
    
    elif loss_function == "mae":
        # Mean Absolute Error (L1)
        diff = pred_masked - target_masked
        loss = torch.abs(diff).mean()
    
    elif loss_function == "huber":
        # Huber loss: combines MSE (small errors) and MAE (large errors)
        huber_loss = nn.HuberLoss(delta=1.0, reduction='mean')
        # Reshape for HuberLoss (expects (N, *) or (N,))
        pred_masked = pred_masked.unsqueeze(-1)  # (N_masked_samples, 1)
        target_masked = target_masked.unsqueeze(-1)  # (N_masked_samples, 1)
        loss = huber_loss(pred_masked, target_masked)
    
    else:
        raise ValueError(f"Unknown loss function: {loss_function}. Options: 'mse', 'mae', 'cosine', 'huber', 'combined'")
    
    return loss


def main() -> None:
    parser = argparse.ArgumentParser(description="EEG Mamba-2 pretraining")
    parser.add_argument("--config", type=str, required=True, help="Path to pretraining YAML config")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--distributed", action="store_true", help="Enable DDP (use with torchrun)")
    parser.add_argument("--backend", type=str, default="nccl", help="DDP backend (default: nccl)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # DDP setup
    distributed = bool(args.distributed) and torch.cuda.is_available()
    if distributed:
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
        else:
            # Fallback for some launchers
            local_rank = int(os.environ.get("RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=args.backend, init_method="env://")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        world_size = 1
        rank = 0
    is_main = (rank == 0)

    model = MambaEEGModel(
        d_model=int(cfg.get("d_model", 128)),
        num_layers=int(cfg.get("num_layers", 6)),
        window_length=int(cfg.get("window_length", 2048)),
        asa_path=cfg.get("asa_path"),
        dropout=0.1,
    ).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    dataset_path = cfg.get("dataset_path")
    if not dataset_path:
        raise ValueError("dataset_path must be set in config")

    ds = EEGPretrainingDataset(dataset_root=dataset_path, window_length=int(cfg.get("window_length", 2048)), split="train", val_ratio=float(cfg.get("val_ratio", 0.2)))
    val_ds = EEGPretrainingDataset(dataset_root=dataset_path, window_length=int(cfg.get("window_length", 2048)), split="val", val_ratio=float(cfg.get("val_ratio", 0.2)))
    
    # Collate function with masking style
    mask_ratio = float(cfg.get("mask_ratio", 0.2))
    masking_style = cfg.get("masking_style", "mae")  # Default to MAE-style
    mask_samples_within_token = bool(cfg.get("mask_samples_within_token", False))
    
    def _collate(batch: List[Dict[str, Any]]):
        return collate_eeg_sequences(
            batch, 
            mask_ratio=mask_ratio, 
            masking_style=masking_style,
            mask_samples_within_token=mask_samples_within_token,
            mask_replacement=mask_replacement,
        )

    per_device_batch = int(cfg.get("batch_size", 8))
    if distributed:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        loader = DataLoader(
            ds,
            batch_size=per_device_batch,
            sampler=sampler,
            num_workers=min(8, os.cpu_count() or 1),
            pin_memory=(device.type == "cuda"),
            collate_fn=_collate,
            drop_last=True,
        )
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        val_loader = DataLoader(
            val_ds,
            batch_size=per_device_batch,
            sampler=val_sampler,
            num_workers=min(8, os.cpu_count() or 1),
            pin_memory=(device.type == "cuda"),
            collate_fn=_collate,
            drop_last=False,
        )
    else:
        loader = DataLoader(
            ds,
            batch_size=per_device_batch,
            shuffle=True,
            num_workers=min(8, os.cpu_count() or 1),
            pin_memory=(device.type == "cuda"),
            collate_fn=_collate,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=per_device_batch,
            shuffle=False,
            num_workers=min(8, os.cpu_count() or 1),
            pin_memory=(device.type == "cuda"),
            collate_fn=_collate,
            drop_last=False,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-4)))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("amp", True)) and device.type == "cuda")
    epochs = int(cfg.get("epochs", 500))
    grad_clip = float(cfg.get("grad_clip_norm", 1.0))
    auto_grad_clip = bool(cfg.get("auto_grad_clip", True))  # Enable auto gradient clipping
    grad_clip_percentile = float(cfg.get("grad_clip_percentile", 95.0))  # Clip at 95th percentile
    
    # For auto gradient clipping: track gradient norm history
    grad_norm_history = []
    grad_history_size = 1000  # Keep last 1000 gradient norms
    
    save_dir = Path(cfg.get("save_dir", "./checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_loss = float("inf")
    
    # Early stopping
    patience = int(cfg.get("patience", 50))  # Lenient patience
    patience_counter = 0
    min_delta = float(cfg.get("min_delta", 1e-4))  # Minimum improvement to reset patience

    # Resume support
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        # DDP state dict handling
        model_state = ckpt["model"]
        if isinstance(model, DDP):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0))
        best_loss = float(ckpt.get("best_loss", best_loss))
        patience_counter = int(ckpt.get("patience_counter", 0))
        if is_main:
            logger.info(f"Resumed from {args.resume} at epoch {start_epoch}")

    # Control experiment flags (defined early for MLflow logging)
    disable_temporal = bool(cfg.get("disable_temporal_encoding", False))
    disable_spatial = bool(cfg.get("disable_spatial_encoding", False))
    
    # Reconstruction mode: Proper MAE (predict signal) vs embedding prediction
    use_signal_reconstruction = bool(cfg.get("reconstruct_signal_space", True))  # Default: True (proper MAE)
    
    # Loss function configuration
    loss_function = cfg.get("loss_function", "mse").lower()  # Default: MSE
    variance_matching_weight = float(cfg.get("variance_matching_weight", 0.0))  # Default: disabled
    mask_replacement = cfg.get("mask_replacement", "zeros").lower()  # Default: zeros
    if is_main:
        logger.info(f"Using loss function: {loss_function}")
        if variance_matching_weight > 0.0:
            logger.info(f"Variance matching weight: {variance_matching_weight}")
        logger.info(f"Mask replacement: {mask_replacement}")
    
    # Anti-position-only learning strategies
    prevent_position_only = bool(cfg.get("prevent_position_only_learning", False))
    position_reg_weight = float(cfg.get("position_regularization_weight", 0.1))
    shuffle_prob = float(cfg.get("shuffle_sequences_prob", 0.0))
    
    if is_main and prevent_position_only:
        logger.info(f"Anti-position-only learning enabled:")
        logger.info(f"  - Position regularization weight: {position_reg_weight}")
        logger.info(f"  - Sequence shuffling probability: {shuffle_prob}")
    
    # SIMPLIFIED ARCHITECTURE: Always reconstruct signal space (no encoder/decoder)
    target_encoder = None  # No encoder - always use raw signal as targets
    if is_main:
        logger.info("Simplified architecture: Direct feed to Mamba (no encoder/decoder)")
        logger.info("Always reconstructing signal space")
    
    # MLflow
    tracking_uri = cfg.get("mlflow_tracking_uri", "mlruns")
    experiment_name = cfg.get("mlflow_experiment", "eeg_pretraining_mamba2")
    if is_main:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        run_ctx = mlflow.start_run(run_name="mamba2_pretrain")
    else:
        run_ctx = nullcontext()
    
    with run_ctx:
        # Log config
        if is_main:
            mlflow.log_params({
                "d_model": cfg.get("d_model", 128),
                "num_layers": cfg.get("num_layers", 6),
                "lr": cfg.get("lr", 1e-4),
                "batch_size": cfg.get("batch_size", 8),
                "epochs": epochs,
                "mask_ratio": cfg.get("mask_ratio", 0.2),
                "masking_style": masking_style,
                "dataset_path": dataset_path,
                "world_size": world_size,
                "patience": patience,
                "min_delta": min_delta,
                "disable_temporal_encoding": disable_temporal,
                "disable_spatial_encoding": disable_spatial,
                "reconstruct_signal_space": use_signal_reconstruction,
                "prevent_position_only_learning": prevent_position_only,
                "position_regularization_weight": position_reg_weight if prevent_position_only else None,
                "shuffle_sequences_prob": shuffle_prob if prevent_position_only else None,
                "grad_clip_norm": grad_clip,
                "auto_grad_clip": auto_grad_clip,
                "grad_clip_percentile": grad_clip_percentile if auto_grad_clip else None,
            })
            mlflow.log_param("total_sequences", len(ds))
            mlflow.log_param("val_sequences", len(val_ds))

        def _tensor_stats(name: str, t: torch.Tensor) -> str:
            try:
                finite = torch.isfinite(t)
                numel = int(t.numel())
                finite_n = int(finite.sum().item())
                nan_n = int(torch.isnan(t).sum().item())
                inf_n = int(torch.isinf(t).sum().item())
                if finite_n > 0:
                    t_f = t[finite]
                    t_min = float(t_f.min().item())
                    t_max = float(t_f.max().item())
                    t_mean = float(t_f.mean().item())
                else:
                    t_min = float("nan")
                    t_max = float("nan")
                    t_mean = float("nan")
                return f"{name}: shape={tuple(t.shape)}, finite={finite_n}/{numel}, nan={nan_n}, inf={inf_n}, min={t_min:.6f}, max={t_max:.6f}, mean={t_mean:.6f}"
            except Exception as e:
                return f"{name}: <stats error: {e}>"

        for epoch in range(start_epoch, epochs):
            if distributed:
                sampler.set_epoch(epoch)  # type: ignore[attr-defined]
            model.train()
            total_loss = 0.0
            total_masked = 0
            for step, batch in enumerate(loader):
                windows = batch["windows"].to(device, non_blocking=True)               # (B, L, W)
                windows_masked = batch["windows_masked"].to(device, non_blocking=True) # (B, L, W)
                mask_bool = batch["mask_bool"].to(device, non_blocking=True)           # (B, L)
                seq_lengths = batch["seq_lengths"].to(device, non_blocking=True)       # (B,)
                channel_names = batch["channel_names"]
                
                # Anti-position-only learning: Randomly shuffle some sequences
                # This breaks position=time mapping, forcing model to learn from context
                if prevent_position_only and shuffle_prob > 0.0:
                    B, L, W = windows.shape
                    for b in range(B):
                        if torch.rand(1).item() < shuffle_prob:
                            # Shuffle this sequence's windows (but keep mask_bool aligned)
                            perm = torch.randperm(L, device=device)
                            windows[b] = windows[b][perm]
                            windows_masked[b] = windows_masked[b][perm]
                            mask_bool[b] = mask_bool[b][perm]
                            # Note: seq_lengths stays same (actual length doesn't change)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    # CRITICAL: Randomize token order to break Mamba's sequential position learning
                    # This prevents model from learning "first token", "second token", etc.
                    B, L, W = windows_masked.shape
                    perms = []  # Store permutations for unpermuting predictions
                    windows_masked_permuted = windows_masked.clone()
                    # Initialize mask_bool_permuted with correct shape (token-level or sample-level)
                    if mask_bool.dim() == 3:  # Sample-level mask (B, L, W)
                        mask_bool_permuted = mask_bool.clone()
                    else:  # Token-level mask (B, L)
                        mask_bool_permuted = mask_bool.clone()
                    target_permuted = None
                    
                    if prevent_position_only and shuffle_prob > 0.0:
                        for b in range(B):
                            seq_len = seq_lengths[b].item()
                            if seq_len > 1:
                                # Random permutation for this sequence
                                perm = torch.randperm(seq_len, device=device)
                                perms.append(perm)
                                
                                # Permute windows, mask, and target
                                windows_masked_permuted[b, :seq_len] = windows_masked[b, perm]
                                
                                # Handle both token-level (B, L) and sample-level (B, L, W) masks
                                if mask_bool.dim() == 3:  # Sample-level mask
                                    mask_bool_permuted[b, :seq_len, :] = mask_bool[b, perm, :]
                                else:  # Token-level mask
                                    mask_bool_permuted[b, :seq_len] = mask_bool[b, perm]
                                
                                if use_signal_reconstruction:
                                    if target_permuted is None:
                                        target_permuted = windows.clone()
                                    target_permuted[b, :seq_len] = windows[b, perm]
                            else:
                                perms.append(None)
                    else:
                        perms = [None] * B
                        target_permuted = windows if use_signal_reconstruction else None
                    
                    # Predict from permuted masked inputs
                    # CRITICAL: Pass mask_bool to prevent positional encoding for masked positions
                    # SIMPLIFIED: Always decode to signal (no encoder/decoder)
                    pred_permuted = (model.module if isinstance(model, DDP) else model)(
                        windows_masked=windows_masked_permuted,
                        channel_names=channel_names,
                        seq_lengths=seq_lengths,
                        disable_temporal=disable_temporal,
                        disable_spatial=disable_spatial,
                        decode_to_signal=True,  # Always true for simplified architecture
                        mask_bool=mask_bool_permuted,  # Zero positional encoding for masked positions
                    )  # (B, L, W) - always signal space
                    
                    # Check for NaN in model output immediately
                    skip_batch = False
                    if torch.isnan(pred_permuted).any():
                        if is_main:
                            logger.error(f"Model output contains NaN! NaN count: {torch.isnan(pred_permuted).sum().item()}")
                            logger.error(f"Input windows_masked stats: {_tensor_stats('windows_masked', windows_masked_permuted)}")
                            # Log more diagnostics
                            logger.error(f"Pred stats: {_tensor_stats('pred', pred_permuted)}")
                            # Check if input has extreme values
                            if windows_masked_permuted.abs().max() > 1e6:
                                logger.error(f"⚠️ Input has extreme values (max abs: {windows_masked_permuted.abs().max().item():.2e})")
                                logger.error("This may cause numerical instability. Consider checking data preprocessing.")
                        # Skip this batch to prevent training instability
                        skip_batch = True
                        # Replace NaN with zeros to prevent downstream errors
                        pred_permuted = torch.where(torch.isnan(pred_permuted), torch.zeros_like(pred_permuted), pred_permuted)
                    
                    # Unpermute predictions back to original order
                    pred = pred_permuted.clone()
                    for b in range(B):
                        if perms[b] is not None:
                            seq_len = seq_lengths[b].item()
                            inv_perm = torch.argsort(perms[b])  # Inverse permutation
                            pred[b, :seq_len] = pred_permuted[b, inv_perm]
                    
                    # Use permuted target if we permuted inputs
                    if target_permuted is not None:
                        windows_for_target = target_permuted
                    else:
                        windows_for_target = windows
                    
                    # Targets
                    if use_signal_reconstruction:
                        # Proper MAE: Target is actual signal (never changes!)
                        # Use permuted windows if we permuted inputs (to match prediction order)
                        target = windows_for_target  # (B, L, W) - raw signal (possibly permuted)
                        
                        # Per-sample normalization (mean=0, std=1)
                        # This preserves signal structure while removing scale differences
                        target_mean = target.mean(dim=-1, keepdim=True)  # (B, L, 1)
                        target_std = target.std(dim=-1, keepdim=True)  # (B, L, 1)
                        # Handle zero-variance windows (all samples same value)
                        # Use larger epsilon to prevent numerical issues
                        target_std = torch.clamp(target_std, min=1e-6)  # Clamp to prevent division by very small numbers
                        target = (target - target_mean) / target_std  # Normalized
                        
                        # Check for NaN after normalization
                        if torch.isnan(target).any():
                            if is_main:
                                logger.warning(f"Target contains NaN after normalization! NaN count: {torch.isnan(target).sum().item()}")
                                # Find which windows have NaN
                                nan_windows = torch.isnan(target).any(dim=-1)  # (B, L)
                                logger.warning(f"Windows with NaN: {nan_windows.sum().item()} out of {target.shape[0] * target.shape[1]}")
                            # Replace NaN with zeros
                            target = torch.where(torch.isnan(target), torch.zeros_like(target), target)
                    else:
                        # SIMPLIFIED: Always use raw signal (no encoder)
                        target = windows  # Raw signal
                    # Compute MSE over masked positions only (avoid NaN propagation by indexing)
                    # Use permuted mask if we permuted inputs
                    mask_bool_for_loss = mask_bool_permuted if (prevent_position_only and shuffle_prob > 0.0) else mask_bool
                    
                    # Handle both token-level (B, L) and sample-level (B, L, W) masks
                    if mask_bool_for_loss.dim() == 3:  # Sample-level mask (B, L, W)
                        # Sample-level masking: mask_bool is (B, L, W), pred/target are (B, L, W)
                        mask_exp = mask_bool_for_loss  # (B, L, W) - already correct shape
                    else:  # Token-level mask (B, L)
                        # Token-level masking: expand to match pred/target shape
                        mask_exp = mask_bool_for_loss.unsqueeze(-1).expand_as(pred)  # (B, L, W)
                    
                    # Control experiment: Check if model is learning dataset mean
                    if disable_temporal and disable_spatial:
                        # With no varying input, model should not be able to reduce loss
                        # If loss reduces, it's learning dataset mean (not sample-specific info)
                        
                        # Option 1: Center targets to prevent learning mean (RECOMMENDED)
                        # Subtract mean from both pred and target so model can't win by predicting mean
                        pred_masked = pred[mask_exp].view(-1, pred.shape[-1])  # (N_masked, D)
                        target_masked = target[mask_exp].view(-1, target.shape[-1])  # (N_masked, D)
                        
                        # Remove mean (model can't reduce loss by learning a constant)
                        target_mean = target_masked.mean(dim=0, keepdim=True)  # (1, D)
                        pred_centered = pred_masked - target_mean  # Center predictions
                        target_centered = target_masked - target_mean  # Center targets
                        
                        loss = (pred_centered - target_centered).pow(2).mean()
                        
                        # Log diagnostics (every 50 steps)
                        if is_main and step % 50 == 0:
                            pred_var = pred_masked.var().item()
                            target_var = target_masked.var().item()
                            pred_mean_norm = pred_masked.mean().item()
                            target_mean_norm = target_masked.mean().item()
                            #logger.info(f"[Control] pred_var={pred_var:.6f}, target_var={target_var:.6f}, "
                            #          f"pred_mean={pred_mean_norm:.6f}, target_mean={target_mean_norm:.6f}")
                    else:
                        # Normal training (configurable loss function)
                        if skip_batch:
                            # Skip batch if NaN was detected in model output
                            # Use a dummy loss that will be skipped by the non-finite check
                            loss = torch.tensor(float('inf'), device=device, requires_grad=False)
                        else:
                            loss = compute_reconstruction_loss(
                                pred=pred,
                                target=target,
                                mask_exp=mask_exp,
                                loss_function=loss_function,
                                variance_matching_weight=variance_matching_weight,
                            )
                            
                            # Check for NaN in loss
                            if torch.isnan(loss) or torch.isinf(loss):
                                if is_main:
                                    logger.error(f"Loss is NaN/Inf! Loss value: {loss.item()}")
                                    logger.error(f"Pred stats: {_tensor_stats('pred', pred)}")
                                    logger.error(f"Target stats: {_tensor_stats('target', target)}")
                                skip_batch = True
                        
                        # KEY METRICS DEBUGGING (every 20000 steps - reduced frequency, only most important)
                        if is_main and step % 20000 == 0:
                            with torch.no_grad():
                                model_eval = model.module if isinstance(model, DDP) else model
                                seq_len = seq_lengths[0].item()
                                
                                # SIMPLIFIED: Check raw window diversity (no encoder)
                                windows_sample = windows_masked[0, :min(5, seq_len)]  # (5, window_length)
                                
                                # Window similarity (raw signal)
                                if windows_sample.shape[0] >= 2:
                                    window_sims = []
                                    for i in range(windows_sample.shape[0]):
                                        for j in range(i+1, windows_sample.shape[0]):
                                            sim = F.cosine_similarity(windows_sample[i:i+1], windows_sample[j:j+1], dim=-1).item()
                                            window_sims.append(sim)
                                    avg_window_sim = np.mean(window_sims) if window_sims else 0.0
                                else:
                                    avg_window_sim = 0.0
                                
                                # Prediction diversity (reconstructed signal)
                                pred_sample = pred[0, :min(5, seq_len)]
                                if pred_sample.shape[0] >= 2:
                                    pred_sims = []
                                    for i in range(pred_sample.shape[0]):
                                        for j in range(i+1, pred_sample.shape[0]):
                                            sim = F.cosine_similarity(pred_sample[i:i+1], pred_sample[j:j+1], dim=-1).item()
                                            pred_sims.append(sim)
                                    avg_pred_sim = np.mean(pred_sims) if pred_sims else 0.0
                                else:
                                    avg_pred_sim = 0.0
                                
                                logger.info(f"\n{'='*80}")
                                logger.info(f"KEY METRICS [Step {step}] | Input window sim: {avg_window_sim:.3f} | Pred sim: {avg_pred_sim:.3f}")
                                logger.info(f"{'='*80}\n")
                        
                        # Decoder output statistics (debug logging every 5000 steps - reduced frequency)
                        if is_main and step % 5000 == 0 and use_signal_reconstruction:
                            pred_masked = pred[mask_exp]  # All masked predictions
                            if pred_masked.numel() > 0:
                                pred_var = pred_masked.var().item()
                                pred_mean = pred_masked.mean().item()
                                pred_std = pred_masked.std().item()
                                pred_min = pred_masked.min().item()
                                pred_max = pred_masked.max().item()
                                
                                # Also check variance across positions (should vary if learning signal)
                                # Handle both token-level and sample-level masks
                                if mask_bool_for_loss.dim() == 3:  # Sample-level mask
                                    # For sample-level masking, check tokens that have masked samples
                                    token_mask = mask_bool_for_loss.any(dim=2)  # (B, L)
                                else:
                                    token_mask = mask_bool_for_loss  # (B, L)
                                
                                B, L = token_mask.shape
                                pred_by_position = []
                                for b in range(B):
                                    seq_len = seq_lengths[b].item()
                                    masked_indices = token_mask[b, :seq_len].nonzero(as_tuple=True)[0]
                                    for idx in masked_indices:
                                        if idx < seq_len:
                                            pred_by_position.append(pred[b, idx].mean().item())
                                
                                if len(pred_by_position) > 1:
                                    pos_variance = np.var(pred_by_position)
                                    mlflow.log_metric("decoder_pred_var", pred_var, step=epoch * len(loader) + step)
                                    mlflow.log_metric("decoder_pred_mean", pred_mean, step=epoch * len(loader) + step)
                                    mlflow.log_metric("decoder_pred_std", pred_std, step=epoch * len(loader) + step)
                                    mlflow.log_metric("decoder_pred_min", pred_min, step=epoch * len(loader) + step)
                                    mlflow.log_metric("decoder_pred_max", pred_max, step=epoch * len(loader) + step)
                                    mlflow.log_metric("decoder_pred_pos_variance", pos_variance, step=epoch * len(loader) + step)
                                    
                                    # Target statistics for comparison
                                    target_masked = target[mask_exp]
                                    target_var = target_masked.var().item()
                                    target_mean = target_masked.mean().item()
                                    target_std = target_masked.std().item()
                                    mlflow.log_metric("target_var", target_var, step=epoch * len(loader) + step)
                                    mlflow.log_metric("target_mean", target_mean, step=epoch * len(loader) + step)
                                    mlflow.log_metric("target_std", target_std, step=epoch * len(loader) + step)
                                    
                                    # Make loss metrics stand out prominently
                                    logger.info(f"\n{'='*80}")
                                    logger.info(f"📊 LOSS METRICS [Step {step}]")
                                    logger.info(f"  Current Loss: {loss.item():.6f}")
                                    logger.info(f"  Predicted: var={pred_var:.6f}, mean={pred_mean:.6f}, std={pred_std:.6f}, pos_var={pos_variance:.6f}")
                                    logger.info(f"  Target:     var={target_var:.6f}, mean={target_mean:.6f}, std={target_std:.6f}")
                                    logger.info(f"{'='*80}\n")
                        
                        # Anti-position-only learning: Position regularization
                        # NOTE: With 100% masking, some position learning is INEVITABLE with sequential models
                        # The goal is to ensure that with partial masking, the model learns SIGNAL CONTENT,
                        # not just position. Position should be a helper feature, not the only feature.
                        # Penalize predictions that correlate too strongly with position
                        # IMPORTANT: Use ORIGINAL positions (before permutation) to detect position learning
                        if prevent_position_only and position_reg_weight > 0.0:
                            # Compute position for each masked token using ORIGINAL positions
                            # Handle both token-level and sample-level masks
                            if mask_bool.dim() == 3:  # Sample-level mask
                                # Convert to token-level: token is masked if ANY sample is masked
                                token_mask_orig = mask_bool.any(dim=2)  # (B, L)
                            else:
                                token_mask_orig = mask_bool  # (B, L)
                            
                            B, L = token_mask_orig.shape
                            positions = []
                            pred_masked_list = []
                            
                            for b in range(B):
                                seq_len = seq_lengths[b].item()
                                # Use original token_mask to get original positions
                                masked_indices = token_mask_orig[b, :seq_len].nonzero(as_tuple=True)[0]
                                
                                for idx in masked_indices:
                                    if idx < seq_len:
                                        # Normalized position in ORIGINAL sequence (0 to 1)
                                        # This detects if model learned original temporal position
                                        pos = float(idx) / float(seq_len)
                                        positions.append(pos)
                                        # Get prediction at this position (already unpermuted)
                                        if use_signal_reconstruction:
                                            # For signal space, use mean of window as summary
                                            pred_masked_list.append(pred[b, idx].mean().item())
                                        else:
                                            # For embedding space, use L2 norm as summary
                                            pred_masked_list.append(pred[b, idx].norm().item())
                            
                            if len(positions) > 10:  # Need enough samples for correlation
                                positions_t = torch.tensor(positions, device=device, dtype=torch.float32)
                                pred_summary_t = torch.tensor(pred_masked_list, device=device, dtype=torch.float32)
                                
                                # Normalize for correlation computation
                                positions_norm = (positions_t - positions_t.mean()) / (positions_t.std() + 1e-8)
                                pred_norm = (pred_summary_t - pred_summary_t.mean()) / (pred_summary_t.std() + 1e-8)
                                
                                # Compute correlation (position dependence)
                                position_corr = (positions_norm * pred_norm).mean()
                                
                                # AGGRESSIVE penalty: Use absolute correlation, scale by reconstruction loss
                                # This ensures penalty is proportional to reconstruction loss magnitude
                                base_loss_magnitude = loss.detach()  # Don't backprop through this
                                
                                # For mask_ratio=1.0, we want ZERO position correlation
                                # With 100% masking, ANY position dependence is wrong
                                if mask_ratio >= 0.95:  # Near 100% masking
                                    # EXTREME penalty: Force position independence
                                    # Penalty = |correlation| * weight * loss * large_factor
                                    position_penalty = torch.abs(position_corr) * position_reg_weight * base_loss_magnitude * 100.0
                                    # Also add quadratic term for very high correlations
                                    if torch.abs(position_corr) > 0.1:
                                        position_penalty = position_penalty + (position_corr.pow(2) * position_reg_weight * 1000.0 * base_loss_magnitude)
                                else:
                                    # Normal penalty for partial masking
                                    position_penalty = torch.abs(position_corr) * position_reg_weight * base_loss_magnitude
                                
                                # Additional constraint: Force predictions to have low variance across positions
                                # If model learns position, predictions will vary by position
                                # Penalize high variance in predictions across positions
                                pred_by_position = {}  # position -> [predictions]
                                for i, pos in enumerate(positions):
                                    pos_bin = int(pos * 10)  # Bin positions into 10 groups
                                    if pos_bin not in pred_by_position:
                                        pred_by_position[pos_bin] = []
                                    pred_by_position[pos_bin].append(pred_masked_list[i])
                                
                                if len(pred_by_position) > 1:
                                    # Compute variance of means across position bins
                                    position_means = [np.mean(pred_by_position[b]) for b in sorted(pred_by_position.keys())]
                                    position_variance = np.var(position_means)
                                    position_variance_penalty = torch.tensor(position_variance, device=device, dtype=torch.float32) * position_reg_weight * base_loss_magnitude * 50.0
                                    position_penalty = position_penalty + position_variance_penalty
                                
                                loss = loss + position_penalty
                                
                                # Log position correlation periodically
                                if is_main and step % 50 == 0:
                                    mlflow.log_metric("position_correlation", float(position_corr.item()), step=epoch * len(loader) + step)
                                    mlflow.log_metric("position_penalty", float(position_penalty.item()), step=epoch * len(loader) + step)
                                    mlflow.log_metric("position_penalty_ratio", float((position_penalty / (loss.detach() + 1e-8)).item()), step=epoch * len(loader) + step)

                # If loss is non-finite, dump diagnostics and skip batch
                if not torch.isfinite(loss):
                    if is_main:
                        logger.error("Non-finite loss detected; dumping diagnostics for current batch")
                        logger.error(_tensor_stats("windows", windows))
                        logger.error(_tensor_stats("windows_masked", windows_masked))
                        logger.error(f"seq_lengths: {seq_lengths.tolist()}")
                        logger.error(f"seq_lengths: {seq_lengths.tolist()}")
                        logger.error(f"mask_bool: shape={tuple(mask_bool.shape)}, masked_tokens={int(mask_bool.sum().item())}")
                        logger.error(f"Total valid tokens (sum of seq_lengths): {int(seq_lengths.sum().item())}")
                        if seq_lengths.sum().item() > 0:
                            logger.error(f"Actual mask ratio: {mask_bool.sum().item() / seq_lengths.sum().item():.3f}")
                        logger.error(f"Total tokens (sum of seq_lengths): {int(seq_lengths.sum().item())}")
                        logger.error(f"Mask ratio: {mask_bool.sum().item() / seq_lengths.sum().item():.3f}")
                        logger.error(_tensor_stats("pred", pred))
                        logger.error(_tensor_stats("target", target))
                        logger.error(_tensor_stats("diff", diff))
                        if 'masked_diff' in locals() and masked_diff.numel() > 0:
                            logger.error(_tensor_stats("masked_diff", masked_diff))
                        else:
                            logger.error("masked_diff: empty (no masked positions selected)")
                        # Check if pred contains NaN before loss computation
                        if torch.isnan(pred).any():
                            logger.error(f"pred contains NaN at positions: {torch.isnan(pred).nonzero()[:10]}")
                        if torch.isnan(target).any():
                            logger.error(f"target contains NaN at positions: {torch.isnan(target).nonzero()[:10]}")
                    continue

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    
                    # Compute current gradient norm (without clipping)
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
                    
                    # Auto gradient clipping: adapt threshold based on gradient history
                    if auto_grad_clip and len(grad_norm_history) >= 100:  # Need history
                        # Compute adaptive threshold from percentile
                        adaptive_threshold = torch.tensor(grad_norm_history).quantile(grad_clip_percentile / 100.0).item()
                        effective_clip = min(grad_clip, adaptive_threshold) if grad_clip > 0 else adaptive_threshold
                    else:
                        effective_clip = grad_clip if grad_clip > 0 else float('inf')
                    
                    # Apply clipping with effective threshold
                    if effective_clip < float('inf'):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=effective_clip)
                    
                    # Update gradient history
                    grad_norm_history.append(float(total_norm))
                    if len(grad_norm_history) > grad_history_size:
                        grad_norm_history.pop(0)
                    
                    # Log gradient statistics periodically
                    if is_main and step % 50 == 0:
                        mlflow.log_metric("grad_norm", float(total_norm), step=epoch * len(loader) + step)
                        mlflow.log_metric("grad_clip_threshold", float(effective_clip), step=epoch * len(loader) + step)
                        mlflow.log_metric("grad_clipped", float(total_norm > effective_clip), step=epoch * len(loader) + step)
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    
                    # Compute current gradient norm (without clipping)
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
                    
                    # Auto gradient clipping: adapt threshold based on gradient history
                    if auto_grad_clip and len(grad_norm_history) >= 100:  # Need history
                        # Compute adaptive threshold from percentile
                        adaptive_threshold = torch.tensor(grad_norm_history).quantile(grad_clip_percentile / 100.0).item()
                        effective_clip = min(grad_clip, adaptive_threshold) if grad_clip > 0 else adaptive_threshold
                    else:
                        effective_clip = grad_clip if grad_clip > 0 else float('inf')
                    
                    # Apply clipping with effective threshold
                    if effective_clip < float('inf'):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=effective_clip)
                    
                    # Update gradient history
                    grad_norm_history.append(float(total_norm))
                    if len(grad_norm_history) > grad_history_size:
                        grad_norm_history.pop(0)
                    
                    # Log gradient statistics periodically
                    if is_main and step % 50 == 0:
                        mlflow.log_metric("grad_norm", float(total_norm), step=epoch * len(loader) + step)
                        mlflow.log_metric("grad_clip_threshold", float(effective_clip), step=epoch * len(loader) + step)
                        mlflow.log_metric("grad_clipped", float(total_norm > effective_clip), step=epoch * len(loader) + step)
                    
                    optimizer.step()

                total_loss += float(loss.detach().cpu().item())
                total_masked += int(mask_bool.sum().cpu().item())

                if is_main and (step + 1) % 50 == 0:
                    mlflow.log_metric("train_loss_step", float(loss.detach().cpu().item()), step=epoch * len(loader) + step)

            # All-reduce for proper global averages
            if distributed:
                loss_tensor = torch.tensor([total_loss, len(loader), total_masked], device=device, dtype=torch.float64)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                total_loss = float(loss_tensor[0].item())
                loader_count = max(1.0, float(loss_tensor[1].item()))
                total_masked = int(loss_tensor[2].item())
            else:
                loader_count = float(max(1, len(loader)))

            avg_loss = total_loss / loader_count
            
            # Validation
            model.eval()
            val_total_loss = 0.0
            val_total_masked = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    windows = val_batch["windows"].to(device, non_blocking=True)
                    windows_masked = val_batch["windows_masked"].to(device, non_blocking=True)
                    mask_bool = val_batch["mask_bool"].to(device, non_blocking=True)
                    seq_lengths = val_batch["seq_lengths"].to(device, non_blocking=True)
                    channel_names = val_batch["channel_names"]
                    
                    pred = (model.module if isinstance(model, DDP) else model)(
                        windows_masked=windows_masked,
                        channel_names=channel_names,
                        seq_lengths=seq_lengths,
                        disable_temporal=disable_temporal,
                        disable_spatial=disable_spatial,
                        decode_to_signal=use_signal_reconstruction,
                    )
                    # Targets
                    if use_signal_reconstruction:
                        target = windows  # Raw signal
                        # Normalize per-sample
                        target_mean = target.mean(dim=-1, keepdim=True)
                        target_std = target.std(dim=-1, keepdim=True) + 1e-8
                        target = (target - target_mean) / target_std
                    else:
                        # SIMPLIFIED: No encoder - targets are always raw signal
                        target = windows  # Raw signal
                    
                    # Handle both token-level (B, L) and sample-level (B, L, W) masks
                    if mask_bool.dim() == 3:  # Sample-level mask (B, L, W)
                        mask_exp = mask_bool  # (B, L, W) - already correct shape
                    else:  # Token-level mask (B, L)
                        mask_exp = mask_bool.unsqueeze(-1).expand_as(pred)  # (B, L, W)
                    
                    # Use same centered loss for validation in control mode
                    if disable_temporal and disable_spatial:
                        pred_masked = pred[mask_exp].view(-1, pred.shape[-1])
                        target_masked = target[mask_exp].view(-1, target.shape[-1])
                        target_mean = target_masked.mean(dim=0, keepdim=True)
                        pred_centered = pred_masked - target_mean
                        target_centered = target_masked - target_mean
                        val_loss = (pred_centered - target_centered).pow(2).mean()
                    else:
                        # Normal validation (configurable loss function)
                        val_loss = compute_reconstruction_loss(
                            pred=pred,
                            target=target,
                            mask_exp=mask_exp,
                            loss_function=loss_function,
                            variance_matching_weight=variance_matching_weight,
                        )
                    
                    if torch.isfinite(val_loss):
                        val_total_loss += float(val_loss.detach().cpu().item())
                        val_total_masked += int(mask_bool.sum().cpu().item())
            
            # All-reduce validation
            if distributed:
                val_tensor = torch.tensor([val_total_loss, len(val_loader), val_total_masked], device=device, dtype=torch.float64)
                dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
                val_total_loss = float(val_tensor[0].item())
                val_loader_count = max(1.0, float(val_tensor[1].item()))
                val_total_masked = int(val_tensor[2].item())
            else:
                val_loader_count = float(max(1, len(val_loader)))
            
            avg_val_loss = val_total_loss / val_loader_count
            
            if is_main:
                mlflow.log_metric("train_loss", avg_loss, step=epoch)
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                mlflow.log_metric("masked_tokens", total_masked, step=epoch)
                mlflow.log_metric("val_masked_tokens", val_total_masked, step=epoch)
                # Make epoch loss metrics stand out prominently
                logger.info(f"\n{'='*80}")
                logger.info(f"📈 EPOCH {epoch+1}/{epochs} SUMMARY")
                logger.info(f"  Train Loss: {avg_loss:.6f}")
                logger.info(f"  Val Loss:   {avg_val_loss:.6f}")
                logger.info(f"  Masked Tokens: {total_masked}")
                logger.info(f"{'='*80}\n")

            # Checkpoint
            is_best = avg_val_loss < (best_loss - min_delta)
            if is_best:
                best_loss = avg_val_loss
                patience_counter = 0  # Reset patience
                if is_main:
                    logger.info(f"🎯 New best val_loss: {best_loss:.6f}")
            else:
                patience_counter += 1
                if is_main:
                    logger.info(f"No improvement for {patience_counter}/{patience} epochs")
            ckpt = {
                "model": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
                "optimizer": optimizer.state_dict(),
                "scaler": (scaler.state_dict() if scaler is not None else None),
                "epoch": epoch + 1,
                "best_loss": best_loss,
                "patience_counter": patience_counter,
                "config": cfg,
            }
            last_path = save_dir / "mamba2_eeg_pretrained_last.pt"
            best_path = save_dir / "mamba2_eeg_pretrained.pt"
            if is_main:
                torch.save(ckpt, last_path)
                if is_best:
                    torch.save(ckpt, best_path)
                    logger.info(f"Saved best checkpoint at epoch {epoch+1} (val_loss={best_loss:.6f})")
                        
                if (epoch + 1) % 10 == 0:
                    mlflow.log_artifact(str(last_path), artifact_path="checkpoints")
            
            # Early stopping check
            if patience_counter >= patience:
                if is_main:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs (patience={patience})")
                    mlflow.log_metric("early_stopped", 1)
                    mlflow.log_metric("stopped_epoch", epoch+1)
                break
        
        # Synchronize all ranks before final operations
        if distributed:
            dist.barrier()
        
        # After training completes, register the best model once
        if is_main:
            logger.info(f"Training completed. Best val_loss: {best_loss:.6f}")
            
            # Log final best checkpoint as artifact
            if best_path.exists():
                mlflow.log_artifact(str(best_path), artifact_path="final_model")
            
            # Register best model in MLflow Model Registry
            d_model = cfg.get("d_model", 128)
            num_layers = cfg.get("num_layers", 6)
            mask_ratio = cfg.get("mask_ratio", 0.2)
            mask_style_short = "mae" if masking_style == "mae" else "bert"
            model_name = f"mamba2_eeg_d{d_model}_l{num_layers}_m{int(mask_ratio*100)}_{mask_style_short}"
            
            try:
                import mlflow.pytorch as mlf_pytorch
                # Load the best checkpoint to register
                best_ckpt = torch.load(best_path, map_location=device)
                model_to_register = model.module if isinstance(model, DDP) else model
                model_to_register.load_state_dict(best_ckpt["model"])
                
                # Register model
                mlf_pytorch.log_model(
                    pytorch_model=model_to_register,
                    artifact_path="model",
                    registered_model_name=model_name,
                )
                logger.info(f"✓ Registered model: {model_name} (final_val_loss={best_loss:.6f})")
            except Exception as e:
                logger.warning(f"Failed to register model in MLflow: {e}")

        # Cleanup DDP
        if distributed:
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()



