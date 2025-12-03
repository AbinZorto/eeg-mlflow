#!/usr/bin/env python3
"""
Diagnostic Script: Analyze What the Model Learns with 100% Masking

This script helps determine whether a model trained with mask_ratio=1.0 is:
1. Learning useful EEG representations (context-based learning)
2. Memorizing positional statistics (position-based learning)
3. Just learning the dataset mean (trivial learning)

Usage:
    python diagnose_100pct_masking.py --checkpoint path/to/model.pt
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys

# Add eeg_analysis to path
EEG_ANALYSIS_ROOT = Path(__file__).parent / "eeg_analysis"
sys.path.insert(0, str(EEG_ANALYSIS_ROOT))

from src.models.mamba_eeg_model import MambaEEGModel
from src.data.eeg_pretraining_dataset import EEGPretrainingDataset, collate_eeg_sequences


def get_masked_token_indices(mask_bool, seq_len=None):
    """
    Extract masked token indices from mask_bool.
    
    Handles both token-level masks (L) or (B, L) and sample-level masks (L, W) or (B, L, W).
    For sample-level masks, a token is considered masked if any sample within it is masked.
    
    Args:
        mask_bool: Mask tensor of shape (L), (B, L), (L, W), or (B, L, W)
        seq_len: Optional sequence length to slice to
    
    Returns:
        Tensor of masked token indices (1D)
    """
    # Convert to token-level mask first
    if mask_bool.dim() == 3:  # Sample-level mask (B, L, W)
        token_mask = (mask_bool.sum(dim=-1) > 0)  # (B, L)
        # Take first batch if batch dimension exists
        if token_mask.shape[0] == 1:
            token_mask = token_mask[0]  # (L,)
        else:
            token_mask = token_mask[0]  # Take first batch
    elif mask_bool.dim() == 2:
        # Check if last dim is large (likely window_length ~2048) vs small (likely batch)
        if mask_bool.shape[-1] > 100:  # Likely (L, W) - sample-level mask
            token_mask = (mask_bool.sum(dim=-1) > 0)  # (L,)
        else:  # Likely (B, L) - token-level mask
            token_mask = mask_bool[0] if mask_bool.shape[0] > 0 else mask_bool  # (L,)
    else:  # Token-level mask (L,)
        token_mask = mask_bool
    
    # Apply seq_len if provided
    if seq_len is not None and token_mask.shape[0] > seq_len:
        token_mask = token_mask[:seq_len]
    
    # Return indices
    return token_mask.nonzero(as_tuple=True)[0]


def get_token_mask_from_mask_bool(mask_bool, seq_len=None):
    """
    Convert mask_bool to token-level mask.
    
    Args:
        mask_bool: Mask tensor of shape (L), (B, L), (L, W), or (B, L, W)
        seq_len: Optional sequence length to slice to
    
    Returns:
        Token-level mask (1D or 2D) where True indicates masked token
    """
    # Convert to token-level mask
    if mask_bool.dim() == 3:  # Sample-level mask (B, L, W)
        token_mask = (mask_bool.sum(dim=-1) > 0)  # (B, L)
    elif mask_bool.dim() == 2:
        # Check if last dim is large (likely window_length ~2048) vs small (likely batch)
        if mask_bool.shape[-1] > 100:  # Likely (L, W) - sample-level mask
            token_mask = (mask_bool.sum(dim=-1) > 0)  # (L,)
        else:  # Likely (B, L) - token-level mask
            token_mask = mask_bool
    else:  # Token-level mask (L,)
        token_mask = mask_bool
    
    # Apply seq_len if provided
    if seq_len is not None:
        if token_mask.dim() == 2:
            token_mask = token_mask[:, :seq_len]
        else:
            token_mask = token_mask[:seq_len]
    
    return token_mask


def analyze_model_predictions(model, dataloader, device, num_samples=100):
    """
    Analyze what the model predicts for masked tokens.
    
    Checks:
    1. Prediction variance across different samples
    2. Prediction consistency for same position
    3. Correlation between predictions and positions
    """
    model.eval()
    
    predictions = []
    positions = []
    channels = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
                
            windows_masked = batch["windows_masked"].to(device)
            mask_bool = batch["mask_bool"].to(device)
            seq_lengths = batch["seq_lengths"].to(device)
            channel_names = batch["channel_names"]
            
            # Get predictions (SIMPLIFIED ARCHITECTURE: always outputs signal space)
            try:
                pred = model(
                    windows_masked=windows_masked,
                    channel_names=channel_names,
                    seq_lengths=seq_lengths,
                    decode_to_signal=True,  # Simplified architecture always outputs signal space
                    mask_bool=mask_bool,  # Pass mask to zero positional encoding for masked positions
                )  # (B, L, W) - always signal space in simplified architecture
            except TypeError as e:
                # Try without mask_bool (old model)
                try:
                    pred = model(
                        windows_masked=windows_masked,
                        channel_names=channel_names,
                        seq_lengths=seq_lengths,
                        decode_to_signal=True,  # Always signal space
                    )  # (B, L, W)
                except TypeError:
                    # Old model without decode_to_signal parameter - simplified arch always outputs signal
                    pred = model(
                        windows_masked=windows_masked,
                        channel_names=channel_names,
                        seq_lengths=seq_lengths,
                    )  # (B, L, W) - simplified architecture
            
            # DEBUG: Log detailed embedding inspection for first batch
            if i == 0:
                print("\n" + "=" * 60)
                print("DETAILED EMBEDDING INSPECTION (First Batch)")
                print("=" * 60)
                b = 0
                seq_len = seq_lengths[b].item()
                # Handle both token-level and sample-level masks
                token_mask = get_token_mask_from_mask_bool(mask_bool[b], seq_len=seq_len)
                masked_indices = token_mask.nonzero(as_tuple=True)[0].cpu().tolist()
                unmasked_indices = (~token_mask).nonzero(as_tuple=True)[0].cpu().tolist()
                
                print(f"\nSequence length: {seq_len}")
                print(f"Masked positions: {len(masked_indices)} (e.g., {masked_indices[:5]})")
                print(f"Unmasked positions: {len(unmasked_indices)} (e.g., {unmasked_indices[:5]})")
                
                # INSPECT MODEL INTERNALS: Check what embeddings look like BEFORE backbone
                print(f"\n" + "-" * 60)
                print("MODEL INTERNAL EMBEDDING INSPECTION")
                print("-" * 60)
                
                # Hook into model to inspect intermediate values
                model.eval()
                with torch.no_grad():
                    # FLEXIBLE ARCHITECTURE: Normalize and project windows to d_model
                    windows_slice = windows_masked[b:b+1, :seq_len]  # (1, seq_len, window_length)
                    token_emb_mean = windows_slice.mean(dim=-1, keepdim=True)  # (1, seq_len, 1)
                    token_emb_std = windows_slice.std(dim=-1, keepdim=True)  # (1, seq_len, 1)
                    token_emb_std = torch.clamp(token_emb_std, min=1e-6)  # Prevent division by very small numbers
                    windows_normalized = (windows_slice - token_emb_mean) / token_emb_std  # (1, seq_len, window_length)
                    
                    # Apply input projection if exists (to match forward pass)
                    if hasattr(model, 'input_proj') and model.input_proj is not None:
                        B_proj, L_proj, W_proj = windows_normalized.shape
                        windows_reshaped = windows_normalized.view(B_proj * L_proj, 1, W_proj)  # (B*L, 1, window_length)
                        token_emb = model.input_proj(windows_reshaped)  # (B*L, 1, d_model)
                        token_emb = token_emb.squeeze(1).view(B_proj, L_proj, model.d_model)  # (1, seq_len, d_model)
                    else:
                        # Direct feed: already correct shape
                        token_emb = windows_normalized  # (1, seq_len, window_length) = (1, seq_len, d_model)
                    
                    # Get temporal encoding
                    temporal = None
                    try:
                        temporal_full = model.temporal_encoder(seq_lengths[b:b+1])
                        temporal = temporal_full[:, :seq_len, :]  # Slice to actual length
                        # Apply mask if model supports it
                        if mask_bool is not None:
                            # Handle both (1, L) and (1, L, W) masks
                            mask_slice = mask_bool[b:b+1, :seq_len]
                            if mask_slice.dim() == 3:  # (1, L, W) - sample-level
                                token_mask_batch = (mask_slice.sum(dim=-1) > 0)  # (1, L)
                            else:  # (1, L) - token-level
                                token_mask_batch = mask_slice
                            temporal_masked = temporal * (~token_mask_batch).unsqueeze(-1)
                        else:
                            temporal_masked = temporal
                    except:
                        temporal_masked = torch.zeros_like(token_emb)
                        temporal = temporal_masked
                    
                    # Get spatial encoding
                    spatial = None
                    d_model = token_emb.shape[-1]  # Get d_model from token_emb shape
                    try:
                        spatial = model.spatial_encoder([channel_names[b]])
                        spatial = spatial.unsqueeze(1).expand(1, seq_len, d_model)
                        # Apply mask if model supports it
                        if mask_bool is not None:
                            # Handle both (1, L) and (1, L, W) masks
                            mask_slice = mask_bool[b:b+1, :seq_len]
                            if mask_slice.dim() == 3:  # (1, L, W) - sample-level
                                token_mask_batch = (mask_slice.sum(dim=-1) > 0)  # (1, L)
                            else:  # (1, L) - token-level
                                token_mask_batch = mask_slice
                            spatial_masked = spatial * (~token_mask_batch).unsqueeze(-1)
                        else:
                            spatial_masked = spatial
                    except:
                        spatial_masked = torch.zeros(1, seq_len, d_model, device=token_emb.device)
                        spatial = spatial_masked
                    
                    # Combined embedding before backbone
                    combined = token_emb + temporal_masked + spatial_masked
                    
                    print(f"\nToken Embeddings (from masked windows):")
                    for pos in masked_indices[:3]:
                        tok_vals = token_emb[0, pos, :5].cpu().numpy()
                        tok_norm = token_emb[0, pos].norm().item()
                        print(f"  Pos {pos} (masked):   {[f'{v:.4f}' for v in tok_vals]}... (norm={tok_norm:.6f})")
                    
                    print(f"\nTemporal Encodings (positional):")
                    for pos in masked_indices[:3]:
                        temp_vals = temporal_masked[0, pos, :5].cpu().numpy()
                        temp_norm = temporal_masked[0, pos].norm().item()
                        if temporal is not None:
                            temp_orig_norm = temporal[0, pos].norm().item()
                        else:
                            temp_orig_norm = 0.0
                        print(f"  Pos {pos} (masked):   {[f'{v:.4f}' for v in temp_vals]}... (norm={temp_norm:.8f}, orig_norm={temp_orig_norm:.6f})")
                    
                    print(f"\nSpatial Encodings (channel):")
                    for pos in masked_indices[:3]:
                        spat_vals = spatial_masked[0, pos, :5].cpu().numpy()
                        spat_norm = spatial_masked[0, pos].norm().item()
                        if spatial is not None:
                            spat_orig_norm = spatial[0, pos].norm().item()
                        else:
                            spat_orig_norm = 0.0
                        print(f"  Pos {pos} (masked):   {[f'{v:.4f}' for v in spat_vals]}... (norm={spat_norm:.8f}, orig_norm={spat_orig_norm:.6f})")
                    
                    print(f"\nCombined Embeddings (before backbone):")
                    for pos in masked_indices[:3]:
                        comb_vals = combined[0, pos, :5].cpu().numpy()
                        comb_norm = combined[0, pos].norm().item()
                        print(f"  Pos {pos} (masked):   {[f'{v:.4f}' for v in comb_vals]}... (norm={comb_norm:.6f})")
                    
                    # Check if temporal/spatial are actually zero for masked positions
                    if len(masked_indices) > 0:
                        masked_temporal_norm = temporal_masked[0, masked_indices].norm().item() / len(masked_indices) if len(masked_indices) > 0 else 0.0
                        masked_spatial_norm = spatial_masked[0, masked_indices].norm().item() / len(masked_indices) if len(masked_indices) > 0 else 0.0
                        print(f"\nAverage norms for masked positions:")
                        print(f"  Temporal: {masked_temporal_norm:.10f} (should be ~0.0)")
                        print(f"  Spatial:  {masked_spatial_norm:.10f} (should be ~0.0)")
                    
                    print("-" * 60)
                
                # Show sample predictions at different positions
                print(f"\nFinal Predictions (after backbone - simplified architecture, no decoder):")
                for pos in masked_indices[:3]:
                    pred_vals = pred[b, pos, :10].detach().cpu().numpy()
                    print(f"  Position {pos} (masked):   {[f'{v:.4f}' for v in pred_vals]}")
                if len(unmasked_indices) > 0:
                    for pos in unmasked_indices[:3]:
                        pred_vals = pred[b, pos, :10].detach().cpu().numpy()
                        print(f"  Position {pos} (unmasked): {[f'{v:.4f}' for v in pred_vals]}")
                
                # Check if predictions vary by position
                if len(masked_indices) >= 5:
                    sample_positions = masked_indices[:5]
                    preds_at_positions = [pred[b, pos].detach().cpu().numpy() for pos in sample_positions]
                    # Compute pairwise similarity
                    print(f"\nPairwise similarity between masked positions:")
                    for i, pos1 in enumerate(sample_positions):
                        for j, pos2 in enumerate(sample_positions[i+1:], start=i+1):
                            sim = np.dot(preds_at_positions[i], preds_at_positions[j]) / (
                                np.linalg.norm(preds_at_positions[i]) * np.linalg.norm(preds_at_positions[j]) + 1e-8
                            )
                            print(f"  Pos {pos1} vs Pos {pos2}: {sim:.4f}")
                
                print("=" * 60 + "\n")
            
            # Extract masked positions
            for b in range(pred.shape[0]):
                seq_len = seq_lengths[b].item()
                masked_indices = get_masked_token_indices(mask_bool[b], seq_len=seq_len)
                
                for idx in masked_indices:
                    if idx < seq_len:
                        pred_emb = pred[b, idx].cpu().numpy()
                        position = float(idx) / float(seq_len)  # Normalized position
                        
                        predictions.append(pred_emb)
                        positions.append(position)
                        channels.append(channel_names[b])
    
    predictions = np.array(predictions)  # (N, D)
    positions = np.array(positions)  # (N,)
    
    print("\n" + "=" * 60)
    print("MODEL PREDICTION ANALYSIS")
    print("=" * 60)
    
    # 1. Overall prediction statistics
    print(f"\nSamples analyzed: {len(predictions)}")
    print(f"Embedding dimension: {predictions.shape[1]}")
    print(f"\nPrediction statistics:")
    print(f"  Mean: {predictions.mean():.6f}")
    print(f"  Std:  {predictions.std():.6f}")
    print(f"  Min:  {predictions.min():.6f}")
    print(f"  Max:  {predictions.max():.6f}")
    
    # 2. Variance analysis
    per_sample_var = np.var(predictions, axis=1)  # Variance within each embedding
    print(f"\nVariance per embedding:")
    print(f"  Mean variance: {per_sample_var.mean():.6f}")
    print(f"  Std variance:  {per_sample_var.std():.6f}")
    
    # 3. Position correlation
    # Check if predictions correlate with temporal position
    position_correlation = np.corrcoef(predictions.T, positions)[:-1, -1]
    mean_pos_corr = np.abs(position_correlation).mean()
    
    print(f"\nPosition correlation:")
    print(f"  Mean |correlation| with position: {mean_pos_corr:.4f}")
    
    if mean_pos_corr > 0.3:
        print("  ⚠️  Strong position dependence - model is learning positional patterns")
    elif mean_pos_corr > 0.1:
        print("  ⚠️  Moderate position dependence")
    else:
        print("  ✅  Low position dependence - predictions vary independently of position")
    
    # 4. Channel-specific patterns
    unique_channels = list(set(channels))
    if len(unique_channels) > 1:
        print(f"\nChannel-specific analysis ({len(unique_channels)} channels):")
        channel_means = {}
        for ch in unique_channels[:5]:  # Show first 5 channels
            ch_mask = np.array([c == ch for c in channels])
            if ch_mask.sum() > 0:
                ch_mean = predictions[ch_mask].mean()
                ch_std = predictions[ch_mask].std()
                channel_means[ch] = ch_mean
                print(f"  {ch}: mean={ch_mean:.4f}, std={ch_std:.4f}")
        
        # Check if different channels have different means
        if len(channel_means) > 1:
            means = np.array(list(channel_means.values()))
            channel_diversity = means.std()
            print(f"\nChannel diversity (std of channel means): {channel_diversity:.4f}")
            
            if channel_diversity > 0.1:
                print("  ✅  Different channels have distinct predictions")
            else:
                print("  ⚠️  All channels predict similar values - may be learning just dataset mean")
    
    # 5. Clustering analysis
    # Check if predictions cluster (indicating memorization of specific patterns)
    # vs. being uniformly distributed (indicating fitting to mean)
    pca_variance = np.var(predictions, axis=0)
    explained_variance_ratio = pca_variance / pca_variance.sum()
    top3_variance = explained_variance_ratio[:3].sum()
    
    print(f"\nRepresentation diversity:")
    print(f"  Top 3 dimensions explain {top3_variance*100:.1f}% of variance")
    
    if top3_variance > 0.9:
        print("  ⚠️  Low diversity - predictions cluster in low-dimensional space")
        print("      Model may be learning simple positional mapping")
    else:
        print("  ✅  High diversity - predictions span high-dimensional space")
        print("      Model is learning complex representations")
    
    return {
        "predictions": predictions,
        "positions": positions,
        "channels": channels,
        "mean_pos_correlation": mean_pos_corr,
    }


def compare_predicted_vs_gt_statistics(model, dataloader, device, num_samples=100, decode_to_signal=True):
    """
    Compare predicted signal statistics (mean, std) to ground truth statistics.
    
    This verifies if the model learns to reconstruct actual EEG signal characteristics,
    not just positional patterns.
    
    IMPORTANT: Accounts for baseline similarity - token vectors for the same window
    across different channels are naturally more similar than vectors from different windows.
    
    Args:
        model: Trained model
        dataloader: DataLoader with batches
        device: torch device
        num_samples: Number of batches to analyze
        decode_to_signal: Whether model outputs signal space (True) or embeddings (False)
    """
    """
    Compare predicted signal statistics (mean, std) to ground truth statistics.
    
    This verifies if the model learns to reconstruct actual signal characteristics,
    not just positional patterns.
    
    Args:
        model: Trained model
        dataloader: DataLoader with batches
        device: torch device
        num_samples: Number of batches to analyze
        decode_to_signal: Whether model outputs signal space (True) or embeddings (False)
    """
    model.eval()
    
    print("\n" + "=" * 60)
    print("PREDICTED vs GROUND TRUTH STATISTICS COMPARISON")
    print("=" * 60)
    
    # Collect statistics per channel
    channel_stats = {}  # channel_name -> {pred_mean: [], pred_std: [], gt_mean: [], gt_std: []}
    
    # NEW: Track windows to compute baseline cross-channel similarity
    window_to_channels = {}  # (file_idx, window_idx) -> {channel: (pred, gt)}
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            
            windows = batch["windows"].to(device)  # (B, L, W) - ground truth
            windows_masked = batch["windows_masked"].to(device)  # (B, L, W) - masked input
            mask_bool = batch["mask_bool"].to(device)  # (B, L)
            seq_lengths = batch["seq_lengths"].to(device)
            channel_names = batch["channel_names"]
            
            # Get predictions
            if decode_to_signal:
                try:
                    pred = model(
                        windows_masked=windows_masked,
                        channel_names=channel_names,
                        seq_lengths=seq_lengths,
                        decode_to_signal=True,
                        mask_bool=mask_bool,  # Pass mask to zero positional encoding
                    )  # (B, L, W) - reconstructed signal
                except TypeError:
                    # Old model without mask_bool parameter
                    pred = model(
                        windows_masked=windows_masked,
                        channel_names=channel_names,
                        seq_lengths=seq_lengths,
                        decode_to_signal=True,
                    )  # (B, L, W) - reconstructed signal
            else:
                # If embeddings, we can't directly compare to signal stats
                print("⚠️  Model outputs embeddings, not signal. Cannot compare signal statistics.")
                print("    Set decode_to_signal=True in model forward or use signal reconstruction.")
                return None
            
            B, L, W = pred.shape
            
            # Compute statistics for each masked window
            for b in range(B):
                channel = channel_names[b]
                seq_len = seq_lengths[b].item()
                
                if channel not in channel_stats:
                    channel_stats[channel] = {
                        "pred_mean": [],
                        "pred_std": [],
                        "gt_mean": [],
                        "gt_std": [],
                        "pred_mean_per_window": [],
                        "pred_std_per_window": [],
                        "gt_mean_per_window": [],
                        "gt_std_per_window": [],
                    }
                
                # Extract masked positions
                masked_indices = get_masked_token_indices(mask_bool[b], seq_len=seq_len)
                
                for idx in masked_indices:
                    if idx < seq_len:
                        # Predicted window (reconstructed signal) - already normalized
                        pred_window = pred[b, idx].cpu().numpy()  # (W,) - 2048 samples
                        
                        # Ground truth window (actual signal) - needs normalization to match training
                        gt_window_raw = windows[b, idx].cpu().numpy()  # (W,) - 2048 samples (raw)
                        
                        # Normalize ground truth the same way as in training (per-window normalization)
                        gt_mean_raw = gt_window_raw.mean()
                        gt_std_raw = gt_window_raw.std() + 1e-8
                        gt_window = (gt_window_raw - gt_mean_raw) / gt_std_raw  # Normalized
                        
                        # After normalization, both should have mean≈0, std≈1
                        # So we compare the RAW statistics (before normalization) to see if model
                        # learns to predict correct signal characteristics
                        # AND we compare normalized signal patterns (correlation)
                        
                        # Raw statistics (what the model should learn to predict)
                        pred_mean_raw = pred_window.mean()  # Predicted normalized mean (should be ~0)
                        pred_std_raw = pred_window.std()   # Predicted normalized std (should be ~1)
                        gt_mean_raw_val = gt_mean_raw       # GT raw mean (varies by window)
                        gt_std_raw_val = gt_std_raw         # GT raw std (varies by window)
                        
                        # Normalized statistics (should be ~0 and ~1 for both)
                        pred_mean = pred_window.mean()
                        pred_std = pred_window.std()
                        gt_mean = gt_window.mean()
                        gt_std = gt_window.std()
                        
                        # Signal pattern correlation (compare normalized waveforms)
                        pattern_corr = np.corrcoef(pred_window, gt_window)[0, 1]
                        pattern_corr = pattern_corr if not np.isnan(pattern_corr) else 0.0
                        
                        # Store statistics
                        # Note: After normalization, means are ~0 and stds are ~1, so we focus on pattern correlation
                        channel_stats[channel]["pred_mean"].append(pred_mean_raw)  # Store raw pred stats
                        channel_stats[channel]["pred_std"].append(pred_std_raw)
                        channel_stats[channel]["gt_mean"].append(gt_mean_raw_val)  # Store raw GT stats
                        channel_stats[channel]["gt_std"].append(gt_std_raw_val)
                        
                        # Also store pattern correlation (normalized waveform similarity)
                        channel_stats[channel].setdefault("pattern_corr", []).append(pattern_corr)
                        
                        # Store normalized windows for cross-channel similarity analysis
                        channel_stats[channel].setdefault("pred_windows", []).append(pred_window)
                        channel_stats[channel].setdefault("gt_windows", []).append(gt_window)
                        channel_stats[channel].setdefault("window_indices", []).append((i, b, idx))  # (batch_idx, sample_idx, window_idx)
                        
                        channel_stats[channel]["pred_mean_per_window"].append(pred_mean)
                        channel_stats[channel]["pred_std_per_window"].append(pred_std)
                        channel_stats[channel]["gt_mean_per_window"].append(gt_mean)
                        channel_stats[channel]["gt_std_per_window"].append(gt_std)
    
    # NEW: Compute baseline cross-channel similarity in ground truth
    # Sample random pairs of GT windows and compute similarity
    print("\n" + "=" * 60)
    print("BASELINE CROSS-CHANNEL SIMILARITY ANALYSIS")
    print("=" * 60)
    print("\nComputing baseline similarity between ground truth windows...")
    
    # Collect all GT windows
    all_gt_windows = []
    all_pred_windows = []
    for stats in channel_stats.values():
        if "gt_windows" in stats:
            all_gt_windows.extend(stats["gt_windows"])
            all_pred_windows.extend(stats["pred_windows"])
    
    if len(all_gt_windows) < 100:
        print("⚠️  Not enough samples for baseline analysis")
        baseline_gt_similarity = 0.0
        baseline_pred_similarity = 0.0
    else:
        # Sample random pairs of GT windows
        n_samples = min(1000, len(all_gt_windows) // 2)
        gt_similarities = []
        pred_similarities = []
        
        np.random.seed(42)
        indices = np.random.choice(len(all_gt_windows), size=n_samples * 2, replace=False)
        
        for j in range(0, len(indices), 2):
            idx1, idx2 = indices[j], indices[j+1]
            gt1, gt2 = all_gt_windows[idx1], all_gt_windows[idx2]
            pred1, pred2 = all_pred_windows[idx1], all_pred_windows[idx2]
            
            # Cosine similarity
            gt_sim = np.dot(gt1, gt2) / (np.linalg.norm(gt1) * np.linalg.norm(gt2) + 1e-8)
            pred_sim = np.dot(pred1, pred2) / (np.linalg.norm(pred1) * np.linalg.norm(pred2) + 1e-8)
            
            gt_similarities.append(gt_sim)
            pred_similarities.append(pred_sim)
        
        baseline_gt_similarity = np.mean(gt_similarities)
        baseline_pred_similarity = np.mean(pred_similarities)
        
        print(f"\nBaseline GT similarity (random window pairs): {baseline_gt_similarity:.4f} ± {np.std(gt_similarities):.4f}")
        print(f"Predicted similarity (random window pairs):    {baseline_pred_similarity:.4f} ± {np.std(pred_similarities):.4f}")
        print(f"\nDifference: {baseline_pred_similarity - baseline_gt_similarity:.4f}")
        
        if abs(baseline_pred_similarity - baseline_gt_similarity) < 0.1:
            print("✅ Model maintains similar baseline similarity structure as GT")
        else:
            print("⚠️  Model's baseline similarity differs from GT")
    
    # Analyze and report
    print(f"\nAnalyzed {sum(len(stats['pred_mean']) for stats in channel_stats.values())} masked windows")
    print(f"Across {len(channel_stats)} channels\n")
    
    # Overall statistics
    all_pred_means = []
    all_pred_stds = []
    all_gt_means = []
    all_gt_stds = []
    all_pattern_corrs = []
    
    for stats in channel_stats.values():
        all_pred_means.extend(stats["pred_mean"])
        all_pred_stds.extend(stats["pred_std"])
        all_gt_means.extend(stats["gt_mean"])
        all_gt_stds.extend(stats["gt_std"])
        if "pattern_corr" in stats:
            all_pattern_corrs.extend(stats["pattern_corr"])
    
    all_pred_means = np.array(all_pred_means)
    all_pred_stds = np.array(all_pred_stds)
    all_gt_means = np.array(all_gt_means)
    all_gt_stds = np.array(all_gt_stds)
    all_pattern_corrs = np.array(all_pattern_corrs) if all_pattern_corrs else np.array([])
    
    print("=" * 60)
    print("OVERALL STATISTICS (All Channels Combined)")
    print("=" * 60)
    print(f"\n⚠️  NOTE: Model outputs normalized signal (mean≈0, std≈1 per window)")
    print(f"    Comparing raw statistics shows what model learns about signal scale.\n")
    
    print(f"Raw Mean (per window, before normalization):")
    print(f"  Predicted (model output): {all_pred_means.mean():.6f} ± {all_pred_means.std():.6f}")
    print(f"  Ground Truth (raw): {all_gt_means.mean():.6f} ± {all_gt_means.std():.6f}")
    print(f"  Difference: {np.abs(all_pred_means.mean() - all_gt_means.mean()):.6f}")
    mean_corr = np.corrcoef(all_pred_means, all_gt_means)[0, 1] if len(all_pred_means) > 1 else 0.0
    mean_corr = mean_corr if not np.isnan(mean_corr) else 0.0
    print(f"  Correlation: {mean_corr:.4f}")
    
    print(f"\nRaw Std (per window, before normalization):")
    print(f"  Predicted (model output): {all_pred_stds.mean():.6f} ± {all_pred_stds.std():.6f}")
    print(f"  Ground Truth (raw): {all_gt_stds.mean():.6f} ± {all_gt_stds.std():.6f}")
    print(f"  Difference: {np.abs(all_pred_stds.mean() - all_gt_stds.mean()):.6f}")
    std_corr = np.corrcoef(all_pred_stds, all_gt_stds)[0, 1] if len(all_pred_stds) > 1 else 0.0
    std_corr = std_corr if not np.isnan(std_corr) else 0.0
    print(f"  Correlation: {std_corr:.4f}")
    
    # Pattern correlation (most important - compares normalized waveforms)
    if len(all_pattern_corrs) > 0:
        print(f"\nPattern Correlation (normalized waveform similarity):")
        print(f"  Mean correlation: {all_pattern_corrs.mean():.6f} ± {all_pattern_corrs.std():.6f}")
        print(f"  Median correlation: {np.median(all_pattern_corrs):.6f}")
        print(f"  % with corr > 0.5: {(all_pattern_corrs > 0.5).sum() / len(all_pattern_corrs) * 100:.1f}%")
        print(f"  % with corr > 0.7: {(all_pattern_corrs > 0.7).sum() / len(all_pattern_corrs) * 100:.1f}%")
    
    # Per-channel statistics
    print("\n" + "=" * 60)
    print("PER-CHANNEL STATISTICS")
    print("=" * 60)
    
    # Sort channels by number of samples
    sorted_channels = sorted(
        channel_stats.items(),
        key=lambda x: len(x[1]["pred_mean"]),
        reverse=True
    )
    
    print(f"\n{'Channel':<12} {'Windows':<10} {'Pattern Corr':<15} {'Mean Corr':<12} {'Std Corr':<12}")
    print("-" * 60)
    
    channel_results = {}
    
    for channel, stats in sorted_channels[:20]:  # Show top 20 channels
        n_windows = len(stats["pred_mean"])
        if n_windows == 0:
            continue
        
        pred_means = np.array(stats["pred_mean"])
        pred_stds = np.array(stats["pred_std"])
        gt_means = np.array(stats["gt_mean"])
        gt_stds = np.array(stats["gt_std"])
        
        # Compute correlations
        mean_corr = np.corrcoef(pred_means, gt_means)[0, 1] if len(pred_means) > 1 else 0.0
        std_corr = np.corrcoef(pred_stds, gt_stds)[0, 1] if len(pred_stds) > 1 else 0.0
        
        # Handle NaN correlations
        mean_corr = mean_corr if not np.isnan(mean_corr) else 0.0
        std_corr = std_corr if not np.isnan(std_corr) else 0.0
        
        # Pattern correlation (most important)
        pattern_corrs = np.array(stats.get("pattern_corr", []))
        pattern_corr_mean = pattern_corrs.mean() if len(pattern_corrs) > 0 else 0.0
        
        print(f"{channel:<12} {n_windows:<10} {pattern_corr_mean:<15.4f} {mean_corr:<12.4f} {std_corr:<12.4f}")
        
        channel_results[channel] = {
            "n_windows": n_windows,
            "pattern_corr": pattern_corr_mean,
            "mean_corr": mean_corr,
            "std_corr": std_corr,
            "pred_mean_mean": pred_means.mean(),
            "pred_std_mean": pred_stds.mean(),
            "gt_mean_mean": gt_means.mean(),
            "gt_std_mean": gt_stds.mean(),
        }
    
    # Summary interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    overall_mean_corr = mean_corr
    overall_std_corr = std_corr
    overall_pattern_corr = all_pattern_corrs.mean() if len(all_pattern_corrs) > 0 else 0.0
    
    print(f"\nRaw Statistics Correlation:")
    print(f"  Mean Correlation: {overall_mean_corr:.4f}")
    print(f"  Std Correlation:  {overall_std_corr:.4f}")
    
    if len(all_pattern_corrs) > 0:
        print(f"\nPattern Correlation (KEY METRIC):")
        print(f"  Mean Pattern Correlation: {overall_pattern_corr:.4f}")
        print(f"  (Compares normalized waveform similarity)")
    
    # Primary interpretation based on pattern correlation
    # Account for baseline similarity: GT windows have inherent similarity structure
    if len(all_pattern_corrs) > 0:
        print(f"\nPattern Correlation Analysis:")
        print(f"  Mean: {overall_pattern_corr:.4f}")
        print(f"  Note: Low correlation may indicate model learns window-level patterns")
        print(f"        rather than exact per-channel reconstruction")
        
        if overall_pattern_corr > 0.7:
            print("\n✅ EXCELLENT: Model accurately reconstructs signal patterns")
            print("   High pattern correlation means predicted waveforms match ground truth")
            print("   Model is learning actual EEG signal structure")
        elif overall_pattern_corr > 0.5:
            print("\n⚠️  MODERATE: Model partially reconstructs signal patterns")
            print("   Some waveform similarity but not perfect")
            print("   Model may be learning some signal structure")
        elif overall_pattern_corr > 0.3:
            print("\n⚠️  POOR: Model has weak signal pattern reconstruction")
            print("   Low pattern correlation suggests model is not learning signal content")
            print("   May be learning positional patterns instead")
        elif overall_pattern_corr > -0.1:
            print("\n⚠️  VERY LOW: Pattern correlation near zero")
            print("   BUT: Consider baseline similarity structure")
            if 'baseline_gt_similarity' in locals() and abs(baseline_pred_similarity - baseline_gt_similarity) < 0.1:
                print("   ✅ Model maintains GT similarity structure (window-level learning)")
                print("   Model may be learning window-level patterns correctly")
                print("   Low per-window correlation doesn't necessarily mean poor learning")
            else:
                print("   Model likely learning positional patterns, not signal content")
        else:
            print("\n❌ NEGATIVE: Pattern correlation is negative")
            print("   Model predictions are anti-correlated with ground truth")
            print("   This suggests fundamental learning problem")
    
    # Secondary interpretation based on raw statistics correlation
    # (Note: Model outputs normalized values, so raw stats correlation may be low)
    if overall_mean_corr > 0.5 or overall_std_corr > 0.5:
        print("\n📊 Note: Raw statistics show some correlation")
        print("   Model may be learning to distinguish windows with different scales")
    else:
        print("\n📊 Note: Raw statistics correlation is low")
        print("   This is expected since model outputs normalized values")
        print("   Focus on pattern correlation for signal reconstruction quality")
    
    # Check if model predicts constant values
    if len(all_pred_stds) > 0 and all_pred_stds.std() < 0.01:
        print("\n⚠️  WARNING: Predicted std values are nearly constant")
        print("   Model may be predicting dataset mean, not sample-specific values")
    
    if len(all_pred_means) > 0 and all_pred_means.std() < 0.01:
        print("\n⚠️  WARNING: Predicted mean values are nearly constant")
        print("   Model may be predicting constant offset, not signal-specific means")
    
    result = {
        "channel_results": channel_results,
        "overall_mean_corr": overall_mean_corr,
        "overall_std_corr": overall_std_corr,
        "overall_pattern_corr": overall_pattern_corr,
        "all_pred_means": all_pred_means,
        "all_pred_stds": all_pred_stds,
        "all_gt_means": all_gt_means,
        "all_gt_stds": all_gt_stds,
        "all_pattern_corrs": all_pattern_corrs,
    }
    
    # Add baseline similarity if computed
    if 'baseline_gt_similarity' in locals():
        result["baseline_gt_similarity"] = baseline_gt_similarity
        result["baseline_pred_similarity"] = baseline_pred_similarity
    
    return result


def analyze_multi_channel_synchronization(dataloader, masking_style, num_samples=50):
    """
    Analyze if multi-channel masking is working correctly.
    
    For multi-channel masking, channels from the same file should have
    synchronized masks (same positions masked).
    
    This function groups channels by file_path and verifies synchronization.
    """
    if masking_style != "multi_channel":
        return None
    
    print("\n" + "=" * 60)
    print("MULTI-CHANNEL MASKING SYNCHRONIZATION ANALYSIS")
    print("=" * 60)
    print("\nAnalyzing mask synchronization by file...")
    print("(Channels from same file should have synchronized masks)")
    
    # Group channels by file_path
    file_to_masks: Dict[str, List[torch.Tensor]] = {}  # file_path -> [masks]
    file_to_channels: Dict[str, List[str]] = {}  # file_path -> [channel_names]
    
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break
        
        mask_bool = batch["mask_bool"]
        channel_names = batch["channel_names"]
        file_paths = batch.get("file_paths", [])
        
        # If file_paths not available, fall back to batch-level analysis
        if not file_paths or all(not fp for fp in file_paths):
            print("\n⚠️  file_paths not available in batch, using batch-level analysis")
            print("    (This is less accurate but still informative)")
            return _analyze_batch_level_synchronization(dataloader, num_samples)
        
        B = mask_bool.shape[0]
        
        # Extract masks and group by file
        for b in range(B):
            fp = file_paths[b] if b < len(file_paths) else f"unknown_{i}_{b}"
            channel = channel_names[b]
            
            if mask_bool.dim() == 3:  # Sample-level (B, L, W)
                # Convert to token-level for comparison
                channel_mask = (mask_bool[b].sum(dim=-1) > 0)  # (L,)
            else:  # Token-level (B, L)
                channel_mask = mask_bool[b]  # (L,)
            
            if fp not in file_to_masks:
                file_to_masks[fp] = []
                file_to_channels[fp] = []
            
            file_to_masks[fp].append(channel_mask)
            file_to_channels[fp].append(channel)
    
    # Analyze synchronization per file
    file_sync_scores = []
    files_with_multiple_channels = 0
    
    for fp, masks in file_to_masks.items():
        if len(masks) < 2:
            continue  # Need at least 2 channels from same file
        
        files_with_multiple_channels += 1
        
        # Compare masks pairwise within file
        file_similarities = []
        for j in range(len(masks)):
            for k in range(j + 1, len(masks)):
                mask1 = masks[j]
                mask2 = masks[k]
                
                # Align lengths
                min_len = min(len(mask1), len(mask2))
                if min_len == 0:
                    continue
                    
                mask1_aligned = mask1[:min_len]
                mask2_aligned = mask2[:min_len]
                
                # Compute similarity (fraction of positions that match)
                similarity = (mask1_aligned == mask2_aligned).float().mean().item()
                file_similarities.append(similarity)
        
        if file_similarities:
            file_avg_sync = np.mean(file_similarities)
            file_sync_scores.append(file_avg_sync)
    
    if file_sync_scores:
        mean_sync = np.mean(file_sync_scores)
        std_sync = np.std(file_sync_scores)
        
        print(f"\nFiles with multiple channels: {files_with_multiple_channels}")
        print(f"Pairwise mask comparisons: {sum(len(masks) * (len(masks) - 1) // 2 for masks in file_to_masks.values() if len(masks) >= 2)}")
        print(f"Mean mask synchronization: {mean_sync:.4f} ± {std_sync:.4f}")
        print("(1.0 = perfect synchronization, ~0.5 = random/independent)")
        
        if mean_sync > 0.95:
            print("\n✅ EXCELLENT: Masks are highly synchronized across channels from same file")
            print("   Multi-channel masking is working correctly!")
        elif mean_sync > 0.8:
            print("\n⚠️  GOOD: Masks show high synchronization")
            print("   Multi-channel masking appears to be working")
            print("   Some variation may be due to sequence length differences")
        elif mean_sync > 0.6:
            print("\n⚠️  MODERATE: Masks show some synchronization")
            print("   Multi-channel masking may be partially working")
            print("   Check sequence lengths and mask generation logic")
        else:
            print("\n❌ LOW: Masks are not synchronized")
            print("   Multi-channel masking may not be working correctly")
            print("   Check that masking_style='multi_channel' in config")
        
        return {
            "mean_synchronization": mean_sync,
            "std_synchronization": std_sync,
            "files_analyzed": files_with_multiple_channels,
            "comparisons": len(file_sync_scores),
        }
    else:
        print("\n⚠️  No files with multiple channels found for analysis")
        print("    (This is expected if batches don't contain multiple channels from same file)")
        return None


def _analyze_batch_level_synchronization(dataloader, num_samples=50):
    """Fallback: Analyze synchronization within batches."""
    batch_mask_similarities = []
    batches_analyzed = 0
    
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break
        
        mask_bool = batch["mask_bool"]
        B = mask_bool.shape[0]
        
        if B < 2:
            continue
        
        batches_analyzed += 1
        masks = []
        for b in range(B):
            if mask_bool.dim() == 3:
                channel_mask = (mask_bool[b].sum(dim=-1) > 0)
            else:
                channel_mask = mask_bool[b]
            masks.append(channel_mask)
        
        for j in range(len(masks)):
            for k in range(j + 1, len(masks)):
                min_len = min(len(masks[j]), len(masks[k]))
                if min_len == 0:
                    continue
                similarity = (masks[j][:min_len] == masks[k][:min_len]).float().mean().item()
                batch_mask_similarities.append(similarity)
    
    if batch_mask_similarities:
        mean_sync = np.mean(batch_mask_similarities)
        std_sync = np.std(batch_mask_similarities)
        print(f"\nBatches analyzed: {batches_analyzed}")
        print(f"Mean mask synchronization: {mean_sync:.4f} ± {std_sync:.4f}")
        return {
            "mean_synchronization": mean_sync,
            "std_synchronization": std_sync,
            "batches_analyzed": batches_analyzed,
        }
    return None


def compare_masked_vs_unmasked_learning(model, dataloader, device, num_samples=50):
    """
    Compare model predictions for same sequence with different masking patterns.
    
    If model is learning from context, different masking should give different predictions.
    If model is learning from position only, same position should give same prediction.
    """
    model.eval()
    
    print("\n" + "=" * 60)
    print("CONTEXT SENSITIVITY TEST")
    print("=" * 60)
    print("\nTesting if predictions depend on masked context...")
    
    position_consistency = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            
            windows = batch["windows"].to(device)
            seq_lengths = batch["seq_lengths"].to(device)
            channel_names = batch["channel_names"]
            
            B, L, W = windows.shape
            
            for b in range(min(B, 2)):  # Check first 2 in batch
                # Test position L//2 with different masking patterns
                test_pos = L // 2
                if test_pos >= seq_lengths[b]:
                    continue
                
                # Get the actual sequence length for this sample
                actual_len = seq_lengths[b].item()
                
                # Mask pattern 1: Mask test position only (trim to actual length)
                masked1 = windows[b:b+1, :actual_len, :].clone()
                masked1[:, test_pos, :] = 0.0
                mask_bool1 = torch.zeros(1, actual_len, dtype=torch.bool, device=device)
                mask_bool1[0, test_pos] = True  # Only test position is masked
                
                # Mask pattern 2: Mask everything (100%)
                masked2 = torch.zeros_like(windows[b:b+1, :actual_len, :])
                mask_bool2 = torch.ones(1, actual_len, dtype=torch.bool, device=device)  # All masked
                
                # Get predictions for same position with different masking
                # Simplified architecture always outputs signal space, so both should use same format
                try:
                    pred1 = model(
                        windows_masked=masked1,
                        channel_names=[channel_names[b]],
                        seq_lengths=torch.tensor([actual_len], device=device),
                        decode_to_signal=True,  # Simplified architecture always outputs signal space
                        mask_bool=mask_bool1,  # Pass mask
                    )
                    pred2 = model(
                        windows_masked=masked2,
                        channel_names=[channel_names[b]],
                        seq_lengths=torch.tensor([actual_len], device=device),
                        decode_to_signal=True,  # Simplified architecture always outputs signal space
                        mask_bool=mask_bool2,  # Pass mask
                    )
                except TypeError:
                    # Try without mask_bool
                    try:
                        pred1 = model(
                            windows_masked=masked1,
                            channel_names=[channel_names[b]],
                            seq_lengths=torch.tensor([actual_len], device=device),
                            decode_to_signal=True,  # Simplified architecture always outputs signal space
                        )
                        pred2 = model(
                            windows_masked=masked2,
                            channel_names=[channel_names[b]],
                            seq_lengths=torch.tensor([actual_len], device=device),
                            decode_to_signal=True,  # Simplified architecture always outputs signal space
                        )
                    except TypeError:
                        # Old model without decode_to_signal parameter - always outputs signal space
                        pred1 = model(
                            windows_masked=masked1,
                            channel_names=[channel_names[b]],
                            seq_lengths=torch.tensor([actual_len], device=device),
                        )
                        pred2 = model(
                            windows_masked=masked2,
                            channel_names=[channel_names[b]],
                            seq_lengths=torch.tensor([actual_len], device=device),
                        )
                
                # Compare predictions at test position
                pred1_vec = pred1[0, test_pos].cpu().numpy()
                pred2_vec = pred2[0, test_pos].cpu().numpy()
                
                # Cosine similarity
                similarity = np.dot(pred1_vec, pred2_vec) / (
                    np.linalg.norm(pred1_vec) * np.linalg.norm(pred2_vec) + 1e-8
                )
                position_consistency.append(similarity)
    
    if position_consistency:
        mean_consistency = np.mean(position_consistency)
        print(f"\nMean prediction similarity: {mean_consistency:.4f}")
        print("(Comparing predictions for same position with different masking)")
        
        if mean_consistency > 0.95:
            print("\n⚠️  VERY HIGH CONSISTENCY")
            print("    Predictions are nearly identical regardless of masking context")
            print("    → Model is learning POSITION-BASED patterns")
            print("    → NOT learning from signal context")
        elif mean_consistency > 0.8:
            print("\n⚠️  HIGH CONSISTENCY")
            print("    Predictions are similar across different masking")
            print("    → Model relies heavily on positional information")
        else:
            print("\n✅  LOW CONSISTENCY")
            print("    Predictions vary with different masking contexts")
            print("    → Model is learning from surrounding context")
    
    return mean_consistency


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose what a model learns with 100% masking"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/home/abin/eeg-mlflow/eeg_analysis/secondarydata/raw",
        help="Path to pretraining data"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples to analyze"
    )
    parser.add_argument(
        "--decode-to-signal",
        action="store_true",
        help="Force decode to signal space for statistics comparison (auto-detected from checkpoint config if not specified)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DIAGNOSING MODEL LEARNING WITH 100% MASKING")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Extract config
    config = checkpoint.get("config", {})
    d_model = config.get("d_model", 128)
    num_layers = config.get("num_layers", 6)
    window_length = config.get("window_length", 2048)
    mask_ratio = config.get("mask_ratio", 1.0)
    masking_style = config.get("masking_style", "mae")
    mask_samples_within_token = config.get("mask_samples_within_token", False)
    # Use command-line arg if provided, otherwise use config
    reconstruct_signal_space = args.decode_to_signal if args.decode_to_signal else config.get("reconstruct_signal_space", True)
    
    print(f"\nModel config:")
    print(f"  d_model: {d_model}")
    print(f"  num_layers: {num_layers}")
    print(f"  window_length: {window_length}")
    print(f"  mask_ratio: {mask_ratio}")
    print(f"  masking_style: {masking_style}")
    if masking_style == "multi_channel":
        print(f"  mask_samples_within_token: {mask_samples_within_token}")
    print(f"  reconstruct_signal_space: {reconstruct_signal_space}")
    
    if mask_ratio < 1.0:
        print(f"\n⚠️  Warning: This model was trained with mask_ratio={mask_ratio}")
        print(f"    This diagnostic is designed for mask_ratio=1.0")
    
    # Load model
    model = MambaEEGModel(
        d_model=d_model,
        num_layers=num_layers,
        window_length=window_length,
    ).to(device)
    
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    print("\n✅ Model loaded successfully")
    
    # Load dataset
    dataset = EEGPretrainingDataset(
        dataset_root=args.data_path,
        window_length=window_length,
        split="val",
    )
    
    def collate_fn(batch):
        # Use same masking style as training (but with 100% masking for diagnostics)
        return collate_eeg_sequences(
            batch, 
            mask_ratio=1.0,  # Always 100% for diagnostics
            masking_style=masking_style,
            mask_samples_within_token=mask_samples_within_token if masking_style == "multi_channel" else False
        )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    print(f"✅ Dataset loaded: {len(dataset)} sequences")
    
    # Run diagnostics
    print("\n" + "=" * 60)
    print("RUNNING DIAGNOSTICS")
    print("=" * 60)
    
    # Test 1: Analyze predictions
    results = analyze_model_predictions(
        model, dataloader, device, num_samples=args.num_samples
    )
    
    # Test 2: Multi-channel synchronization (if multi-channel masking)
    sync_results = analyze_multi_channel_synchronization(
        dataloader, masking_style, num_samples=args.num_samples
    )
    
    # Test 3: Context sensitivity
    consistency = compare_masked_vs_unmasked_learning(
        model, dataloader, device, num_samples=args.num_samples
    )
    
    # Test 4: Predicted vs Ground Truth Statistics (if signal reconstruction)
    stats_comparison = None
    if reconstruct_signal_space:
        stats_comparison = compare_predicted_vs_gt_statistics(
            model, dataloader, device, num_samples=args.num_samples, decode_to_signal=True
        )
    else:
        print("\n" + "=" * 60)
        print("SKIPPING STATISTICS COMPARISON")
        print("=" * 60)
        print("\n⚠️  Model uses embedding space, not signal reconstruction")
        print("    Cannot compare signal statistics (mean/std)")
        print("    Enable reconstruct_signal_space=true for signal-level analysis")
    
    # Final summary
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)
    
    pos_corr = results["mean_pos_correlation"]
    
    # Include multi-channel synchronization if available
    if sync_results:
        print(f"\nMulti-Channel Masking: Synchronization = {sync_results['mean_synchronization']:.3f}")
        if sync_results['mean_synchronization'] > 0.8:
            print("  ✅ Multi-channel masking appears to be working correctly")
    
    # Include statistics comparison if available
    stats_summary = ""
    if stats_comparison:
        mean_corr = stats_comparison["overall_mean_corr"]
        std_corr = stats_comparison["overall_std_corr"]
        stats_summary = f"\n  - Mean correlation: {mean_corr:.4f}"
        stats_summary += f"\n  - Std correlation: {std_corr:.4f}"
    
    # Get pattern correlation if available
    pattern_corr_val = stats_comparison.get("overall_pattern_corr", 0.0) if stats_comparison else 0.0
    
    if consistency > 0.95 and pos_corr > 0.3:
        print("\n❌ MODEL IS LEARNING POSITIONAL STATISTICS ONLY")
        print("\nEvidence:")
        print(f"  - Very high prediction consistency ({consistency:.3f})")
        print(f"  - Strong position correlation ({pos_corr:.3f})")
        if stats_comparison:
            if pattern_corr_val > 0:
                print(f"  - Low pattern correlation ({pattern_corr_val:.3f})")
            print(f"  - Low signal statistics correlation (mean={stats_comparison['overall_mean_corr']:.3f}, std={stats_comparison['overall_std_corr']:.3f})")
        print("\nInterpretation:")
        print("  The model is learning: position → average embedding")
        print("  It is NOT learning EEG signal structure from context")
        print("  Representations may have limited utility for downstream tasks")
        print("\nRecommendation:")
        print("  Try training with mask_ratio=0.75 or 0.5")
        print("  Compare downstream task performance")
    
    elif pos_corr > 0.3:
        print("\n⚠️  MODEL IS HEAVILY POSITION-DEPENDENT")
        print("\nEvidence:")
        print(f"  - Strong position correlation ({pos_corr:.3f})")
        if stats_comparison:
            if pattern_corr_val > 0:
                print(f"  - Pattern correlation: {pattern_corr_val:.3f}")
            print(f"  - Signal statistics correlation: mean={stats_comparison['overall_mean_corr']:.3f}, std={stats_comparison['overall_std_corr']:.3f}")
            
            # Check baseline similarity
            if "baseline_gt_similarity" in stats_comparison:
                baseline_gt = stats_comparison["baseline_gt_similarity"]
                baseline_pred = stats_comparison["baseline_pred_similarity"]
                print(f"  - Baseline GT similarity: {baseline_gt:.3f}")
                print(f"  - Baseline pred similarity: {baseline_pred:.3f}")
                if abs(baseline_pred - baseline_gt) < 0.1:
                    print("  ✅ Model maintains GT similarity structure")
        print("\nInterpretation:")
        print("  Model relies on positional information")
        if stats_comparison:
            if pattern_corr_val > 0.5:
                print("  BUT: Model DOES reconstruct signal patterns (pattern corr > 0.5)")
                print("  May still learn useful representations despite position dependence")
            elif "baseline_gt_similarity" in stats_comparison and abs(stats_comparison["baseline_pred_similarity"] - stats_comparison["baseline_gt_similarity"]) < 0.1:
                print("  BUT: Model maintains GT similarity structure")
                print("  May be learning window-level patterns (which is correct!)")
                print("  Low per-window correlation doesn't necessarily mean poor learning")
            else:
                print("  Model does NOT accurately reconstruct signal patterns")
                print("  May have limited utility for downstream tasks")
        print("\nRecommendation:")
        print("  Test on downstream tasks to validate representation quality")
    
    else:
        print("\n✅ MODEL APPEARS TO LEARN BEYOND SIMPLE POSITIONS")
        print("\nEvidence:")
        print(f"  - Low position correlation ({pos_corr:.3f})")
        print(f"  - Moderate context sensitivity ({consistency:.3f})")
        if stats_comparison:
            if pattern_corr_val > 0:
                print(f"  - Pattern correlation: {pattern_corr_val:.3f}")
                if pattern_corr_val > 0.7:
                    print("  ✅ Model accurately reconstructs signal patterns!")
                elif pattern_corr_val > 0.5:
                    print("  ⚠️  Model partially reconstructs signal patterns")
                else:
                    print("  ⚠️  Model has weak signal pattern reconstruction")
            print(f"  - Signal statistics correlation: mean={stats_comparison['overall_mean_corr']:.3f}, std={stats_comparison['overall_std_corr']:.3f}")
            
            # Check baseline similarity
            if "baseline_gt_similarity" in stats_comparison:
                baseline_gt = stats_comparison["baseline_gt_similarity"]
                baseline_pred = stats_comparison["baseline_pred_similarity"]
                print(f"  - Baseline GT similarity: {baseline_gt:.3f}")
                print(f"  - Baseline pred similarity: {baseline_pred:.3f}")
                if abs(baseline_pred - baseline_gt) < 0.1:
                    print("  ✅ Model maintains GT similarity structure (window-level learning)")
        print("\nInterpretation:")
        print("  Model may be learning useful EEG representations")
        if stats_comparison:
            if pattern_corr_val > 0 and pattern_corr_val < 0.5:
                if "baseline_gt_similarity" in stats_comparison and abs(stats_comparison["baseline_pred_similarity"] - stats_comparison["baseline_gt_similarity"]) < 0.1:
                    print("  ✅ Low per-window correlation BUT maintains similarity structure")
                    print("  Model likely learning window-level patterns (correct behavior!)")
                else:
                    print("  BUT: Pattern correlation is low")
                    print("  May need more training or different mask ratio")
        print("\nRecommendation:")
        print("  Validate with downstream tasks")
        print("  Compare with mask_ratio=0.75 model")


if __name__ == "__main__":
    main()

