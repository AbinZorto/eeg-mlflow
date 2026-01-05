"""
Supervised Fine-Tuning (SFT) for Mamba EEG Model
Fine-tunes pretrained Mamba model for remission classification.
"""

from __future__ import annotations

import argparse
import os
import sys
import logging
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path
import yaml
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import mlflow
import mlflow.pytorch

# Path bootstrap
CURRENT_FILE = Path(__file__).resolve()
EEG_ANALYSIS_ROOT = CURRENT_FILE.parents[2]
if str(EEG_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(EEG_ANALYSIS_ROOT))

from src.models.mamba_sft_model import MambaEEGClassifier, MambaEEGWindowClassifier
from src.data.eeg_sft_dataset import EEGSFTDataset, collate_sft_batch
from src.processing.data_loader import load_eeg_data
from src.processing.upsampler import upsample_eeg_data
from src.processing.filter import filter_eeg_data
from src.processing.downsampler import downsample_eeg_data
from src.processing.window_slicer import slice_eeg_windows
from src.processing.dc_offset import remove_dc_offset_eeg_data
from src.processing.primary_dataset import create_primary_dataset
from src.utils.logger import setup_logger
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

os.environ.setdefault("LOG_PLAIN", "1")
logger = setup_logger(__name__)


@contextmanager
def suppress_output(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        yield


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_processing_config() -> Dict[str, Any]:
    processing_path = EEG_ANALYSIS_ROOT / "configs" / "processing_config.yaml"
    with open(processing_path, "r") as f:
        return yaml.safe_load(f)


def compute_window_length(processing_cfg: Dict[str, Any]) -> int:
    window_seconds = float(processing_cfg["window_slicer"]["window_seconds"])
    sampling_rate = float(processing_cfg["window_slicer"]["sampling_rate"])
    return int(round(window_seconds * sampling_rate))


def build_primary_dataset_path(processing_cfg: Dict[str, Any]) -> str:
    window_seconds = processing_cfg["window_slicer"]["window_seconds"]
    channels = processing_cfg["data_loader"]["channels"]
    channels_str = "-".join(channels)
    primary_dir = Path(processing_cfg["paths"]["features"]["window"]).parent / "primary"
    output_filename = f"{window_seconds}s_{channels_str}_primary_dataset.parquet"
    return str(primary_dir / output_filename)


def ensure_primary_dataset(processing_cfg: Dict[str, Any]) -> str:
    dataset_path = build_primary_dataset_path(processing_cfg)
    if Path(dataset_path).exists():
        logger.info(f"Using existing primary dataset: {dataset_path}")
        return dataset_path

    response = input(
        f"Primary dataset not found at {dataset_path}. Build it now? [y/N]: "
    ).strip().lower()
    if response not in {"y", "yes"}:
        raise FileNotFoundError(
            f"Primary dataset missing and build declined: {dataset_path}"
        )
    logger.info(f"Building primary dataset at: {dataset_path}")
    raw_data = load_eeg_data(processing_cfg)
    upsampled = upsample_eeg_data(processing_cfg, raw_data)
    filtered = filter_eeg_data(processing_cfg, upsampled)
    downsampled = downsample_eeg_data(processing_cfg, filtered)
    windowed = slice_eeg_windows(processing_cfg, downsampled)
    _ = remove_dc_offset_eeg_data(processing_cfg, windowed)
    windowed_path = processing_cfg["paths"]["interim"]["windowed"]
    primary_df, _ = create_primary_dataset(processing_cfg, windowed_path)
    logger.info(f"Primary dataset created: rows={len(primary_df)} path={dataset_path}")
    return dataset_path


def find_pretrained_checkpoint(pretrain_config_path: str, checkpoints_dir: str) -> str:
    """
    Find the pretrained checkpoint based on pretraining config parameters.
    
    Args:
        pretrain_config_path: Path to pretraining config
        checkpoints_dir: Directory containing checkpoints
        
    Returns:
        Path to best pretrained checkpoint
    """
    pretrain_cfg = load_config(pretrain_config_path)
    
    d_model = pretrain_cfg.get("d_model", 128)
    num_layers = pretrain_cfg.get("num_layers", 6)
    mask_ratio = pretrain_cfg.get("mask_ratio", 0.2)
    masking_style = pretrain_cfg.get("masking_style", "mae")
    
    # Construct expected checkpoint name
    mask_style_short = "mae" if masking_style == "mae" else "bert"
    checkpoint_name = f"mamba2_eeg_d{d_model}_l{num_layers}_m{int(mask_ratio*100)}_{mask_style_short}"
    
    # Look for checkpoint in checkpoints directory
    checkpoints_path = Path(checkpoints_dir)
    best_checkpoint = checkpoints_path / "mamba2_eeg_pretrained.pt"
    
    if not best_checkpoint.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {best_checkpoint}\n"
            f"Expected model: {checkpoint_name}\n"
            f"Make sure pretraining completed successfully."
        )
    
    logger.info(f"Found pretrained checkpoint: {best_checkpoint}")
    logger.info(f"Model config: d_model={d_model}, num_layers={num_layers}, "
                f"mask_ratio={mask_ratio}, masking_style={masking_style}")
    
    return str(best_checkpoint), checkpoint_name


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(loader):
        windows = batch["windows"].to(device)  # [B, C, W, L]
        labels = batch["labels"].to(device)  # [B]
        seq_lengths = batch["seq_lengths"].to(device)  # [B]
        channel_names = batch["channel_names"]  # List[str] - channel names
        
        # Forward pass
        logits = model(windows, channel_names, seq_lengths)  # [B, num_classes]
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Epoch {epoch} [{batch_idx+1}/{len(loader)}] - loss: {loss.item():.4f}")
    
    return total_loss / num_batches


def _flatten_valid_windows(
    logits: torch.Tensor,
    labels: torch.Tensor,
    seq_lengths: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten valid windows for window-level loss/metrics."""
    B, W, _ = logits.shape
    device = logits.device
    mask = torch.arange(W, device=device)[None, :] < seq_lengths[:, None]
    flat_logits = logits[mask]  # [N_valid, num_classes]
    flat_labels = labels.unsqueeze(1).expand(-1, W)[mask]  # [N_valid]
    return flat_logits, flat_labels


def train_epoch_window(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    for _, batch in enumerate(loader):
        windows = batch["windows"].to(device)
        labels = batch["labels"].to(device)
        seq_lengths = batch["seq_lengths"].to(device)
        channel_names = batch["channel_names"]
        logits = model(windows, channel_names, seq_lengths)  # [B, W, num_classes]
        flat_logits, flat_labels = _flatten_valid_windows(logits, labels, seq_lengths)
        loss = criterion(flat_logits, flat_labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float | None = None,
    return_details: bool = False,
) -> Dict[str, float] | Tuple[Dict[str, float], List[int], List[float]]:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            windows = batch["windows"].to(device)
            labels = batch["labels"].to(device)
            seq_lengths = batch["seq_lengths"].to(device)
            channel_names = batch["channel_names"]  # List[str] - channel names
            
            logits = model(windows, channel_names, seq_lengths)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            # Get predictions
            probs = torch.softmax(logits, dim=1)
            if threshold is None:
                preds = torch.argmax(logits, dim=1)
            else:
                preds = (probs[:, 1] >= threshold).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    metrics = {
        "loss": total_loss / len(loader),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }
    if return_details:
        return metrics, all_labels, all_probs
    return metrics


def evaluate_window(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float | None = None,
    return_details: bool = False,
    patient_vote: str = "majority",
) -> Dict[str, float] | Tuple[Dict[str, float], List[int], List[float], List[int], List[int], List[int]]:
    """Evaluate window-level model; returns window metrics and patient summaries."""
    model.eval()
    total_loss = 0.0
    window_preds = []
    window_labels = []
    window_probs = []
    patient_labels = []
    patient_probs = []
    patient_preds = []

    with torch.no_grad():
        for batch in loader:
            windows = batch["windows"].to(device)
            labels = batch["labels"].to(device)
            seq_lengths = batch["seq_lengths"].to(device)
            channel_names = batch["channel_names"]
            logits = model(windows, channel_names, seq_lengths)  # [B, W, num_classes]
            flat_logits, flat_labels = _flatten_valid_windows(logits, labels, seq_lengths)
            loss = criterion(flat_logits, flat_labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=2)
            if threshold is None:
                preds = torch.argmax(logits, dim=2)
            else:
                preds = (probs[:, :, 1] >= threshold).long()

            # Window-level metrics
            for i in range(labels.shape[0]):
                valid_w = int(seq_lengths[i].item())
                window_labels.extend([int(labels[i].item())] * valid_w)
                window_preds.extend(preds[i, :valid_w].cpu().tolist())
                window_probs.extend(probs[i, :valid_w, 1].cpu().tolist())

                # Patient-level aggregation
                p_prob = float(probs[i, :valid_w, 1].mean().item())
                if patient_vote == "prob":
                    p_pred = int(p_prob >= (threshold if threshold is not None else 0.5))
                else:
                    p_pred = int(preds[i, :valid_w].float().mean().item() >= 0.5)
                patient_labels.append(int(labels[i].item()))
                patient_probs.append(p_prob)
                patient_preds.append(p_pred)

    window_accuracy = accuracy_score(window_labels, window_preds) if window_labels else 0.0
    metrics = {
        "loss": total_loss / len(loader),
        "window_accuracy": window_accuracy,
    }
    if return_details:
        return metrics, patient_labels, patient_probs, patient_preds, window_labels, window_preds
    return metrics


def build_weighted_sampler(train_ds: EEGSFTDataset, num_classes: int) -> WeightedRandomSampler:
    """Create a participant-level weighted sampler to balance classes per batch."""
    labels = [int(train_ds.labels[p]) for p in train_ds.participants]
    counts = [0] * num_classes
    for label in labels:
        if 0 <= label < num_classes:
            counts[label] += 1
    if any(count == 0 for count in counts):
        raise ValueError(f"Cannot build weighted sampler with empty class: {counts}")
    total = sum(counts)
    class_weights = [total / (num_classes * count) for count in counts]
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def tune_threshold(
    labels: List[int],
    probs: List[float],
    threshold_min: float,
    threshold_max: float,
    steps: int,
) -> Tuple[float, Dict[str, float]]:
    """Tune decision threshold to maximize F1 on validation data."""
    if len(set(labels)) < 2:
        return 0.5, {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    thresholds = np.linspace(threshold_min, threshold_max, steps)
    best = {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": -1.0}
    y_true = np.array(labels)
    y_prob = np.array(probs)
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        if f1 > best["f1"]:
            best = {"threshold": float(t), "precision": float(precision), "recall": float(recall), "f1": float(f1)}
    return best["threshold"], {"precision": best["precision"], "recall": best["recall"], "f1": best["f1"]}


def split_train_val_participants(
    participants: List[str],
    labels: Dict[str, int],
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Split participants into train/val with stratification when possible."""
    if val_ratio <= 0.0 or len(participants) < 2:
        return participants, []
    y = [labels[p] for p in participants]
    stratify = y if len(set(y)) > 1 else None
    try:
        train_participants, val_participants = train_test_split(
            participants,
            test_size=val_ratio,
            random_state=seed,
            stratify=stratify,
        )
        return list(train_participants), list(val_participants)
    except ValueError:
        train_participants, val_participants = train_test_split(
            participants,
            test_size=val_ratio,
            random_state=seed,
            stratify=None,
        )
        return list(train_participants), list(val_participants)


def compute_class_weights(
    train_ds: EEGSFTDataset,
    num_classes: int,
    device: torch.device,
    cfg: Dict[str, Any],
) -> torch.Tensor | None:
    """Compute class weights for imbalanced training."""
    weighting = cfg.get("class_weighting", "none")
    if weighting == "none":
        return None

    if weighting == "custom":
        weights = cfg.get("class_weights")
        if not weights or len(weights) != num_classes:
            raise ValueError("class_weights must match num_classes when class_weighting=custom")
        return torch.tensor(weights, dtype=torch.float32, device=device)

    if weighting != "balanced":
        raise ValueError(f"Unsupported class_weighting value: {weighting}")

    # Balanced weights: total / (num_classes * class_count)
    labels = list(train_ds.labels.values())
    counts = [0] * num_classes
    for label in labels:
        if 0 <= int(label) < num_classes:
            counts[int(label)] += 1
    if any(count == 0 for count in counts):
        raise ValueError(f"Cannot compute balanced weights with empty class: {counts}")

    total = sum(counts)
    weights = [total / (num_classes * count) for count in counts]
    return torch.tensor(weights, dtype=torch.float32, device=device)


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional class weights."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((1.0 - target_probs) ** self.gamma) * target_log_probs
        if self.weight is not None:
            loss = loss * self.weight.gather(0, targets)
        return loss.mean()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Mamba EEG model for classification")
    parser.add_argument("--config", type=str, required=True, help="Path to fine-tuning config")
    parser.add_argument("--pretrain-config", type=str, required=True, help="Path to pretraining config")
    parser.add_argument("--data-path", type=str, default=None, help="Path to primary dataset")
    parser.add_argument("--output-dir", type=str, default="./finetuned_models", help="Output directory")
    
    args = parser.parse_args()
    
    # Load configs
    cfg = load_config(args.config)
    pretrain_cfg = load_config(args.pretrain_config)
    processing_cfg = load_processing_config()
    log_fold_summary_only = bool(cfg.get("log_fold_summary_only", False))
    if log_fold_summary_only:
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")
        logging.getLogger("mlflow").setLevel(logging.ERROR)
        logging.getLogger("src.models.mamba_eeg_model").setLevel(logging.ERROR)
        logging.getLogger("src.models.mamba_sft_model").setLevel(logging.ERROR)
        logger.setLevel(logging.WARNING)

    expected_window_length = compute_window_length(processing_cfg)
    if "window_length" in pretrain_cfg:
        cfg_window_length = int(pretrain_cfg["window_length"])
        if cfg_window_length != expected_window_length:
            raise ValueError(
                "window_length mismatch: pretrain config has "
                f"{cfg_window_length}, but processing_config.yaml implies "
                f"{expected_window_length}"
            )
    pretrain_cfg["window_length"] = expected_window_length
    if not log_fold_summary_only:
        logger.info(
            "Window settings from processing_config.yaml: window_seconds=%.3f sampling_rate=%.3f window_length=%d",
            float(processing_cfg["window_slicer"]["window_seconds"]),
            float(processing_cfg["window_slicer"]["sampling_rate"]),
            expected_window_length,
        )
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not log_fold_summary_only:
        logger.info(f"Using device: {device}")
    
    # Find pretrained checkpoint
    checkpoints_dir = pretrain_cfg.get("save_dir", "./checkpoints")
    pretrained_path, model_name = find_pretrained_checkpoint(args.pretrain_config, checkpoints_dir)
    
    # Resolve primary dataset path from processing config if not provided
    if args.data_path is None:
        data_path = ensure_primary_dataset(processing_cfg)
    else:
        data_path = args.data_path

    # MLflow setup
    tracking_uri = cfg.get("mlflow_tracking_uri", "mlruns")
    experiment_name = cfg.get("mlflow_experiment", "eeg_finetuning_mamba2")
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    split_strategy = cfg.get("split_strategy", "random")
    if split_strategy not in {"random", "leave_one_out"}:
        raise ValueError(f"Unsupported split_strategy: {split_strategy}")

    training_mode = cfg.get("training_mode", "participant")
    window_strategy = cfg.get("window_strategy", "frozen")
    val_ratio = float(cfg.get("val_ratio", 0.0))
    tune_threshold = bool(cfg.get("tune_threshold", False))
    log_fold_summary_only = bool(cfg.get("log_fold_summary_only", False))
    if not log_fold_summary_only:
        if training_mode == "participant" and window_strategy is not None:
            logger.info("Note: window_strategy is ignored when training_mode=participant.")
        if training_mode == "window" and cfg.get("freeze_backbone") is not None:
            logger.info("Note: freeze_backbone is ignored when training_mode=window; use window_strategy.")
        if split_strategy == "leave_one_out" and cfg.get("test_ratio") is not None:
            logger.info("Note: test_ratio is ignored when split_strategy=leave_one_out.")
        if tune_threshold and val_ratio <= 0.0:
            logger.warning("tune_threshold is true but val_ratio <= 0; threshold tuning will be skipped.")

    # Preload participants for leave-one-out
    participants = None
    labels = None
    if split_strategy == "leave_one_out":
        df = pd.read_parquet(data_path)
        participants = sorted(df["Participant"].unique())
        labels = df.groupby("Participant")["Remission"].first().to_dict()
        if not log_fold_summary_only:
            logger.info("Using leave-one-out across %d participants", len(participants))

    if log_fold_summary_only:
        logger.setLevel(logging.INFO)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=f"sft_{model_name}_{split_strategy}"):
        # Loop folds
        folds = participants if split_strategy == "leave_one_out" else [None]
        all_test_labels: List[int] = []
        all_test_probs: List[float] = []
        all_test_preds: List[int] = []
        all_window_labels: List[int] = []
        all_window_preds: List[int] = []
        dataset_verbose = bool(cfg.get("dataset_verbose", True))
        for fold_idx, holdout in enumerate(folds, start=1):
            if split_strategy == "leave_one_out":
                holdout_str = str(holdout).replace("/", "_")
                train_pool = [p for p in participants if p != holdout]
                train_participants, val_participants = split_train_val_participants(
                    train_pool,
                    labels,
                    float(cfg.get("val_ratio", 0.2)),
                    int(cfg.get("seed", 42)),
                )
                test_participants = [holdout]
            else:
                holdout_str = "random"
                train_participants = val_participants = test_participants = None

            logger.info("Loading datasets for fold %s...", holdout_str)
            train_ds = EEGSFTDataset(
                data_path=data_path,
                window_length=pretrain_cfg.get("window_length", 2048),
                split="train",
                val_ratio=cfg.get("val_ratio", 0.2),
                test_ratio=cfg.get("test_ratio", 0.1),
                seed=cfg.get("seed", 42),
                participants=train_participants,
                labels=labels,
                verbose=dataset_verbose,
            )
            val_ds = None
            if split_strategy == "leave_one_out":
                if float(cfg.get("val_ratio", 0.0)) > 0.0:
                    val_participants_use = val_participants
                    if not val_participants_use:
                        if not log_fold_summary_only:
                            logger.warning(
                                "No validation participants available for fold %s; skipping validation.",
                                holdout_str,
                            )
                        val_participants_use = None
                    if val_participants_use:
                        val_ds = EEGSFTDataset(
                            data_path=data_path,
                            window_length=pretrain_cfg.get("window_length", 2048),
                            split="val",
                            val_ratio=cfg.get("val_ratio", 0.2),
                            test_ratio=cfg.get("test_ratio", 0.1),
                            seed=cfg.get("seed", 42),
                            participants=val_participants_use,
                            labels=labels,
                            verbose=dataset_verbose,
                        )
            else:
                val_ds = EEGSFTDataset(
                    data_path=data_path,
                    window_length=pretrain_cfg.get("window_length", 2048),
                    split="val",
                    val_ratio=cfg.get("val_ratio", 0.2),
                    test_ratio=cfg.get("test_ratio", 0.1),
                    seed=cfg.get("seed", 42),
                    verbose=dataset_verbose,
                )

            test_ds = EEGSFTDataset(
                data_path=data_path,
                window_length=pretrain_cfg.get("window_length", 2048),
                split="test",
                val_ratio=cfg.get("val_ratio", 0.2),
                test_ratio=cfg.get("test_ratio", 0.1),
                seed=cfg.get("seed", 42),
                participants=test_participants,
                labels=labels,
                verbose=dataset_verbose,
            )
            
            # Create dataloaders
            use_weighted_sampler = bool(cfg.get("use_weighted_sampler", False))
            sampler = build_weighted_sampler(train_ds, cfg.get("num_classes", 2)) if use_weighted_sampler else None
            train_loader = DataLoader(
                train_ds,
                batch_size=cfg.get("batch_size", 8),
                shuffle=(sampler is None),
                sampler=sampler,
                collate_fn=collate_sft_batch,
                num_workers=0
            )
            
            val_loader = None
            if val_ds is not None:
                val_loader = DataLoader(
                    val_ds,
                    batch_size=cfg.get("batch_size", 8),
                    shuffle=False,
                    collate_fn=collate_sft_batch,
                    num_workers=0
                )
            
            test_loader = DataLoader(
                test_ds,
                batch_size=cfg.get("batch_size", 8),
                shuffle=False,
                collate_fn=collate_sft_batch,
                num_workers=0
            )
            
            # Create model
            if not log_fold_summary_only:
                logger.info("Creating model...")
            with suppress_output(log_fold_summary_only):
                if training_mode == "window":
                    freeze_backbone = window_strategy == "frozen"
                    model = MambaEEGWindowClassifier(
                        d_model=pretrain_cfg.get("d_model", 128),
                        num_layers=pretrain_cfg.get("num_layers", 6),
                        window_length=pretrain_cfg.get("window_length", 2048),
                        asa_path=pretrain_cfg.get("asa_path"),
                        num_classes=2,
                        dropout=cfg.get("dropout", 0.1),
                        pretrained_path=pretrained_path,
                        freeze_backbone=freeze_backbone
                    ).to(device)
                else:
                    model = MambaEEGClassifier(
                        d_model=pretrain_cfg.get("d_model", 128),
                        num_layers=pretrain_cfg.get("num_layers", 6),
                        window_length=pretrain_cfg.get("window_length", 2048),
                        asa_path=pretrain_cfg.get("asa_path"),
                        num_classes=2,
                        dropout=cfg.get("dropout", 0.1),
                        pretrained_path=pretrained_path,
                        freeze_backbone=cfg.get("freeze_backbone", True)
                    ).to(device)
            
            param_counts = model.get_trainable_parameters()
            if not log_fold_summary_only:
                logger.info(f"Model parameters: {param_counts}")
            
            # Optimizer and scheduler (separate lrs for backbone vs head)
            head_lr = float(cfg.get("lr", 1e-4))
            backbone_lr = cfg.get("backbone_lr")
            if backbone_lr is None:
                backbone_lr = head_lr * 0.1
            else:
                backbone_lr = float(backbone_lr)
            optimizer = AdamW(
                [
                    {"params": model.backbone.parameters(), "lr": backbone_lr},
                    {"params": model.classifier.parameters(), "lr": head_lr},
                ],
                weight_decay=cfg.get("weight_decay", 0.01),
            )
            
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=cfg.get("epochs", 50),
                eta_min=cfg.get("min_lr", 1e-6)
            )

            class_weights = compute_class_weights(train_ds, cfg.get("num_classes", 2), device, cfg)
            if not log_fold_summary_only:
                if class_weights is not None:
                    logger.info(f"Using class weights: {class_weights.tolist()}")
                else:
                    logger.info("Using unweighted loss")
            loss_type = cfg.get("loss_type", "cross_entropy")
            if loss_type == "focal":
                focal_gamma = float(cfg.get("focal_gamma", 2.0))
                if not log_fold_summary_only:
                    logger.info(f"Using focal loss (gamma={focal_gamma})")
                criterion = FocalLoss(gamma=focal_gamma, weight=class_weights)
            elif loss_type == "cross_entropy":
                label_smoothing = float(cfg.get("label_smoothing", 0.0))
                if not log_fold_summary_only:
                    logger.info("Using cross-entropy loss")
                criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
            else:
                raise ValueError(f"Unsupported loss_type: {loss_type}")

            with mlflow.start_run(run_name=f"fold_{fold_idx}_{holdout_str}", nested=True):
                # Log config
                mlflow.log_params({
                    "pretrained_model": model_name,
                    "d_model": pretrain_cfg.get("d_model", 128),
                    "num_layers": pretrain_cfg.get("num_layers", 6),
                    "mask_ratio": pretrain_cfg.get("mask_ratio", 0.2),
                    "masking_style": pretrain_cfg.get("masking_style", "mae"),
                    "lr": cfg.get("lr", 1e-4),
                    "batch_size": cfg.get("batch_size", 8),
                    "epochs": cfg.get("epochs", 50),
                    "freeze_backbone": cfg.get("freeze_backbone", True),
                    "training_mode": training_mode,
                    "window_strategy": cfg.get("window_strategy", "frozen"),
                    "class_weighting": cfg.get("class_weighting", "none"),
                    "loss_type": cfg.get("loss_type", "cross_entropy"),
                    "focal_gamma": cfg.get("focal_gamma", 2.0),
                    "trainable_params": param_counts["trainable"],
                    "total_params": param_counts["total"],
                    "train_samples": len(train_ds),
                    "val_samples": len(val_ds) if val_ds is not None else 0,
                    "test_samples": len(test_ds),
                    "use_weighted_sampler": use_weighted_sampler,
                    "backbone_lr": backbone_lr,
                    "label_smoothing": cfg.get("label_smoothing", 0.0),
                    "unfreeze_epoch": cfg.get("unfreeze_epoch"),
                    "split_strategy": split_strategy,
                    "fold_index": fold_idx,
                    "holdout_participant": holdout_str
                })
                
                best_val_f1 = 0.0
                best_path = None
                best_threshold = None
                
                first_train_loss = None
                last_train_loss = None
                for epoch in range(1, cfg.get("epochs", 50) + 1):
                    if cfg.get("unfreeze_epoch") is not None and epoch == int(cfg["unfreeze_epoch"]):
                        model.unfreeze_backbone_layers()
                        if not log_fold_summary_only:
                            logger.info("Unfroze backbone parameters for fold %s", holdout_str)
                    
                    # Train
                    if training_mode == "window":
                        train_loss = train_epoch_window(model, train_loader, criterion, optimizer, device, epoch)
                    else:
                        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
                    if first_train_loss is None:
                        first_train_loss = train_loss
                    last_train_loss = train_loss
                    mlflow.log_metric("train_loss", train_loss, step=epoch)
                    
                    # Validate (if available)
                    if val_loader is not None:
                        tune_thr = bool(cfg.get("tune_threshold", False))
                        if tune_thr:
                            val_metrics, val_labels, val_probs = evaluate(
                                model, val_loader, criterion, device, return_details=True
                            )
                            threshold, _ = tune_threshold(
                                val_labels,
                                val_probs,
                                float(cfg.get("threshold_min", 0.05)),
                                float(cfg.get("threshold_max", 0.95)),
                                int(cfg.get("threshold_steps", 19)),
                            )
                            best_threshold = threshold
                            val_metrics_thr = evaluate(
                                model, val_loader, criterion, device, threshold=threshold
                            )
                            mlflow.log_metric("val_threshold", threshold, step=epoch)
                            for key, value in val_metrics.items():
                                mlflow.log_metric(f"val_{key}", value, step=epoch)
                            for key, value in val_metrics_thr.items():
                                mlflow.log_metric(f"val_thr_{key}", value, step=epoch)
                            val_metrics = val_metrics_thr
                        else:
                            val_metrics = evaluate(model, val_loader, criterion, device)
                            for key, value in val_metrics.items():
                                mlflow.log_metric(f"val_{key}", value, step=epoch)
                    
                    # Step scheduler
                    scheduler.step()
                    lrs = scheduler.get_last_lr()
                    mlflow.log_metric("lr_backbone", lrs[0], step=epoch)
                    mlflow.log_metric("lr_head", lrs[1], step=epoch)
                    
                    # Save best model
                    if val_loader is not None and val_metrics["f1"] > best_val_f1:
                        best_val_f1 = val_metrics["f1"]
                        best_path = output_dir / f"{model_name}_fold_{fold_idx}_{holdout_str}_best.pt"
                        torch.save({
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "val_f1": best_val_f1,
                            "val_threshold": best_threshold,
                            "config": cfg,
                            "pretrain_config": pretrain_cfg
                        }, best_path)
                        if not log_fold_summary_only:
                            logger.info(f"Saved best model for fold {holdout_str}: {best_path}")
                
                # Test evaluation (patient-level in LOPO)
                if training_mode == "window":
                    patient_vote = cfg.get("patient_vote", "majority")
                    if best_threshold is not None:
                        test_metrics, patient_labels, patient_probs, patient_preds, window_labels, window_preds = evaluate_window(
                            model, test_loader, criterion, device, threshold=best_threshold, return_details=True, patient_vote=patient_vote
                        )
                    else:
                        test_metrics, patient_labels, patient_probs, patient_preds, window_labels, window_preds = evaluate_window(
                            model, test_loader, criterion, device, return_details=True, patient_vote=patient_vote
                        )
                    for key, value in test_metrics.items():
                        mlflow.log_metric(f"test_{key}", value)
                    if split_strategy == "leave_one_out":
                        patient_true = int(patient_labels[0]) if patient_labels else 0
                        patient_prob = float(patient_probs[0]) if patient_probs else 0.0
                        patient_pred = int(patient_preds[0]) if patient_preds else 0
                        patient_accuracy = int(patient_true == patient_pred)
                        all_test_labels.append(patient_true)
                        all_test_probs.append(patient_prob)
                        all_test_preds.append(patient_pred)
                        all_window_labels.extend(window_labels)
                        all_window_preds.extend(window_preds)
                        mlflow.log_metric("fold_patient_accuracy", patient_accuracy)
                        mlflow.log_metric("fold_patient_prob", patient_prob)
                        mlflow.log_metric("fold_window_accuracy", test_metrics.get("window_accuracy", 0.0))
                        mlflow.log_param("fold_patient_true", patient_true)
                        mlflow.log_param("fold_patient_pred", patient_pred)
                else:
                    if best_threshold is not None:
                        test_metrics, test_labels, test_probs = evaluate(
                            model, test_loader, criterion, device, threshold=best_threshold, return_details=True
                        )
                    else:
                        test_metrics, test_labels, test_probs = evaluate(
                            model, test_loader, criterion, device, return_details=True
                        )
                    for key, value in test_metrics.items():
                        mlflow.log_metric(f"test_{key}", value)
                    if split_strategy == "leave_one_out":
                        patient_true = int(test_labels[0]) if test_labels else 0
                        patient_prob = float(test_probs[0]) if test_probs else 0.0
                        patient_pred = int(patient_prob >= (best_threshold if best_threshold is not None else 0.5))
                        patient_accuracy = int(patient_true == patient_pred)
                        all_test_labels.append(patient_true)
                        all_test_probs.append(patient_prob)
                        all_test_preds.append(patient_pred)
                        mlflow.log_metric("fold_patient_accuracy", patient_accuracy)
                        mlflow.log_metric("fold_patient_prob", patient_prob)
                        mlflow.log_param("fold_patient_true", patient_true)
                        mlflow.log_param("fold_patient_pred", patient_pred)
                
                # Log best model
                if best_path is not None:
                    mlflow.log_artifact(str(best_path))
                else:
                    if not log_fold_summary_only:
                        logger.warning("No best model checkpoint was saved; skipping artifact logging.")
                
                # Register model
                try:
                    mlflow.pytorch.log_model(
                        pytorch_model=model,
                        artifact_path="model",
                        registered_model_name=f"{model_name}_finetuned"
                    )
                    if not log_fold_summary_only:
                        logger.info(f"Registered model: {model_name}_finetuned")
                except Exception as e:
                    if not log_fold_summary_only:
                        logger.warning(f"Failed to register model: {e}")
                
                loss_change_pct = None
                if first_train_loss is not None and first_train_loss != 0:
                    loss_change_pct = ((last_train_loss - first_train_loss) / first_train_loss) * 100.0
                logger.info(f"Fold {fold_idx}/{len(folds)} complete: holdout={holdout_str}")
                if loss_change_pct is not None:
                    logger.info(f"Train loss change: {loss_change_pct:.2f}% (first={first_train_loss:.4f} last={last_train_loss:.4f})")
                if val_loader is not None:
                    logger.info(f"Best val F1: {best_val_f1:.4f}")
                if split_strategy == "leave_one_out":
                    logger.info(f"Fold patient accuracy: {patient_accuracy}")
                    logger.info(f"Fold patient prob: {patient_prob:.4f}")
                    logger.info(f"Fold patient true: {patient_true}")
                    logger.info(f"Fold patient pred: {patient_pred}")
                    if training_mode == "window":
                        logger.info(f"Fold patient vote: {cfg.get('patient_vote', 'majority')}")
                    if training_mode == "window":
                        logger.info(f"Fold window accuracy: {test_metrics.get('window_accuracy', 0.0):.4f}")
                if training_mode != "window":
                    logger.info(f"Test F1: {test_metrics['f1']:.4f}")
                    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

        # Aggregate patient-level metrics across folds
        if split_strategy == "leave_one_out" and all_test_labels:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_test_labels, all_test_preds, average="binary", zero_division=0
            )
            overall_accuracy = accuracy_score(all_test_labels, all_test_preds)
            try:
                overall_auc = roc_auc_score(all_test_labels, all_test_probs)
            except Exception:
                overall_auc = 0.0
            logger.info("Overall LOPO patient metrics:")
            logger.info(f"Accuracy: {overall_accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1: {f1:.4f}")
            logger.info(f"AUC: {overall_auc:.4f}")
            if training_mode == "window" and all_window_labels:
                overall_window_accuracy = accuracy_score(all_window_labels, all_window_preds)
                logger.info(f"Overall LOPO window accuracy: {overall_window_accuracy:.4f}")
            mlflow.log_metrics({
                "lopo_patient_accuracy": overall_accuracy,
                "lopo_patient_precision": precision,
                "lopo_patient_recall": recall,
                "lopo_patient_f1": f1,
                "lopo_patient_auc": overall_auc,
            })
            if training_mode == "window" and all_window_labels:
                mlflow.log_metric("lopo_window_accuracy", overall_window_accuracy)


if __name__ == "__main__":
    main()
