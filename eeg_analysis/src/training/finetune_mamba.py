"""
Supervised Fine-Tuning (SFT) for Mamba EEG Model
Fine-tunes pretrained Mamba model for remission classification.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import mlflow
import mlflow.pytorch

# Path bootstrap
CURRENT_FILE = Path(__file__).resolve()
EEG_ANALYSIS_ROOT = CURRENT_FILE.parents[2]
if str(EEG_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(EEG_ANALYSIS_ROOT))

from src.models.mamba_sft_model import MambaEEGClassifier
from src.data.eeg_sft_dataset import EEGSFTDataset, collate_sft_batch
from src.utils.logger import setup_logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

logger = setup_logger(__name__)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


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


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
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
            preds = torch.argmax(logits, dim=1)
            
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
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Mamba EEG model for classification")
    parser.add_argument("--config", type=str, required=True, help="Path to fine-tuning config")
    parser.add_argument("--pretrain-config", type=str, required=True, help="Path to pretraining config")
    parser.add_argument("--data-path", type=str, required=True, help="Path to primary dataset")
    parser.add_argument("--output-dir", type=str, default="./finetuned_models", help="Output directory")
    
    args = parser.parse_args()
    
    # Load configs
    cfg = load_config(args.config)
    pretrain_cfg = load_config(args.pretrain_config)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Find pretrained checkpoint
    checkpoints_dir = pretrain_cfg.get("save_dir", "./checkpoints")
    pretrained_path, model_name = find_pretrained_checkpoint(args.pretrain_config, checkpoints_dir)
    
    # Create datasets
    logger.info("Loading datasets...")
    train_ds = EEGSFTDataset(
        data_path=args.data_path,
        window_length=pretrain_cfg.get("window_length", 2048),
        split="train",
        val_ratio=cfg.get("val_ratio", 0.2),
        test_ratio=cfg.get("test_ratio", 0.1),
        seed=cfg.get("seed", 42)
    )
    
    val_ds = EEGSFTDataset(
        data_path=args.data_path,
        window_length=pretrain_cfg.get("window_length", 2048),
        split="val",
        val_ratio=cfg.get("val_ratio", 0.2),
        test_ratio=cfg.get("test_ratio", 0.1),
        seed=cfg.get("seed", 42)
    )
    
    test_ds = EEGSFTDataset(
        data_path=args.data_path,
        window_length=pretrain_cfg.get("window_length", 2048),
        split="test",
        val_ratio=cfg.get("val_ratio", 0.2),
        test_ratio=cfg.get("test_ratio", 0.1),
        seed=cfg.get("seed", 42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 8),
        shuffle=True,
        collate_fn=collate_sft_batch,
        num_workers=0
    )
    
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
    logger.info("Creating model...")
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
    logger.info(f"Model parameters: {param_counts}")
    
    # Optimizer and scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 0.01)
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.get("epochs", 50),
        eta_min=cfg.get("min_lr", 1e-6)
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # MLflow setup
    tracking_uri = cfg.get("mlflow_tracking_uri", "mlruns")
    experiment_name = cfg.get("mlflow_experiment", "eeg_finetuning_mamba2")
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    # Training loop
    with mlflow.start_run(run_name=f"sft_{model_name}"):
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
            "trainable_params": param_counts["trainable"],
            "total_params": param_counts["total"],
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": len(test_ds)
        })
        
        best_val_f1 = 0.0
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(1, cfg.get("epochs", 50) + 1):
            logger.info(f"\n=== Epoch {epoch}/{cfg.get('epochs', 50)} ===")
            
            # Train
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
            logger.info(f"Train loss: {train_loss:.4f}")
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            
            # Validate
            val_metrics = evaluate(model, val_loader, criterion, device)
            logger.info(f"Val metrics: {val_metrics}")
            for key, value in val_metrics.items():
                mlflow.log_metric(f"val_{key}", value, step=epoch)
            
            # Step scheduler
            scheduler.step()
            mlflow.log_metric("lr", scheduler.get_last_lr()[0], step=epoch)
            
            # Save best model
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_path = output_dir / f"{model_name}_finetuned_best.pt"
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_f1": best_val_f1,
                    "config": cfg,
                    "pretrain_config": pretrain_cfg
                }, best_path)
                logger.info(f"Saved best model (F1={best_val_f1:.4f}) to {best_path}")
        
        # Test evaluation
        logger.info("\n=== Final Test Evaluation ===")
        test_metrics = evaluate(model, test_loader, criterion, device)
        logger.info(f"Test metrics: {test_metrics}")
        for key, value in test_metrics.items():
            mlflow.log_metric(f"test_{key}", value)
        
        # Log best model
        mlflow.log_artifact(str(best_path))
        
        # Register model
        try:
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name=f"{model_name}_finetuned"
            )
            logger.info(f"Registered model: {model_name}_finetuned")
        except Exception as e:
            logger.warning(f"Failed to register model: {e}")
        
        logger.info(f"\nFine-tuning completed!")
        logger.info(f"Best val F1: {best_val_f1:.4f}")
        logger.info(f"Test F1: {test_metrics['f1']:.4f}")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()

