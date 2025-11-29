from __future__ import annotations

from typing import Any, Dict, List, Optional
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import os
import math
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

    ds = EEGPretrainingDataset(dataset_root=dataset_path, window_length=int(cfg.get("window_length", 2048)))
    def _collate(batch: List[Dict[str, Any]]):
        return collate_eeg_sequences(batch, mask_ratio=float(cfg.get("mask_ratio", 0.2)))

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-4)))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("amp", True)) and device.type == "cuda")
    epochs = int(cfg.get("epochs", 500))
    grad_clip = float(cfg.get("grad_clip_norm", 1.0))
    save_dir = Path(cfg.get("save_dir", "./checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_loss = float("inf")

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
        if is_main:
            logger.info(f"Resumed from {args.resume} at epoch {start_epoch}")

    # MLflow
    tracking_uri = cfg.get("mlflow_tracking_uri", "mlruns")
    experiment_name = cfg.get("mlflow_experiment", "eeg_pretraining_mamba2")
    if is_main:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    run_ctx = mlflow.start_run(run_name="mamba2_pretrain") if is_main else nullcontext()
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
                "dataset_path": dataset_path,
                "world_size": world_size,
            })
            mlflow.log_param("total_sequences", len(ds))

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

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    # Predict embeddings from masked inputs
                    pred = (model.module if isinstance(model, DDP) else model)(
                        windows_masked=windows_masked,
                        channel_names=channel_names,
                        seq_lengths=seq_lengths,
                    )  # (B, L, D)
                    # Targets: token projection of original windows
                    target = (model.module if isinstance(model, DDP) else model).encode_tokens_only(windows)  # (B, L, D)
                    # Compute MSE over masked positions only (avoid NaN propagation by indexing)
                    mask_exp = mask_bool.unsqueeze(-1).expand_as(pred)  # (B, L, D)
                    diff = pred - target
                    masked_diff = diff[mask_exp]  # (N_masked * D,)
                    # mean over all masked token dimensions
                    loss = masked_diff.pow(2).mean()

                # If loss is non-finite, dump diagnostics and skip batch
                if not torch.isfinite(loss):
                    if is_main:
                        logger.error("Non-finite loss detected; dumping diagnostics for current batch")
                        logger.error(_tensor_stats("windows", windows))
                        logger.error(_tensor_stats("windows_masked", windows_masked))
                        logger.error(f"mask_bool: shape={tuple(mask_bool.shape)}, masked_tokens={int(mask_bool.sum().item())}")
                        logger.error(_tensor_stats("pred", pred))
                        logger.error(_tensor_stats("target", target))
                        logger.error(_tensor_stats("diff", diff))
                        if masked_diff.numel() > 0:
                            logger.error(_tensor_stats("masked_diff", masked_diff))
                        else:
                            logger.error("masked_diff: empty (no masked positions selected)")
                    continue

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if grad_clip and grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
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
            if is_main:
                mlflow.log_metric("train_loss", avg_loss, step=epoch)
                mlflow.log_metric("masked_tokens", total_masked, step=epoch)
                logger.info(f"Epoch {epoch+1}/{epochs} - loss={avg_loss:.6f}, masked_tokens={total_masked}")

            # Checkpoint
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
            ckpt = {
                "model": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
                "optimizer": optimizer.state_dict(),
                "scaler": (scaler.state_dict() if scaler is not None else None),
                "epoch": epoch + 1,
                "best_loss": best_loss,
                "config": cfg,
            }
            last_path = save_dir / "mamba2_eeg_pretrained_last.pt"
            best_path = save_dir / "mamba2_eeg_pretrained.pt"
            if is_main:
                torch.save(ckpt, last_path)
                if is_best:
                    torch.save(ckpt, best_path)
                    mlflow.log_artifact(str(best_path))
                if (epoch + 1) % 10 == 0:
                    mlflow.log_artifact(str(last_path))

        # Cleanup DDP
        if distributed:
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()


