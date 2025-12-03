#!/usr/bin/env python3
"""
Sweep script for mask_ratio hyperparameter search.
Runs multiple pretraining jobs with different mask ratios.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List
import yaml

# Add eeg_analysis to path
SCRIPT_DIR = Path(__file__).resolve().parent
EEG_ANALYSIS_ROOT = SCRIPT_DIR.parents[1]
if str(EEG_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(EEG_ANALYSIS_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def run_sweep(
    base_config: str,
    mask_ratios: List[float],
    use_torchrun: bool = False,
    num_gpus: int = None,
    distributed: bool = False,
    backend: str = "nccl",
) -> None:
    """
    Run pretraining sweep over mask_ratio values.
    
    Args:
        base_config: Path to base config YAML
        mask_ratios: List of mask ratio values to sweep
        use_torchrun: Whether to use torchrun for multi-GPU
        num_gpus: Number of GPUs (auto-detect if None)
        distributed: Whether to use DDP
        backend: DDP backend
    """
    logger.info(f"Starting mask_ratio sweep with {len(mask_ratios)} values: {mask_ratios}")
    
    # Auto-detect number of GPUs if using torchrun
    if use_torchrun and num_gpus is None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
            )
            num_gpus = len([line for line in result.stdout.strip().split("\n") if line.strip()])
            logger.info(f"Auto-detected {num_gpus} GPUs")
        except Exception as e:
            logger.warning(f"Failed to auto-detect GPUs: {e}. Falling back to 1.")
            num_gpus = 1
    
    # Load base config
    with open(base_config, "r") as f:
        base_cfg = yaml.safe_load(f)
    
    # Create temp configs for each mask ratio
    sweep_dir = Path("./sweep_configs")
    sweep_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for mask_ratio in mask_ratios:
        logger.info(f"\n{'='*80}")
        logger.info(f"Running experiment with mask_ratio={mask_ratio:.2f}")
        logger.info(f"{'='*80}\n")
        
        # Update config with current mask_ratio
        sweep_cfg = base_cfg.copy()
        sweep_cfg["mask_ratio"] = mask_ratio
        
        # Update MLflow experiment name to group sweep runs
        sweep_cfg["mlflow_experiment"] = f"eeg_pretraining_mamba2_sweep_mask"
        
        # Save temp config
        temp_config_path = sweep_dir / f"pretrain_mask{int(mask_ratio*100)}.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(sweep_cfg, f)
        
        # Build command
        train_script = SCRIPT_DIR / "pretrain_mamba.py"
        
        if use_torchrun:
            # Use torchrun for multi-GPU DDP
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={num_gpus}",
                str(train_script),
                "--config",
                str(temp_config_path),
                "--distributed",
                "--backend",
                backend,
            ]
        else:
            # Single process (single GPU or CPU)
            cmd = [sys.executable, str(train_script), "--config", str(temp_config_path)]
            
            if distributed:
                cmd.extend(["--distributed", "--backend", backend])
        
        # Run training
        try:
            result = subprocess.run(cmd, check=True, capture_output=False, text=True)
            logger.info(f"✓ Completed mask_ratio={mask_ratio:.2f}")
            results.append((mask_ratio, "success"))
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed mask_ratio={mask_ratio:.2f}: {e}")
            results.append((mask_ratio, "failed"))
            # Continue with next ratio instead of stopping
            continue
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("Sweep Summary:")
    logger.info(f"{'='*80}")
    for mask_ratio, status in results:
        status_icon = "✓" if status == "success" else "✗"
        logger.info(f"{status_icon} mask_ratio={mask_ratio:.2f} - {status}")
    logger.info(f"{'='*80}\n")
    
    successes = sum(1 for _, s in results if s == "success")
    logger.info(f"Completed {successes}/{len(mask_ratios)} experiments successfully")


def main():
    parser = argparse.ArgumentParser(description="Sweep mask_ratio for EEG Mamba-2 pretraining")
    parser.add_argument("--config", type=str, required=True, help="Base config YAML path")
    parser.add_argument(
        "--mask-ratios",
        type=str,
        default="0.2,0.4,0.6,0.8",
        help="Comma-separated mask ratios to sweep (default: 0.2,0.4,0.6,0.8)",
    )
    parser.add_argument("--torchrun", action="store_true", help="Use torchrun for multi-GPU DDP")
    parser.add_argument("--num-gpus", type=int, default=None, help="Number of GPUs (auto-detect if not specified)")
    parser.add_argument("--distributed", action="store_true", help="Use DDP for each run (ignored if --torchrun is set)")
    parser.add_argument("--backend", type=str, default="nccl", help="DDP backend")
    args = parser.parse_args()
    
    # Parse mask ratios
    mask_ratios = [float(x.strip()) for x in args.mask_ratios.split(",")]
    
    run_sweep(
        base_config=args.config,
        mask_ratios=mask_ratios,
        use_torchrun=args.torchrun,
        num_gpus=args.num_gpus,
        distributed=args.distributed,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()

