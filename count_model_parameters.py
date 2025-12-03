#!/usr/bin/env python3
"""
Count parameters in MambaEEGModel based on config.
"""

import torch
import yaml
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "eeg_analysis"))

from src.models.mamba_eeg_model import MambaEEGModel


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count by component
    component_counts = {}
    
    # Input projection (lightweight conv/pooling if window_length != d_model)
    if hasattr(model, 'input_proj') and model.input_proj is not None:
        component_counts["input_projection"] = sum(p.numel() for p in model.input_proj.parameters())
    else:
        component_counts["input_projection"] = 0
    
    # Output projection (for reconstruction if window_length != d_model)
    if hasattr(model, 'output_proj') and model.output_proj is not None:
        component_counts["output_projection"] = sum(p.numel() for p in model.output_proj.parameters())
    else:
        component_counts["output_projection"] = 0
    
    # Spatial encoder (fixed projection - no parameters)
    component_counts["spatial_encoder"] = sum(p.numel() for p in model.spatial_encoder.parameters())
    
    # Temporal encoder (minimal parameters)
    component_counts["temporal_encoder"] = sum(p.numel() for p in model.temporal_encoder.parameters())
    
    # Backbone (Mamba) - all complexity here
    component_counts["backbone"] = sum(p.numel() for p in model.backbone.parameters())
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "components": component_counts
    }


def format_number(num):
    """Format number with commas and units."""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)


def main():
    # Load config
    config_path = Path("eeg_analysis/configs/pretrain.yaml")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    d_model = cfg.get("d_model", 128)
    num_layers = cfg.get("num_layers", 6)
    window_length = cfg.get("window_length", 2048)
    asa_path = cfg.get("asa_path")
    
    print("="*80)
    print("MODEL PARAMETER COUNT")
    print("="*80)
    print(f"\nConfig:")
    print(f"  d_model: {d_model}")
    print(f"  num_layers: {num_layers}")
    print(f"  window_length: {window_length}")
    print(f"  asa_path: {asa_path}")
    
    # Create model
    print("\nCreating model...")
    model = MambaEEGModel(
        d_model=d_model,
        num_layers=num_layers,
        window_length=window_length,
        asa_path=asa_path,
        dropout=0.1
    )
    
    # Count parameters
    counts = count_parameters(model)
    
    print("\n" + "="*80)
    print("PARAMETER BREAKDOWN")
    print("="*80)
    
    print(f"\nTotal Parameters: {counts['total']:,} ({format_number(counts['total'])})")
    print(f"Trainable Parameters: {counts['trainable']:,} ({format_number(counts['trainable'])})")
    
    print("\nBy Component:")
    print("-" * 80)
    for component, count in sorted(counts['components'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / counts['total']) * 100
        print(f"  {component:25s}: {count:12,} ({format_number(count):>8s}) - {percentage:5.2f}%")
    
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    # Estimate memory usage (assuming float32)
    bytes_per_param = 4  # float32
    model_size_mb = (counts['total'] * bytes_per_param) / (1024 ** 2)
    print(f"\nModel Size (float32): {model_size_mb:.2f} MB")
    
    # Compare to common model sizes
    print("\nSize Comparison:")
    if counts['total'] < 1e6:
        print(f"  ✅ Small model (< 1M params)")
    elif counts['total'] < 10e6:
        print(f"  ✅ Medium model (1M - 10M params)")
    elif counts['total'] < 100e6:
        print(f"  ⚠️  Large model (10M - 100M params)")
    elif counts['total'] < 1e9:
        print(f"  ⚠️  Very large model (100M - 1B params)")
    else:
        print(f"  ❌ Extremely large model (> 1B params)")
    
    print(f"\nArchitecture Notes:")
    if d_model == window_length:
        print(f"  ✅ Direct feed: d_model == window_length ({d_model})")
        print(f"  ✅ All complexity in Mamba backbone")
    else:
        print(f"  ✅ Flexible: window_length={window_length}, d_model={d_model}")
        print(f"  ✅ Lightweight projection: {window_length} → {d_model}")
        print(f"  ✅ Most complexity in Mamba backbone")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

