"""
Mamba EEG Model for Supervised Fine-Tuning (SFT)
Loads pretrained Mamba model and adds classification head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
from .mamba_eeg_model import MambaEEGModel


class MambaEEGClassifier(nn.Module):
    """
    Mamba EEG model with classification head for supervised fine-tuning.
    
    Architecture:
    - Loads pretrained Mamba backbone
    - Freezes all layers except classification head
    - Adds classification head: [d_model] -> [num_classes]
    """
    
    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 6,
        window_length: int = 2048,
        asa_path: Optional[str] = None,
        num_classes: int = 2,
        dropout: float = 0.1,
        pretrained_path: Optional[str] = None,
        freeze_backbone: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        
        # Load pretrained Mamba backbone
        self.backbone = MambaEEGModel(
            d_model=d_model,
            num_layers=num_layers,
            window_length=window_length,
            asa_path=asa_path,
            dropout=dropout
        )
        
        # Load pretrained weights if provided
        if pretrained_path:
            self.load_pretrained_weights(pretrained_path)
        
        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone_layers()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Initialize classification head
        self._init_classifier()
    
    def load_pretrained_weights(self, checkpoint_path: str):
        """Load pretrained weights from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract model state dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Load weights (strict=False to allow missing classifier weights)
        missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
        
        print(f"Loaded pretrained weights from {checkpoint_path}")
        if missing_keys:
            print(f"  Missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
        if unexpected_keys:
            print(f"  Unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
    
    def freeze_backbone_layers(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Froze all backbone parameters")
    
    def unfreeze_backbone_layers(self):
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Unfroze all backbone parameters")
    
    def _init_classifier(self):
        """Initialize classifier weights."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        windows: torch.Tensor,
        channel_names: list[str],
        seq_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for classification.
        
        Args:
            windows: [B, C, W, L] - batch, channels, windows, window_length
            channel_names: List of channel names (length C)
            seq_lengths: [B] - actual sequence lengths (for masking padding)
            
        Returns:
            logits: [B, num_classes]
        """
        B, C, W, L = windows.shape
        
        # Process each channel separately through backbone
        # Then aggregate across channels and windows
        channel_embeddings = []
        
        for c_idx in range(C):
            channel_name = channel_names[c_idx] if c_idx < len(channel_names) else f"CH{c_idx}"
            channel_windows = windows[:, c_idx, :, :]  # [B, W, L]
            
            # Create channel_names list for this batch (all same channel)
            batch_channel_names = [channel_name] * B
            
            # Forward through backbone for this channel
            # backbone.forward(windows_masked, channel_names, seq_lengths)
            # expects: windows_masked [B, W, L], channel_names List[str] length B, seq_lengths [B]
            if seq_lengths is None:
                seq_lengths_tensor = torch.full((B,), W, dtype=torch.long, device=windows.device)
            else:
                seq_lengths_tensor = seq_lengths
            
            channel_emb = self.backbone(
                windows_masked=channel_windows,
                channel_names=batch_channel_names,
                seq_lengths=seq_lengths_tensor
            )  # [B, W, d_model]
            
            # Average pool over windows (considering seq_lengths if provided)
            if seq_lengths is not None:
                # Create mask for valid windows
                mask = torch.arange(W, device=windows.device)[None, :] < seq_lengths[:, None]  # [B, W]
                mask = mask.unsqueeze(-1)  # [B, W, 1]
                
                # Masked mean
                channel_emb = (channel_emb * mask).sum(dim=1) / seq_lengths.unsqueeze(-1).float()  # [B, d_model]
            else:
                # Simple mean
                channel_emb = channel_emb.mean(dim=1)  # [B, d_model]
            
            channel_embeddings.append(channel_emb)
        
        # Stack and average across channels
        channel_embeddings = torch.stack(channel_embeddings, dim=1)  # [B, C, d_model]
        pooled = channel_embeddings.mean(dim=1)  # [B, d_model]
        
        # Classification
        logits = self.classifier(pooled)  # [B, num_classes]
        
        return logits
    
    def get_trainable_parameters(self) -> Dict[str, int]:
        """Get count of trainable vs frozen parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {
            "trainable": trainable,
            "frozen": total - trainable,
            "total": total
        }

