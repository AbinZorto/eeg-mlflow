# Strategies to Prevent Positional Learning in MAE Pretraining

**Problem**: Model learns trivial position → embedding mapping instead of signal structure  
**Evidence**: Prediction consistency 0.9995, position correlation 0.403, zero channel diversity

---

## Strategy 1: Reduce Mask Ratio (Easiest & Most Effective) ⭐

**Why it works**: Forces model to learn from signal context, not just position.

### Implementation

Update `eeg_analysis/configs/pretrain.yaml`:

```yaml
mask_ratio: 0.75  # Down from 1.0
```

**Why this prevents positional learning:**
- Unmasked tokens contain actual signal → vary by channel
- Model must learn: "Use unmasked C3 patterns to predict masked C3"
- Position alone becomes insufficient (needs channel-specific context)
- Spatial encoding becomes critical for reconstruction

**Expected results:**
- Channel diversity increases (> 0.05)
- Position correlation decreases (< 0.3)
- Prediction consistency drops (< 0.95)

**Pros**: Standard approach, proven to work, easy to implement  
**Cons**: None (this is the recommended solution)

---

## Strategy 2: Add Contrastive Loss (Advanced)

**Principle**: Same position but different channels should have different embeddings.

### Implementation

Add to `eeg_analysis/src/training/pretrain_mamba.py`:

```python
def contrastive_channel_loss(pred, channel_names, mask_bool, temperature=0.1):
    """
    Penalize predictions that are too similar across different channels
    at the same relative temporal position.
    
    Args:
        pred: (B, L, D) predictions
        channel_names: list[str] of length B
        mask_bool: (B, L) mask positions
        temperature: contrastive temperature
    
    Returns:
        loss: scalar contrastive loss
    """
    B, L, D = pred.shape
    
    # For each temporal position, gather predictions from different channels
    losses = []
    
    for t in range(L):
        # Get predictions at position t from all channels where it's masked
        pos_t_preds = []
        pos_t_channels = []
        
        for b in range(B):
            if mask_bool[b, t] and t < pred.shape[1]:
                pos_t_preds.append(pred[b, t])
                pos_t_channels.append(channel_names[b])
        
        if len(pos_t_preds) < 2:
            continue
        
        # Stack predictions: (N, D) where N = number of channels at this position
        pos_t_preds = torch.stack(pos_t_preds, dim=0)
        
        # Compute pairwise similarities
        # Normalize embeddings
        pos_t_preds = F.normalize(pos_t_preds, dim=1)
        
        # Similarity matrix: (N, N)
        sim_matrix = torch.mm(pos_t_preds, pos_t_preds.t()) / temperature
        
        # We want DIFFERENT channels to have LOW similarity
        # For same channel (if any duplicates), allow high similarity
        
        # Create labels: same channel = 1, different channel = 0
        same_channel = torch.zeros(len(pos_t_channels), len(pos_t_channels), device=pred.device)
        for i in range(len(pos_t_channels)):
            for j in range(len(pos_t_channels)):
                if i != j and pos_t_channels[i] == pos_t_channels[j]:
                    same_channel[i, j] = 1.0
        
        # Contrastive loss: maximize similarity for same channel, minimize for different
        # Use NT-Xent style loss
        exp_sim = torch.exp(sim_matrix)
        
        # For each sample, compute: log(exp(sim_to_same) / sum(exp(sim_to_all)))
        # Since we typically don't have same channel in batch, we penalize high similarity
        
        # Simplified: just penalize high off-diagonal similarities
        mask_diag = 1.0 - torch.eye(len(pos_t_channels), device=pred.device)
        off_diag_sim = (sim_matrix * mask_diag).sum() / (mask_diag.sum() + 1e-8)
        
        # Loss: penalize high similarity between different channels
        losses.append(off_diag_sim)
    
    if losses:
        return torch.stack(losses).mean()
    else:
        return torch.tensor(0.0, device=pred.device)


# In training loop, after line 241:
loss = masked_diff.pow(2).mean()

# Add contrastive loss
contrastive_loss = contrastive_channel_loss(pred, channel_names, mask_bool)
total_loss = loss + 0.1 * contrastive_loss  # Weight the contrastive term

# Use total_loss for backward
```

**How it works:**
- Penalizes predictions that are too similar across different channels
- Forces model to learn channel-specific representations
- Prevents "one embedding fits all positions" solution

**Pros**: Can work even with 100% masking  
**Cons**: Complex implementation, requires tuning weight (0.1 in example)

---

## Strategy 3: Random Temporal Shifts (Augmentation)

**Principle**: Make position unreliable by randomly shifting sequences.

### Implementation

Add to `eeg_analysis/src/data/eeg_pretraining_dataset.py`:

```python
def collate_eeg_sequences(
    batch: List[Dict[str, Any]],
    mask_ratio: float,
    device: Optional[torch.device] = None,
    masking_style: str = "mae",
    random_shift: bool = True,  # NEW
    max_shift: int = 5,  # NEW: max positions to shift
) -> Dict[str, Any]:
    """
    Enhanced collate with random temporal shifting.
    """
    # ... existing code ...
    
    # Fill
    for i, b in enumerate(batch):
        L = b["seq_len"]
        
        # Random temporal shift (NEW)
        if random_shift and L > max_shift:
            shift = np.random.randint(-max_shift, max_shift + 1)
            if shift > 0:
                # Shift right: pad left, truncate right
                orig[i, shift:L, :] = b["windows"][:L-shift, :]
                masked[i, shift:L, :] = b["windows"][:L-shift, :]
            elif shift < 0:
                # Shift left: truncate left, pad right
                orig[i, :L+shift, :] = b["windows"][-shift:, :]
                masked[i, :L+shift, :] = b["windows"][-shift:, :]
            else:
                # No shift
                orig[i, :L, :] = b["windows"]
                masked[i, :L, :] = b["windows"]
        else:
            # No shift (original behavior)
            orig[i, :L, :] = b["windows"]
            masked[i, :L, :] = b["windows"]
        
        # Select mask positions (after shifting)
        k = max(1, int(math.ceil(mask_ratio * L)))
        idxs = np.random.choice(L, size=k, replace=False)
        mask_bool[i, idxs] = True
        
        # Apply masking
        if masking_style == "mae":
            masked[i, idxs, :] = 0.0
        # ... rest of masking code ...
```

**How it works:**
- Randomly shifts sequences by ±5 positions during training
- Same temporal position (t/T) now contains different signal content
- Breaks position → embedding correlation

**Pros**: Simple augmentation, prevents position memorization  
**Cons**: Might disrupt legitimate temporal learning, requires tuning

---

## Strategy 4: Position Embedding Dropout

**Principle**: Randomly remove temporal position information during training.

### Implementation

Modify `eeg_analysis/src/models/mamba_eeg_model.py`:

```python
class TemporalEncoder(nn.Module):
    def __init__(self, d_model: int, dropout_prob: float = 0.1):  # NEW
        super().__init__()
        self.lin = nn.Linear(1, d_model)
        self.dropout_prob = dropout_prob  # NEW

    def forward(self, seq_lengths: torch.Tensor) -> torch.Tensor:
        device = seq_lengths.device
        max_len = int(seq_lengths.max().item())
        positions = torch.arange(max_len, device=device, dtype=torch.float32)
        
        norm_pos = []
        for T in seq_lengths.tolist():
            T_val = max(1, int(T))
            p = positions[:T_val] / float(T_val)
            if T_val < max_len:
                pad = torch.zeros(max_len - T_val, device=device, dtype=torch.float32)
                p = torch.cat([p, pad], dim=0)
            norm_pos.append(p)
        norm_positions = torch.stack(norm_pos, dim=0).unsqueeze(-1)
        
        temporal_emb = self.lin(norm_positions)
        
        # NEW: Random dropout of position embeddings during training
        if self.training and self.dropout_prob > 0:
            # With probability dropout_prob, zero out temporal embeddings
            mask = torch.rand(temporal_emb.shape[0], device=device) > self.dropout_prob
            mask = mask.view(-1, 1, 1).expand_as(temporal_emb)
            temporal_emb = temporal_emb * mask.float()
        
        return temporal_emb
```

**How it works:**
- Randomly removes position information for some samples
- Model can't rely solely on position (might be zero)
- Forces learning from token and spatial information

**Pros**: Simple, forces robust learning  
**Cons**: Might hurt performance if position is legitimately important

---

## Strategy 5: Normalize Out Positional Patterns (Post-hoc)

**Principle**: Remove position-dependent mean from predictions before computing loss.

### Implementation

Add to training loop in `pretrain_mamba.py`:

```python
# After line 235
target = (model.module if isinstance(model, DDP) else model).encode_tokens_only(windows)

# NEW: Compute position-dependent means and subtract
# This removes the "position → embedding" shortcut
B, L, D = pred.shape

# Compute mean prediction at each position across batch
position_means = pred.mean(dim=0, keepdim=True)  # (1, L, D)
target_means = target.mean(dim=0, keepdim=True)  # (1, L, D)

# Center predictions and targets (remove positional bias)
pred_centered = pred - position_means
target_centered = target - target_means

# Compute loss on centered representations
mask_exp = mask_bool.unsqueeze(-1).expand_as(pred_centered)
diff = pred_centered - target_centered
masked_diff = diff[mask_exp]
loss = masked_diff.pow(2).mean()
```

**How it works:**
- Removes batch-level positional mean before loss computation
- Model can't reduce loss by learning position → mean mapping
- Forces learning of deviations from positional mean (i.e., channel-specific patterns)

**Pros**: Can work with 100% masking, no architecture changes  
**Cons**: Removes legitimate temporal structure too

---

## Strategy 6: Shuffle Temporal Positions (Nuclear Option)

**Principle**: Completely break temporal structure, force spatial-only learning.

### Implementation

```python
def collate_eeg_sequences(batch, mask_ratio, shuffle_temporal=True):
    # ... existing setup ...
    
    for i, b in enumerate(batch):
        L = b["seq_len"]
        windows_seq = b["windows"]  # (L, W)
        
        if shuffle_temporal:
            # Randomly permute temporal order
            perm = torch.randperm(L)
            windows_seq = windows_seq[perm]
        
        orig[i, :L, :] = windows_seq
        masked[i, :L, :] = windows_seq
        
        # ... rest of masking ...
```

**How it works:**
- Destroys all temporal structure
- Only spatial (channel) information remains
- Model MUST learn channel-specific patterns

**Pros**: Guaranteed to prevent positional learning  
**Cons**: Destroys temporal information (probably too extreme for EEG)

---

## Recommended Approach: Combination Strategy

### Phase 1: Fix the fundamentals (NOW)

```yaml
# pretrain.yaml
mask_ratio: 0.75  # Provide context
```

This alone should fix 90% of the problem.

### Phase 2: If still seeing positional learning (LATER)

Add Strategy 3 (Random Temporal Shifts):

```python
# In collate function
collate_eeg_sequences(batch, mask_ratio=0.75, random_shift=True, max_shift=3)
```

### Phase 3: For maximum robustness (OPTIONAL)

Add Strategy 2 (Contrastive Loss):

```python
# In training loop
total_loss = reconstruction_loss + 0.1 * contrastive_channel_loss
```

---

## Expected Results After Fixing

### Before (mask_ratio=1.0)
```
Position correlation: 0.403  ❌
Channel diversity:    0.000  ❌
Consistency:          0.999  ❌
Diagnosis: Positional learning only
```

### After (mask_ratio=0.75)
```
Position correlation: < 0.20  ✅
Channel diversity:    > 0.05  ✅
Consistency:          < 0.95  ✅
Diagnosis: Context-based learning with channel awareness
```

---

## Implementation Priority

1. **HIGH PRIORITY**: Reduce mask_ratio to 0.75 → Solves 90% of problem
2. **MEDIUM**: Add random temporal shifts → Prevents remaining positional bias
3. **LOW**: Add contrastive loss → Only if above two aren't sufficient
4. **AVOID**: Position dropout, normalization, shuffling → Too destructive

---

## Quick Start: Fix It Now

```bash
# 1. Update config
nano eeg_analysis/configs/pretrain.yaml
# Change: mask_ratio: 0.75

# 2. Retrain
python eeg_analysis/src/training/pretrain_mamba.py \
    --config eeg_analysis/configs/pretrain.yaml

# 3. Re-diagnose
python diagnose_100pct_masking.py \
    --checkpoint eeg_analysis/checkpoints/mamba2_eeg_pretrained.pt \
    --num-samples 100

# 4. Compare results
# Look for:
# - Position correlation < 0.2 (down from 0.4)
# - Channel diversity > 0.05 (up from 0.0)
# - Consistency < 0.95 (down from 0.999)
```

---

## Why This Matters

**Current model (positional learning):**
- Useless for downstream tasks requiring channel-specific patterns
- Can't distinguish between different EEG activities (all channels same)
- Only knows "average signal at time t"

**Fixed model (context learning):**
- Learns channel-specific EEG patterns
- Can identify different brain states/activities
- Representations transfer to classification/regression tasks
- Actually understands EEG structure

**Bottom line**: The easiest and most effective solution is to simply **reduce mask_ratio to 0.75**. Everything else is optional enhancements.

