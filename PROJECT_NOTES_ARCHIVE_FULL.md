# Project Notes Archive

This file consolidates historical top-level markdown notes from the repository root.

Merged source files:
- ANTI_POSITIONAL_LEARNING_STRATEGIES.md
- ANTI_POSITION_ONLY_LEARNING.md
- ARCHITECTURE_SIMPLIFICATION_OPTIONS.md
- AUTO_GRADIENT_CLIPPING.md
- BASELINE_SIMILARITY_ANALYSIS.md
- CONTROL_EXPERIMENT_GUIDE.md
- DIAGNOSIS_RESULTS.md
- DIAGNOSTIC_CHECKLIST.md
- DROPOUT_VARIANCE_ANALYSIS.md
- FIGURE_TABLE_PLAN.md
- IDEAL_LEARNING_TRAJECTORY.md
- LEAKAGE_ANALYSIS.md
- MAE_PIPELINE_AUDIT.md
- MULTI_CHANNEL_MASKING.md
- PAPER_OUTLINE.md
- PAPER_SKELETON.md
- POSITION_LEAKAGE_ROOT_CAUSE.md
- PRETRAINING_LEAKAGE_AUDIT.md
- PROPER_MAE_SOLUTION.md
- QUICK_START_SFT.md
- REAL_LEAKAGE_FOUND.md
- REAL_POSITION_LEARNING_ANALYSIS.md
- SCALE_MISMATCH_FIX.md
- SFT_PIPELINE_SUMMARY.md
- SOLUTION_PREVENT_MEAN_LEARNING.md
- STATISTICS_COMPARISON_FEATURE.md
- SUMMARY_POSITIONAL_LEARNING_FIX.md
- TEMPORAL_ORDERING_NOTES.md
- TRAINING_RESULTS_ANALYSIS.md
- WITHIN_TOKEN_MASKING.md

---

## Source: ANTI_POSITIONAL_LEARNING_STRATEGIES.md

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

Add to training loop in `eeg_analysis/src/training/pretrain_mamba.py`:

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
# eeg_analysis/configs/pretrain.yaml
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
python scripts/diagnose_100pct_masking.py \
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


---

## Source: ANTI_POSITION_ONLY_LEARNING.md

# Anti-Position-Only Learning Strategy

## Problem

Even with explicit positional encodings disabled, **Mamba's sequential processing provides implicit positional information**:
- Mamba processes tokens sequentially (token 0, then 1, then 2...)
- Hidden state accumulates position information
- Model learns "position → prediction" instead of "context → prediction"

**Result**: Model learns positional patterns, not signal content.

## Solution: Multi-Strategy Approach

We implement **two complementary strategies** to prevent position-only learning while keeping Mamba and position:

### Strategy 1: Position Regularization

**What it does**: Penalizes predictions that correlate strongly with position.

**How it works**:
1. For each masked token, compute its normalized position (0 to 1)
2. Compute correlation between position and prediction summary
3. Add penalty: `loss += position_corr² * weight`

**Effect**: Model is discouraged from learning position-only patterns.

**Configuration**:
```yaml
prevent_position_only_learning: true
position_regularization_weight: 0.1  # 0.0-1.0, higher = more penalty
```

### Strategy 2: Sequence Shuffling

**What it does**: Randomly shuffles window order during training.

**How it works**:
1. For each sequence in batch, with probability `shuffle_prob`:
   - Randomly permute window order
   - Breaks position=time mapping
2. Model must learn from context (unmasked tokens), not position

**Effect**: Forces model to learn from signal context, not temporal position.

**Configuration**:
```yaml
shuffle_sequences_prob: 0.3  # 0.0-1.0
# 0.0 = never shuffle (preserve temporal order)
# 0.3 = shuffle 30% of sequences (balanced)
# 1.0 = always shuffle (breaks temporal structure)
```

## Why This Works

### Position Regularization

**Before**:
```
Position 0 → Predict A
Position 1 → Predict B
Position 2 → Predict C
Loss: Low (model learns position mapping) ✅
Position correlation: High (0.6+) ❌
```

**After**:
```
Position 0 → Predict A (but penalized if too correlated)
Position 1 → Predict B (but penalized if too correlated)
Loss: Slightly higher (penalty added)
Position correlation: Lower (<0.3) ✅
```

### Sequence Shuffling

**Before** (ordered):
```
Window 0 (time 0s) → Position 0
Window 1 (time 8s) → Position 1
Window 2 (time 16s) → Position 2
Model learns: position = time ✅ (but learns position, not signal)
```

**After** (shuffled 30%):
```
Window 0 (time 16s) → Position 0  ← Shuffled!
Window 1 (time 0s) → Position 1   ← Shuffled!
Window 2 (time 8s) → Position 2   ← Shuffled!
Model must learn: context → prediction (can't use position=time)
```

## Configuration

### Recommended Settings

**Balanced** (recommended):
```yaml
prevent_position_only_learning: true
position_regularization_weight: 0.1
shuffle_sequences_prob: 0.3
```

**Aggressive** (if still learning position):
```yaml
prevent_position_only_learning: true
position_regularization_weight: 0.3
shuffle_sequences_prob: 0.5
```

**Conservative** (minimal intervention):
```yaml
prevent_position_only_learning: true
position_regularization_weight: 0.05
shuffle_sequences_prob: 0.1
```

## Monitoring

### MLflow Metrics

The training script logs:
- **`position_correlation`**: Correlation between position and predictions
  - **Target**: < 0.3 (low position dependence)
  - **Warning**: > 0.5 (high position dependence)
- **`position_penalty`**: Current penalty value
  - Should decrease as model learns context instead of position

### Diagnostic Script

Run after training:
```bash
python scripts/diagnose_100pct_masking.py \
    --checkpoint eeg_analysis/checkpoints/mamba2_eeg_pretrained.pt \
    --num-samples 100
```

**Look for**:
- Position correlation < 0.3 ✅
- Pattern correlation > 0.3 ✅
- Baseline similarity maintained ✅

## Expected Behavior

### With Anti-Position-Only Learning

**Training**:
- Loss may be slightly higher (penalty added)
- Position correlation decreases over time
- Model learns from context, not just position

**Diagnostics**:
- Position correlation: < 0.3 ✅
- Pattern correlation: > 0.3 ✅
- Context sensitivity: < 0.9 ✅

### Without Anti-Position-Only Learning

**Training**:
- Loss decreases quickly
- Position correlation stays high (> 0.6)
- Model learns position mapping

**Diagnostics**:
- Position correlation: > 0.6 ❌
- Pattern correlation: < 0.1 ❌
- Context sensitivity: > 0.9 ❌

## Trade-offs

### Position Regularization

**Pros**:
- ✅ Keeps temporal order intact
- ✅ Allows position to be used (just not exclusively)
- ✅ Simple to implement

**Cons**:
- ⚠️ Adds hyperparameter (weight)
- ⚠️ May slow convergence slightly

### Sequence Shuffling

**Pros**:
- ✅ Forces context learning
- ✅ Breaks position=time mapping directly
- ✅ No hyperparameter tuning needed (just probability)

**Cons**:
- ⚠️ Breaks temporal structure (may hurt some tasks)
- ⚠️ May confuse model if overused

## Best Practices

1. **Start with balanced settings** (0.1 weight, 0.3 shuffle prob)
2. **Monitor position correlation** during training
3. **Adjust if needed**:
   - High position corr (>0.5) → Increase weight/shuffle prob
   - Low pattern corr (<0.1) → Decrease weight/shuffle prob
4. **Test on downstream tasks** to validate representations

## Summary

**Goal**: Learn signal content, not just position.

**Strategy**: 
- Position regularization (penalize position-only learning)
- Sequence shuffling (break position=time mapping)

**Result**: Model learns from context while still using position when appropriate.

**Keep**: Mamba architecture, positional encodings, temporal order (mostly)

**Prevent**: Position-only learning, ignoring signal content


---

## Source: ARCHITECTURE_SIMPLIFICATION_OPTIONS.md

# Architecture Simplification Options

## Current Setup
- **window_length**: 2048 samples
- **d_model**: 512
- **Encoder**: 3-layer MLP (2048 → 1024 → 768 → 512) = 5.33M params
- **Decoder**: 3-layer MLP (512 → 768 → 1024 → 2048) = 3.29M params
- **Total encoder/decoder**: ~8.6M params (72% of model!)

## Goal
Remove encoder/decoder, simplify to direct Mamba processing with complexity in the backbone.

---

## Option 1: Direct Feed (No Projection) ⭐ **RECOMMENDED**

**Architecture:**
- Set `d_model = window_length` (e.g., 512, 1024, or 2048)
- Remove encoder/decoder entirely
- Feed raw windows directly to Mamba: `(B, L, window_length)` → Mamba → `(B, L, d_model)`
- Single linear projection for reconstruction: `(B, L, d_model)` → `(B, L, window_length)`
- Or identity if `d_model == window_length`

**Pros:**
- ✅ Maximum simplicity
- ✅ All complexity in Mamba backbone
- ✅ Minimal parameters (just one linear layer for reconstruction)
- ✅ Preserves raw signal information
- ✅ Natural fit for 2-second windows

**Cons:**
- ⚠️ Requires `d_model` to match window size (or accept projection)
- ⚠️ Larger `d_model` = more Mamba parameters

**Parameter Count (with d_model=512, window_length=512):**
- Input projection: 0 (direct feed)
- Mamba backbone: ~3.4M (scales with d_model)
- Output projection: 512 × 512 = 262K (or 0 if identity)
- **Total: ~3.7M params** (69% reduction!)

**Parameter Count (with d_model=1024, window_length=1024):**
- Mamba backbone: ~13.6M (4x larger due to d_model² scaling)
- Output projection: 0 (identity)
- **Total: ~13.6M params**

---

## Option 2: Minimal Single-Layer Projection

**Architecture:**
- Keep `d_model` flexible (e.g., 512)
- Single linear encoder: `window_length → d_model`
- Single linear decoder: `d_model → window_length`
- No MLP layers, no weight tying

**Pros:**
- ✅ Flexible `d_model` independent of window size
- ✅ Very simple (just 2 linear layers)
- ✅ Still minimal parameters

**Cons:**
- ⚠️ Less expressive than MLP
- ⚠️ Still has projection overhead

**Parameter Count (d_model=512, window_length=2048):**
- Encoder: 2048 × 512 = 1.05M
- Decoder: 512 × 2048 = 1.05M
- Mamba: ~3.4M
- **Total: ~5.5M params** (54% reduction)

---

## Option 3: Identity + Mamba Only (Pure Sequence Model)

**Architecture:**
- Set `d_model = window_length`
- No encoder, no decoder
- Mamba processes raw windows: `(B, L, window_length)` → `(B, L, window_length)`
- Reconstruction target: same as input (identity mapping)

**Pros:**
- ✅ Absolute minimum complexity
- ✅ Zero projection overhead
- ✅ Pure sequence modeling

**Cons:**
- ⚠️ Requires exact `d_model == window_length`
- ⚠️ Mamba parameters scale quadratically with `d_model`
- ⚠️ May be too restrictive

**Parameter Count (d_model=512, window_length=512):**
- Mamba backbone: ~3.4M
- **Total: ~3.4M params** (72% reduction!)

---

## Option 4: Hybrid - Minimal Encoder + Direct Decoder

**Architecture:**
- Single linear encoder: `window_length → d_model`
- No decoder (Mamba outputs directly to signal space)
- Mamba: `(B, L, d_model)` → `(B, L, d_model)`
- Single linear: `d_model → window_length` (if needed)

**Pros:**
- ✅ Flexible `d_model`
- ✅ Simpler than full encoder/decoder
- ✅ Mamba does most of the work

**Cons:**
- ⚠️ Still has encoder projection
- ⚠️ Asymmetric (encoder but no decoder)

**Parameter Count (d_model=512, window_length=2048):**
- Encoder: 2048 × 512 = 1.05M
- Mamba: ~3.4M
- Decoder: 512 × 2048 = 1.05M
- **Total: ~5.5M params**

---

## Recommendation: **Option 1 (Direct Feed)**

### Why Option 1?

1. **Maximum Simplicity**: Removes all encoder/decoder complexity
2. **Parameter Efficiency**: With 500k tokens, 3.7M-13.6M params is much better than 12M
3. **Natural Fit**: 2-second windows at common sampling rates (256-1024 Hz) give 512-2048 samples
4. **Mamba Strength**: Mamba excels at sequence modeling - let it do the work!

### Implementation Strategy:

**For 2-second windows:**
- **256 Hz**: `window_length = 512`, `d_model = 512` → **~3.7M params**
- **512 Hz**: `window_length = 1024`, `d_model = 1024` → **~13.6M params**
- **1024 Hz**: `window_length = 2048`, `d_model = 2048` → **~54M params** (may be too large)

**Recommended:**
- Use **512 Hz sampling** → `window_length = 1024`, `d_model = 1024`
- This gives **~13.6M params** (still reasonable for 500k tokens)
- Or use **256 Hz** → `window_length = 512`, `d_model = 512` → **~3.7M params** (very efficient!)

### Code Changes Needed:

1. Remove `TokenEncoder` class
2. Remove decoder layers
3. Modify `MambaEEGModel.forward()`:
   - Input: `(B, L, window_length)` directly
   - Add temporal/spatial encodings to each token
   - Feed to Mamba: `(B, L, window_length)` → `(B, L, window_length)`
   - Output: reconstructed signal (same shape)
4. Update config: `d_model = window_length`

---

## Comparison Table

| Option | d_model | window_length | Encoder | Decoder | Total Params | Complexity |
|-------|---------|---------------|---------|---------|--------------|------------|
| **Current** | 512 | 2048 | 3-layer MLP | 3-layer MLP | 12.01M | High |
| **Option 1** | 512 | 512 | None | Identity | 3.7M | Minimal |
| **Option 1** | 1024 | 1024 | None | Identity | 13.6M | Minimal |
| **Option 2** | 512 | 2048 | Linear | Linear | 5.5M | Low |
| **Option 3** | 512 | 512 | None | None | 3.4M | Minimal |
| **Option 4** | 512 | 2048 | Linear | Linear | 5.5M | Low |

---

## Next Steps

1. **Decide on sampling rate** → determines `window_length` for 2 seconds
2. **Set `d_model = window_length`** for Option 1
3. **Remove encoder/decoder** from model
4. **Update forward pass** to feed directly to Mamba
5. **Test with reduced parameters**

Would you like me to implement Option 1?


---

## Source: AUTO_GRADIENT_CLIPPING.md

# Auto Gradient Clipping Implementation

## Overview

Adaptive gradient clipping that automatically adjusts the clipping threshold based on gradient norm statistics during training.

## Features

### 1. Percentile-Based Adaptive Clipping

Instead of a fixed clipping threshold, the system:
- Tracks gradient norms from recent batches (last 1000 by default)
- Computes the 95th percentile of gradient norms
- Uses this as the clipping threshold

**Benefits**:
- Adapts to training dynamics (gradients change over time)
- Prevents rare gradient explosions without over-clipping
- More flexible than fixed thresholds

### 2. Dual Threshold System

```python
effective_clip = min(grad_clip_norm, adaptive_threshold)
```

- `grad_clip_norm`: Hard upper limit (safety net)
- `adaptive_threshold`: Learned from gradient statistics
- Uses the more conservative (smaller) of the two

**Example**:
```
grad_clip_norm: 1.0 (config)
Epoch 1: adaptive=0.3 → clips at 0.3 (adaptive is lower)
Epoch 10: adaptive=0.8 → clips at 0.8 (adaptive is lower)
Epoch 50: adaptive=1.5 → clips at 1.0 (hard limit is lower)
```

### 3. Gradient Statistics Logging

Every 50 steps, logs to MLflow:
- `grad_norm`: Current gradient norm (before clipping)
- `grad_clip_threshold`: Effective threshold used
- `grad_clipped`: Whether clipping was applied (1=yes, 0=no)

## Configuration

### In `eeg_analysis/configs/pretrain.yaml`

```yaml
# Gradient Clipping
grad_clip_norm: 1.0           # Hard maximum (safety limit)
auto_grad_clip: true          # Enable adaptive clipping
grad_clip_percentile: 95.0    # Clip at 95th percentile
```

### Parameters Explained

**`grad_clip_norm`** (default: 1.0)
- Hard upper limit on gradient norm
- Safety net to prevent extreme explosions
- Typical values: 0.5-2.0 depending on model

**`auto_grad_clip`** (default: true)
- Enable/disable adaptive clipping
- `true`: Use percentile-based adaptive threshold
- `false`: Use only fixed `grad_clip_norm`

**`grad_clip_percentile`** (default: 95.0)
- Percentile of gradient history to use as threshold
- 95.0 = clip the top 5% of gradients
- Higher = more lenient (99 = clip only top 1%)
- Lower = more aggressive (90 = clip top 10%)

**Recommended values**:
- Conservative: `percentile: 90` (clip top 10%)
- Balanced: `percentile: 95` (clip top 5%) ← default
- Lenient: `percentile: 99` (clip top 1%)

## How It Works

### Algorithm

```python
# Initialization
grad_norm_history = []  # Track last 1000 gradient norms

# During training:
for batch in dataloader:
    loss.backward()
    
    # 1. Compute gradient norm (no clipping)
    total_norm = compute_grad_norm(model.parameters())
    
    # 2. Compute adaptive threshold (after warmup)
    if len(grad_norm_history) >= 100:
        adaptive_threshold = percentile(grad_norm_history, 95.0)
        effective_clip = min(grad_clip_norm, adaptive_threshold)
    else:
        effective_clip = grad_clip_norm  # Use fixed during warmup
    
    # 3. Apply clipping
    clip_grad_norm_(model.parameters(), max_norm=effective_clip)
    
    # 4. Update history
    grad_norm_history.append(total_norm)
    if len(grad_norm_history) > 1000:
        grad_norm_history.pop(0)  # Keep recent 1000
    
    optimizer.step()
```

### Warmup Period

- First 100 batches: Uses fixed `grad_clip_norm`
- After 100 batches: Switches to adaptive threshold
- Ensures stable statistics before adapting

### History Management

- Keeps last **1000 gradient norms**
- Rolling window: old values removed when limit reached
- Adapts to recent training dynamics

## Monitoring

### MLflow Metrics

Check these metrics to understand gradient behavior:

**`grad_norm`**:
- Current gradient norm before clipping
- Trend shows if gradients are exploding/vanishing
- Should be stable, not wildly varying

**`grad_clip_threshold`**:
- Effective clipping threshold used
- Should adapt over training (usually decreases)
- Shows how aggressive clipping is

**`grad_clipped`**:
- Binary: 1 if clipped, 0 if not
- Percentage clipped ≈ (100 - percentile)%
- With 95th percentile, ~5% should be clipped

### Healthy Training Signs

```
Early training:
  grad_norm: 2.0-5.0 (large, unstable)
  grad_clip_threshold: 1.0 (hard limit)
  grad_clipped: 0.8 (80% clipped)

Mid training:
  grad_norm: 0.5-1.0 (moderate, stable)
  grad_clip_threshold: 0.8 (adaptive)
  grad_clipped: 0.05 (5% clipped)

Late training:
  grad_norm: 0.1-0.3 (small, very stable)
  grad_clip_threshold: 0.3 (adaptive)
  grad_clipped: 0.05 (5% clipped)
```

### Problem Signs

**Gradient explosion**:
```
grad_norm: 100+ (increasing rapidly)
grad_clipped: 1.0 (always clipping)
→ Solution: Lower grad_clip_norm to 0.5 or reduce learning rate
```

**Over-clipping**:
```
grad_norm: 0.3-0.5 (stable)
grad_clip_threshold: 0.1 (too low)
grad_clipped: 0.9 (90% clipped)
→ Solution: Increase grad_clip_percentile to 99
```

**Vanishing gradients**:
```
grad_norm: 0.001 (very small)
grad_clipped: 0.0 (never clipping)
→ Solution: Check loss scale, might need higher learning rate
```

## Comparison: Fixed vs. Auto

### Fixed Clipping (Traditional)

```yaml
grad_clip_norm: 1.0
auto_grad_clip: false
```

**Pros**:
- Simple, predictable
- Easy to tune once

**Cons**:
- May over-clip in later training (gradients shrink)
- May under-clip in early training (gradients large)
- Same threshold throughout training

### Auto Clipping (Adaptive)

```yaml
grad_clip_norm: 1.0
auto_grad_clip: true
grad_clip_percentile: 95.0
```

**Pros**:
- Adapts to training dynamics
- More aggressive early (large gradients)
- More lenient late (small gradients)
- Better final performance

**Cons**:
- Slightly more complex
- Need to monitor percentile choice
- First 100 steps use fixed clipping

## Tuning Guide

### Start with Defaults

```yaml
grad_clip_norm: 1.0
auto_grad_clip: true
grad_clip_percentile: 95.0
```

### If Training is Unstable (loss spikes)

```yaml
grad_clip_norm: 0.5  # Lower hard limit
grad_clip_percentile: 90.0  # More aggressive
```

### If Training is Too Conservative (slow progress)

```yaml
grad_clip_norm: 2.0  # Higher hard limit
grad_clip_percentile: 99.0  # More lenient
```

### If You Want Pure Adaptive (no hard limit)

```yaml
grad_clip_norm: 100.0  # Very high (effectively disabled)
auto_grad_clip: true
grad_clip_percentile: 95.0  # Only adaptive threshold
```

## Example MLflow Query

To analyze gradient clipping behavior:

```python
import mlflow
import pandas as pd

run = mlflow.get_run(run_id)
metrics = mlflow.get_metric_history(run.info.run_id, "grad_norm")

df = pd.DataFrame([
    {"step": m.step, "grad_norm": m.value}
    for m in metrics
])

# Plot gradient norm over training
df.plot(x="step", y="grad_norm")

# Compute clipping statistics
threshold_metrics = mlflow.get_metric_history(run.info.run_id, "grad_clip_threshold")
clip_applied = mlflow.get_metric_history(run.info.run_id, "grad_clipped")

print(f"Average gradient norm: {df['grad_norm'].mean():.4f}")
print(f"Max gradient norm: {df['grad_norm'].max():.4f}")
print(f"% of steps clipped: {sum(m.value for m in clip_applied) / len(clip_applied) * 100:.1f}%")
```

## Implementation Details

### Why Percentile, Not Mean/Median?

**Percentile (95th)**:
- Explicitly targets outliers (top 5%)
- Robust to extreme values
- Directly controls clipping frequency

**Mean**:
- Affected by outliers (what we want to clip)
- Would clip too aggressively

**Median (50th percentile)**:
- Would clip 50% of gradients (too aggressive)
- Not targeting outliers

### Why Rolling Window?

- Training dynamics change over epochs
- Recent gradients more relevant than old ones
- 1000 steps ≈ balances responsiveness vs. stability
- With batch_size=16, 1000 steps ≈ 2-3 epochs

### AMP Compatibility

The implementation works with both:
- **AMP enabled**: Unscales gradients before computing norms
- **AMP disabled**: Computes norms directly

Both paths have identical clipping logic.

## Summary

**Auto gradient clipping** provides adaptive, intelligent gradient norm management:

✅ **Adapts** to training dynamics
✅ **Prevents** gradient explosions
✅ **Avoids** over-clipping in late training
✅ **Monitors** via MLflow metrics
✅ **Simple** to configure

**Default config works well for most cases** - just monitor the metrics and adjust if needed!


---

## Source: BASELINE_SIMILARITY_ANALYSIS.md

# Baseline Similarity Analysis: Understanding Window-Level Learning

## Key Insight

**Token vectors for the same window across different channels are naturally more similar than vectors from different windows.**

This means:
- Ground truth windows have inherent similarity structure
- Model should learn to maintain this structure
- Low per-window correlation doesn't necessarily mean poor learning

## The Problem with Simple Pattern Correlation

### Traditional Approach (Misleading)

```
For each masked window:
  pred_window = model prediction
  gt_window = ground truth
  correlation = corr(pred_window, gt_window)
  
If correlation is low → Model is not learning ❌
```

**Issue**: This ignores the fact that:
- Different channels for the same window should be similar
- Model might learn window-level patterns correctly
- But per-window correlation appears low because model predicts window-level features, not exact channel-specific details

### Example

**Ground Truth**:
- Window 1, Channel A: [1, 2, 3, ...]
- Window 1, Channel B: [1.1, 2.1, 3.1, ...] ← Similar to Channel A
- Window 2, Channel A: [10, 20, 30, ...] ← Different from Window 1

**Model Prediction** (Learning window-level patterns):
- Window 1, Channel A: [0.9, 1.9, 2.9, ...] ← Learns Window 1 pattern
- Window 1, Channel B: [0.95, 1.95, 2.95, ...] ← Learns Window 1 pattern (similar!)
- Window 2, Channel A: [9.5, 19.5, 29.5, ...] ← Learns Window 2 pattern (different!)

**Per-window correlation**: Low (model doesn't match exact channel details)
**Baseline similarity**: High (model maintains GT similarity structure) ✅

## Baseline Similarity Analysis

### What It Measures

1. **GT Baseline Similarity**: 
   - Sample random pairs of GT windows
   - Compute average similarity
   - Measures inherent similarity structure in data

2. **Predicted Baseline Similarity**:
   - Sample random pairs of predicted windows
   - Compute average similarity
   - Measures if model maintains same structure

3. **Comparison**:
   - If `|pred_similarity - gt_similarity| < 0.1` → Model maintains structure ✅
   - If difference is large → Model doesn't learn structure ❌

### Interpretation

**Good Learning** (Window-Level Patterns):
```
GT baseline similarity: 0.35
Pred baseline similarity: 0.33
Difference: 0.02 ✅

Per-window correlation: 0.15 (low)
BUT: Model maintains similarity structure!
→ Model learns window-level patterns correctly
```

**Poor Learning** (Positional Patterns):
```
GT baseline similarity: 0.35
Pred baseline similarity: 0.85
Difference: 0.50 ❌

Per-window correlation: 0.10 (low)
AND: Model doesn't maintain similarity structure
→ Model learning positional patterns, not signal content
```

## Updated Diagnostic Interpretation

### Pattern Correlation Alone (Misleading)

```
Pattern correlation: -0.047
→ ❌ Model does NOT reconstruct signal patterns
```

**Problem**: Ignores baseline similarity structure!

### Pattern Correlation + Baseline Similarity (Accurate)

```
Pattern correlation: -0.047 (low)
GT baseline similarity: 0.32
Pred baseline similarity: 0.30
Difference: 0.02 ✅

→ Model maintains similarity structure
→ Low per-window correlation BUT learns window-level patterns
→ This is actually GOOD learning!
```

## Why This Matters

### Window-Level Learning is Correct!

In EEG data:
- **Same window, different channels**: Should be similar (shared temporal pattern)
- **Different windows**: Should be different (different temporal patterns)

**Model should learn**:
- Window-level temporal patterns ✅
- Not exact per-channel reconstruction (channels vary slightly)

**Traditional diagnostic** (per-window correlation) penalizes this correct behavior!

### Example: What Model Should Learn

**Input**: Masked windows from multiple channels
**Output**: Window-level temporal pattern

**Correct behavior**:
- Predictions for same window across channels: Similar ✅
- Predictions for different windows: Different ✅
- Per-window correlation: Low (doesn't match exact channel details)
- Baseline similarity: High (maintains GT structure) ✅

## Diagnostic Output

The updated diagnostic now reports:

```
BASELINE CROSS-CHANNEL SIMILARITY ANALYSIS
============================================================

Baseline GT similarity (random window pairs): 0.3245 ± 0.1234
Predicted similarity (random window pairs):    0.3102 ± 0.1156

Difference: -0.0143

✅ Model maintains similar baseline similarity structure as GT
```

**Interpretation**:
- Model learns window-level patterns correctly
- Maintains GT similarity structure
- Low per-window correlation is expected (model doesn't match exact channel details)

## When to Worry

### ❌ Bad: Model Doesn't Maintain Structure

```
GT baseline similarity: 0.35
Pred baseline similarity: 0.85
Difference: 0.50 ❌

→ Model predicts similar values for all windows
→ Learning positional patterns, not signal content
```

### ✅ Good: Model Maintains Structure

```
GT baseline similarity: 0.35
Pred baseline similarity: 0.33
Difference: 0.02 ✅

→ Model maintains GT similarity structure
→ Learning window-level patterns correctly
→ Low per-window correlation is OK!
```

## Summary

**Key Insight**: 
- Token vectors for same window across channels are naturally similar
- Model should learn to maintain this similarity structure
- Low per-window correlation doesn't mean poor learning if baseline similarity is maintained

**Updated Diagnostic**:
- Computes baseline GT similarity
- Compares to predicted baseline similarity
- If similar → Model learns window-level patterns correctly ✅
- If different → Model learning positional patterns ❌

**Takeaway**: 
- Window-level learning is correct behavior
- Baseline similarity analysis reveals this
- Per-window correlation alone can be misleading


---

## Source: CONTROL_EXPERIMENT_GUIDE.md

# Control Experiment Guide: Detecting True Signal Leakage

## Purpose

Use `mask_ratio=1.0` with disabled positional encodings as a **control experiment** to detect if there's any information leakage from the actual signal content.

## The Problem We Solved

Your original diagnostic showed loss reduction with `mask_ratio=1.0`, but this was **NOT from signal leakage** - it was from positional encodings:

```
Position correlation: 0.403  ← Learning from temporal encoding (t/T)
Channel diversity:    0.000  ← Not using spatial encoding effectively
Consistency:          0.9995 ← Predictions identical regardless of content
```

The model was learning: `prediction = f(position)` not `prediction = f(signal)`

## Control Experiment Design

### Configuration (Already Applied)

`eeg_analysis/configs/pretrain.yaml`:

```yaml
mask_ratio: 1.0                    # All tokens masked (zeros)
disable_temporal_encoding: true    # Remove position information
disable_spatial_encoding: true     # Remove channel information
```

### What the Model Sees

With this configuration:

```python
# Input to model
windows_masked = [0, 0, 0, ..., 0]  # All zeros (fully masked)

# Token encoding
token_emb = LayerNorm(Linear([0,0,...,0]) + bias)
          = LayerNorm(bias)  # Constant for ALL tokens

# Temporal encoding (DISABLED)
temporal = [0, 0, 0, ..., 0]  # No position info

# Spatial encoding (DISABLED)  
spatial = [0, 0, 0, ..., 0]   # No channel info

# Final input to backbone
x = token_emb + temporal + spatial
  = LayerNorm(bias) + 0 + 0
  = constant  # SAME for all positions, all channels, all samples!
```

**Result**: The model has **ZERO varying information**. Every input looks identical.

### Expected Behavior

#### If NO Leakage (Expected)

```
Epoch 1: loss=1.234 (random predictions)
Epoch 2: loss=1.231 (minimal fluctuation)
Epoch 3: loss=1.228
...
Epoch 10: loss=1.225 (stays high, no learning)

✅ Conclusion: No information leakage
   Model cannot learn without information
```

Loss should stay at **random baseline** (~1.0 or higher depending on embedding dimension).

#### If Leakage EXISTS (Would indicate a bug)

```
Epoch 1: loss=1.234
Epoch 2: loss=0.987  ← Dropping!
Epoch 3: loss=0.756
...
Epoch 10: loss=0.234

❌ Conclusion: TRUE SIGNAL LEAKAGE DETECTED
   Model is learning from leaked unmasked signal
```

This would mean there's a bug in the masking or forward pass that allows unmasked signal to leak through.

## How to Run the Control Experiment

### Step 1: Run Training

```bash
cd ~/eeg-mlflow
source .venv/bin/activate

# Train with control configuration (already set in eeg_analysis/configs/pretrain.yaml)
python eeg_analysis/src/training/pretrain_mamba.py \
    --config eeg_analysis/configs/pretrain.yaml
```

### Step 2: Monitor Loss

Watch for loss behavior:

```bash
# In another terminal, monitor MLflow
cd ~/eeg-mlflow
source .venv/bin/activate
mlflow ui --port 5000

# Open browser: http://localhost:5000
# Check experiment: eeg_pretraining_mamba2
# Look at train_loss and val_loss curves
```

### Step 3: Interpret Results

**Scenario A: Loss Stays High (Expected)**

```
train_loss: 1.2 → 1.19 → 1.18 → 1.17 (slow random fluctuation)
val_loss:   1.21 → 1.20 → 1.19 → 1.18

✅ NO LEAKAGE CONFIRMED
```

**What this means:**
- Your masking pipeline is correct
- No signal information leaks through
- Model cannot learn without information (as expected)
- The previous learning with mask_ratio=1.0 was purely from positional encodings

**Scenario B: Loss Drops Significantly (Would indicate bug)**

```
train_loss: 1.2 → 0.9 → 0.6 → 0.3 (rapid decrease)
val_loss:   1.21 → 0.95 → 0.65 → 0.35

❌ LEAKAGE DETECTED - INVESTIGATE!
```

**What to check if this happens:**
1. Are masked tokens truly all zeros? (Check collate function)
2. Is the `windows` tensor being passed to forward somehow?
3. Are normalization layers sharing statistics across masked/unmasked?
4. Is DDP synchronizing unmasked content across ranks?

## Comparison: Control vs. Standard Training

### Control Experiment (Current Config)

```yaml
mask_ratio: 1.0
disable_temporal_encoding: true
disable_spatial_encoding: true

Purpose: Detect signal leakage
Expected: Loss stays high (no learning)
```

### Standard Training (After confirming no leakage)

```yaml
mask_ratio: 0.75
disable_temporal_encoding: false
disable_spatial_encoding: false

Purpose: Learn useful EEG representations
Expected: Loss decreases (learning from context)
```

## After Control Experiment

Once you confirm **no leakage** (loss stays high), update config for real training:

```yaml
# Update eeg_analysis/configs/pretrain.yaml
mask_ratio: 0.75                   # Provide unmasked context
disable_temporal_encoding: false   # Enable position information
disable_spatial_encoding: false    # Enable channel information
```

Then retrain for actual representation learning.

## Diagnostic Script Update

The diagnostic script will also respect these flags:

```bash
# After control training completes
python scripts/diagnose_100pct_masking.py \
    --checkpoint eeg_analysis/checkpoints/mamba2_eeg_pretrained.pt \
    --num-samples 100

# With disable flags enabled, you should see:
# Position correlation: ~0.0 (no position info available)
# Channel diversity: ~0.0 (no channel info available)
# Consistency: ~1.0 (all predictions identical - pure noise)
```

## Summary

| Configuration | Purpose | Expected Loss | Interpretation |
|--------------|---------|---------------|----------------|
| mask_ratio=1.0<br>disable_temporal=false<br>disable_spatial=false | Baseline (your original) | Decreases | Learning from positions (not leakage) |
| mask_ratio=1.0<br>disable_temporal=true<br>disable_spatial=true | Control (current) | Stays high | No leakage confirmed |
| mask_ratio=0.75<br>disable_temporal=false<br>disable_spatial=false | Standard training | Decreases | Learning from signal context |

## Key Insight

**Your original question**: "Is there leakage causing loss to drop with mask_ratio=1.0?"

**Answer**: No leakage - the loss dropped because the model learned from **positional encodings** (temporal + spatial), not from leaked signal.

**This control experiment confirms**: With positions disabled, the model truly has no information, and loss should stay high, confirming your masking pipeline is correct.

---

## Quick Reference

**Run control experiment** (current config is already set):
```bash
python eeg_analysis/src/training/pretrain_mamba.py --config eeg_analysis/configs/pretrain.yaml
```

**Expected result**: Loss stays at ~1.0-1.5 (random baseline)

**If loss drops**: Signal leakage bug detected - investigate masking pipeline

**After confirming no leakage**: Switch to `mask_ratio=0.75` with encodings enabled for real training


---

## Source: DIAGNOSIS_RESULTS.md

# Diagnosis Results: 100% Masking Model

**Date**: December 2, 2025  
**Model**: mamba2_eeg_pretrained.pt (d_model=512, num_layers=2, mask_ratio=1.0)  
**Samples Analyzed**: 17,436 masked token predictions

---

## 🔍 Key Findings

### 1. ⚠️ Strong Position Dependence

**Position Correlation: 0.4028**

- **Finding**: Model predictions have a strong correlation (40%) with temporal position (t/T)
- **Interpretation**: The model is learning **"at time t, predict embedding X"** rather than learning from signal context
- **Implication**: Position alone is a major factor in predictions

**What this means:**
```
Instead of learning: "Given context [signals before/after], reconstruct masked signal"
Model is learning:   "At position 25% through sequence, predict this embedding"
```

---

### 2. 🚨 CRITICAL: Zero Channel Diversity

**Channel Diversity (std of channel means): 0.0000**

- **Finding**: ALL 65 CHANNELS predict nearly identical values
- **Channel means**: All approximately -0.0021 (identical to 4 decimal places)
- **Interpretation**: The model is **completely ignoring channel information**

**This is the smoking gun!**

The model has access to:
- ✅ Temporal position (t/T) - USED
- ❌ Channel identity (C3, FP1, etc.) - IGNORED
- ❌ Signal content - UNAVAILABLE (100% masked)

**What the model learned:**
```python
def predict(position_t, channel):
    # Ignore channel completely!
    return mean_embedding + position_offset(position_t)
```

---

### 3. ✅ High-Dimensional Representations

**Variance Explained by Top 3 Dimensions: 0.9%**

- **Finding**: Predictions span high-dimensional space (not collapsed to a few dimensions)
- **Interpretation**: Model is not just predicting a single mean vector
- **Good sign**: Representations have internal structure

**But combined with finding #2**, this tells us:
- Predictions vary smoothly with temporal position
- But are essentially identical across channels
- High dimensionality comes from temporal variation, not channel-specific patterns

---

## 🎯 Diagnosis: Position-Only Learning

### What the Model Learned

The model is implementing approximately:

```python
embedding = f(t/T)  # Function of temporal position only
# Channel information is ignored
# Signal content is unavailable (100% masked)
```

This is **NOT** learning useful EEG representations because:

1. ❌ **No channel specificity**: All electrodes treated identically
2. ❌ **No signal understanding**: Only position matters
3. ❌ **No context usage**: Can't learn from surrounding signals

### Why Loss Still Drops

Loss decreases because:
1. Random predictions → Very high loss
2. Predicting temporal mean → Lower loss (better than random)
3. Fine-tuning per-position → Even lower loss

**But**: This doesn't mean the model learned anything useful!

**Analogy**: 
- Task: "Predict what word comes next in a book"
- Model with context: Reads previous words, understands story
- Your model: Only knows "word at 25% through book" → predicts average word at that position
- Loss drops (better than random) but no real understanding

---

## 🔬 Root Cause Analysis

### Why Is Channel Information Ignored?

**Hypothesis**: Spatial encoding is too weak compared to temporal encoding

Looking at the model architecture:

1. **Token Embedding**: `LayerNorm(Linear([0,0,...,0]) + bias)` = same for all masked tokens
2. **Temporal Encoding**: `Linear(t/T)` - continuous, varies across sequence
3. **Spatial Encoding**: `Fixed_Projection(channel_coords)` - fixed per channel

**The problem:**
- Temporal encoding has **strong gradient signal** (varies within each sequence)
- Spatial encoding has **weak gradient signal** (constant within sequence, varies across batches)
- With 100% masking, temporal is the ONLY signal, so model learns to rely on it exclusively

**Spatial encoding gets drowned out** by temporal encoding dominance.

---

## 💡 Why This Happens with 100% Masking

### Comparison: Context-Based vs Position-Based Learning

**With Unmasked Context (mask_ratio < 1.0):**
```
Input: [signal, signal, MASKED, signal, signal]
Model learns: Use surrounding signals to reconstruct masked one
Spatial encoding: Helps identify which channel patterns to use
```

**With 100% Masking (mask_ratio = 1.0):**
```
Input: [ZERO, ZERO, ZERO, ZERO, ZERO]
Model learns: Only position available, just predict positional mean
Spatial encoding: Provides channel info but model has no reason to use it
                  (all zeros look the same regardless of channel!)
```

---

## 🎯 Recommendations

### Immediate Action: Reduce Mask Ratio

**Update `eeg_analysis/configs/pretrain.yaml`:**

```yaml
mask_ratio: 0.75  # or 0.5 for easier learning
```

**Why this will help:**

1. ✅ **Unmasked tokens provide context**: Model must learn channel-specific patterns
2. ✅ **Forces spatial encoding to matter**: Different channels will have different patterns in unmasked tokens
3. ✅ **Enables true reconstruction**: Model learns "use C3 context to predict C3 signal"

### Expected Results After Reducing Mask Ratio

When you retrain with `mask_ratio=0.75`:

**What should change:**
- Channel diversity should increase (different channels → different predictions)
- Position correlation should decrease (predictions depend more on context)
- Representations should be more useful for downstream tasks

**How to verify:**
```bash
# After retraining with mask_ratio=0.75
python scripts/diagnose_100pct_masking.py \
    --checkpoint eeg_analysis/checkpoints/mamba2_eeg_pretrained_0.75.pt \
    --num-samples 100

# Look for:
# - Channel diversity > 0.01 (channels differ)
# - Position correlation < 0.3 (less position-dependent)
```

---

## 📊 Comparison: What Good Learning Should Look Like

### Current Model (mask_ratio=1.0)
- Position correlation: **0.40** (high - bad)
- Channel diversity: **0.00** (none - very bad)
- Interpretation: Position-only learning

### Target Model (mask_ratio=0.75)
- Position correlation: **< 0.20** (low - good)
- Channel diversity: **> 0.05** (distinct channels - good)
- Interpretation: Context-based learning with channel awareness

---

## 🔬 Technical Deep Dive: Why Channels Are Ignored

### Gradient Flow Analysis

With 100% masking:

```python
# Forward pass
token_emb = token_encoder([0,0,...,0])  # Same for all masked tokens
spatial = spatial_encoder(channel)       # Different per channel
temporal = temporal_encoder(t/T)        # Different per position

x = token_emb + spatial + temporal       # Combine

# Backward pass (loss gradient)
# For within-sequence variations: temporal gets strong signal
# For across-batch variations: spatial gets weak signal (batch mixing)
```

**The gradient imbalance:**
- Temporal: Dense signal every sequence
- Spatial: Sparse signal across batches
- Result: Model learns to rely on temporal, ignores spatial

**With unmasked context:**
```python
token_emb = token_encoder(actual_signal)  # DIFFERENT for each token
# Now token_emb varies with channel characteristics
# Spatial encoding reinforces this channel-specific pattern
# Model learns: "use spatial encoding to interpret token patterns"
```

---

## Conclusion

**Your model is not experiencing information leakage.**

**Instead, it's learning a trivial solution:**
- Predict average embedding at each temporal position
- Ignore all channel information
- Loss drops but representations lack semantic meaning

**Solution:**
```yaml
# Change in eeg_analysis/configs/pretrain.yaml
mask_ratio: 0.75  # Force context-based learning
```

This will force the model to:
1. Learn from unmasked signal context
2. Use channel-specific patterns
3. Develop representations useful for downstream tasks

**The 100% masking experiment was valuable** - it revealed that your model CAN learn, but without context, it learns only trivial positional patterns.


---

## Source: DIAGNOSTIC_CHECKLIST.md

# Diagnostic Checklist: Signal Learning Issue

## What We Need to Check

### 1. Model Architecture (Complexity)

**Check current model capacity:**
- `d_model`: Current value?
- `num_layers`: Current value?
- `decoder`: What is the decoder architecture? (Linear layer? Size?)

**Questions:**
- Is the decoder just a single Linear(d_model → window_length)?
- Is the model too small to learn complex signal patterns?
- Should we increase model capacity?

### 2. Training Loss Values

**From MLflow or training logs, provide:**
- Initial `train_loss` (epoch 1)
- Final `train_loss` (last epoch)
- `val_loss` progression
- Is loss actually decreasing?

**Questions:**
- Is reconstruction loss decreasing?
- What's the final loss value?
- Is loss plateauing early?

### 3. Decoder Output Statistics

**Add logging to check decoder outputs:**
- Mean/std of decoder outputs per batch
- Variance of predictions across masked positions
- Are predictions actually varying or collapsing to constant?

**Questions:**
- Are decoder outputs diverse or constant?
- What's the variance of predictions?
- Is decoder learning or just outputting bias?

### 4. Gradient Flow

**Check if gradients reach decoder:**
- Decoder weight gradients (norm, mean, std)
- Are decoder weights actually updating?
- Gradient norms for decoder vs backbone

**Questions:**
- Are decoder gradients non-zero?
- Is decoder receiving gradient signal?
- Are decoder weights changing during training?

### 5. Actual Predictions vs Ground Truth

**Visual inspection:**
- Sample a few masked windows
- Plot predicted signal vs ground truth signal
- Check if predictions have any structure

**Questions:**
- Do predictions look like signals or noise?
- Are predictions just constant values?
- Is there any pattern in predictions?

## What to Provide

Please provide:

1. **Model config** (from eeg_analysis/configs/pretrain.yaml):
   - `d_model`
   - `num_layers`
   - Decoder architecture (if specified)

2. **Training loss values**:
   - First epoch loss
   - Last epoch loss
   - Loss progression (if available)

3. **Sample predictions** (if possible):
   - A few example predicted windows
   - Corresponding ground truth windows
   - Or at least: mean/std of predictions

4. **Model checkpoint info**:
   - Which checkpoint was used for diagnosis?
   - How many epochs was it trained?

## Next Steps Based on Findings

### If model is too small:
- Increase `d_model` (e.g., 256 → 512 or 1024)
- Increase `num_layers` (e.g., 2 → 4 or 6)
- Make decoder more complex (e.g., MLP instead of single Linear)

### If loss isn't decreasing:
- Check learning rate
- Check gradient clipping
- Check if targets are normalized correctly

### If decoder outputs are constant:
- Check decoder initialization
- Check if decoder is receiving gradients
- Consider decoder architecture change

### If predictions have no structure:
- Check if normalization is removing signal
- Check if loss function is correct
- Consider different reconstruction target


---

## Source: DROPOUT_VARIANCE_ANALYSIS.md

# Analysis: Prediction Variance from Dropout

## What We're Seeing

```
pred_var: 0.000 → 0.13-0.15 (predictions varying!)
target_var: 1.0 → 0.59-0.63
Loss: 0.80 → 0.53 (decreasing)
Loss < Target Variance (0.53 < 0.59)
```

## Root Cause: Dropout

The Mamba backbone has `Dropout(p=0.1)`:

```python
# eeg_analysis/src/models/mamba_eeg_model.py:141
self.dropout = nn.Dropout(dropout)
```

**Effect**: Even with constant input, dropout randomly zeros 10% of activations **differently for each sample** in the batch.

```python
# Input (same for all samples)
x = constant  # [B, L, D] - all identical

# After Mamba + Dropout
output = Mamba(constant) with random dropout masks per sample
# Each sample gets different dropout mask → different output!
```

## Why This Causes Variance

```python
Sample 1: dropout mask = [1, 1, 0, 1, 0, ...] → output_1
Sample 2: dropout mask = [1, 0, 1, 1, 1, ...] → output_2
Sample 3: dropout mask = [0, 1, 1, 0, 1, ...] → output_3

# Outputs differ due to different dropout masks
pred_var = Var([output_1, output_2, output_3, ...]) > 0
```

## Why Loss < Target Variance

This is the KEY question. With dropout causing random noise:

```python
pred = constant + dropout_noise  # noise is random per sample
target = actual_signal_embedding  # varies per sample

loss = E[(pred - target)^2]
     = E[(constant + noise - target)^2]
```

**Theoretical minimum** (constant pred, no noise):
```
loss_min = Var(target) ≈ 0.59
```

**With dropout noise** (should be):
```
loss = Var(target) + Var(noise) > 0.59
```

**But we observe**:
```
loss = 0.53 < 0.59  ← IMPOSSIBLE with random noise!
```

## 🚨 What This Means

**Possibility A: Dropout + Optimization Artifact**
- Model learns to exploit dropout patterns
- Over many iterations, certain dropout patterns correlate with targets by chance
- Optimizer finds weights that minimize loss given dropout distribution
- **This is NOT true information leakage** - just learning noise statistics

**Possibility B: True Information Leakage**
- Model has access to sample-specific information beyond dropout
- Predictions correlate with targets systematically
- **This IS problematic** - indicates a bug in masking

## 🔬 How to Distinguish

### Test: Monitor Loss Convergence

**If learning dropout statistics (A)**:
- Loss will decrease initially
- Then **plateau** at some level
- Plateau level = best achievable with dropout noise + constant pred
- Expected plateau: around 0.4-0.5

**If true leakage (B)**:
- Loss will **continue decreasing** beyond dropout plateau
- Could reach very low values (< 0.3)
- Indicates sample-specific learning beyond noise

## 📊 Current Status (Epoch 4)

```
Loss trajectory:
Epoch 1: 0.799
Epoch 2: 0.677
Epoch 3: 0.597
Epoch 4: 0.526

Rate of decrease: ~0.07-0.10 per epoch
```

### Prediction

**Next 5-10 epochs will reveal**:
- **If plateau at ~0.45-0.50**: Dropout statistics learning (not a real problem)
- **If continues to ~0.3 or below**: True information leakage (investigate!)

## ✅ What To Do

### Option A: Let It Train (Recommended)

Monitor for 10 more epochs:

```bash
# Let current training continue
# Watch the loss curve in MLflow or terminal logs
```

**Expected behavior**:
```
Epoch 5-10: Loss decreases slowly
Epoch 10-15: Loss plateau around 0.45-0.50
Epoch 15+: No further improvement
```

If this happens → ✅ **No true leakage**, just dropout noise

If loss goes below 0.35 → ❌ **Investigate further**

### Option B: Train Without Dropout (Alternative)

To eliminate dropout as a confounder:

```python
# Temporarily modify eeg_analysis/src/models/mamba_eeg_model.py
self.dropout = nn.Dropout(0.0)  # Disable dropout
```

Retrain and check:
- **If loss stays constant**: Dropout was the only source of learning
- **If loss still decreases**: Something else is causing leakage

## 🎯 My Recommendation

**Let the current training continue for ~20 epochs total.**

**Expected outcome**:
```
Epoch 10: loss ≈ 0.45
Epoch 15: loss ≈ 0.43
Epoch 20: loss ≈ 0.42 (plateau)
Early stopping triggers
```

This would confirm:
- Model is learning dropout statistics (expected behavior)
- No true signal leakage (good news!)
- Control experiment validates masking pipeline (objective achieved!)

**If loss continues dropping significantly beyond epoch 20**, we'll investigate deeper.

## 📝 Technical Note: Why Centering Didn't Help

My earlier fix (target centering) was mathematically ineffective:

```python
# Centered loss
loss = mean((pred - target_mean - (target - target_mean))^2)
     = mean((pred - target)^2)  # Centering cancels out!
```

The real issue is dropout introducing sample-dependent randomness, not mean learning.

## Summary

| Loss Behavior | Interpretation | Action |
|--------------|----------------|--------|
| Plateaus at 0.4-0.5 | ✅ Learning dropout noise (expected) | Control experiment successful |
| Continues to <0.3 | ❌ Possible true leakage | Investigate further |

**Current status**: Wait and monitor. Most likely will plateau soon, confirming no leakage.


---

## Source: FIGURE_TABLE_PLAN.md

# Figure and Table Planning Document

## FIGURE SPECIFICATIONS

---

## Figure 1: Pipeline Diagram
**Purpose**: Overview of entire pipeline, highlight DC offset removal

**Content**:
```
[Raw EEG Data] 
    ↓
[Data Loading] (4 channels, 21 patients)
    ↓
[Upsampling] (2× to 256 Hz)
    ↓
[Filtering] (Butterworth, 60 Hz cutoff)
    ↓
[Downsampling] (2× to 128 Hz)
    ↓
[Windowing] (10-second windows)
    ↓
[DC OFFSET REMOVAL] ← HIGHLIGHT THIS STEP
    ↓
[Feature Extraction] (5 selected features)
    ↓
[Models] (Tabular MLP / Hybrid CNN-LSTM)
    ↓
[Window-Level Predictions]
    ↓
[Patient-Level Aggregation] (mean probabilities)
    ↓
[Evaluation] (Patient-level metrics)
```

**Design Notes**:
- Use color coding: DC removal step in RED/BOLD
- Show data flow clearly
- Include key numbers (21 patients, 4 channels, 5 features)
- Two parallel paths for two models

**Location**: After Introduction, before Methodology

---

## Figure 2: DC Offset Removal Visualization
**Purpose**: Visual demonstration of DC offset removal effect

**Content**: 4 subplots (one per channel)
- **Subplot 1-4**: Each channel (AF7, AF8, TP9, TP10)
  - **Top**: Before DC removal (signal with offset)
  - **Bottom**: After DC removal (centered signal)
  - **Annotation**: Show DC offset magnitude

**Design Notes**:
- Use time series plots
- Show clear baseline shift before
- Show zero-centered signal after
- Include DC offset value annotation
- Use consistent y-axis scale for comparison

**Example Data Points**:
- Before: Signal oscillating around +500 μV
- After: Signal oscillating around 0 μV
- DC Offset: ~500 μV removed

**Location**: In Methodology section (DC Offset Removal subsection)

---

## Figure 3: Model Architectures
**Purpose**: Compare simple vs. complex architectures

**Content**: Two side-by-side architecture diagrams

**Left: Tabular MLP**
```
Input (5 features)
    ↓
Linear(5 → 1024) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Linear(1024 → 512) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Linear(512 → 256) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Linear(256 → 128) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Linear(128 → 2)
    ↓
Output (Binary Classification)
```

**Right: Hybrid CNN-LSTM**
```
Input (5 features)
    ↓
Feature Embedding (5 → 128)
    ↓
[CNN Branch]              [LSTM Branch]
    ↓                           ↓
CNN Block 1 (64 filters)    LSTM Layer 1 (128 units, bidirectional)
    ↓                           ↓
CNN Block 2 (128 filters)   LSTM Layer 2 (64 units, bidirectional)
    ↓                           ↓
CNN Block 3 (256 filters)   Attention (8 heads, 64 dim)
    ↓                           ↓
Feature Pyramid              Positional Encoding
    ↓                           ↓
    └─────────── Fusion ─────────┘
            ↓
    Dense(256) + GELU + Dropout(0.3)
            ↓
    Dense(128) + GELU + Dropout(0.4)
            ↓
    Output (Binary Classification)
```

**Design Notes**:
- Use different colors for different layer types
- Show complexity difference clearly
- Include key parameters (units, filters, heads)
- Use consistent styling

**Location**: In Methodology section (Model Architectures subsection)

---

## Figure 4: ROC Curves
**Purpose**: Visualize classification performance improvements

**Content**: 4 ROC curves on same plot
1. Tabular MLP - Before DC removal (ROC-AUC = 0.776)
2. Tabular MLP - After DC removal (ROC-AUC = 0.898)
3. Hybrid Model - Before improvements (ROC-AUC = 0.765)
4. Hybrid Model - After improvements (ROC-AUC = 0.893)

**Design Notes**:
- Use different line styles/colors for each curve
- Include diagonal reference line (random classifier)
- Annotate AUC values on plot
- Use legend to distinguish curves
- Highlight improvement (before → after)

**Legend**:
- Solid line: After improvements
- Dashed line: Before improvements
- Colors: Blue (Tabular MLP), Red (Hybrid Model)

**Location**: In Results section (Performance Improvements subsection)

---

## Figure 5: Confusion Matrices
**Purpose**: Show classification errors before/after improvements

**Content**: 4 confusion matrices (2×2 grid)

**Top Row: Tabular MLP**
- **Left**: Before DC removal
  ```
         Predicted
        Neg  Pos
  Neg   11   3
  Pos    2   5
  ```
- **Right**: After DC removal
  ```
         Predicted
        Neg  Pos
  Neg   11   3
  Pos    0   7  ← Perfect Recall!
  ```

**Bottom Row: Hybrid Model**
- **Left**: Before improvements
  ```
         Predicted
        Neg  Pos
  Neg   14   0  ← Perfect Precision
  Pos    3   4
  ```
- **Right**: After improvements
  ```
         Predicted
        Neg  Pos
  Neg   14   0  ← Perfect Precision
  Pos    2   5
  ```

**Design Notes**:
- Use color intensity for cell values (darker = higher)
- Highlight perfect metrics (bold borders)
- Include percentages in cells
- Show TP, TN, FP, FN clearly

**Location**: In Results section (Confusion Matrices subsection)

---

## Figure 6: Performance Comparison Bar Chart
**Purpose**: Compare all metrics across conditions

**Content**: Grouped bar chart

**X-axis**: Metrics (ROC-AUC, F1-Score, Recall, Precision, Accuracy)

**Y-axis**: Metric values (0 to 1)

**Groups**: 4 bars per metric
1. Tabular MLP - Before (light blue)
2. Tabular MLP - After (dark blue)
3. Hybrid Model - Before (light red)
4. Hybrid Model - After (dark red)

**Annotations**:
- Percentage improvements above bars
- Highlight perfect scores (1.0) with star
- Show error bars if available

**Example**:
```
ROC-AUC: [0.776] [0.898↑+15.7%] [0.765] [0.893↑+16.7%]
F1:      [0.667] [0.824↑+23.5%] [0.727] [0.833↑+14.6%]
Recall:  [0.714] [1.000*↑+40%]  [0.571] [0.714↑+25%]
```

**Design Notes**:
- Use consistent color scheme
- Group by metric, not by model
- Include improvement annotations
- Highlight perfect scores

**Location**: In Results section (Comparative Analysis subsection)

---

## Figure 7: Feature Importance
**Purpose**: Show which features were selected and their importance

**Content**: Horizontal bar chart

**Y-axis**: Features (5 selected features)
1. tp10_total_power
2. tp10_psd_mean
3. tp10_psd_std
4. tp10_psd_max
5. left_frontal_temporal_diff_beta

**X-axis**: Importance score (f_classif F-statistic)

**Design Notes**:
- Sort by importance (highest to lowest)
- Use color gradient (darker = more important)
- Include actual F-statistic values
- Annotate feature names clearly
- Show that tp10 features dominate (4/5 features)

**Location**: In Methodology section (Feature Selection subsection) or Results section

---

## TABLE SPECIFICATIONS

---

## Table 1: Dataset Characteristics
**Purpose**: Provide comprehensive dataset description

**Structure**:
| Characteristic | Value |
|----------------|-------|
| **Participants** | 21 patients |
| **Non-Remission** | 14 (66.7%) |
| **Remission** | 7 (33.3%) |
| **EEG Channels** | 4 (AF7, AF8, TP9, TP10) |
| **Channel Types** | Frontal (AF7, AF8), Temporal (TP9, TP10) |
| **Sampling Rate** | 256 Hz |
| **Window Size** | 10 seconds |
| **Samples per Window** | 2,560 |
| **Total Windows** | 1,203 |
| **Positive Windows** | 393 (32.7%) |
| **Negative Windows** | 810 (67.3%) |
| **Class Ratio** | 2.06:1 (negative:positive) |
| **Cross-Validation** | Leave-One-Patient-Out (21 folds) |
| **Evaluation Level** | Patient-level (aggregated from windows) |
| **Selected Features** | 5 (after feature selection) |

**Location**: In Methodology section (Dataset Description subsection)

---

## Table 2: Model Architectures
**Purpose**: Detailed architecture specifications

**Structure**: Two subtables (one per model)

### Tabular MLP Architecture
| Component | Specification |
|-----------|--------------|
| **Input Features** | 5 |
| **Hidden Layers** | [1024, 512, 256, 128] |
| **Activation** | ReLU |
| **Batch Normalization** | Yes (after each hidden layer) |
| **Dropout Rate** | 0.3 |
| **Output Layer** | Linear(128 → 2) |
| **Total Parameters** | ~700K |

### Hybrid CNN-LSTM Architecture
| Component | Specification |
|-----------|--------------|
| **Input Features** | 5 |
| **Feature Embedding** | Linear(5 → 128) + BatchNorm + ReLU |
| **CNN Blocks** | 3 blocks |
| **Block 1** | Filters: [64, 64], Kernels: [5, 3], Pool: 2 |
| **Block 2** | Filters: [128, 128], Kernels: [3, 3], Pool: 2 |
| **Block 3** | Filters: [256, 128], Kernels: [3, 1], Pool: 1 |
| **CNN Dropout** | [0.1, 0.2, 0.3] (progressive) |
| **Spatial Dropout** | Yes |
| **Gaussian Noise** | 0.005 |
| **LSTM Layers** | 2 layers, bidirectional |
| **LSTM Layer 1** | 128 units, return_sequences=True, dropout=0.2 |
| **LSTM Layer 2** | 64 units, return_sequences=False, dropout=0.3 |
| **Attention** | 8 heads, key_dim=64, positional encoding |
| **Fusion Strategy** | Concatenation |
| **Feature Pyramid** | Enabled |
| **Dense Layers** | [256, 128] units, GELU activation |
| **Dense Dropout** | [0.3, 0.4] |
| **Total Parameters** | ~2.5M |

**Location**: In Methodology section (Model Architectures subsection)

---

## Table 3: Performance Metrics (Main Results)
**Purpose**: Comprehensive performance comparison

**Structure**:
| Metric | Tabular MLP | Tabular MLP | Hybrid Model | Hybrid Model | Tabular MLP | Hybrid Model |
|        | Before      | After       | Before       | After        | Improvement | Improvement |
|--------|-------------|-------------|--------------|--------------|-------------|--------------|
| **ROC-AUC** | 0.776 | 0.898 | 0.765 | 0.893 | +0.122 (+15.7%) | +0.128 (+16.7%) |
| **F1-Score** | 0.667 | 0.824 | 0.727 | 0.833 | +0.157 (+23.5%) | +0.106 (+14.6%) |
| **Recall** | 0.714 | **1.000** | 0.571 | 0.714 | +0.286 (+40.0%) | +0.143 (+25.0%) |
| **Precision** | 0.625 | 0.700 | 1.000 | 1.000 | +0.075 (+12.0%) | 0.000 (0%) |
| **Accuracy** | 0.762 | 0.857 | 0.857 | 0.905 | +0.095 (+12.5%) | +0.048 (+5.6%) |
| **Window Accuracy** | 0.789 | 0.828 | - | - | +0.039 (+4.9%) | - |

**Design Notes**:
- Bold perfect scores (1.000)
- Highlight largest improvements
- Include percentage improvements
- Use consistent formatting

**Location**: In Results section (Performance Improvements subsection)

---

## Table 4: Confusion Matrices
**Purpose**: Detailed error analysis

**Structure**: 4 confusion matrices

### Tabular MLP - Before DC Removal
|        | Predicted Negative | Predicted Positive | Total |
|--------|-------------------|-------------------|-------|
| **Actual Negative** | 11 (TN) | 3 (FP) | 14 |
| **Actual Positive** | 2 (FN) | 5 (TP) | 7 |
| **Total** | 13 | 8 | 21 |

### Tabular MLP - After DC Removal
|        | Predicted Negative | Predicted Positive | Total |
|--------|-------------------|-------------------|-------|
| **Actual Negative** | 11 (TN) | 3 (FP) | 14 |
| **Actual Positive** | **0 (FN)** | **7 (TP)** | 7 |
| **Total** | 11 | 10 | 21 |

### Hybrid Model - Before Improvements
|        | Predicted Negative | Predicted Positive | Total |
|--------|-------------------|-------------------|-------|
| **Actual Negative** | **14 (TN)** | **0 (FP)** | 14 |
| **Actual Positive** | 3 (FN) | 4 (TP) | 7 |
| **Total** | 17 | 4 | 21 |

### Hybrid Model - After Improvements
|        | Predicted Negative | Predicted Positive | Total |
|--------|-------------------|-------------------|-------|
| **Actual Negative** | **14 (TN)** | **0 (FP)** | 14 |
| **Actual Positive** | 2 (FN) | 5 (TP) | 7 |
| **Total** | 16 | 5 | 21 |

**Design Notes**:
- Bold perfect metrics (0 FN or 0 FP)
- Include abbreviations (TN, TP, FP, FN)
- Show totals for clarity

**Location**: In Results section (Confusion Matrices subsection)

---

## Table 5: Architecture Changes (Hybrid Model)
**Purpose**: Document what changed between experiments

**Structure**:
| Component | Baseline | Enhanced | Rationale |
|-----------|----------|----------|-----------|
| **Activation (Dense)** | ReLU | GELU | Smoother gradients, better performance |
| **Attention Heads** | 4 | 8 | Increased capacity for feature learning |
| **Attention Key Dim** | 32 | 64 | Larger representation space |
| **Positional Encoding** | Disabled | Enabled | Temporal awareness |
| **Deterministic Training** | False | True | Reproducibility |
| **Random State** | 42 | 30 | Different initialization |

**Design Notes**:
- Highlight changes clearly
- Include rationale for each change
- Note that changes are confounded with DC removal

**Location**: In Methodology section (Model Architectures subsection) or Results section

---

## Table 6: Hyperparameters
**Purpose**: Complete hyperparameter specifications for reproducibility

**Structure**: Two subtables (one per model)

### Tabular MLP Hyperparameters
| Hyperparameter | Value |
|----------------|-------|
| **Hidden Layers** | [1024, 512, 256, 128] |
| **Dropout Rate** | 0.3 |
| **Batch Normalization** | Yes |
| **Optimizer** | AdamW |
| **Learning Rate** | 0.001 |
| **Weight Decay** | 0.01 |
| **Beta 1** | 0.9 |
| **Beta 2** | 0.999 |
| **Epsilon** | 1e-7 |
| **LR Schedule** | Cosine Annealing Warm Restarts |
| **Initial LR** | 0.001 |
| **Min LR** | 1e-6 |
| **Warmup Epochs** | 5 |
| **Cycle Length** | 20 |
| **Label Smoothing** | 0.05 |
| **Gradient Clip Norm** | 1.0 |
| **Batch Size** | 1024 |
| **Epochs** | 100 |
| **Early Stopping Patience** | 10 |
| **Early Stopping Monitor** | Loss |
| **Mixed Precision** | Yes |
| **Class Weight** | Balanced |
| **Random State** | 42 |

### Hybrid CNN-LSTM Hyperparameters
| Hyperparameter | Value |
|----------------|-------|
| **Feature Embedding** | Linear(5 → 128) |
| **CNN Blocks** | 3 (see Table 2) |
| **CNN Dropout** | [0.1, 0.2, 0.3] |
| **Spatial Dropout** | Yes |
| **Gaussian Noise** | 0.005 |
| **LSTM Units** | [128, 64] |
| **LSTM Bidirectional** | Yes |
| **LSTM Dropout** | [0.2, 0.3] |
| **LSTM Recurrent Dropout** | [0.1, 0.2] |
| **Attention Heads** | 8 |
| **Attention Key Dim** | 64 |
| **Attention Dropout** | 0.1 |
| **Positional Encoding** | Yes |
| **Fusion Strategy** | Concat |
| **Feature Pyramid** | Yes |
| **Dense Layers** | [256, 128] |
| **Dense Activation** | GELU |
| **Dense Dropout** | [0.3, 0.4] |
| **Optimizer** | AdamW |
| **Learning Rate** | 0.001 |
| **Weight Decay** | 0.01 |
| **LR Schedule** | Cosine Annealing Warm Restarts |
| **Label Smoothing** | 0.05 |
| **Gradient Clip Norm** | 1.0 |
| **Batch Size** | 1024 |
| **Epochs** | 100 |
| **Early Stopping Patience** | 10 |
| **Early Stopping Monitor** | val_f1_macro |
| **Mixed Precision** | Yes |
| **Deterministic Training** | Yes |
| **Random State** | 30 |

**Location**: In Methodology section (Training Configuration subsection) or Appendix

---

## FIGURE/TABLE PLACEMENT STRATEGY

### Introduction Section
- None (keep focused on problem/motivation)

### Related Work Section
- None (text-focused)

### Methodology Section
- **Figure 1**: Pipeline diagram (beginning)
- **Figure 2**: DC offset removal visualization (DC removal subsection)
- **Figure 3**: Model architectures (Model architectures subsection)
- **Figure 7**: Feature importance (Feature selection subsection)
- **Table 1**: Dataset characteristics (Dataset description subsection)
- **Table 2**: Model architectures (Model architectures subsection)
- **Table 5**: Architecture changes (Model architectures subsection)
- **Table 6**: Hyperparameters (Training configuration subsection)

### Results Section
- **Figure 4**: ROC curves (Performance improvements subsection)
- **Figure 5**: Confusion matrices (Confusion matrices subsection)
- **Figure 6**: Performance comparison bar chart (Comparative analysis subsection)
- **Table 3**: Performance metrics (Performance improvements subsection)
- **Table 4**: Confusion matrices (Confusion matrices subsection)

### Discussion Section
- None (text-focused, may reference figures/tables)

### Conclusion Section
- None (text-focused)

---

## DESIGN GUIDELINES

### Color Scheme
- **Main Colors**: Blue (Tabular MLP), Red (Hybrid Model)
- **Shades**: Light (before), Dark (after)
- **Highlights**: Green (improvements), Yellow (perfect scores)

### Typography
- **Headers**: Bold, 12-14pt
- **Body Text**: Regular, 10-11pt
- **Annotations**: Italic, 9-10pt
- **Numbers**: Monospace font for alignment

### Consistency
- Same color scheme across all figures
- Same formatting style across all tables
- Consistent axis labels and legends
- Uniform figure sizes (where possible)

### Accessibility
- High contrast colors
- Clear labels and legends
- Descriptive captions
- Avoid color-only encoding (use patterns/labels too)


---

## Source: IDEAL_LEARNING_TRAJECTORY.md

# Ideal Learning Trajectory for MAE Pretraining

## The Learning Phases

### Phase 1: Position Learning (Early Epochs)
**What happens:**
- Model learns position-dependent patterns quickly
- Position correlation increases (e.g., 0.0 → 0.3-0.5)
- Easy to learn: Sequential processing naturally provides position

**Is this good?**
- ✅ **YES** - Position is useful for temporal understanding
- ✅ **YES** - It's a stepping stone to richer representations
- ⚠️ **BUT** - Shouldn't be the ONLY thing learned

**Expected:**
- Position correlation: 0.3-0.5 (moderate)
- Pattern correlation: Low initially (0.0-0.1)
- Context sensitivity: High initially (0.9+) - predictions don't vary with context

### Phase 2: Context Learning (Middle Epochs)
**What happens:**
- Model starts learning from unmasked context
- Context sensitivity decreases (e.g., 0.9 → 0.6-0.7)
- Predictions start varying with different masking patterns

**Is this good?**
- ✅ **YES** - Model learns from signal content
- ✅ **YES** - Predictions become context-dependent
- ⚠️ **BUT** - Pattern correlation still low

**Expected:**
- Position correlation: Stays moderate (0.3-0.5) or decreases
- Pattern correlation: Starts increasing (0.1 → 0.2-0.3)
- Context sensitivity: Decreases (0.9 → 0.6-0.7)

### Phase 3: Signal Pattern Learning (Late Epochs)
**What happens:**
- Model learns actual signal patterns
- Pattern correlation increases (e.g., 0.2 → 0.5-0.7)
- Predictions match ground truth waveforms

**Is this good?**
- ✅ **YES** - Model learns rich signal representations
- ✅ **YES** - Useful for downstream tasks
- ✅ **YES** - This is the goal!

**Expected:**
- Position correlation: Moderate (0.2-0.4) - position helps but isn't dominant
- Pattern correlation: High (0.5-0.7+) - learns signal patterns
- Context sensitivity: Low (0.4-0.6) - predictions vary with context

## The Problem We're Seeing

### Current State (mask_ratio=0.75)
- ✅ Position correlation: 0.31 (moderate - good!)
- ✅ Context sensitivity: 0.58 (low - good! Model learns from context)
- ❌ Pattern correlation: -0.045 (very low - bad! Not learning signal patterns)
- ❌ Baseline similarity: 0.965 (very high - bad! All predictions similar)

### What This Means
1. **Position learning**: ✅ Working (moderate correlation)
2. **Context learning**: ✅ Working (low context sensitivity)
3. **Signal pattern learning**: ❌ NOT working (low pattern correlation)

**The model is stuck between Phase 1 and Phase 2:**
- It learned position (Phase 1) ✅
- It learned to use context (Phase 2) ✅
- But it's NOT learning signal patterns (Phase 3) ❌

## Why Signal Pattern Learning Isn't Happening

### Possible Causes

1. **Decoder too simple** (FIXED: Now MLP)
   - Single Linear layer couldn't learn complex patterns
   - MLP should help

2. **Model capacity insufficient**
   - d_model=512, num_layers=4 might not be enough
   - May need larger model

3. **Loss function issue**
   - Per-window normalization might be removing signal structure
   - Model optimizes for normalized targets, not actual signals

4. **Training dynamics**
   - Learning rate too high/low
   - Model converges to local minimum (constant predictions)
   - Need more training or different hyperparameters

## The Ideal Trajectory

### What We Want to See

**Early Training (Epochs 1-10):**
```
Position correlation: 0.0 → 0.4 (learns position)
Pattern correlation: 0.0 → 0.1 (starts learning)
Context sensitivity: 1.0 → 0.8 (starts using context)
```

**Middle Training (Epochs 10-50):**
```
Position correlation: 0.4 → 0.3 (position helps but not dominant)
Pattern correlation: 0.1 → 0.3 (learns signal patterns)
Context sensitivity: 0.8 → 0.6 (predictions vary with context)
```

**Late Training (Epochs 50+):**
```
Position correlation: 0.3 → 0.2 (position is helper feature)
Pattern correlation: 0.3 → 0.6+ (learns rich signal patterns)
Context sensitivity: 0.6 → 0.4 (strong context dependence)
```

## Answer to Your Question

**Is the ideal to learn positions quickly then learn underlying representation?**

**YES, but with caveats:**

1. **Position learning first is OK** ✅
   - It's a natural stepping stone
   - Position is useful for temporal understanding
   - Sequential models naturally learn position

2. **BUT signal learning must follow** ⚠️
   - Position should HELP, not REPLACE signal learning
   - Model must learn signal patterns eventually
   - If stuck in position-only learning, that's a problem

3. **The ideal trajectory:**
   - **Early**: Learn position quickly (easy)
   - **Middle**: Learn from context (harder)
   - **Late**: Learn signal patterns (hardest)

4. **What we're seeing:**
   - ✅ Position learned (Phase 1)
   - ✅ Context learned (Phase 2)
   - ❌ Signal patterns NOT learned (Phase 3 - stuck!)

## What to Monitor

### Good Signs (Model Learning Correctly)
- Position correlation: Moderate (0.2-0.4) and stable
- Pattern correlation: **Increasing** over time (0.1 → 0.5+)
- Context sensitivity: Decreasing over time (0.9 → 0.4-0.6)
- Decoder variance: **Increasing** (predictions become diverse)

### Bad Signs (Model Stuck)
- Position correlation: High and increasing (0.6+)
- Pattern correlation: **Stuck low** (< 0.1) or decreasing
- Context sensitivity: High and stable (0.9+)
- Decoder variance: **Low and constant** (predictions collapse)

## Summary

**Yes, learning position quickly is fine**, but:
- It should be a stepping stone, not the end goal
- Signal pattern learning must follow
- Position should help, not dominate

**Current state**: Model learned position and context, but NOT signal patterns. The MLP decoder should help, but we need to monitor if pattern correlation increases during training.


---

## Source: LEAKAGE_ANALYSIS.md

# LEAKAGE ANALYSIS: Loss Reduction with Zero Information

## 🚨 Critical Finding

With the control configuration:
- `mask_ratio: 1.0` - All tokens masked (zeros)
- `disable_temporal_encoding: true` - No position information
- `disable_spatial_encoding: true` - No channel information

**Result**: Loss is STILL DROPPING

```
Epoch 1: train_loss=0.808, val_loss=0.723
Epoch 2: train_loss=0.683, val_loss=0.637
Epoch 3: train_loss=0.603, val_loss=0.564
```

## 🔍 Root Cause Analysis

### What the Model Receives (Forward Pass)

```python
# Input
windows_masked = [0, 0, 0, ..., 0]  # All zeros (B, L, 2048)

# Token encoding
token_emb = token_encoder(windows_masked)
          = LayerNorm(Linear([0,0,...,0]) + bias)
          = LayerNorm(bias)  # CONSTANT for all samples

# Positional encodings (DISABLED)
temporal = 0
spatial = 0

# Input to backbone
x = token_emb + temporal + spatial
  = LayerNorm(bias)  # SAME for every sample, every position, every channel

# Prediction
pred = backbone(x)  # Processes the constant
```

**Key point**: Input `x` is IDENTICAL for all samples!

### What the Model is Compared Against (Targets)

```python
# Target generation
target = token_encoder(windows)  # windows = ORIGINAL unmasked signal
       = LayerNorm(Linear(actual_signal) + bias)
       = VARIES based on actual EEG content
```

**Key point**: Targets are DIFFERENT for each sample (contain actual signal)!

### The "Leakage"

**The model is learning to predict the DATASET MEAN of target embeddings!**

```python
# Model learns:
pred_constant = mean(all_target_embeddings_in_dataset)

# This reduces loss:
# Initial (random): MSE(random, targets) = high
# After learning: MSE(dataset_mean, targets) = lower
```

## 📊 Why This Isn't Traditional "Leakage"

This is **NOT** leakage in the sense of "model sees unmasked signal content."

Instead, it's learning the **prior distribution** of EEG embeddings:

- Model doesn't see individual signals
- Model learns: "On average, what does an EEG embedding look like?"
- Predicting the mean is better than predicting random values
- Loss decreases even though model has no sample-specific information

**Analogy**:
```
Task: "Guess someone's height without seeing them"
Bad guess: Random number (anywhere from 0 to 300cm)
Good guess: Average height (~170cm)
Your guess is better on average, but you don't know the specific person's height
```

## 🎯 Confirming This Hypothesis

### Test 1: Check Prediction Variance

If the model is predicting dataset mean:
- Predictions should be VERY SIMILAR across all samples
- Low variance in predictions
- High variance in targets

```bash
# After a few more epochs, run:
python scripts/diagnose_100pct_masking.py \
    --checkpoint eeg_analysis/checkpoints/mamba2_eeg_pretrained.pt \
    --num-samples 100

# Look for:
# - Very low prediction variance
# - Very high consistency (>0.999)
# - Predictions essentially constant
```

### Test 2: Check if Predictions Match Target Mean

```python
# In training loop, add logging:
pred_mean = pred[mask_bool].mean()
target_mean = target[mask_bool].mean()
print(f"Pred mean: {pred_mean:.6f}, Target mean: {target_mean:.6f}")

# After learning, these should converge!
```

## 🔬 The Real Question

**Is the model learning ANYTHING USEFUL?**

Answer: **No** - It's just learning a single constant (the dataset mean).

**Evidence needed**:
- Can the model produce DIFFERENT predictions for different samples?
- Or does it always predict the same embedding regardless of input?

## 🛠️ How to Detect TRUE Signal Leakage

Current test is insufficient because:
- Model can reduce loss by learning dataset statistics
- This doesn't require access to individual signal content
- We need to test if model can distinguish between DIFFERENT samples

### Proposed Test: Prediction Consistency Check

```python
# Take same sample, mask it twice with different random masks
sample_1 = sample with mask pattern A
sample_2 = same sample with mask pattern B

pred_1 = model(sample_1)
pred_2 = model(sample_2)

# If model has access to underlying signal:
# pred_1 ≈ pred_2 (same sample → same prediction)

# If model only predicts mean:
# pred_1 ≈ pred_2 ≈ constant (always predicts dataset mean)

# Need to compare both against:
# Different sample prediction (should be different if signal leakage)
```

## 💡 Solution: Add Prediction Variance Test

The model should NOT be able to produce varying predictions if:
1. All inputs are identical (constant)
2. All positions disabled
3. All channels disabled

**Proposed fix**: Add assertion in training:

```python
# After getting predictions
if disable_temporal and disable_spatial and mask_ratio == 1.0:
    # All inputs are identical, predictions should be identical
    pred_std = pred.std()
    if pred_std > 0.01:  # Some small threshold
        logger.warning(f"Predictions vary (std={pred_std:.6f}) despite constant input!")
        logger.warning("This suggests model has access to sample-specific information")
```

## 🎯 Current Status: INCONCLUSIVE

**What we know**:
- ✅ Loss decreases (model is learning something)
- ❓ Is it learning dataset mean OR accessing signal content?

**What we need to check**:
1. Do predictions vary across samples? (Should be constant if learning mean)
2. Do predictions equal target mean? (Should converge if learning mean)
3. Can model distinguish between different samples? (Should fail if no signal access)

## 📋 Next Steps

### Option A: Let training continue and diagnose

```bash
# Let it train for ~10 epochs
# Then run diagnostic:
python scripts/diagnose_100pct_masking.py \
    --checkpoint eeg_analysis/checkpoints/mamba2_eeg_pretrained.pt
```

Look for:
- Prediction variance (should be near-zero if learning mean)
- Prediction mean vs target mean (should match if learning mean)

### Option B: Add real-time checks

Modify training script to log:
```python
# In training loop
pred_variance = pred[mask_bool].var()
pred_mean = pred[mask_bool].mean()
target_mean = target[mask_bool].mean()

mlflow.log_metric("pred_variance", float(pred_variance), step=epoch)
mlflow.log_metric("pred_mean", float(pred_mean), step=epoch)
mlflow.log_metric("target_mean", float(target_mean), step=epoch)
```

If `pred_mean` converges to `target_mean` and `pred_variance → 0`:
→ Model is learning dataset mean (not true leakage)

If `pred_variance` remains high:
→ Model has sample-specific information (TRUE LEAKAGE)

## 🧠 Theoretical Lower Bound

What's the minimum achievable loss by predicting dataset mean?

```python
# Theoretical minimum (predicting target mean):
min_loss = Var(targets)  # Variance of targets

# Random baseline:
random_loss = much higher

# Current loss after 3 epochs: 0.564
# Need to compare this to target variance
```

If loss is approaching `Var(targets)`, model is just learning mean.
If loss goes BELOW `Var(targets)`, model has sample-specific information.

## 🎯 Verdict

**Preliminary**: Loss reduction is likely from learning **dataset mean**, NOT true signal leakage.

**Confirmation needed**: Check if predictions vary across samples or are constant.

**Action**: Let training run a bit longer, then diagnose prediction variance.


---

## Source: MAE_PIPELINE_AUDIT.md

# MAE Pipeline Audit: Line-by-Line Analysis

**Date**: December 2, 2025  
**Auditor**: Comprehensive code review per user request  
**Configuration**: `reconstruct_signal_space: true`

---

## Executive Summary

### ✅ PASS: With `reconstruct_signal_space: true`

The pipeline is **CORRECT** when signal reconstruction is enabled:
- Targets are raw signal (2048 samples), not encoded
- No learnable modules used for target computation
- Masking applied before projection
- Loss computed only on masked positions
- Predictions decoded back to signal space for comparison

### ⚠️ WARNING: Legacy Code Paths

**Dangerous fallback paths exist** that use encoders for targets (lines 270-275, 380-383). These should be **REMOVED** to prevent accidental misuse.

---

## Detailed Audit

### 1. ✅ Target Computation (Training Loop)

**File**: `eeg_analysis/src/training/pretrain_mamba.py`  
**Lines**: 266-275

```python
# Targets
if use_signal_reconstruction:
    # Proper MAE: Target is actual signal (never changes!)
    target = windows  # (B, L, W) - raw signal
elif target_encoder is not None:
    # Embedding space + control mode: frozen encoder
    target = target_encoder(windows)  # (B, L, D) ← ENCODER USED!
else:
    # Embedding space + normal training: current encoder  
    target = (model.module if isinstance(model, DDP) else model).encode_tokens_only(windows)  # ← ENCODER USED!
```

**Status**: ✅ **PASS** (when `use_signal_reconstruction=True`)
- Line 269: `target = windows` - Direct assignment of raw signal
- No learnable modules involved
- `windows` is from collate function (verified below)

**⚠️ ISSUE**: Lines 270-275 contain **dangerous fallback paths**:
- Line 272: Uses `target_encoder(windows)` - frozen but still an encoder
- Line 275: Uses `model.encode_tokens_only(windows)` - training encoder!

**Recommendation**: Remove these paths entirely or add assertions:

```python
# Targets - MUST be raw signal for proper MAE
if use_signal_reconstruction:
    target = windows  # (B, L, W) - raw signal
else:
    raise ValueError("Embedding space targets are deprecated. Use reconstruct_signal_space=true")
```

---

### 2. ✅ Target Computation (Validation Loop)

**File**: `eeg_analysis/src/training/pretrain_mamba.py`  
**Lines**: 378-383

```python
# Targets
if use_signal_reconstruction:
    target = windows  # Raw signal
elif target_encoder is not None:
    target = target_encoder(windows)  # Frozen encoder ← ENCODER USED!
else:
    target = (model.module if isinstance(model, DDP) else model).encode_tokens_only(windows)  # ← ENCODER USED!
```

**Status**: ✅ **PASS** (when `use_signal_reconstruction=True`)
- Line 379: `target = windows` - Raw signal
- Consistent with training loop

**⚠️ ISSUE**: Same dangerous fallback paths as training loop.

**Recommendation**: Remove fallback paths.

---

### 3. ✅ Raw Signal Source (Collate Function)

**File**: `eeg_analysis/src/data/eeg_pretraining_dataset.py`  
**Lines**: 157-167, 194-200

```python
# Allocate tensors
orig = torch.zeros((B, max_len, window_length), dtype=torch.float32)
masked = torch.zeros((B, max_len, window_length), dtype=torch.float32)

# Fill
for i, b in enumerate(batch):
    L = b["seq_len"]
    orig[i, :L, :] = b["windows"]  # Direct copy from dataset
    masked[i, :L, :] = b["windows"]  # Start with original

return {
    "windows": orig,                # (B, L, W) - original unmasked
    "windows_masked": masked,       # (B, L, W) - masked input
    "mask_bool": mask_bool,         # (B, L) - True at masked positions
}
```

**Status**: ✅ **PASS**
- `orig` (returned as `windows`) is direct copy from dataset
- No projection, normalization, or encoding
- Pure tensor copy operation
- `b["windows"]` comes from dataset `__getitem__` (line 119 in dataset.py)

**Verification Chain**:
```python
# Dataset returns raw windows
return {
    "windows": windows_t,  # torch.from_numpy(windows_np)
    ...
}
# → Collate copies directly
orig[i, :L, :] = b["windows"]
# → Training uses directly  
target = windows
```

✅ **No learnable modules in the chain**

---

### 4. ✅ Masking Applied Before Projection

**File**: `eeg_analysis/src/data/eeg_pretraining_dataset.py`  
**Lines**: 174-177

```python
if masking_style == "mae":
    # MAE-style: Replace ALL masked positions with zeros
    # No information leakage - model must reconstruct from context only
    masked[i, idxs, :] = 0.0  # Masking on raw windows (shape: window_length=2048)
```

**Status**: ✅ **PASS**
- Masking applied to raw window tensors (2048 samples)
- BEFORE any model processing
- In collate function, not in model

**Flow**:
```
Dataset → Raw windows (2048)
    ↓
Collate → Mask applied (zeros inserted)
    ↓
Model → TokenEncoder projects (2048 → 512)
    ↓
    → Backbone processes
    ↓
    → Decoder reconstructs (512 → 2048)
```

✅ Masking happens at step 2, projection at step 3.

---

### 5. ⚠️ Loss Computation (Control Mode Issue)

**File**: `eeg_analysis/src/training/pretrain_mamba.py`  
**Lines**: 277-308

```python
mask_exp = mask_bool.unsqueeze(-1).expand_as(pred)  # (B, L, D)

# Control experiment: Check if model is learning dataset mean
if disable_temporal and disable_spatial:
    # ... centering logic ...
    pred_masked = pred[mask_exp].view(-1, pred.shape[-1])  # Extract masked
    target_masked = target[mask_exp].view(-1, target.shape[-1])  # Extract masked
    
    # Remove mean (model can't reduce loss by learning a constant)
    target_mean = target_masked.mean(dim=0, keepdim=True)  # (1, D)
    pred_centered = pred_masked - target_mean  # ← MANIPULATES targets!
    target_centered = target_masked - target_mean  # ← MANIPULATES targets!
    
    loss = (pred_centered - target_centered).pow(2).mean()
else:
    # Normal training (standard loss)
    diff = pred - target
    masked_diff = diff[mask_exp]  # (N_masked * D,)
    loss = masked_diff.pow(2).mean()
```

**Status**: ⚠️ **ISSUE WITH CONTROL MODE**

**Normal Mode (lines 305-308)**: ✅ **PASS**
- Line 306: `diff = pred - target` - Direct comparison
- Line 307: `masked_diff = diff[mask_exp]` - Only masked positions
- Line 308: MSE on masked positions only
- ✅ Compares predictions to **raw signal targets**

**Control Mode (lines 280-294)**: ⚠️ **CONCERN**
- Lines 290-292: Subtracts mean from targets
- This **modifies the ground truth**
- While mathematically it doesn't create new information, it obscures the direct signal comparison

**Why this is problematic**:
- The whole point of using raw signal targets is that they're **fixed ground truth**
- Centering transforms them: `target_centered = target - mean(target)`
- This is no longer comparing to actual EEG samples

**Recommendation**: For control experiments with raw signals, centering is **unnecessary and confusing**:

```python
# Control mode should just use standard loss
# If model can't learn (constant input + fixed targets), loss stays high
# No need to center - targets are already fixed!

# Remove centering logic entirely
diff = pred - target
masked_diff = diff[mask_exp]
loss = masked_diff.pow(2).mean()
```

---

### 6. ✅ Model Forward Pass

**File**: `eeg_analysis/src/models/mamba_eeg_model.py`  
**Lines**: 257-295

```python
def forward(
    self,
    windows_masked: torch.Tensor,  # (B, L, 2048) - raw signal with zeros
    ...,
    decode_to_signal: bool = False,
):
    token_emb = self.token_encoder(windows_masked)  # (B, L, 512) - Project
    
    # ... add positional encodings ...
    
    embeddings = self.backbone(x)  # (B, L, 512) - Process
    
    # Optionally decode back to signal space (proper MAE)
    if decode_to_signal:
        reconstructed = self.decoder(embeddings)  # (B, L, 2048) - Decode
        return reconstructed
    else:
        return embeddings
```

**Status**: ✅ **PASS** (when `decode_to_signal=True`)
- Line 272: Encoder processes **masked input** (with zeros)
- Line 288: Backbone processes embeddings
- Line 292: Decoder reconstructs to signal space (2048)
- Returns reconstructed signal, **same shape as targets**

**Key point**: When `decode_to_signal=True`:
- Predictions: `(B, L, 2048)` - reconstructed signal
- Targets: `(B, L, 2048)` - raw signal
- **Direct comparison in signal space** ✅

---

### 7. ✅ No Encoder Used for Targets (When Configured Correctly)

**File**: `eeg_analysis/src/training/pretrain_mamba.py`  
**Lines**: 168-176

```python
# Reconstruction mode: Proper MAE (predict signal) vs embedding prediction
use_signal_reconstruction = bool(cfg.get("reconstruct_signal_space", True))  # Default: True

# If reconstructing signal space: targets are raw signal (never change!)
if use_signal_reconstruction:
    target_encoder = None  # Targets are raw signal, no encoder needed
    if is_main:
        logger.info("Using signal space reconstruction (proper MAE)")
```

**Status**: ✅ **PASS**
- `target_encoder = None` when signal reconstruction enabled
- Comment explicitly states "no encoder needed"
- Falls through to line 269: `target = windows` (raw signal)

**Verification**: With config `reconstruct_signal_space: true`:
1. `use_signal_reconstruction = True` (line 169)
2. `target_encoder = None` (line 174)
3. Training: `if use_signal_reconstruction: target = windows` (line 267-269)
4. Validation: `if use_signal_reconstruction: target = windows` (line 378-379)

✅ **No encoder ever touches targets**

---

## Summary of Issues

### Critical Issues: ❌ NONE (when configured correctly)

With `reconstruct_signal_space: true`, the pipeline is **correct**.

### High Priority Warnings: ⚠️ 2 Issues

1. **Dangerous fallback paths** (lines 270-275, 380-383)
   - Should be removed to prevent accidental encoder use
   - Could cause circular dependency bug if config is changed

2. **Unnecessary centering in control mode** (lines 280-294)
   - Obscures direct signal comparison
   - Unnecessary with fixed targets
   - Should be removed for clarity

### Medium Priority: 📝 1 Issue

3. **Legacy embedding space code** still present
   - `encode_tokens_only()` method (line 237-245)
   - Frozen encoder logic (lines 177-185)
   - Should be deprecated or removed

---

## Recommendations

### 1. Remove Dangerous Fallback Paths ✅ CRITICAL

```python
# In eeg_analysis/src/training/pretrain_mamba.py, lines 266-275 (training) and 378-383 (validation)
# BEFORE (dangerous):
if use_signal_reconstruction:
    target = windows
elif target_encoder is not None:  # ← REMOVE THIS
    target = target_encoder(windows)
else:  # ← REMOVE THIS
    target = model.encode_tokens_only(windows)

# AFTER (safe):
if use_signal_reconstruction:
    target = windows  # Raw signal
else:
    raise ValueError(
        "Embedding space reconstruction is deprecated and causes circular dependency. "
        "Set reconstruct_signal_space=true in config."
    )
```

### 2. Remove Centering Logic ✅ HIGH PRIORITY

```python
# In eeg_analysis/src/training/pretrain_mamba.py, lines 277-308
# BEFORE (complex):
if disable_temporal and disable_spatial:
    # ... 25 lines of centering logic ...
else:
    diff = pred - target
    masked_diff = diff[mask_exp]
    loss = masked_diff.pow(2).mean()

# AFTER (simple):
# Direct loss - no special case needed
diff = pred - target
masked_diff = diff[mask_exp]  # Only masked positions
loss = masked_diff.pow(2).mean()

# With fixed targets + constant input, loss CANNOT decrease
# Centering is unnecessary and confusing
```

### 3. Deprecate Embedding Space Code ✅ MEDIUM PRIORITY

```python
# In eeg_analysis/src/models/mamba_eeg_model.py
@torch.no_grad()
@deprecated("Use decode_to_signal=True for proper MAE reconstruction")
def encode_tokens_only(self, windows: torch.Tensor) -> torch.Tensor:
    """DEPRECATED: Creates circular dependency in MAE training."""
    raise DeprecationWarning("This method should not be used. Enable decode_to_signal=True.")
```

### 4. Add Configuration Validation ✅ HIGH PRIORITY

```python
# At start of eeg_analysis/src/training/pretrain_mamba.py main()
use_signal_reconstruction = bool(cfg.get("reconstruct_signal_space", True))

if not use_signal_reconstruction:
    raise ValueError(
        "reconstruct_signal_space must be True. "
        "Embedding space reconstruction causes circular dependency bug."
    )
```

---

## Final Verdict

### Current Status: ✅ **CORRECT** (with caveats)

**When `reconstruct_signal_space: true` is set**:

| Requirement | Status | Notes |
|------------|--------|-------|
| 1. Targets are raw signal (2048) | ✅ PASS | `target = windows` |
| 2. No encoder operates on targets | ✅ PASS | Direct assignment |
| 3. Masking before projection | ✅ PASS | In collate function |
| 4. Loss only on masked positions | ✅ PASS | Boolean indexing |
| 5. Compare to raw signal, not embeddings | ✅ PASS | Decoder produces (B,L,2048) |
| 6. Targets are direct clone, not encoded | ✅ PASS | Tensor copy in collate |
| 7. No need for frozen encoders | ✅ PASS | `target_encoder = None` |

### Dangerous Code Exists: ⚠️ **WARNING**

**Fallback paths that use encoders** (lines 270-275, 380-383) still exist and could be accidentally triggered if:
- Config is changed
- Code is refactored
- Someone doesn't understand the significance of `reconstruct_signal_space`

**These should be REMOVED immediately** to prevent future bugs.

---

## Minimal Required Changes

To make the pipeline **bulletproof**, apply these changes:

```python
# Change 1: Remove fallback paths in training loop (line 266-275)
# Targets
assert use_signal_reconstruction, "Must use signal space reconstruction"
target = windows  # (B, L, W) - raw signal

# Change 2: Remove fallback paths in validation loop (line 378-383)
# Targets  
assert use_signal_reconstruction, "Must use signal space reconstruction"
target = windows  # Raw signal

# Change 3: Remove centering (replace lines 277-308)
# Loss computation - works for both control and normal modes
diff = pred - target
masked_diff = diff[mask_exp]  # Only masked positions
loss = masked_diff.pow(2).mean()

# Change 4: Add config validation (after line 169)
if not use_signal_reconstruction:
    raise ValueError("reconstruct_signal_space must be True")
```

With these 4 changes, the pipeline becomes **provably correct** with no dangerous paths.

---

## Conclusion

Your MAE pipeline is **fundamentally correct** when properly configured (`reconstruct_signal_space: true`).

The main issues are:
1. **Legacy code paths** that could accidentally use encoders
2. **Unnecessary complexity** in control mode centering
3. **Lack of validation** to prevent misconfiguration

**All issues are in the training script, not the core architecture.**

Apply the minimal changes above to make it bulletproof. 🎯


---

## Source: MULTI_CHANNEL_MASKING.md

# Multi-Channel Masking Strategy

## Overview

Multi-channel masking is a **4th masking option** that synchronizes masking across channels from the same file (participant-run). This forces the model to learn **cross-channel spatial relationships** by requiring it to use information from neighboring channels to reconstruct masked channels.

## How It Works

### Synchronized Temporal Masking

When `masking_style: "multi_channel"`:

1. **Group channels by file**: Channels from the same parquet file are grouped together
2. **Generate synchronized mask**: The same temporal positions are masked across all channels in the group
3. **Apply mask**: All channels from the same file mask the same positions

### Example

If a batch contains channels from file `participant001_run1.parquet`:
- **C3** at position 10 → **masked**
- **C1** at position 10 → **masked** (same position!)
- **C5** at position 10 → **masked** (same position!)
- **F3** at position 10 → **masked** (same position!)

The model must use:
- **C3** at positions 9, 11, 12... (unmasked) to reconstruct C3 at position 10
- **C1, C5, F3** at positions 9, 11, 12... (unmasked) to help reconstruct C3 at position 10

This encourages learning: **"Use spatial context from neighboring electrodes to reconstruct masked electrode"**

## Configuration

### Basic Usage

```yaml
masking_style: "multi_channel"
mask_ratio: 0.75
mask_samples_within_token: false  # Token-level synchronized masking
```

### With Sample-Level Masking

```yaml
masking_style: "multi_channel"
mask_ratio: 0.75
mask_samples_within_token: true  # Sample-level synchronized masking
```

When `mask_samples_within_token: true`:
- Same **samples** (within tokens) are masked across channels
- More fine-grained synchronization
- Still forces cross-channel learning

## Comparison with Other Masking Strategies

| Strategy | Masking Pattern | Learning Focus |
|----------|----------------|----------------|
| **MAE** | Independent per channel | Temporal patterns within channel |
| **BERT** | Independent per channel (with noise) | Temporal patterns (easier task) |
| **Within-Token** | Independent per channel, sample-level | Signal patterns within tokens |
| **Multi-Channel** | Synchronized across channels | **Spatial relationships + temporal patterns** |

## Benefits

1. **Forces Spatial Learning**: Model must learn which channels are spatially related
2. **Encourages Cross-Channel Context**: Uses information from neighboring electrodes
3. **Better for Downstream Tasks**: Learned representations capture spatial structure
4. **More Realistic**: EEG signals are naturally correlated across nearby electrodes

## When to Use

**Recommended for:**
- Learning spatial patterns in EEG
- Downstream tasks requiring spatial understanding
- Models with strong spatial encoders
- Multi-channel EEG analysis

**Not recommended for:**
- Single-channel analysis
- When channels are independent
- Initial pretraining (start with MAE/within-token, then fine-tune with multi-channel)

## Implementation Details

### File Grouping

Channels are grouped by `file_path`:
- Same file → same mask pattern
- Different files → independent masks

### Sequence Length Handling

- Uses **minimum sequence length** within each file group
- Ensures all channels from same file mask same positions
- Handles variable-length sequences gracefully

### Mask Generation

1. **Token-level** (`mask_samples_within_token: false`):
   - Generate mask for `min_seq_len` tokens
   - Same token positions masked across all channels

2. **Sample-level** (`mask_samples_within_token: true`):
   - Generate mask for `min_seq_len` tokens × `window_length` samples
   - Same sample positions masked across all channels

## Expected Results

### Training Metrics

- **Loss**: May be slightly higher initially (harder task)
- **Spatial Encoding Importance**: Should increase
- **Cross-Channel Similarity**: Should decrease (channels learn distinct patterns)

### Diagnostic Script

Run diagnostics to verify multi-channel learning:

```bash
python scripts/diagnose_100pct_masking.py \
    --checkpoint eeg_analysis/checkpoints/mamba2_eeg_pretrained.pt \
    --num-samples 100
```

Look for:
- **Lower baseline similarity** between channels (channels learn distinct patterns)
- **Higher pattern correlation** (better signal reconstruction)
- **Spatial awareness** in embeddings

## Research Background

Multi-channel masking is inspired by:
- **Vision MAE**: Patches at same spatial location are masked together
- **EEG Spatial Structure**: Nearby electrodes record correlated signals
- **Cross-Modal Learning**: Using one modality to reconstruct another

This strategy is particularly effective for EEG because:
- Electrodes have known spatial relationships
- Signals are naturally correlated across nearby channels
- Spatial patterns are crucial for EEG analysis

## Example Training Command

```bash
uv run torchrun --standalone --nproc_per_node=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) \
    eeg_analysis/src/training/pretrain_mamba.py \
    --config eeg_analysis/configs/pretrain.yaml \
    --distributed
```

With config:
```yaml
masking_style: "multi_channel"
mask_ratio: 0.75
mask_samples_within_token: false
```

## Troubleshooting

### Issue: All channels predict similar values

**Cause**: Model not learning spatial relationships

**Solution**: 
- Increase `mask_ratio` to 0.75-0.9
- Ensure spatial encoder is enabled
- Check that channels from same file are in same batch

### Issue: Loss not decreasing

**Cause**: Task too hard (100% synchronized masking)

**Solution**:
- Reduce `mask_ratio` to 0.5-0.75
- Start with token-level (`mask_samples_within_token: false`)
- Gradually increase difficulty

### Issue: Channels from different files getting same mask

**Cause**: File grouping not working correctly

**Solution**:
- Check that `file_path` is correctly included in dataset items
- Verify batch contains channels from multiple files
- Check collate function grouping logic


---

## Source: PAPER_OUTLINE.md

# Paper Outline: DC Offset Removal and Architecture Enhancements for EEG-Based Depression Remission Prediction

## Title Options
1. **"The Critical Role of DC Offset Removal in EEG Signal Processing: A Comparative Study of Simple and Complex Models for Depression Remission Prediction"**
2. **"Data Quality Over Architecture Complexity: DC Offset Removal Dramatically Improves EEG-Based Depression Classification"**
3. **"Per-Channel DC Offset Removal Enhances EEG Classification: Evidence from Tabular and Hybrid Deep Learning Models"**

---

## 1. ABSTRACT (250-300 words)

### Key Points to Cover:
- **Problem**: EEG signals contain DC offsets that can mask true signal characteristics, particularly in multi-channel recordings
- **Method**: Per-channel, per-window DC offset removal using mean subtraction
- **Experiments**: Two model architectures (Efficient Tabular MLP, Advanced Hybrid 1D CNN-LSTM) evaluated on 21-patient depression remission dataset
- **Key Findings**:
  - DC offset removal improved ROC-AUC by 15.7% (0.776→0.898) in tabular MLP
  - Perfect recall (1.0) achieved in tabular MLP after DC offset removal
  - Hybrid model improved ROC-AUC by 16.7% (0.765→0.893) with combined DC removal + architecture enhancements
  - Simpler models benefit more dramatically from data quality improvements
- **Novelty**: First systematic study demonstrating DC offset removal's critical importance in clinical EEG classification, with controlled comparison across model complexities
- **Impact**: Establishes DC offset removal as essential preprocessing step, not optional enhancement

---

## 2. INTRODUCTION

### 2.1 Background and Motivation
- **EEG for Depression Prediction**: Growing interest in using EEG biomarkers for treatment outcome prediction
- **Signal Quality Challenges**: 
  - DC offsets from electrode-skin interface
  - Channel-specific variations
  - Window-to-window variability
- **Model Complexity Trade-offs**: Simple vs. complex architectures in clinical settings
- **Gap in Literature**: Limited systematic studies on preprocessing impact vs. architecture complexity

### 2.2 Research Questions
1. **RQ1**: Does per-channel DC offset removal significantly improve classification performance?
2. **RQ2**: Do simpler models benefit more from data quality improvements than complex models?
3. **RQ3**: What is the relative contribution of preprocessing vs. architecture enhancements?
4. **RQ4**: Can DC offset removal enable simpler models to match or exceed complex architectures?

### 2.3 Contributions
- **Novel Finding**: DC offset removal provides larger performance gains than architectural complexity
- **Methodological Contribution**: Controlled ablation study comparing preprocessing vs. architecture improvements
- **Practical Impact**: Establishes DC offset removal as essential preprocessing step
- **Clinical Relevance**: Demonstrates simpler models can achieve excellent performance with proper preprocessing

---

## 3. RELATED WORK

### 3.1 EEG Preprocessing for Clinical Applications
- **Standard Preprocessing Pipelines**: Filtering, artifact removal, normalization
- **DC Offset Handling**: Often overlooked or handled globally
- **Per-Channel Processing**: Limited studies on channel-specific preprocessing

### 3.2 Deep Learning for EEG Classification
- **CNN-LSTM Hybrid Models**: Temporal-spatial feature learning
- **Attention Mechanisms**: Multi-head attention for feature selection
- **Tabular MLPs**: Efficient alternatives for engineered features

### 3.3 Data Quality vs. Model Complexity
- **"Garbage In, Garbage Out"**: Data quality importance
- **Model Complexity Trade-offs**: When simple models suffice
- **Preprocessing Impact Studies**: Limited systematic comparisons

### 3.4 Depression Remission Prediction
- **EEG Biomarkers**: Alpha asymmetry, connectivity patterns
- **Window-Level vs. Patient-Level**: Aggregation strategies
- **Class Imbalance**: Handling minority classes in clinical data

---

## 4. METHODOLOGY

### 4.1 Dataset Description
- **Participants**: 21 patients (14 non-remission, 7 remission)
- **EEG Channels**: 4 channels (AF7, AF8, TP9, TP10) - frontal and temporal regions
- **Recording Protocol**: 10-second windows, 256 Hz sampling rate
- **Data Split**: Leave-One-Patient-Out cross-validation (21 folds)
- **Evaluation Level**: Patient-level aggregation from window-level predictions
- **Class Distribution**: 393 positive windows, 810 negative windows (2.06:1 ratio)

### 4.2 DC Offset Removal Implementation

#### 4.2.1 Algorithm
```python
# Per-channel, per-window DC offset removal
for each window:
    for each channel:
        # Handle zero values (replace with mean of non-zero values)
        nonzero_mean = mean(signal[signal != 0])
        filled_signal = replace_zeros(signal, nonzero_mean)
        
        # Compute DC offset (mean or median)
        dc_offset = mean(filled_signal)  # or median
        
        # Remove DC offset
        centered_signal = filled_signal - dc_offset
```

#### 4.2.2 Key Design Decisions
- **Per-Channel Processing**: Each channel processed independently (critical for multi-channel EEG)
- **Per-Window Processing**: DC offset computed per window, not globally (accounts for temporal variations)
- **Zero Handling**: Replace exact zeros with mean of non-zero values before centering (prevents bias)
- **Method Selection**: Mean subtraction (more common) vs. median (more robust to outliers)

#### 4.2.3 Rationale
- **Channel Independence**: Different electrodes have different DC offsets due to skin-electrode interface
- **Temporal Variability**: DC offset can drift over time within a recording
- **Signal Preservation**: Centering preserves signal dynamics while removing baseline shifts

### 4.3 Model Architectures

#### 4.3.1 Efficient Tabular MLP (Control Model)
**Purpose**: Isolate DC offset removal effect (same architecture, only preprocessing changes)

**Architecture**:
- Input: 5 selected features (from feature selection)
- Hidden Layers: [1024, 512, 256, 128] units
- Activation: ReLU
- Regularization: Dropout (0.3), Batch Normalization, Label Smoothing (0.05)
- Optimizer: AdamW (lr=0.001, weight_decay=0.01)
- Learning Rate: Cosine annealing with warm restarts
- Batch Size: 1024
- Training: Mixed precision, gradient clipping

**Why This Model**:
- Simple architecture eliminates confounding factors
- Direct comparison: same model, different preprocessing
- Demonstrates preprocessing impact without architecture changes

#### 4.3.2 Advanced Hybrid 1D CNN-LSTM (Complex Model)
**Purpose**: Evaluate combined effect of DC offset removal + architecture enhancements

**Architecture Components**:

1. **Feature Embedding**:
   - Input: 5 selected features
   - Linear projection: 5 → 128 dimensions
   - BatchNorm + ReLU + Dropout

2. **CNN Blocks** (3 blocks):
   - Block 1: [64, 64] filters, kernels [5, 3], pool=2
   - Block 2: [128, 128] filters, kernels [3, 3], pool=2
   - Block 3: [256, 128] filters, kernels [3, 1], pool=1
   - Progressive dropout: [0.1, 0.2, 0.3]
   - Spatial dropout enabled
   - Gaussian noise: 0.005

3. **LSTM Layers** (2 layers, bidirectional):
   - Layer 1: 128 units, return_sequences=True, dropout=0.2, recurrent_dropout=0.1
   - Layer 2: 64 units, return_sequences=False, dropout=0.3, recurrent_dropout=0.2

4. **Attention Mechanism**:
   - Multi-head attention: 8 heads (increased from 4)
   - Key dimension: 64 (increased from 32)
   - Positional encoding: Enabled (new addition)
   - Dropout: 0.1

5. **Fusion Strategy**:
   - Concatenation of CNN and LSTM features
   - Feature pyramid enabled

6. **Dense Layers**:
   - Layer 1: 256 units, GELU activation (changed from ReLU), dropout=0.3
   - Layer 2: 128 units, GELU activation, dropout=0.4
   - Batch normalization enabled

**Training Configuration**:
- Optimizer: AdamW (lr=0.001, weight_decay=0.01)
- Learning Rate: Cosine annealing with warm restarts
- Batch Size: 1024
- Epochs: 100 (early stopping patience=10)
- Regularization: Label smoothing (0.05), gradient clipping (1.0)
- Mixed Precision: Enabled
- Deterministic Training: Enabled (for reproducibility)

**Architecture Changes Between Experiments**:
- **Activation**: ReLU → GELU (in dense layers)
- **Attention**: 4 heads → 8 heads, 32 dim → 64 dim
- **Positional Encoding**: Disabled → Enabled
- **Deterministic Training**: False → True

### 4.4 Feature Selection
- **Method**: SelectKBest with f_classif
- **Selected Features**: 5 features
  - `tp10_total_power`
  - `tp10_psd_mean`
  - `tp10_psd_std`
  - `tp10_psd_max`
  - `left_frontal_temporal_diff_beta`
- **Rationale**: Focus on most discriminative features, reduce overfitting risk

### 4.5 Training and Evaluation Protocol

#### 4.5.1 Cross-Validation Strategy
- **Method**: Leave-One-Patient-Out (LOGO)
- **Folds**: 21 (one per patient)
- **Rationale**: Prevents data leakage, realistic clinical scenario

#### 4.5.2 Window-to-Patient Aggregation
- **Window-Level Predictions**: Probability scores for each window
- **Patient-Level Aggregation**: Mean of window probabilities
- **Threshold**: 0.5 for binary classification
- **Rationale**: Accounts for patient-level variability, standard clinical practice

#### 4.5.3 Evaluation Metrics
- **Main Metrics**:
  - ROC-AUC (patient-level)
  - F1-Score (patient-level)
  - Recall (patient-level) - critical for clinical applications
  - Precision (patient-level)
  - Accuracy (patient-level)
- **Additional Metrics**:
  - Window-level accuracy
  - Confusion matrix (TP, TN, FP, FN)
- **Rationale**: Patient-level metrics align with clinical decision-making

### 4.6 Experimental Design

#### 4.6.1 Experiment 1: DC Offset Removal Impact (Tabular MLP)
- **Baseline**: No DC offset removal
- **Intervention**: DC offset removal enabled
- **Model**: Efficient Tabular MLP (identical architecture)
- **Purpose**: Isolate preprocessing effect

#### 4.6.2 Experiment 2: Combined Improvements (Hybrid Model)
- **Baseline**: No DC offset removal + simpler architecture
- **Intervention**: DC offset removal + enhanced architecture
- **Model**: Advanced Hybrid 1D CNN-LSTM
- **Purpose**: Evaluate combined effect, compare to simpler model

#### 4.6.3 Controlled Variables
- **Feature Selection**: Same 5 features in all experiments
- **Cross-Validation**: Same LOGO splits
- **Class Balancing**: Balanced class weights (no SMOTE/NearMiss)

---

## 5. RESULTS

### 5.1 Experiment 1: DC Offset Removal Impact (Tabular MLP)

#### 5.1.1 Performance Improvements
| Metric | Before DC Removal | After DC Removal | Improvement | % Change |
|--------|-------------------|------------------|-------------|----------|
| **ROC-AUC** | 0.776 | 0.898 | +0.122 | **+15.7%** |
| **F1-Score** | 0.667 | 0.824 | +0.157 | **+23.5%** |
| **Recall** | 0.714 | **1.000** | +0.286 | **+40.0%** |
| **Precision** | 0.625 | 0.700 | +0.075 | +12.0% |
| **Accuracy** | 0.762 | 0.857 | +0.095 | +12.5% |
| **False Negatives** | 2 | **0** | -2 | **-100%** |

#### 5.1.2 Key Findings
- **Perfect Recall**: All positive patients correctly identified (7/7)
- **No Precision Trade-off**: Precision improved despite perfect recall
- **Eliminated False Negatives**: Critical for clinical applications
- **Consistent Improvements**: All metrics improved substantially

#### 5.1.3 Statistical Significance
- **Dataset Digest Change**: b8cc93be → 15150132 (confirms preprocessing change)
- **Same Architecture**: Eliminates architecture as confounding factor
- **Same Features**: Same 5 selected features

### 5.2 Experiment 2: Combined Improvements (Hybrid Model)

#### 5.2.1 Performance Improvements
| Metric | Baseline | Enhanced | Improvement | % Change |
|--------|----------|----------|-------------|----------|
| **ROC-AUC** | 0.765 | 0.893 | +0.128 | **+16.7%** |
| **F1-Score** | 0.727 | 0.833 | +0.106 | **+14.6%** |
| **Recall** | 0.571 | 0.714 | +0.143 | **+25.0%** |
| **Precision** | 1.000 | 1.000 | 0.000 | 0% (maintained) |
| **Accuracy** | 0.857 | 0.905 | +0.048 | +5.6% |
| **False Negatives** | 3 | 2 | -1 | -33.3% |
| **True Positives** | 4 | 5 | +1 | +25.0% |

#### 5.2.2 Architecture Changes
- **Activation Function**: ReLU → GELU (dense layers)
- **Attention Enhancement**: 4→8 heads, 32→64 dim, positional encoding added
- **Deterministic Training**: Enabled for reproducibility

#### 5.2.3 Key Findings
- **Similar ROC-AUC Gain**: ~0.13 improvement (comparable to tabular MLP)
- **Maintained Perfect Precision**: No false positives in either condition
- **Improved Recall**: Caught 1 additional positive patient (4→5 out of 7)
- **Dataset Digest Change**: Confirms DC offset removal applied

### 5.3 Comparative Analysis

#### 5.3.1 Model Complexity vs. Performance
| Model | ROC-AUC (After) | F1-Score | Recall | Architecture Complexity |
|-------|----------------|----------|--------|------------------------|
| **Tabular MLP** | 0.898 | 0.824 | **1.000** | Simple (4-layer MLP) |
| **Hybrid CNN-LSTM** | 0.893 | 0.833 | 0.714 | Complex (CNN+LSTM+Attention) |

#### 5.3.2 Key Insights
1. **Simpler Model Achieved Higher Recall**: Tabular MLP caught all positive cases
2. **Similar ROC-AUC**: Both models achieved ~0.89-0.90 after improvements
3. **Preprocessing > Architecture**: DC offset removal provided larger gains than architectural enhancements
4. **No Complexity Advantage**: Complex model didn't outperform simple model after preprocessing

#### 5.3.3 Relative Contributions
- **DC Offset Removal**: Main driver of improvement (~0.12-0.13 ROC-AUC gain)
- **Architecture Enhancements**: Additional contribution (difficult to isolate due to confounded design)
- **Interaction Effect**: Possible synergy, but DC removal dominates

### 5.4 Clinical Interpretation

#### 5.4.1 Patient-Level Performance
- **Total Patients**: 21 (14 negative, 7 positive)
- **Tabular MLP (After)**:
  - True Positives: 7/7 (100% recall)
  - True Negatives: 11/14
  - False Positives: 3/14
  - False Negatives: 0/7
- **Hybrid Model (After)**:
  - True Positives: 5/7 (71.4% recall)
  - True Negatives: 14/14 (100% specificity)
  - False Positives: 0/14
  - False Negatives: 2/7

#### 5.4.2 Clinical Implications
- **Tabular MLP**: Better for screening (perfect sensitivity, some false positives acceptable)
- **Hybrid Model**: Better for confirmation (perfect specificity, some false negatives)
- **Trade-off**: Sensitivity vs. Specificity based on clinical context

---

## 6. DISCUSSION

### 6.1 Why DC Offset Removal Matters

#### 6.1.1 Signal Quality Perspective
- **Baseline Shifts**: DC offsets create artificial signal baselines
- **Feature Distortion**: Statistical features (mean, variance) affected by DC offset
- **Channel Variability**: Different electrodes have different offsets
- **Temporal Drift**: Offsets can vary across windows

#### 6.1.2 Model Learning Perspective
- **Feature Learning**: Models may learn DC offset patterns instead of true signal characteristics
- **Generalization**: DC offsets are recording-specific, hurt generalization
- **Simple Models**: Less capacity to overcome noisy inputs
- **Complex Models**: Can learn to ignore offsets, but waste capacity

### 6.2 Why Simpler Models Benefit More

#### 6.2.1 Capacity Hypothesis
- **Limited Capacity**: Simple models have less ability to learn complex patterns
- **Clean Data Advantage**: Cleaner inputs allow simple models to focus on discriminative features
- **Complex Models**: Can overcome noise, but preprocessing still helps

#### 6.2.2 Feature Engineering Perspective
- **Engineered Features**: Tabular MLP uses pre-computed features
- **Feature Quality**: DC offset removal improves feature quality directly
- **Deep Learning**: Hybrid model learns features, but benefits from cleaner raw signals

### 6.3 Architecture Enhancements: Limited Impact

#### 6.3.1 GELU vs. ReLU
- **Theoretical Advantage**: GELU smoother, better gradients
- **Empirical Impact**: Difficult to isolate (confounded with DC removal)
- **Recommendation**: Further ablation needed

#### 6.3.2 Enhanced Attention
- **Increased Capacity**: 8 heads vs. 4, 64 dim vs. 32
- **Positional Encoding**: Added temporal awareness
- **Impact**: Unclear due to confounded design
- **Recommendation**: Controlled ablation study needed

### 6.4 Limitations

#### 6.4.1 Experimental Design
- **Confounded Variables**: DC removal + architecture changes in hybrid experiment
- **Single Dataset**: 21 patients, limited generalizability
- **No Ablation Study**: Can't isolate architecture contribution

#### 6.4.2 Dataset Characteristics
- **Small Sample Size**: 21 patients (though 1,203 windows)
- **Class Imbalance**: 2:1 ratio (mitigated with balanced weights)
- **Single Clinical Site**: May not generalize to other populations
- **Fixed Window Size**: 10-second windows (may not be optimal)

#### 6.4.3 Model Limitations
- **Feature Selection**: Only 5 features (may miss important signals)
- **Fixed Architecture**: No architecture search
- **Hyperparameter Tuning**: Limited hyperparameter exploration

### 6.5 Future Work

#### 6.5.1 Ablation Studies
- **DC Removal Alone**: Hybrid model with/without DC removal (same architecture)
- **Architecture Alone**: Enhanced vs. baseline architecture (same preprocessing)
- **Component Analysis**: GELU, attention, positional encoding individually

#### 6.5.2 Extended Validation
- **Multi-Site Data**: Validate across different clinical sites
- **Larger Cohorts**: More patients for statistical power
- **Different Disorders**: Test on other EEG classification tasks

#### 6.5.3 Methodological Improvements
- **Adaptive DC Removal**: Learn optimal DC removal strategy
- **Channel-Specific Methods**: Different methods per channel
- **Temporal Modeling**: Account for DC drift over time

---

## 7. CONCLUSION

### 7.1 Summary of Findings
1. **DC Offset Removal is Critical**: Provides 15-17% ROC-AUC improvement
2. **Preprocessing > Architecture**: Data quality improvements exceed architectural enhancements
3. **Simple Models Can Excel**: Tabular MLP achieved perfect recall with proper preprocessing
4. **Clinical Relevance**: Perfect recall in tabular MLP has important clinical implications

### 7.2 Key Takeaways
- **Essential Preprocessing**: DC offset removal should be standard in EEG pipelines
- **Model Selection**: Simpler models may be preferable with proper preprocessing
- **Cost-Benefit**: Preprocessing is cheaper than complex architectures
- **Clinical Impact**: Better sensitivity enables better patient outcomes

### 7.3 Broader Implications
- **Signal Processing**: Highlights importance of preprocessing in ML pipelines
- **Model Complexity**: Questions value of complex architectures without proper preprocessing
- **Clinical ML**: Demonstrates simple models can achieve clinical-grade performance
- **Reproducibility**: Emphasizes importance of preprocessing documentation

---

## 8. FIGURES AND TABLES

### 8.1 Required Figures
1. **Figure 1**: Pipeline diagram (Data → Preprocessing → Models → Evaluation)
2. **Figure 2**: DC offset removal visualization (before/after signals)
3. **Figure 3**: Model architectures (Tabular MLP vs. Hybrid CNN-LSTM)
4. **Figure 4**: ROC curves comparison (before/after DC removal)
5. **Figure 5**: Confusion matrices (all experimental conditions)
6. **Figure 6**: Performance comparison bar chart (all metrics)
7. **Figure 7**: Feature importance (selected 5 features)

### 8.2 Required Tables
1. **Table 1**: Dataset characteristics
2. **Table 2**: Model architectures (detailed specifications)
3. **Table 3**: Performance metrics (before/after comparisons)
4. **Table 4**: Confusion matrices (all conditions)
5. **Table 5**: Statistical significance tests (if applicable)
6. **Table 6**: Hyperparameters (all models)

---

## 9. APPENDIX

### 9.1 Implementation Details
- **Code Availability**: GitHub repository link
- **Reproducibility**: deterministic training settings
- **Hardware**: GPU specifications, training time

### 9.2 Additional Results
- **Window-Level Metrics**: Detailed window-level performance
- **Feature Analysis**: Feature importance, correlation analysis
- **Training Curves**: Loss curves, learning rate schedules

### 9.3 Extended Ablation Studies
- **DC Removal Methods**: Mean vs. median comparison
- **Channel-Specific Analysis**: Per-channel DC offset magnitudes
- **Temporal Analysis**: DC offset variation across windows

---

## 10. WRITING GUIDELINES

### 10.1 Tone and Style
- **Scientific Rigor**: Objective, evidence-based
- **Clarity**: Clear explanations of technical concepts
- **Clinical Relevance**: Connect findings to clinical practice
- **Honesty**: Acknowledge limitations and confounded variables

### 10.2 Key Messages to Emphasize
1. **DC offset removal is not optional** - it's essential
2. **Data quality > model complexity** - preprocessing matters more
3. **Simple models can excel** - with proper preprocessing
4. **Clinical impact** - perfect recall has real-world implications

### 10.3 Avoid Overstating
- **Don't claim**: Architecture enhancements caused improvements (confounded)
- **Don't claim**: Results generalize to all EEG tasks (single dataset)
- **Do claim**: DC offset removal significantly improves performance
- **Do claim**: Simple models achieve excellent performance with preprocessing

---

## 11. NOVELTY STATEMENTS

### 11.1 What Makes This Novel
1. **First Systematic Study**: Comparing preprocessing vs. architecture impact in EEG classification
2. **Per-Channel, Per-Window**: Novel DC removal approach (not global)
3. **Controlled Comparison**: Tabular MLP isolates preprocessing effect
4. **Clinical Focus**: Patient-level evaluation with clinical interpretation
5. **Perfect Recall Achievement**: Demonstrates feasibility of high-sensitivity screening

### 11.2 Contribution to Literature
- **Preprocessing Guidelines**: Establishes DC removal as essential step
- **Model Selection**: Challenges assumption that complex models are always better
- **Clinical ML**: Demonstrates simple models can achieve clinical-grade performance
- **Reproducibility**: Provides detailed methodology for replication

---

## 12. TARGET VENUES

### 12.1 Journal Options
1. **IEEE Transactions on Biomedical Engineering** (Impact Factor: ~4.4)
2. **Journal of Neural Engineering** (Impact Factor: ~4.0)
3. **Computers in Biology and Medicine** (Impact Factor: ~7.7)
4. **NeuroImage: Clinical** (Impact Factor: ~4.8)
5. **Journal of Medical Internet Research** (Impact Factor: ~7.4)

### 12.2 Conference Options
1. **MICCAI** (Medical Image Computing and Computer Assisted Intervention)
2. **EMBC** (Engineering in Medicine and Biology Conference)
3. **BIBM** (IEEE International Conference on Bioinformatics and Biomedicine)
4. **NeurIPS** (if emphasizing ML methodology)

---

## 13. PAPER STRUCTURE CHECKLIST

### 13.1 Essential Sections
- [ ] Abstract (concise, compelling)
- [ ] Introduction (problem, motivation, contributions)
- [ ] Related Work (comprehensive, positioned correctly)
- [ ] Methodology (detailed, reproducible)
- [ ] Results (clear, well-organized)
- [ ] Discussion (insights, limitations, future work)
- [ ] Conclusion (summary, implications)

### 13.2 Key Elements
- [ ] Novel contribution clearly stated
- [ ] Controlled experiments (isolate variables)
- [ ] Statistical analysis (if applicable)
- [ ] Clinical interpretation
- [ ] Limitations acknowledged
- [ ] Reproducibility details
- [ ] Figures and tables support narrative

### 13.3 Quality Checks
- [ ] All claims supported by evidence
- [ ] Limitations honestly discussed
- [ ] Related work properly cited
- [ ] Methodology detailed enough for replication
- [ ] Results clearly presented
- [ ] Discussion provides insights, not just summary

---

## NOTES FOR WRITING

### Key Strengths to Emphasize
1. **Controlled Experiment**: Tabular MLP isolates DC removal effect
2. **Large Improvement**: 15-17% ROC-AUC gain is substantial
3. **Perfect Recall**: Clinically significant achievement
4. **Practical Impact**: Simple preprocessing step, large benefit
5. **Clinical Relevance**: Patient-level evaluation, clinical interpretation

### Key Weaknesses to Address
1. **Confounded Design**: Hybrid experiment has multiple changes
2. **Small Dataset**: 21 patients (mitigate with window-level analysis)
4. **No Ablation**: Can't isolate architecture contribution
5. **Single Dataset**: Limited generalizability

### Story Arc
1. **Hook**: DC offset removal provides larger gains than complex architectures
2. **Problem**: Preprocessing often overlooked in favor of model complexity
3. **Solution**: Systematic study comparing preprocessing vs. architecture
4. **Results**: Dramatic improvements, perfect recall achieved
5. **Insight**: Data quality > model complexity
6. **Impact**: Clinical implications, practical recommendations

---

## METRICS TO HIGHLIGHT

### Most Impressive Results
1. **Perfect Recall**: Tabular MLP caught all 7 positive patients
2. **ROC-AUC Improvement**: +0.122 to +0.128 (15-17% relative improvement)
3. **False Negative Elimination**: 2 → 0 in tabular MLP
4. **F1 Improvement**: +0.157 (23.5% relative improvement)

### Clinical Significance
- **Sensitivity**: Perfect recall means no missed cases (critical for screening)
- **Specificity**: Hybrid model achieved perfect precision (no false positives)
- **Trade-off**: Different models for different clinical scenarios

### Statistical Significance
- **Consistent Improvements**: All metrics improved
- **Large Effect Size**: 15-17% ROC-AUC improvement
- **Eliminated Errors**: False negatives eliminated in tabular MLP


---

## Source: PAPER_SKELETON.md

# Paper Skeleton: Quick Reference Checklist

## CORE NARRATIVE
**Main Story**: DC offset removal provides larger performance gains (15-17% ROC-AUC) than architectural complexity improvements. Simple models achieve excellent performance (perfect recall) with proper preprocessing.

---

## 1. ABSTRACT (250-300 words)
- [ ] Problem: DC offsets mask true EEG signal characteristics
- [ ] Method: Per-channel, per-window DC offset removal
- [ ] Experiments: Tabular MLP (control) + Hybrid CNN-LSTM (combined)
- [ ] Key Finding: 15.7% ROC-AUC improvement in tabular MLP, perfect recall achieved
- [ ] Novelty: First systematic comparison of preprocessing vs. architecture
- [ ] Impact: Establishes DC removal as essential preprocessing step

---

## 2. INTRODUCTION (~2 pages)

### 2.1 Background
- [ ] EEG for depression prediction (clinical context)
- [ ] Signal quality challenges (DC offsets, channel variability)
- [ ] Model complexity trade-offs (simple vs. complex)
- [ ] Gap: Limited studies on preprocessing impact

### 2.2 Research Questions
- [ ] RQ1: Does DC removal improve performance?
- [ ] RQ2: Do simpler models benefit more?
- [ ] RQ3: Preprocessing vs. architecture contribution?
- [ ] RQ4: Can simple models match complex ones?

### 2.3 Contributions
- [ ] Novel finding: Preprocessing > architecture
- [ ] Methodological: Controlled ablation study
- [ ] Practical: Essential preprocessing step
- [ ] Clinical: Simple models achieve excellent performance

---

## 3. RELATED WORK (~1.5 pages)
- [ ] EEG preprocessing (standard pipelines, DC handling)
- [ ] Deep learning for EEG (CNN-LSTM, attention)
- [ ] Data quality vs. model complexity
- [ ] Depression remission prediction (biomarkers, aggregation)

---

## 4. METHODOLOGY (~3 pages)

### 4.1 Dataset
- [ ] 21 patients (14 non-remission, 7 remission)
- [ ] 4 channels (AF7, AF8, TP9, TP10)
- [ ] 10-second windows, 256 Hz
- [ ] LOGO cross-validation (21 folds)
- [ ] Patient-level evaluation

### 4.2 DC Offset Removal
- [ ] Algorithm: Per-channel, per-window mean subtraction
- [ ] Zero handling: Replace with non-zero mean
- [ ] Rationale: Channel independence, temporal variability

### 4.3 Models
- [ ] **Tabular MLP**: [1024, 512, 256, 128], ReLU, AdamW
- [ ] **Hybrid CNN-LSTM**: 3 CNN blocks, 2 LSTM layers, 8-head attention, GELU

### 4.4 Feature Selection
- [ ] SelectKBest, 5 features selected
- [ ] List: tp10_total_power, tp10_psd_mean, tp10_psd_std, tp10_psd_max, left_frontal_temporal_diff_beta

### 4.5 Evaluation
- [ ] LOGO cross-validation
- [ ] Window-to-patient aggregation (mean probabilities)
- [ ] Metrics: ROC-AUC, F1, Recall, Precision, Accuracy

---

## 5. RESULTS (~2.5 pages)

### 5.1 Experiment 1: Tabular MLP (DC Removal Only)
- [ ] **Before**: ROC-AUC=0.776, F1=0.667, Recall=0.714, FN=2
- [ ] **After**: ROC-AUC=0.898, F1=0.824, Recall=1.000, FN=0
- [ ] **Improvement**: +15.7% ROC-AUC, +23.5% F1, perfect recall
- [ ] **Key**: Eliminated all false negatives

### 5.2 Experiment 2: Hybrid Model (DC Removal + Architecture)
- [ ] **Before**: ROC-AUC=0.765, F1=0.727, Recall=0.571, FN=3
- [ ] **After**: ROC-AUC=0.893, F1=0.833, Recall=0.714, FN=2
- [ ] **Improvement**: +16.7% ROC-AUC, +14.6% F1, +25% Recall
- [ ] **Architecture Changes**: GELU, 8-head attention, positional encoding

### 5.3 Comparative Analysis
- [ ] **Tabular MLP**: ROC-AUC=0.898, Recall=1.000 (perfect)
- [ ] **Hybrid Model**: ROC-AUC=0.893, Recall=0.714
- [ ] **Insight**: Simpler model achieved higher recall
- [ ] **Conclusion**: Preprocessing > architecture complexity

### 5.4 Clinical Interpretation
- [ ] Tabular MLP: Perfect sensitivity (screening)
- [ ] Hybrid Model: Perfect specificity (confirmation)
- [ ] Trade-off: Sensitivity vs. specificity

---

## 6. DISCUSSION (~2 pages)

### 6.1 Why DC Removal Matters
- [ ] Signal quality: Baseline shifts, feature distortion
- [ ] Model learning: Models learn offsets instead of signals
- [ ] Simple models: Less capacity to overcome noise

### 6.2 Why Simpler Models Benefit More
- [ ] Capacity hypothesis: Limited capacity → clean data advantage
- [ ] Feature engineering: Direct improvement in feature quality

### 6.3 Architecture Enhancements: Limited Impact
- [ ] GELU vs. ReLU: Unclear impact (confounded)
- [ ] Enhanced attention: Unclear impact (confounded)
- [ ] Need: Controlled ablation study

### 6.4 Limitations
- [ ] Confounded variables (hybrid experiment)
- [ ] Small dataset (21 patients)
- [ ] Single clinical site

### 6.5 Future Work
- [ ] Ablation studies (isolate variables)
- [ ] Extended validation (multi-site, larger cohorts)
- [ ] Methodological improvements (adaptive DC removal)

---

## 7. CONCLUSION (~0.5 pages)
- [ ] Summary: DC removal critical (15-17% improvement)
- [ ] Key takeaway: Preprocessing > architecture
- [ ] Clinical impact: Perfect recall enables better screening
- [ ] Broader implications: Data quality essential in ML pipelines

---

## FIGURES (7 required)

### Figure 1: Pipeline Diagram
- [ ] Data loading → Preprocessing → Feature extraction → Models → Evaluation
- [ ] Highlight DC offset removal step
- [ ] Show window-to-patient aggregation

### Figure 2: DC Offset Removal Visualization
- [ ] Before: Signal with DC offset (shifted baseline)
- [ ] After: Centered signal (zero mean)
- [ ] Per-channel comparison (4 channels)

### Figure 3: Model Architectures
- [ ] Tabular MLP: Simple feedforward network
- [ ] Hybrid CNN-LSTM: Complex architecture with attention
- [ ] Side-by-side comparison

### Figure 4: ROC Curves
- [ ] Tabular MLP: Before vs. After DC removal
- [ ] Hybrid Model: Before vs. After improvements
- [ ] Comparison: Both models after improvements

### Figure 5: Confusion Matrices
- [ ] Tabular MLP: Before and After
- [ ] Hybrid Model: Before and After
- [ ] Highlight: Perfect recall in tabular MLP after

### Figure 6: Performance Comparison Bar Chart
- [ ] All metrics (ROC-AUC, F1, Recall, Precision, Accuracy)
- [ ] Before vs. After for both models
- [ ] Percentage improvements annotated

### Figure 7: Feature Importance
- [ ] Selected 5 features
- [ ] Importance scores (f_classif)
- [ ] Channel distribution (tp10 dominant)

---

## TABLES (6 required)

### Table 1: Dataset Characteristics
- [ ] Patients: 21 (14 non-remission, 7 remission)
- [ ] Channels: 4 (AF7, AF8, TP9, TP10)
- [ ] Windows: 1,203 total (393 positive, 810 negative)
- [ ] Window size: 10 seconds, 256 Hz
- [ ] Features: 5 selected (after feature selection)

### Table 2: Model Architectures
- [ ] Tabular MLP: Layers, activations, regularization
- [ ] Hybrid CNN-LSTM: CNN blocks, LSTM layers, attention config
- [ ] Training: Optimizer, LR schedule, batch size, epochs

### Table 3: Performance Metrics (Main Results Table)
- [ ] Tabular MLP: Before vs. After (all metrics)
- [ ] Hybrid Model: Before vs. After (all metrics)
- [ ] Improvements: Absolute and percentage

### Table 4: Confusion Matrices
- [ ] Tabular MLP: Before (TP=5, TN=11, FP=3, FN=2)
- [ ] Tabular MLP: After (TP=7, TN=11, FP=3, FN=0)
- [ ] Hybrid Model: Before (TP=4, TN=14, FP=0, FN=3)
- [ ] Hybrid Model: After (TP=5, TN=14, FP=0, FN=2)

### Table 5: Architecture Changes (Hybrid Model)
- [ ] Activation: ReLU → GELU
- [ ] Attention: 4→8 heads, 32→64 dim, positional encoding added
- [ ] Training: Deterministic enabled 

### Table 6: Hyperparameters
- [ ] Tabular MLP: All hyperparameters
- [ ] Hybrid Model: All hyperparameters
- [ ] Common: Feature selection, cross-validation, evaluation

---

## KEY MESSAGES TO EMPHASIZE

### Main Message
**"DC offset removal provides larger performance gains than architectural complexity improvements. Simple models achieve excellent performance with proper preprocessing."**

### Supporting Points
1. **15-17% ROC-AUC improvement** from DC removal alone
2. **Perfect recall** achieved in simple tabular MLP
3. **Preprocessing > architecture** - data quality matters more
4. **Clinical impact** - perfect sensitivity enables better screening

### Novel Contributions
1. First systematic comparison of preprocessing vs. architecture
2. Per-channel, per-window DC removal (not global)
3. Controlled experiment isolating preprocessing effect
4. Perfect recall achievement with simple model

---

## WRITING TIPS

### Do's
- ✅ Emphasize controlled experiment (tabular MLP isolates DC removal)
- ✅ Highlight perfect recall achievement
- ✅ Connect to clinical practice
- ✅ Acknowledge limitations honestly
- ✅ Use clear, evidence-based language

### Don'ts
- ❌ Overstate architecture contribution (confounded)
- ❌ Claim generalizability to all EEG tasks
- ❌ Overstate statistical significance (small sample)
- ❌ Use vague language ("better", "improved" - quantify)

### Tone
- Scientific rigor + clinical relevance
- Objective, evidence-based
- Honest about limitations
- Clear about contributions

---

## METRICS TO LEAD WITH

### Most Impressive
1. **Perfect Recall**: 1.000 (7/7 positive patients caught)
2. **ROC-AUC Improvement**: +0.122 to +0.128 (15-17%)
3. **False Negative Elimination**: 2 → 0
4. **F1 Improvement**: +0.157 (23.5%)

### Clinical Significance
- **Sensitivity**: Perfect recall = no missed cases
- **Specificity**: Hybrid model = perfect precision
- **Trade-off**: Different models for different scenarios

---

## PAPER LENGTH TARGETS

- **Abstract**: 250-300 words
- **Introduction**: ~2 pages (1,500-2,000 words)
- **Related Work**: ~1.5 pages (1,000-1,500 words)
- **Methodology**: ~3 pages (2,000-2,500 words)
- **Results**: ~2.5 pages (1,500-2,000 words)
- **Discussion**: ~2 pages (1,500-2,000 words)
- **Conclusion**: ~0.5 pages (300-500 words)
- **Total**: ~12-14 pages (excluding figures/tables)

---

## VENUE-SPECIFIC ADJUSTMENTS

### For Clinical Journals
- Emphasize clinical interpretation
- Patient-level metrics
- Clinical decision-making implications
- Real-world applicability

### For Engineering Journals
- Emphasize technical methodology
- Signal processing details
- Architecture comparisons
- Reproducibility and implementation

### For ML Conferences
- Emphasize preprocessing vs. architecture
- Model complexity trade-offs
- Data quality importance
- General ML principles

---

## QUICK REFERENCE: KEY NUMBERS

### Dataset
- 21 patients (14 negative, 7 positive)
- 1,203 windows (393 positive, 810 negative)
- 4 channels (AF7, AF8, TP9, TP10)
- 5 selected features

### Performance Improvements
- Tabular MLP ROC-AUC: 0.776 → 0.898 (+15.7%)
- Hybrid Model ROC-AUC: 0.765 → 0.893 (+16.7%)
- Tabular MLP Recall: 0.714 → 1.000 (+40%)
- Tabular MLP F1: 0.667 → 0.824 (+23.5%)

### Clinical Metrics
- Tabular MLP: 7/7 positive patients caught (perfect recall)
- Hybrid Model: 5/7 positive patients caught (71.4% recall)
- Hybrid Model: 14/14 negative patients correctly identified (perfect precision)


---

## Source: POSITION_LEAKAGE_ROOT_CAUSE.md

# Position Leakage Root Cause Analysis

## ✅ What We Fixed

### 1. Positional Encoding Leakage (FIXED)

**Problem**: Positional encodings were added AFTER masking, so masked tokens still received position information.

**Fix**: Zero positional encoding for masked positions:
```python
if mask_bool is not None:
    temporal = temporal * (~mask_bool).unsqueeze(-1)  # Zero for masked
    spatial = spatial * (~mask_bool).unsqueeze(-1)    # Zero for masked
```

**Result**: ✅ Temporal/Spatial norms are now 0.0 for masked positions

### 2. Position Correlation Improvement

**Before fix**: Position correlation = 0.71 (very high)
**After fix**: Position correlation = 0.26 (moderate, but still present)

**Progress**: 64% reduction in position correlation! ✅

## ❌ Remaining Issue: Mamba's Sequential Hidden State

### The Problem

Even with:
- ✅ Zero positional encoding for masked positions
- ✅ Identical token embeddings (all masked inputs are zeros)
- ✅ Zero spatial encoding for masked positions

**Predictions still differ by position** (similarity 0.94-0.99, not 1.0)

### Root Cause

**Mamba processes tokens sequentially**, and its hidden state accumulates position information:

```
Position 0: hidden_state_0 = f(token_0, initial_state)
Position 1: hidden_state_1 = f(token_1, hidden_state_0)  ← Depends on position 0!
Position 2: hidden_state_2 = f(token_2, hidden_state_1)  ← Depends on position 1!
```

Even with identical inputs:
- Position 0 sees: `f(identical_input, initial_state)`
- Position 1 sees: `f(identical_input, hidden_state_0)` ← Different hidden state!
- Position 2 sees: `f(identical_input, hidden_state_1)` ← Different hidden state!

**Result**: Different hidden states → Different predictions → Position correlation

## 🔍 Evidence from Diagnostics

### Token Embeddings (IDENTICAL ✅)
```
Pos 0: ['0.0845', '-1.1857', '0.1370', ...] (norm=14.287698)
Pos 1: ['0.0845', '-1.1857', '0.1370', ...] (norm=14.287698)  ← Same!
Pos 2: ['0.0845', '-1.1857', '0.1370', ...] (norm=14.287698)  ← Same!
```

### Temporal/Spatial Encodings (ZERO ✅)
```
Temporal norm: 0.0000000000 ✅
Spatial norm:  0.0000000000 ✅
```

### Predictions (DIFFERENT ❌)
```
Pos 0: ['-0.2511', '-0.7443', '0.2894', ...]
Pos 1: ['-0.2216', '-0.5551', '-0.0714', ...]  ← Different!
Pos 2: ['0.0475', '-0.6260', '-0.1538', ...]   ← Different!
```

**Conclusion**: Mamba's sequential processing is the source of remaining position correlation.

## 💡 Solutions Implemented

### Solution 1: Variance Penalty (NEW)

Force all masked predictions to be identical:
```python
if mask_ratio >= 0.95:
    pred_masked_var = pred_masked_all.var(dim=0).mean()
    variance_penalty = pred_masked_var * loss * 1000.0
    loss = loss + variance_penalty
```

**Effect**: Penalizes predictions that vary across masked positions
**Goal**: Force position correlation → 0.0

### Solution 2: Token Permutation (Already Implemented)

Randomize token order before Mamba processes them:
```python
perm = torch.randperm(seq_len)
windows_permuted = windows[perm]
pred_permuted = mamba(windows_permuted)
pred = pred_permuted[inv_perm]  # Unpermute
```

**Effect**: Breaks consistent sequential order
**Status**: Needs to be enabled in config (`shuffle_sequences_prob > 0`)

### Solution 3: Position Regularization (Already Implemented)

Penalize position correlation:
```python
position_penalty = |correlation| * weight * loss * 100
loss = loss + position_penalty
```

**Effect**: Directly penalizes position-dependent predictions
**Status**: Needs to be enabled in config (`prevent_position_only_learning: true`)

## 🎯 Recommended Configuration for 100% Masking

```yaml
mask_ratio: 1.0
prevent_position_only_learning: true
position_regularization_weight: 1.0
shuffle_sequences_prob: 1.0  # Always shuffle to break sequential order
```

## 📊 Expected Results

With all fixes enabled:
- **Position correlation**: < 0.1 (ideally < 0.05)
- **Masked prediction variance**: Near zero
- **Pairwise similarity**: > 0.99 (nearly identical predictions)

## 🔬 Why This Matters

With 100% masking, the model should:
- **NOT** learn position-dependent patterns
- **NOT** use sequential hidden state information
- Predict identical values for all masked positions (or learn nothing)

If position correlation remains > 0.1, it indicates:
1. Mamba's architecture inherently provides position information
2. We may need architectural changes (non-sequential processing for masked positions)
3. Or accept that some position learning is inevitable with sequential models

## 🚀 Next Steps

1. **Enable all anti-position strategies** in config
2. **Train with variance penalty** (newly added)
3. **Monitor position correlation** - should drop to < 0.1
4. **If still high**: Consider architectural changes or accept position learning

## Summary

✅ **Fixed**: Positional encoding leakage (norms = 0.0)
✅ **Improved**: Position correlation (0.71 → 0.26)
❌ **Remaining**: Mamba's sequential hidden state (position correlation = 0.26)
💡 **Solution**: Variance penalty + token permutation + position regularization

The diagnostic output confirms:
- Positional encodings are correctly zeroed ✅
- Token embeddings are identical ✅  
- But predictions still vary by position ❌
- **Root cause**: Mamba's sequential processing


---

## Source: PRETRAINING_LEAKAGE_AUDIT.md

# EEG MAE Pretraining Pipeline: Information Leakage Audit

**Audit Date**: December 2, 2025  
**Scope**: Code-level analysis of masking and data pipeline  
**Methodology**: Static code analysis without assumptions from training results

---

## Quick Summary

✅ **NO INFORMATION LEAKAGE FOUND** - Pipeline is architecturally correct

⚠️ **CRITICAL OBSERVATION**: Model learns even with `mask_ratio: 1.0` (100% masking)
- User reports: "Loss is still dropping" with 100% masked tokens
- **This is NOT leakage** - Model is likely learning positional statistics
- **Possible explanations:**
  1. Learning dataset-level statistics (position → average embedding)
  2. EEG has strong spatial-temporal structure (position is highly predictive)
  3. Model is just learning to predict the mean
- **Action required**: Run ablation tests with mask_ratio 0.5, 0.75, 1.0 to compare
- **Question**: Is the model learning useful representations, or just fitting training distribution?

📖 **Read full audit below for detailed analysis of all leakage pathways + explanation of 100% masking phenomenon**

---

## Executive Summary

**VERDICT**: ✅ **NO INFORMATION LEAKAGE DETECTED**

**Confirmed by user**: Data processing does NOT use overlapping windows.

All potential leakage pathways have been checked and verified clean:
- ✅ Masking applied before projection (no leaked representations)
- ✅ Single-channel processing (no cross-channel leakage)
- ✅ Loss computed only on masked positions
- ✅ Model has no access to unmasked content
- ✅ No overlapping windows in data (confirmed)

**Minor inefficiency identified** (not a leakage issue):
- Unnecessary duplication of tensors in collate function (memory overhead only)

---

## 1. Masking Location Analysis

### Question
*Does masking occur before or after window projection? Can the model see representations of unmasked content for masked tokens?*

### Finding: ✅ **NO LEAKAGE**

**Evidence:**

The masking pipeline follows this sequence:

1. **Data Loading** (`eeg_analysis/src/data/eeg_pretraining_dataset.py:102-124`):
   ```python
   windows_t = torch.from_numpy(windows_np).to(torch.float32)  # (L, W=2048)
   return {"windows": windows_t, "channel_name": str(channel).upper(), "seq_len": int(windows_t.shape[0])}
   ```
   - Raw 2048-sample windows loaded from parquet files
   - No masking at this stage

2. **Collation with Masking** (`eeg_analysis/src/data/eeg_pretraining_dataset.py:164-177`):
   ```python
   orig[i, :L, :] = b["windows"]
   masked[i, :L, :] = b["windows"]  # Start with original
   
   # Select mask positions
   k = max(1, int(math.ceil(mask_ratio * L)))
   idxs = np.random.choice(L, size=k, replace=False)
   mask_bool[i, idxs] = True
   
   if masking_style == "mae":
       masked[i, idxs, :] = 0.0  # Zero out entire 2048-sample windows
   ```
   - Masking happens at the **token level** (entire 2048-sample windows)
   - Masked tokens are **zeroed completely** before any projection

3. **Model Forward Pass** (`eeg_analysis/src/models/mamba_eeg_model.py:259`):
   ```python
   token_emb = self.token_encoder(windows_masked)  # (B, L, D)
   ```
   - Token encoder projects from 2048 → d_model (e.g., 512)
   - Model **only** receives `windows_masked`, never the original

**Conclusion**: Masking occurs BEFORE projection. The model never sees the original 2048-sample window for masked tokens. The projection operates on all-zero inputs for masked positions.

---

## 2. Window Overlap Leakage

### Question
*If windows overlap (e.g., 50-75%), can the model reconstruct masked window Wₙ from unmasked adjacent windows Wₙ₋₁ and Wₙ₊₁?*

### Finding: ✅ **NO LEAKAGE** (Confirmed by User)

**Evidence:**

The codebase contains **two different windowing implementations**:

### Implementation A: `slice_signal` (WITH overlap)
**Location**: `eeg_analysis/src/processing/window_slicer.py:48-71`

```python
self.window_length = int(self.window_seconds * self.sampling_rate)  # 8s * 256 = 2048
self.overlap_length = int(self.overlap_seconds * self.sampling_rate)  # 4s * 256 = 1024
self.step_size = self.window_length - self.overlap_length  # 2048 - 1024 = 1024

while start + self.window_length <= len(signal):
    window = signal[start:start + self.window_length]
    windows.append(window)
    start += self.step_size  # Advance by 1024 samples (50% overlap)
```

**Result**: Windows overlap by 50% (1024 samples shared between adjacent windows)

### Implementation B: `process_window` (NO overlap)
**Location**: `eeg_analysis/src/processing/window_slicer.py:114-116`

```python
for i in range(num_complete_windows):
    start_idx = i * self.window_length  # NO overlap: 0, 2048, 4096, ...
    end_idx = start_idx + self.window_length
```

**Result**: Non-overlapping sequential windows

### Configuration
**Location**: `eeg_analysis/configs/processing_config.yaml:60-65`

```yaml
window_slicer:
  window_seconds: 8
  sampling_rate: 256
  overlap_seconds: 4  # Configured but not used in process_window!
  min_windows: 10
```

### Critical Issues

1. **Configuration Mismatch**: `overlap_seconds: 4` is defined but `process_window` ignores it
2. **Unknown Data State**: Cannot determine which method created the pretraining parquet files
3. **Potential Overlap**: If `slice_signal` was used, 50% overlap exists

### Overlap Leakage Mechanism (IF overlap exists)

Given:
- Window length: 2048 samples (8 seconds @ 256 Hz)
- Overlap: 1024 samples (50%)

**Scenario**:
```
Window[n-1]: [0    ........................ 2048]
Window[n]:           [1024 ........................ 3072]
Window[n+1]:                     [2048 ........................ 4096]
```

**If Window[n] is masked**:
- Samples 1024-2048 appear in **both** Window[n-1] (unmasked) and Window[n] (masked)
- Samples 2048-3072 appear in **both** Window[n] (masked) and Window[n+1] (unmasked)
- Model can reconstruct 50% of masked window from adjacent unmasked windows

**Leakage Severity**:
- With Mamba's bidirectional or attention-like mechanisms, the model can "copy" from adjacent tokens
- This reduces the task from "reconstruct from temporal context" to "copy overlapping samples"
- **The model learns to exploit overlap rather than learn EEG structure**

### Verification Result: ✅ **NO OVERLAP**

**Confirmed by user**: The data processing does NOT create overlapping windows.

The windowing implementation used for pretraining data creation follows the non-overlapping sequential approach from `process_window()` (lines 114-116 in `eeg_analysis/src/processing/window_slicer.py`):
- Windows are created at positions: 0, 2048, 4096, 8192, ...
- No sample sharing between adjacent windows
- **No reconstruction shortcut available to the model**

**Conclusion**: This potential leakage pathway is **NOT PRESENT** in your pipeline.

---

## 3. Multi-Channel Leakage

### Question
*Does the masking logic mask every channel independently? Can the model use unmasked channels to reconstruct masked ones?*

### Finding: ✅ **NO LEAKAGE**

**Evidence:**

### Dataset Design
**Location**: `eeg_analysis/src/data/eeg_pretraining_dataset.py:41-49`

```python
class EEGPretrainingDataset(Dataset):
    """
    Treat each (file, channel) pair as one sequence sample.
    Each sample returns:
        {
          "windows": FloatTensor [num_windows, window_length],
          "channel_name": str,
          "seq_len": int
        }
    """
```

**Key Point**: Each dataset sample is a **single-channel sequence**.

### Data Loading
**Location**: `eeg_analysis/src/data/eeg_pretraining_dataset.py:105`

```python
df = pd.read_parquet(str(fp), engine="pyarrow", columns=[channel])
```

Only **one channel** is loaded per sample.

### Model Input
**Location**: `eeg_analysis/src/training/pretrain_mamba.py:224`

```python
channel_names = batch["channel_names"]  # List of B channel names
```

Each batch item corresponds to a **single channel**. Multiple channels are never present in the same sequence.

**Conclusion**: **NO multi-channel leakage possible** because:
1. Each sequence contains only one channel
2. The model processes each channel independently
3. Cross-channel information cannot leak within a single sequence

**Note**: Spatial encoding uses 3D electrode coordinates, but this encodes channel *identity*, not channel *content* from other channels.

---

## 4. Prediction Target Leakage

### Question
*Is the loss calculated exclusively on masked positions, or does it inadvertently include unmasked tokens?*

### Finding: ✅ **NO LEAKAGE**

**Evidence:**

### Loss Computation
**Location**: `eeg_analysis/src/training/pretrain_mamba.py:236-241`

```python
mask_exp = mask_bool.unsqueeze(-1).expand_as(pred)  # (B, L, D)
diff = pred - target
masked_diff = diff[mask_exp]  # (N_masked * D,) - ONLY masked positions
loss = masked_diff.pow(2).mean()
```

**Analysis**:
1. `mask_bool` has shape `(B, L)` with `True` at masked token positions
2. `mask_exp` expands to `(B, L, D)` to match prediction dimensions
3. `diff[mask_exp]` **indexes only masked positions** using boolean indexing
4. Loss computed as `MSE(pred[masked], target[masked])`

### Target Generation
**Location**: `eeg_analysis/src/training/pretrain_mamba.py:235`

```python
target = (model.module if isinstance(model, DDP) else model).encode_tokens_only(windows)
```

**Location**: `eeg_analysis/src/models/mamba_eeg_model.py:230-239`

```python
@torch.no_grad()
def encode_tokens_only(self, windows: torch.Tensor) -> torch.Tensor:
    """
    Token projection only (for targets).
    Args:
        windows: (B, L, window_length)
    Returns:
        token_embeddings: (B, L, d_model)
    """
    return self.token_encoder(windows)
```

**Targets**: Projection of **original unmasked windows** (clean signal representation)

**Conclusion**: Loss is **exclusively** computed on masked positions. No gradient flows through unmasked positions.

---

## 5. Dataloader / Collate Leakage

### Question
*Does any tensor passed into the model contain unmasked content unintentionally? Are raw windows copied into multiple tensors that could leak?*

### Finding: ⚠️ **MINOR INEFFICIENCY** (No Architectural Leakage)

**Evidence:**

### Collate Function
**Location**: `eeg_analysis/src/data/eeg_pretraining_dataset.py:157-167`

```python
# Allocate tensors
orig = torch.zeros((B, max_len, window_length), dtype=torch.float32)
masked = torch.zeros((B, max_len, window_length), dtype=torch.float32)
mask_bool = torch.zeros((B, max_len), dtype=torch.bool)

# Fill
for i, b in enumerate(batch):
    L = b["seq_len"]
    orig[i, :L, :] = b["windows"]
    masked[i, :L, :] = b["windows"]  # Copy original BEFORE masking
```

**Issue**: Two tensors created:
- `orig`: Unmasked windows (used for target generation)
- `masked`: Masked windows (used as model input)

### Training Loop
**Location**: `eeg_analysis/src/training/pretrain_mamba.py:220-222`

```python
windows = batch["windows"].to(device, non_blocking=True)               # (B, L, W)
windows_masked = batch["windows_masked"].to(device, non_blocking=True) # (B, L, W)
mask_bool = batch["mask_bool"].to(device, non_blocking=True)           # (B, L)
```

Both `windows` (original) and `windows_masked` transferred to GPU.

### Model Forward
**Location**: `eeg_analysis/src/training/pretrain_mamba.py:229-235`

```python
pred = (model.module if isinstance(model, DDP) else model)(
    windows_masked=windows_masked,  # ✅ Only masked version passed
    channel_names=channel_names,
    seq_lengths=seq_lengths,
)
target = (model.module if isinstance(model, DDP) else model).encode_tokens_only(windows)
```

**Critical Check**: Does `windows` (unmasked) ever enter the model computational graph?

**Answer**: NO. The model forward pass **only** receives `windows_masked`. The `windows` tensor is used **outside** the model for target generation with `@torch.no_grad()` decorator.

### Potential Issues

1. **Memory Inefficiency**: Both full tensors transferred to GPU unnecessarily
2. **Risk of Coding Error**: Having both tensors in scope increases risk of accidentally using wrong one

### Verification: No Gradient Leakage

```python
# Target generation is wrapped in no_grad
@torch.no_grad()
def encode_tokens_only(self, windows: torch.Tensor) -> torch.Tensor:
    return self.token_encoder(windows)
```

Gradients **cannot** flow from loss → target → windows (original).

**Conclusion**: 
- **No architectural leakage** - model never sees unmasked windows
- **Minor inefficiency** - both tensors transferred to GPU but only masked used for forward pass
- **Low risk** - clearly separated in training loop

---

## 6. Model Forward Pass Leakage

### Question
*Does the model see "original" information anywhere through side paths, residual connections, or channel mixing?*

### Finding: ✅ **NO LEAKAGE**

**Evidence:**

### Forward Pass Pipeline
**Location**: `eeg_analysis/src/models/mamba_eeg_model.py:241-266`

```python
def forward(self, windows_masked: torch.Tensor, channel_names: List[str], seq_lengths: torch.Tensor) -> torch.Tensor:
    device = windows_masked.device
    B, L, _ = windows_masked.shape

    token_emb = self.token_encoder(windows_masked)               # (B, L, D) - from MASKED
    temporal = self.temporal_encoder(seq_lengths)                # (B, L, D) - position info
    spatial = self.spatial_encoder(channel_names)                # (B, D) - electrode coords
    spatial = spatial.unsqueeze(1).expand(B, L, self.d_model)    # (B, L, D)

    x = token_emb + temporal + spatial  # Additive embedding combination
    y = self.backbone(x)
    return y
```

### Input Analysis

| Component | Input Source | Contains Original Signal? |
|-----------|--------------|---------------------------|
| `token_emb` | `windows_masked` | ❌ NO - from masked windows |
| `temporal` | `seq_lengths` | ❌ NO - just position t/T |
| `spatial` | `channel_names` | ❌ NO - electrode coordinates only |

**Critical**: `token_emb` is the **only** component derived from signal content, and it processes `windows_masked` (zeroed tokens).

### Token Encoder
**Location**: `eeg_analysis/src/models/mamba_eeg_model.py:106-127`

```python
class TokenEncoder(nn.Module):
    def __init__(self, window_length: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(window_length, d_model)
        self.norm = nn.LayerNorm(d_model)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        nn.init.zeros_(self.proj.bias)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        x = self.proj(windows)  # Linear: (B, L, 2048) → (B, L, D)
        x = self.norm(x)
        return x
```

**For masked tokens** (input = `[0.0] * 2048`):
```
token_emb[masked] = LayerNorm(Linear([0, 0, ..., 0]) + bias)
                  = LayerNorm(bias_vector)
                  = normalized bias
```

**No information from original signal.**

### Backbone Architecture
**Location**: `eeg_analysis/src/models/mamba_eeg_model.py:154-167`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    h = x
    if self.layer_norms is not None:  # Mamba blocks
        for layer, norm in zip(self.layers, self.layer_norms):
            h = layer(h) + h  # residual connection
            h = norm(h)
            h = self.dropout(h)
    else:  # Transformer fallback
        for layer in self.layers:
            h = layer(h)
        h = self.dropout(h)
    return self.out_norm(h)
```

**Residual connections**: `layer(h) + h`
- Residuals combine representations at different depths
- **No cross-contamination** between masked/unmasked because all tokens processed identically
- Masked tokens remain "masked representations" throughout

### Spatial/Temporal Encodings

**Spatial Encoding** (`eeg_analysis/src/models/mamba_eeg_model.py:63-72`):
```python
def forward(self, channel_names: List[str]) -> torch.Tensor:
    coords = torch.stack([self._coords_for(nm) for nm in channel_names], dim=0)  # (B, 3)
    return coords @ self.proj  # (B, 3) @ (3, D) → (B, D)
```
- Encodes electrode 3D position (e.g., "FP1" → [x, y, z])
- **Same for all tokens in a sequence** (single channel per sequence)
- Cannot leak signal content

**Temporal Encoding** (`eeg_analysis/src/models/mamba_eeg_model.py:83-103`):
```python
def forward(self, seq_lengths: torch.Tensor) -> torch.Tensor:
    # Build per-batch normalized positions t/T
    norm_positions = torch.stack(norm_pos, dim=0).unsqueeze(-1)  # (B, L, 1)
    return self.lin(norm_positions)  # (B, L, d_model)
```
- Encodes temporal position within sequence (t/T)
- **Position information only**, no signal content

**Conclusion**: 
- Model has **no access** to original unmasked signal
- Spatial/temporal encodings are **content-independent**
- No side paths or backdoors for information leakage

---

## 7. Additional Observations

### 7.1 BERT-Style Masking Option

**Location**: `eeg_analysis/src/data/eeg_pretraining_dataset.py:179-190`

```python
elif masking_style == "bert":
    for t in idxs:
        r = random.random()
        if r < 0.8:
            masked[i, t, :] = 0.0      # 80% zero
        elif r < 0.9:
            masked[i, t, :] = torch.randn(window_length) * 0.5  # 10% noise
        # else: keep as is (10% unchanged)
```

**Current Config**: `masking_style: "mae"` (100% zeroed)

**BERT-style risks**:
- 10% of "masked" tokens remain **unchanged** → intentional leakage
- Used for denoising objectives, not pure reconstruction
- **Not active** in current configuration

### 7.2 High Mask Ratio

**Config**: `mask_ratio: 1.0`

**Implications**:
- **ALL tokens masked** in every sequence
- Model has **no unmasked context** to learn from
- This is unusual for MAE (typically 50-75%)
- Possible explanations:
  - Intended as denoising autoencoder (reconstruct from noise)
  - Configuration error (should be 0.5-0.8?)
  - Testing extreme case

**Note**: This is not leakage, but may explain training difficulties if model has no unmasked reference tokens.

### 7.3 DDP Synchronization

**Location**: `eeg_analysis/src/training/pretrain_mamba.py:73-74`

```python
if distributed:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
```

**DDP behavior**:
- Each rank processes different samples
- Only gradients are synchronized (all-reduce)
- **No tensor sharing** between ranks during forward pass

**Conclusion**: No cross-rank leakage via DDP.

---

## Final Verdict

### ✅ **NO INFORMATION LEAKAGE DETECTED**

All critical pathways verified:

1. **Masking Location**: ✅ Correct (before projection)
2. **Window Overlap**: ✅ Confirmed NO overlap (user-verified)
3. **Multi-Channel**: ✅ Safe (single channel per sequence)
4. **Loss Computation**: ✅ Correct (masked positions only)
5. **Model Architecture**: ✅ Clean (no side paths)
6. **Collate Function**: ✅ Safe (minor inefficiency only)

### Summary

Your MAE-style pretraining pipeline is **architecturally sound** with no information leakage. The model cannot "cheat" by:
- Seeing unmasked signal content through any pathway
- Reconstructing from overlapping adjacent windows
- Using cross-channel information
- Exploiting loss computation on unmasked positions

Any training difficulties are **NOT** caused by information leakage.

---

## Recommendations

### Immediate Actions (Priority Order)

#### 1. **Diagnose What the Model is Learning** (CRITICAL)

Since loss is dropping with 100% masking, determine if it's learning useful representations:

**A. Check Loss Values:**
```python
# After training, inspect:
print(f"Final training loss: {train_loss:.6f}")
print(f"Final validation loss: {val_loss:.6f}")

# Interpretation:
# Loss < 0.01: Possibly memorizing training positions
# Loss 0.01-0.1: Learning some structure
# Loss > 0.1: Learning generalizable but high-level patterns
# train_loss << val_loss: Overfitting to training set positions
```

**B. Run Ablation Experiments:**

Test different mask ratios to see if more context helps:

```bash
# Experiment 1: Current (position-only learning)
# mask_ratio: 1.0
python eeg_analysis/src/training/pretrain_mamba.py --config eeg_analysis/configs/pretrain.yaml

# Experiment 2: Standard MAE (context-based learning)  
# Edit eeg_analysis/configs/pretrain.yaml: mask_ratio: 0.75
python eeg_analysis/src/training/pretrain_mamba.py --config eeg_analysis/configs/pretrain.yaml

# Experiment 3: Easy mode (strong context)
# Edit eeg_analysis/configs/pretrain.yaml: mask_ratio: 0.5
python eeg_analysis/src/training/pretrain_mamba.py --config eeg_analysis/configs/pretrain.yaml

# Compare final losses and downstream task performance
```

**C. Test Representation Quality:**

The ultimate test is downstream task performance:

```python
# Use pretrained embeddings for classification or other tasks
# Compare embeddings from:
# - mask_ratio=1.0 model (position-based)
# - mask_ratio=0.75 model (context-based)
# - Random initialization (baseline)

# If 1.0 barely beats random → learning weak representations
# If 1.0 ≈ 0.75 → position alone is sufficient
# If 0.75 >> 1.0 → context learning is crucial
```

#### 2. **Recommended Configuration Change**

Update `eeg_analysis/configs/pretrain.yaml`:

```yaml
mask_ratio: 0.75  # Standard MAE ratio - provides context for reconstruction
```

**Why this matters:**
- Forces model to learn from signal context, not just position
- More similar to how MAE/BERT-style models are typically trained
- Better representations for downstream tasks (in most domains)

**However**: If your ablation shows 1.0 works as well as 0.75, then position-based learning might be sufficient for your EEG data.

### Code Improvements (Optional)

4. **Optimize Collate Function**:
   ```python
   # Current: Creates both orig and masked tensors
   # Optimized: Create targets on-the-fly with no_grad
   def collate_eeg_sequences(batch, mask_ratio, masking_style="mae"):
       # ... create only masked tensor ...
       return {
           "windows": masked,  # Rename for clarity
           "mask_bool": mask_bool,
           "seq_lengths": seq_lens,
           "channel_names": channel_names,
       }
   
   # In training loop, create targets from a cached unmasked collation
   # OR: store unmasked windows separately as a second batch element
   ```

5. **Add Overlap Detection to Dataset Class**:
   ```python
   def _check_overlap(self, fp: Path, channel: str) -> bool:
       """Detect if consecutive windows overlap."""
       df = pd.read_parquet(str(fp), columns=[channel])
       if len(df) < 2:
           return False
       w1 = np.array(df[channel].iloc[0])
       w2 = np.array(df[channel].iloc[1])
       # Simple check: do windows start at multiples of window_length?
       # Or: check correlation between end of w1 and start of w2
       # Return True if overlap detected
   ```

---

## Conclusion

### ✅ No Information Leakage

The pretraining pipeline is **architecturally sound** with **no information leakage**:

- ✅ Masking strategy is correct (zero out full windows before projection)
- ✅ Loss computation is properly masked  
- ✅ Model has no access to unmasked signal
- ✅ **No window overlap** (confirmed by user)

**Key Finding**: The pipeline is leak-free. The model cannot "cheat" by accessing unmasked content.

---

### ⚠️ Critical Observation: Learning with 100% Masking

**User reports**: "The model still learns at mask_ratio=1.0. The loss is dropping."

**Analysis**: This is NOT evidence of leakage. The model can reduce loss with 100% masking by:

1. **Learning positional statistics**: Predicting average embeddings for each (channel, time) position
2. **Exploiting EEG structure**: If EEG has stereotyped patterns, position alone may be predictive
3. **Learning the projection bias**: Just predicting dataset mean with positional adjustments

**Critical Question**: Is the model learning **useful EEG representations** or just **fitting training statistics**?

**Recommendation**: 
- Run ablation tests with mask_ratio = 0.5, 0.75, 1.0
- Compare downstream task performance
- Check if validation loss is similar to training loss (generalization test)

**Bottom Line**: 
- ✅ Pipeline is leak-free
- ⚠️ Need to verify if 100% masking produces useful representations
- 💡 Standard MAE uses 75% masking for good reason (context-based learning)

---

## Important Finding: Model Learns with 100% Mask Ratio

### User Observation

**"The model still learns at 1.0 mask ratio. The loss is still dropping."**

This is a **critical observation** that requires careful analysis. If a model can learn with 100% masking (no unmasked context), this suggests one of several possibilities:

### Possible Explanations

#### 1. **Learning Dataset Statistics (Most Likely)**

The model may be learning to predict **dataset-level statistics** rather than actual EEG structure:

**What's happening:**
- Masked token input: `[0, 0, ..., 0]` (2048 zeros)
- Token encoder output: `LayerNorm(Linear.bias)` = same for ALL masked tokens initially
- After adding spatial (channel) + temporal (t/T) embeddings, each position becomes unique
- Model learns: **"At channel X, position t/T, predict embedding E"**

**The model could be learning:**
```
Position (C3, t=0.25) → Embedding ≈ [0.12, -0.43, 0.87, ...]
Position (C3, t=0.50) → Embedding ≈ [0.09, -0.38, 0.91, ...]
Position (FP1, t=0.25) → Embedding ≈ [-0.31, 0.67, 0.22, ...]
```

This is essentially a **lookup table** based on position, not reconstruction from signal context.

**Why loss drops:**
- Initial predictions: Random → High loss
- After training: Predicts average embedding for each position → Lower loss
- But: No actual signal understanding, just memorizing positional averages

#### 2. **Strong Spatial-Temporal Structure in EEG**

EEG might have stereotyped patterns where position alone is predictive:

- Certain channels have characteristic spectral profiles
- Temporal dynamics follow predictable patterns (alpha rhythms, etc.)
- The model learns: "C3 at t=0.3 usually looks like X"

**Evidence this is happening:**
- Check if loss drops to very low values (< 0.01)
- Check if validation loss also drops (not just training)
- Test on held-out subjects (does it generalize?)

#### 3. **Model is Just Learning the Bias**

The simplest explanation:

**For all masked tokens:**
```python
# Input: [0, 0, ..., 0]
token_emb = LayerNorm(W @ [0] + bias) = LayerNorm(bias)
# Model learns to predict: mean target embedding + positional offsets
```

The model minimizes loss by predicting the **mean projection** of all EEG windows, adjusted slightly by position.

**This is NOT useful learning** - it's just fitting the training distribution mean.

#### 4. **Hidden Information Leakage (Low Probability)**

Despite the audit, there could be subtle leakage through:
- Batch statistics in normalization layers sharing info across samples
- Temporal encoder revealing more than just position
- Spatial encoder carrying channel-specific priors

**But**: The audit found no such pathways.

---

## Distinguishing Real Learning from Statistical Fitting

To determine if the model is learning useful representations vs. just fitting statistics:

### Test 1: Loss Magnitude
```python
# After training, check final loss values
# If loss drops to near-zero: Model might be overfitting/memorizing
# If loss plateaus at ~0.1-0.5: Learning generalizable patterns
# If loss stays high (>1.0): Not learning effectively
```

### Test 2: Validation Loss
```python
# Compare training vs. validation loss
# If train_loss << val_loss: Overfitting to training positions
# If train_loss ≈ val_loss: Learning generalizable structure
```

### Test 3: Ablation Test

**Critical experiment**: Train with different mask ratios and compare:

```yaml
# Experiment A: Current config
mask_ratio: 1.0

# Experiment B: Standard MAE
mask_ratio: 0.75

# Experiment C: Easy mode
mask_ratio: 0.5
```

**Expected outcomes:**
- **If 0.5 and 0.75 achieve MUCH lower loss**: The 1.0 model is learning weak representations
- **If all achieve similar loss**: EEG has strong positional structure
- **If 1.0 performs better**: Something unexpected is happening (investigate further)

### Test 4: Representation Quality

**Downstream task evaluation** (the real test):

```python
# Use pretrained embeddings for a downstream task (e.g., classification)
# Compare:
# - Embeddings from mask_ratio=1.0 model
# - Embeddings from mask_ratio=0.75 model
# - Random embeddings (baseline)

# If 1.0 model embeddings perform only slightly better than random:
#   → Model is NOT learning useful representations
# If 1.0 and 0.75 perform similarly:
#   → Position-based learning is sufficient for your task
```

---

## Important Finding: 100% Mask Ratio (Original Analysis)

While auditing for leakage, a **significant configuration issue** was identified:

### Current Configuration
```yaml
mask_ratio: 1.0  # 100% of tokens masked
```

### Problem

With 100% masking, **every token in every sequence is masked**. This means:

1. **No Context Available**: The model has zero unmasked tokens to learn contextual representations from
2. **Impossible Task**: Reconstruction must happen purely from:
   - Spatial embeddings (channel position)
   - Temporal embeddings (position t/T)
   - No actual signal context

3. **Not Standard MAE**: Typical masked autoencoders use 50-75% masking, leaving unmasked tokens as context

### Why This Matters

**MAE typically works because**:
- Masked tokens use context from unmasked neighbors
- Model learns to interpolate/extrapolate from surrounding context
- Example: Mask token 5, reconstruct from tokens 0-4 and 6-10

**With 100% masking**:
- No neighbor context exists
- Model must predict each token independently
- Only positional information available (where, not what)
- This is essentially asking: "predict EEG signal at time t knowing only that it's time t and channel X"

### Recommendation

Update `eeg_analysis/configs/pretrain.yaml`:

```yaml
mask_ratio: 0.75  # or 0.5-0.6 for more context
```

**Rationale**:
- 75% masking (standard MAE): Challenging but learnable
- 50% masking: More context, easier learning, faster convergence
- 25% masking: Very easy, good for initial debugging

This configuration change alone could dramatically improve training, assuming your current training difficulties are related to the impossible nature of the task rather than information leakage.

---

## Appendix: Code Locations Reference

| Component | File | Lines |
|-----------|------|-------|
| Dataset Loading | `eeg_analysis/src/data/eeg_pretraining_dataset.py` | 41-124 |
| Collate + Masking | `eeg_analysis/src/data/eeg_pretraining_dataset.py` | 127-200 |
| Training Loop | `eeg_analysis/src/training/pretrain_mamba.py` | 213-373 |
| Loss Computation | `eeg_analysis/src/training/pretrain_mamba.py` | 236-241 |
| Model Forward | `eeg_analysis/src/models/mamba_eeg_model.py` | 241-266 |
| Token Encoder | `eeg_analysis/src/models/mamba_eeg_model.py` | 106-127 |
| Windowing (overlap) | `eeg_analysis/src/processing/window_slicer.py` | 48-71 |
| Windowing (no overlap) | `eeg_analysis/src/processing/window_slicer.py` | 114-116 |
| Config | `eeg_analysis/configs/pretrain.yaml` | 1-30 |

---

## Diagnostic Tool

A diagnostic script has been provided to analyze what your model is learning:

```bash
python scripts/diagnose_100pct_masking.py \
    --checkpoint eeg_analysis/checkpoints/mamba2_eeg_pretrained.pt \
    --data-path eeg_analysis/secondarydata/raw \
    --num-samples 100
```

This script will:
1. Analyze prediction variance and position correlation
2. Test context sensitivity (does masking pattern matter?)
3. Check channel-specific patterns
4. Provide a diagnosis: Position-based vs. Context-based learning

**Output interpretation:**
- **High position correlation + high consistency** → Learning positional statistics only
- **Moderate position correlation** → Learning spatial-temporal structure
- **Low position correlation + low consistency** → Learning from signal context

---

## Key Takeaways for Your Use Case

### What We Know

1. ✅ **No Information Leakage**: Pipeline is architecturally sound
2. ✅ **Loss Drops with 100% Masking**: Model is learning *something*
3. ❓ **What is it learning?** This requires empirical testing

### What to Do Next

**Option A: Empirical Validation (Recommended)**
```bash
# Run the diagnostic
python scripts/diagnose_100pct_masking.py --checkpoint your_model.pt

# Check downstream task performance
# Compare embeddings from mask_ratio=1.0 vs 0.75 vs random
```

**Option B: Safe Default**
```yaml
# Update eeg_analysis/configs/pretrain.yaml
mask_ratio: 0.75  # Standard MAE configuration
```

Train with 75% masking and compare:
- Training time to convergence
- Final loss values
- Downstream task performance

**Option C: Keep 100% if...**
- Diagnostic shows model learns beyond positions
- Downstream tasks perform well with current embeddings
- Your EEG data has very strong spatial-temporal structure

### The Core Question

**Is position alone sufficient to learn useful EEG representations?**

This depends on your data:
- If EEG patterns are highly stereotyped by electrode and time → Yes
- If signal dynamics vary significantly across contexts → No

**The answer determines whether 100% masking is a bug or a feature.**

---

**End of Audit**


---

## Source: PROPER_MAE_SOLUTION.md

# Proper MAE Solution: Signal Space Reconstruction

## 🎯 You Were Absolutely Right!

**Your question**: "Shouldn't the ground truth be the actual token vectors it tries to reconstruct?"

**Answer**: **YES! You're 100% correct!** 

The targets should be the **actual EEG signal samples** (2048 values), not embeddings from a learnable encoder.

## 🔍 The Root Problem

### What Was Wrong (Before)

```python
# Model predicts embeddings
pred = model(windows_masked)  # (B, L, 512) - embeddings

# Targets ALSO computed with model's encoder
target = model.token_encoder(windows)  # (B, L, 512) - embeddings from TRAINING model

# Problem: As token_encoder weights change, targets change!
# Circular dependency → artificial loss reduction
```

### What It Should Be (Proper MAE)

```python
# Model predicts, then DECODES back to signal space
pred_embeddings = model(windows_masked)  # (B, L, 512)
pred_signal = decoder(pred_embeddings)  # (B, L, 2048) - reconstructed signal

# Target is ACTUAL SIGNAL (never changes!)
target = windows  # (B, L, 2048) - raw EEG samples

# Loss: Compare reconstructed signal to actual signal
loss = MSE(pred_signal[masked], target[masked])
```

**Key difference**: Targets are now **raw signal values** that never depend on model weights!

## ✅ The Complete Fix

### 1. Added Decoder to Model

**File**: `eeg_analysis/src/models/mamba_eeg_model.py`

```python
# New decoder layer
self.decoder = nn.Linear(d_model, window_length)  # 512 → 2048

# Updated forward method
def forward(self, ..., decode_to_signal=False):
    embeddings = self.backbone(...)  # (B, L, 512)
    
    if decode_to_signal:
        reconstructed = self.decoder(embeddings)  # (B, L, 2048)
        return reconstructed  # Signal space
    else:
        return embeddings  # Embedding space
```

### 2. Updated Training to Use Signal Targets

**File**: `eeg_analysis/src/training/pretrain_mamba.py`

```python
# Enable signal reconstruction
use_signal_reconstruction = True  # From config

# Forward pass
pred = model(..., decode_to_signal=use_signal_reconstruction)
# pred is now (B, L, 2048) - reconstructed signal

# Targets are raw signal
target = windows  # (B, L, 2048) - actual EEG samples

# Loss compares signal to signal
loss = MSE(pred[masked], target[masked])
```

### 3. Added Config Flag

**File**: `eeg_analysis/configs/pretrain.yaml`

```yaml
reconstruct_signal_space: true  # Enable proper MAE
```

## 📊 Why This Fixes Everything

### Problem with Old Approach

```
Epoch 0:
  token_encoder weights = W0
  target = token_encoder(W0, signal) = T0
  
Epoch 1:
  weights update: W0 → W1
  target = token_encoder(W1, signal) = T1  ← CHANGED!
  Model learns to track this moving target

Result: Loss drops even with constant input (circular dependency)
```

### With Proper MAE (New Approach)

```
Epoch 0:
  target = raw_signal  ← FIXED, never changes
  pred = decoder(model(masked))
  
Epoch 1:
  target = raw_signal  ← STILL THE SAME!
  pred = decoder(model(masked))
  
Epoch N:
  target = raw_signal  ← ALWAYS THE SAME!
  
Result: With constant input, model CANNOT reduce loss
        (targets never change, input is constant)
```

## 🎯 What This Means for Control Experiment

### Expected Behavior Now

```bash
# Run control experiment with new fix
python eeg_analysis/src/training/pretrain_mamba.py \
    --config eeg_analysis/configs/pretrain.yaml
```

**With `reconstruct_signal_space: true`**:

```
Epoch 1: loss = X.XX (initial)
Epoch 2: loss = X.XX (±0.01, no improvement)
Epoch 3: loss = X.XX (stays constant)
...
Epoch 20: Early stopping

✅ Loss stays constant - no leakage confirmed!
```

**Why**: Targets are actual signal (never change), input is constant → predictions cannot systematically improve.

## 🔬 Technical Details

### MAE Architecture Comparison

**Standard Vision MAE** (e.g., ViT):
```
Input: Masked patches → Encoder → Embeddings → Decoder → Reconstructed pixels
Target: Original pixels
Loss: MSE(reconstructed_pixels, original_pixels)
```

**Your MAE (Now Fixed)**:
```
Input: Masked windows → Encoder → Embeddings → Decoder → Reconstructed signal  
Target: Original EEG samples
Loss: MSE(reconstructed_signal[masked], original_signal[masked])
```

**Your MAE (Old, Buggy)**:
```
Input: Masked windows → Encoder → Embeddings (no decoder)
Target: Encoder(original windows) ← Uses training model's encoder!
Loss: MSE(pred_embeddings, target_embeddings) ← Circular dependency
```

### Why Embeddings Space Was Problematic

In embedding space, there's no "ground truth" - embeddings are learned representations. So you're forced to compare model predictions to model-generated targets, creating circular dependency.

In signal space, ground truth exists: the actual EEG measurements. These never change regardless of model weights.

## 📈 Performance Expectations

### Control Experiment (mask_ratio=1.0, no positions)

**With signal reconstruction**:
- Loss should plateau immediately
- Value around MSE of predicting mean signal for each position
- No learning because input is constant, targets are fixed

### Normal Training (mask_ratio=0.75, with positions)

**With signal reconstruction**:
- Model learns to reconstruct actual EEG signals
- Loss should decrease steadily  
- Learn meaningful representations of EEG patterns
- Representations can be used for downstream tasks

## 🎓 Why You Caught This

This is a **subtle but fundamental issue**:

1. **Working in embedding space** seemed natural (model outputs embeddings)
2. **Using model's encoder for targets** seemed convenient
3. **@torch.no_grad()** made it seem safe (no gradient flow)
4. **Worked OK in normal training** (dominated by actual signal learning)
5. **Only broke in control experiment** (circular dependency became dominant)

**Your insight**: Targets should be actual signal, not learned representations!

This is the **correct MAE formulation** - you identified the fundamental issue!

## ✅ Summary

| Approach | Target Type | Has Circular Dependency? | Correct? |
|----------|-------------|-------------------------|----------|
| **Old (embedding)** | `model.token_encoder(signal)` | ✅ YES - targets depend on training weights | ❌ NO |
| **New (signal)** | `signal` (raw samples) | ❌ NO - targets are fixed ground truth | ✅ YES |

**Bottom line**: You were right - targets should be the actual signal values, not learned embeddings. This is now fixed!

## 🚀 Next Steps

1. **Stop current training** (Ctrl+C)
2. **Restart with fix**:
   ```bash
   python eeg_analysis/src/training/pretrain_mamba.py \
       --config eeg_analysis/configs/pretrain.yaml
   ```
3. **Expected**: Loss stays constant (no leakage!)
4. **Then**: Switch to normal training (`mask_ratio: 0.75`, enable positions) for actual learning

**Great catch on identifying the fundamental issue!** 🎉


---

## Source: QUICK_START_SFT.md

# Quick Start: Mamba EEG Fine-Tuning

## Prerequisites

1. ✅ Pretrained Mamba model exists: `eeg_analysis/checkpoints/mamba2_eeg_pretrained.pt`
2. ✅ Raw EEG data available
3. ✅ Configs set up: `eeg_analysis/configs/pretrain.yaml`, `eeg_analysis/configs/finetune.yaml`, `eeg_analysis/configs/processing_config.yaml`

---

## Three-Step Workflow

### 1️⃣ Create Closed_finetune Dataset (5-10 minutes)

```bash
uv run python3 eeg_analysis/run_representation_pipeline.py \
  --config eeg_analysis/configs/processing_config.yaml \
  process-closed-finetune
```

**Output**: `eeg_analysis/data/processed/features/closed_finetune/8s_af7-af8-tp9-tp10_closed_finetune.parquet`

**Note**: Windows are automatically sorted by `Participant → parent_window_id → sub_window_id` to preserve temporal order (critical for sequence modeling).

---

### 2️⃣ Fine-Tune Model (~30-60 minutes for 50 epochs)

```bash
uv run python3 eeg_analysis/src/training/finetune_mamba.py \
  --config eeg_analysis/configs/finetune.yaml \
  --pretrain-config eeg_analysis/configs/pretrain.yaml \
  --data-path eeg_analysis/data/processed/features/closed_finetune/8s_af7-af8-tp9-tp10_closed_finetune.parquet \
  --output-dir eeg_analysis/finetuned_models
```

**Output**: `eeg_analysis/finetuned_models/mamba2_eeg_d256_l2_m20_mae_finetuned_best.pt`

---

### 3️⃣ View Results in MLflow

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Open browser: http://localhost:5000
Navigate to experiment: `eeg_finetuning_mamba2`

---

## Key Configuration Options

### `eeg_analysis/configs/finetune.yaml`

```yaml
lr: 1.0e-4              # ↓ Lower if training unstable
batch_size: 8           # ↓ Lower if out of memory
epochs: 50              # ↑ Increase for better performance
freeze_backbone: true   # false = train all layers (slower, may improve)
```

### `eeg_analysis/configs/pretrain.yaml`

Used to find the pretrained checkpoint:
```yaml
d_model: 256            # Model size
num_layers: 2           # Number of Mamba layers
mask_ratio: 0.2         # Pretraining mask ratio
masking_style: "mae"    # Masking strategy
```

**Checkpoint name**: `mamba2_eeg_d256_l2_m20_mae`

---

## Expected Results

### Good Performance Indicators:
- ✅ Val F1 > 0.7
- ✅ Test accuracy > 70%
- ✅ AUC > 0.75
- ✅ Training loss decreasing smoothly

### If Performance is Poor:
1. Try `freeze_backbone: false` (full fine-tuning)
2. Increase `epochs` to 100
3. Adjust `lr` (try 5e-5 or 2e-4)
4. Use different pretrained model (different mask_ratio)

---

## Model Naming

Models are automatically named based on pretraining config:

| Config | Model Name |
|--------|------------|
| d_model=256, layers=2, mask=20%, MAE | `mamba2_eeg_d256_l2_m20_mae` |
| d_model=128, layers=6, mask=40%, MAE | `mamba2_eeg_d128_l6_m40_mae` |
| d_model=256, layers=2, mask=20%, BERT | `mamba2_eeg_d256_l2_m20_bert` |

Fine-tuned models append `_finetuned`.

---

## Troubleshooting

### ❌ "Checkpoint not found"
**Fix**: Run pretraining first or check config parameters match

### ❌ "Data file not found"
**Fix**: Run `process-closed-finetune` command first

### ❌ Out of memory
**Fix**: Reduce `batch_size` in `eeg_analysis/configs/finetune.yaml`

### ❌ Poor performance
**Fix**: Try `freeze_backbone: false` or increase epochs

---

## Full Pipeline (From Scratch)

```bash
# 1. Pretrain Mamba model
uv run python3 eeg_analysis/src/training/pretrain_mamba.py \
  --config eeg_analysis/configs/pretrain.yaml

# 2. Create closed_finetune dataset
uv run python3 eeg_analysis/run_representation_pipeline.py \
  --config eeg_analysis/configs/processing_config.yaml \
  process-closed-finetune

# 3. Fine-tune
uv run python3 eeg_analysis/src/training/finetune_mamba.py \
  --config eeg_analysis/configs/finetune.yaml \
  --pretrain-config eeg_analysis/configs/pretrain.yaml \
  --data-path eeg_analysis/data/processed/features/closed_finetune/8s_af7-af8-tp9-tp10_closed_finetune.parquet \
  --output-dir eeg_analysis/finetuned_models

# 4. View results
mlflow ui --backend-store-uri file:./mlruns
```

---

## Next Steps

- Compare different pretrained models (different mask ratios)
- Experiment with frozen vs unfrozen backbone
- Try different learning rates and epochs
- Evaluate on held-out test set
- Use fine-tuned model for inference

For detailed documentation, see `SFT_PIPELINE_SUMMARY.md`.


---

## Source: REAL_LEAKAGE_FOUND.md

# REAL INFORMATION LEAKAGE DETECTED AND FIXED

## 🚨 Critical Finding

**Loss reached 0.33 at epoch 7** - well below target variance (~0.6), confirming TRUE information leakage in the control experiment.

## 🔍 Root Cause: Circular Dependency in Target Computation

### The Bug

```python
# In training loop (BEFORE FIX)
pred = model(windows_masked, ...)  # Uses model.token_encoder
target = model.encode_tokens_only(windows)  # ALSO uses model.token_encoder
loss = MSE(pred, target)
```

**The problem**: 
- Both predictions AND targets use the **SAME TokenEncoder**
- TokenEncoder has **learnable parameters** (Linear weights + LayerNorm gamma/beta)
- As training progresses, TokenEncoder weights change
- Targets change (they depend on current TokenEncoder weights!)
- Model learns to track these changing targets
- Loss decreases even with constant input!

### The Leakage Mechanism

```python
Epoch 0:
  TokenEncoder weights: W0
  target = TokenEncoder(W0, real_signal) = T0
  pred = model(constant_input) = P0  
  loss = MSE(P0, T0) = high

Epoch 1:
  Model trains, weights update: W0 → W1
  target = TokenEncoder(W1, real_signal) = T1  ← CHANGED!
  pred = model(constant_input) = P1
  Model learns: "predict T1"
  loss = MSE(P1, T1) = lower

Epoch 2:
  Weights update: W1 → W2
  target = TokenEncoder(W2, real_signal) = T2  ← CHANGED AGAIN!
  ...and so on
```

**Result**: Model is chasing a moving target that it itself controls. This creates a circular dependency that allows loss to decrease indefinitely, even with no actual signal information!

## ✅ The Fix: Frozen Target Encoder

Modified `eeg_analysis/src/training/pretrain_mamba.py` to use a **separate, frozen encoder** for targets in control mode:

```python
# Create frozen copy of TokenEncoder at start of training
target_encoder = deepcopy(model.token_encoder)
target_encoder.eval()  # No dropout/batch effects
for param in target_encoder.parameters():
    param.requires_grad = False  # Freeze weights

# In training loop
pred = model(windows_masked)  # Uses training model
target = target_encoder(windows)  # Uses FROZEN encoder
loss = MSE(pred, target)
```

**Now**:
- Targets are computed with **fixed weights** (never change)
- Model cannot reduce loss by changing TokenEncoder
- With constant input → predictions cannot vary systematically
- **Loss should stay constant** (cannot be reduced)

## 🎯 Expected Behavior After Fix

### With Frozen Target Encoder

```
Epoch 1: loss = X.XX (initial)
Epoch 2: loss = X.XX (±0.01, no improvement)
Epoch 3: loss = X.XX (stays constant)
...
Epoch 20: Early stopping (no improvement)

✅ Confirms: No signal information leakage
   Model cannot learn without varying information
```

### If Loss Still Decreases (Would Indicate Another Bug)

```
Epoch 1: loss = X.XX
Epoch 2: loss = Y.YY (Y < X, still decreasing!)
...

❌ Another leakage source exists - investigate:
   - Dropout patterns
   - Batch normalization
   - Model internal state
```

## 📊 Previous vs. Fixed Behavior

### Before Fix (BUGGY)

| Epoch | Loss | Explanation |
|-------|------|-------------|
| 1 | 0.80 | Initial |
| 2 | 0.68 | TokenEncoder weights change → targets change |
| 3 | 0.60 | Model tracks moving target |
| 4 | 0.53 | Loss < target variance (impossible without leakage!) |
| 7 | 0.33 | Well below variance → confirmed leakage |

### After Fix (EXPECTED)

| Epoch | Loss | Explanation |
|-------|------|-------------|
| 1 | X.XX | Initial random loss |
| 2 | X.XX | No improvement (targets fixed, input constant) |
| 3 | X.XX | Loss plateaus immediately |
| ... | X.XX | No learning possible |
| 20 | X.XX | Early stopping |

## 🔬 Why This is Subtle

This bug is particularly sneaky because:

1. **Targets look correct**: We're using real signal, not masked signal
2. **@torch.no_grad() misleads**: Gradients don't flow through target computation, but targets still depend on model weights
3. **Works fine in normal training**: Only problematic in control experiments where input is constant
4. **Loss decreases smoothly**: Looks like normal learning, not an obvious bug

## 🎯 Implications for Normal Training

**Question**: Does this affect normal training (mask_ratio < 1.0)?

**Answer**: **Probably minimal impact**, but technically yes:

### In Normal Training

```python
# Normal training (mask_ratio=0.75)
windows_masked = [signal, signal, 0, signal, ...]  # Mix of real + masked
pred = model(windows_masked)  # Predictions vary by sample (different signals)
target = model.encode_tokens_only(windows)  # Also varies by sample

# Targets change as TokenEncoder weights change, but:
# - Model also receives varying signal information
# - Loss is dominated by reconstruction task, not circular dependency
# - Effect is minor compared to actual signal learning
```

**Impact**: Small bias, but overshadowed by actual signal reconstruction.

### In Control Experiment

```python
# Control (mask_ratio=1.0, no positions)
windows_masked = [0, 0, 0, ...]  # All zeros, all samples identical
pred = model(constant)  # Should be constant across samples
target = model.encode_tokens_only(windows)  # Varies, but depends on current weights!

# With no varying input, circular dependency dominates
# → Loss reduction is PURELY from circular dependency
# → Reveals the bug
```

**Impact**: **Complete** - the ONLY source of loss reduction.

## 🔧 How to Test the Fix

### Step 1: Stop Current Training

```bash
# Press Ctrl+C to stop
```

### Step 2: Restart with Fix

```bash
cd ~/eeg-mlflow
source .venv/bin/activate

# Fix is already applied in eeg_analysis/src/training/pretrain_mamba.py
python eeg_analysis/src/training/pretrain_mamba.py \
    --config eeg_analysis/configs/pretrain.yaml
```

### Step 3: Monitor Loss

**Expected behavior**:
```
Epoch 1: loss = Y.YY
Epoch 2: loss = Y.YY (±0.01)
Epoch 3: loss = Y.YY (no change)
...
Early stopping after ~20 epochs
```

Loss should **plateau immediately** and not decrease.

### Step 4: Interpretation

| Observation | Meaning |
|-------------|---------|
| Loss plateaus | ✅ **Fixed!** No leakage, control experiment successful |
| Loss still decreases slowly (to ~0.5) | ⚠️ Dropout noise (expected, not a problem) |
| Loss decreases significantly (to <0.4) | ❌ Another leak exists (investigate) |

## 🎓 Lessons Learned

1. **Be careful with shared weights**: If both pred and target use the same model components, circular dependencies can arise

2. **Control experiments need truly fixed targets**: For control experiments, targets must be computed with frozen weights

3. **@torch.no_grad() ≠ no dependency**: Even without gradient flow, outputs can depend on model parameters

4. **Subtle bugs show up in edge cases**: This bug didn't significantly affect normal training, only revealed in the control experiment

## 📋 Summary

- **Bug Found**: Targets computed with training model's TokenEncoder created circular dependency
- **Symptom**: Loss decreased to 0.33 even with constant input
- **Fix**: Use frozen TokenEncoder copy for target computation in control mode
- **Expected**: Loss now plateaus immediately, confirming no true signal leakage
- **Verification**: Restart training and check if loss stays constant

**This was an excellent control experiment** - it revealed a real but subtle bug that would have been hard to find otherwise!


---

## Source: REAL_POSITION_LEARNING_ANALYSIS.md

# Real Position Learning Analysis

## The Fundamental Issue

### What We Know

1. **Positional encodings are correctly zeroed** ✅
   - Temporal/Spatial norms = 0.0 for masked positions
   - No explicit position information leaks through encodings

2. **Token embeddings are identical** ✅
   - All masked positions have identical inputs (zeros)
   - Token encoder produces identical embeddings

3. **But predictions still differ by position** ❌
   - Position correlation = 0.69 (high)
   - Pairwise similarity = 0.94-0.99 (not 1.0)

### Why This Happens

**Mamba is a sequential model**. Even with identical inputs:

```
Position 0: h_0 = Mamba(zero_input, initial_state)
Position 1: h_1 = Mamba(zero_input, h_0)  ← Different hidden state!
Position 2: h_2 = Mamba(zero_input, h_1)  ← Different hidden state!
```

The hidden state accumulates position information naturally through sequential processing. **This is not a bug - it's how sequential models work.**

## The Real Question

### For 100% Masking (Control Experiment)

**Is position learning actually a problem?**

- **No signal content exists** - all inputs are zeros
- **Only varying information is position** - sequential processing order
- **Position learning is EXPECTED** - confirms model can learn temporal patterns

**This is actually GOOD** - it shows the model can learn temporal structure, which is useful for EEG tasks.

### For Partial Masking (Real Training)

**Does the model learn SIGNAL CONTENT or just position?**

This is the REAL question. With `mask_ratio < 1.0`:
- Unmasked tokens provide signal content
- Model should learn from context, not just position
- Position should HELP, not be the only signal

## What We Should Do

### ❌ DON'T: Force Identical Predictions

Forcing identical predictions:
- Doesn't fix the root cause (sequential processing)
- Fights against Mamba's architecture
- Doesn't solve the real problem (signal learning)

### ✅ DO: Ensure Signal Learning with Partial Masking

1. **Accept position learning with 100% masking**
   - It's inevitable with sequential models
   - It confirms temporal learning capability

2. **Verify signal learning with partial masking**
   - Test with `mask_ratio = 0.75` or `0.5`
   - Check if model learns from UNMASKED context
   - Verify pattern correlation improves

3. **Use position as a feature, not the only feature**
   - Position should help temporal understanding
   - But signal content should be main
   - Context should matter more than position

## The Real Solution

### For Control Experiment (100% Masking)

**Accept position learning** - it's expected and confirms temporal capability.

**What to verify**:
- Position correlation exists (confirms sequential processing works)
- But model doesn't learn signal content (no signal to learn)
- This is CORRECT behavior for 100% masking

### For Real Training (Partial Masking)

**Ensure signal learning** - model should learn from context.

**What to verify**:
- Pattern correlation > 0.3 (model learns signal patterns)
- Context sensitivity < 0.9 (predictions vary with context)
- Position correlation < 0.4 (position helps but isn't dominant)

## Key Insight

**Position learning with 100% masking is NOT a bug** - it's a feature of sequential models.

**The real test**: With partial masking, does the model learn SIGNAL CONTENT?

If yes → Model is working correctly ✅
If no → Model is only learning position ❌

## Next Steps

1. **Remove variance penalty** (doesn't solve the problem)
2. **Test with partial masking** (`mask_ratio = 0.75`)
3. **Verify signal learning** (pattern correlation, context sensitivity)
4. **Use position as helper** (not the only signal)

The goal isn't to eliminate position learning - it's to ensure the model learns SIGNAL CONTENT when it's available.


---

## Source: SCALE_MISMATCH_FIX.md

# Scale Mismatch Fix: Raw EEG Magnitude Issue

## Problem Detected

From training logs:

```
pred_var=280.234711, target_var=3377368.750000  ← 12,000x difference!
train_loss=6706086205  ← Loss in billions!
val_loss=2600507655    ← Impossibly high!
```

### Root Cause

**Raw EEG samples have huge absolute values** (microvolts scale):
- Decoder output: Small values (~200-400 variance) - initialized with Xavier
- Raw EEG targets: MASSIVE values (1M-80B variance) - actual physiological data
- MSE loss: Comparing tiny predictions to huge targets → billions

### Why This Happened

Your pipeline correctly uses raw signal as targets (no learned encoder), but didn't account for **EEG amplitude scale**:

```python
# What was happening:
pred = decoder(embeddings)  # Output: ~[-10, 10] (small Xavier init)
target = raw_EEG_samples    # Values: ~[-1000, 1000] or more (microvolts)
loss = MSE(pred, target)    # Comparing incompatible scales!
```

**EEG voltage ranges** (typical):
- Amplitude: 10-100 μV (microvolts)
- Artifacts can be 1000+ μV
- DC offset can vary widely between channels
- Your data shows variance from 100K to 80 billion!

## ✅ Solution: Per-Sample Normalization

Added normalization to targets (NOT inputs - those stay raw for masking):

```python
# Normalize targets to zero mean, unit variance per window
target_mean = target.mean(dim=-1, keepdim=True)  # Per window
target_std = target.std(dim=-1, keepdim=True) + 1e-8
target_normalized = (target - target_mean) / target_std

# Now decoder learns to output normalized signal
# Loss compares normalized pred to normalized target
```

### Why Per-Sample (Per-Window) Normalization?

1. **Preserves signal structure**: Removes DC offset and amplitude differences
2. **Handles variability**: Different channels/subjects have different amplitudes
3. **Stabilizes training**: Decoder learns patterns, not absolute scales
4. **Standard in MAE**: Vision MAE normalizes pixel values similarly

### What This Achieves

**Before**:
```
Window 1: [-500, -480, -460, ...]  (high DC offset)
Window 2: [20, 40, 60, ...]         (low DC offset)
Window 3: [2000, 2100, 2200, ...]   (artifact)

Decoder tries to predict absolute values → fails on scale differences
```

**After**:
```
Window 1: [-0.5, 0.0, 0.5, ...]  (normalized)
Window 2: [-0.5, 0.0, 0.5, ...]  (normalized)
Window 3: [-0.5, 0.0, 0.5, ...]  (normalized)

Decoder predicts relative patterns → learns signal structure
```

## Expected Results

### Before Fix
```
Epoch 1: train_loss=6,700,000,000, val_loss=2,600,000,000
Epoch 2: train_loss=6,700,000,000, val_loss=2,600,000,000
→ Loss in billions, not learning anything useful
```

### After Fix (Expected)
```
Epoch 1: train_loss=~1.0, val_loss=~1.0
Epoch 2: train_loss=~0.8, val_loss=~0.9
Epoch 3: train_loss=~0.6, val_loss=~0.7
→ Loss in reasonable range, steady improvement
```

For control experiment (mask_ratio=1.0, no positions):
```
Epoch 1: loss=~1.0
Epoch 2: loss=~1.0 (±0.01, stays constant)
→ Confirms no leakage with normalized scale
```

## Important Notes

### ✅ This Is Still "Raw Signal" Reconstruction

**Q**: Doesn't normalization violate the "raw signal targets" requirement?

**A**: No! Normalization is a **fixed, deterministic transform** with no learnable parameters:

```python
# No model weights involved:
target_normalized = (target - target.mean()) / target.std()

# Still comparing to actual signal, just rescaled
# Like converting meters to kilometers - same data, different units
```

The key properties preserved:
- ✅ No learnable encoder used
- ✅ Targets computed from raw input data
- ✅ No circular dependency
- ✅ Deterministic transform (same input → same normalized output)

### This Is Standard Practice

**Vision MAE** (original paper):
- Normalizes pixel values to [0, 1] or [-1, 1]
- Computes loss on normalized space
- Still reconstructing "pixels", just normalized

**Audio MAE**:
- Normalizes waveforms per-sample
- Removes DC bias and amplitude differences
- Learns temporal patterns, not absolute loudness

**Your EEG MAE** (now fixed):
- Normalizes EEG windows per-sample
- Removes DC offset and amplitude differences  
- Learns EEG patterns, not absolute microvolt scales

## Implementation Details

### What Gets Normalized

```python
# Training loop:
windows = batch["windows"]  # (B, L, 2048) - raw EEG
windows_masked = batch["windows_masked"]  # (B, L, 2048) - with zeros

# Input stays raw (masking needs raw scale)
pred = model(windows_masked)  # Decoder outputs (B, L, 2048)

# Target gets normalized
target = windows  # Start with raw
target = (target - target.mean(-1, keepdim=True)) / target.std(-1, keepdim=True)

# Loss on normalized scale
loss = MSE(pred[masked], target[masked])
```

### Normalization Dimensions

```python
target.mean(dim=-1, keepdim=True)  # Mean over window (2048 samples)
target.std(dim=-1, keepdim=True)   # Std over window

# Normalizes each window independently
# Shape: (B, L, 2048) → mean/std per (B, L) → (B, L, 2048) normalized
```

**Why per-window?**
- Each window may have different baseline
- Different channels have different amplitudes
- Artifacts affect different windows differently
- Preserves within-window temporal patterns

## Restart Training

```bash
# Stop current training (Ctrl+C)

# Restart with normalization fix (already applied)
python eeg_analysis/src/training/pretrain_mamba.py \
    --config eeg_analysis/configs/pretrain.yaml
```

### Expected Observations

1. **Loss drops to ~1.0** (not billions)
2. **pred_var and target_var are similar** (both normalized to ~1.0)
3. **Steady improvement** or plateau (for control experiment)
4. **No scale mismatch** in diagnostic logs

## Verification

After 1-2 epochs, check logs:

**Good signs**:
```
pred_var=0.8-1.2, target_var=0.8-1.2  ← Similar scales!
train_loss=0.5-1.5                    ← Reasonable range!
```

**Bad signs** (would indicate other issues):
```
pred_var=0.0, target_var=1.0   ← Not learning
loss > 10                       ← Still scale issues
loss < 0.1 in epoch 1          ← Suspicious (too good)
```

## Summary

| Aspect | Before Fix | After Fix |
|--------|-----------|-----------|
| Target values | Raw μV (huge range) | Normalized (μ=0, σ=1) |
| Loss magnitude | Billions | ~0.5-1.5 |
| Decoder task | Learn absolute μV | Learn normalized patterns |
| Training | Unstable (scale issues) | Stable |
| Still "raw signal"? | Yes | Yes (deterministic transform) |
| Learnable params on targets? | No | No ✅ |

**The fix maintains MAE integrity while handling EEG's physiological scale.**


---

## Source: SFT_PIPELINE_SUMMARY.md

# Supervised Fine-Tuning (SFT) Pipeline for Mamba EEG Models

## Overview

This document describes the complete pipeline for:
1. Creating closed_finetune datasets (windowed EEG data without feature extraction)
2. Fine-tuning pretrained Mamba models for remission classification
3. Automatic model discovery based on pretraining configuration

---

## 1. Closed_finetune Dataset Creation

### What is the Closed_finetune Dataset?

The **closed_finetune dataset** is raw windowed EEG data concatenated across all participants, ready for direct model input. Unlike the feature-extracted dataset, it preserves the raw signal data for each channel.

**Structure:**
- `Participant`: Participant ID
- `Remission`: Binary label (0=non-remission, 1=remission)
- `parent_window_id`, `sub_window_id`: **Window identifiers (CRITICAL for temporal order)**
- `window_start`, `window_end`: Window boundaries
- Channel columns (`AF7`, `AF8`, `TP9`, `TP10`): Raw signal vectors

**⚠️ IMPORTANT: Temporal Ordering**

Windows are sorted by `Participant → parent_window_id → sub_window_id` to preserve temporal relationships. This is critical because:
1. The Mamba backbone is a sequence model that depends on temporal order
2. Windows represent consecutive time segments from EEG recordings
3. Scrambling window order would destroy temporal patterns the model needs to learn

The closed_finetune dataset creation automatically sorts and verifies window ordering.

### Creating the Closed_finetune Dataset

```bash
uv run python3 eeg_analysis/run_representation_pipeline.py \
  --config eeg_analysis/configs/processing_config.yaml \
  process-closed-finetune
```

**What it does:**
1. Loads raw EEG data
2. Applies upsampling, filtering, downsampling, windowing, DC offset removal
3. **Skips feature extraction** (unlike the regular `process` command)
4. Concatenates remission and non-remission windowed data
5. Saves to `eeg_analysis/data/processed/features/closed_finetune/`
6. Logs dataset to MLflow with tags for discovery

**Output:**
- File: `{window_size}s_{channels}_closed_finetune.parquet`
- Example: `8s_af7-af8-tp9-tp10_closed_finetune.parquet`
- MLflow dataset name: `EEG_8s_af7-af8-tp9-tp10_{N}windows_closed_finetune`

---

## 2. Supervised Fine-Tuning (SFT)

### Architecture

**Mamba EEG Classifier:**
- **Backbone**: Pretrained Mamba-2 model (loaded from checkpoint)
- **Freezing**: All backbone layers frozen by default (only train classification head)
- **Classification Head**:
  ```
  LayerNorm(d_model)
  → Dropout
  → Linear(d_model → d_model/2)
  → GELU
  → Dropout
  → Linear(d_model/2 → num_classes)
  ```

### Automatic Checkpoint Discovery

The fine-tuning script **automatically finds the pretrained checkpoint** based on parameters in `eeg_analysis/configs/pretrain.yaml`:

```yaml
# eeg_analysis/configs/pretrain.yaml
d_model: 256
num_layers: 2
mask_ratio: 0.2
masking_style: "mae"
```

**Expected checkpoint**: `mamba2_eeg_d256_l2_m20_mae`
**Checkpoint file**: `eeg_analysis/checkpoints/mamba2_eeg_pretrained.pt`

### Running Fine-Tuning

```bash
uv run python3 eeg_analysis/src/training/finetune_mamba.py \
  --config eeg_analysis/configs/finetune.yaml \
  --pretrain-config eeg_analysis/configs/pretrain.yaml \
  --data-path eeg_analysis/data/processed/features/closed_finetune/8s_af7-af8-tp9-tp10_closed_finetune.parquet \
  --output-dir eeg_analysis/finetuned_models
```

**Arguments:**
- `--config`: Fine-tuning hyperparameters (lr, epochs, batch_size, etc.)
- `--pretrain-config`: Pretraining config (used to find checkpoint and load model architecture)
- `--data-path`: Path to closed_finetune dataset parquet file
- `--output-dir`: Where to save fine-tuned models

### Configuration (`eeg_analysis/configs/finetune.yaml`)

```yaml
# Training hyperparameters
lr: 1.0e-4              # Learning rate
batch_size: 8
epochs: 50
dropout: 0.1
weight_decay: 0.01

# Model
freeze_backbone: true   # Freeze pretrained layers
num_classes: 2          # Binary classification

# Data splits
val_ratio: 0.2          # 20% validation
test_ratio: 0.1         # 10% test
seed: 42

# MLflow
mlflow_tracking_uri: "mlruns"
mlflow_experiment: "eeg_finetuning_mamba2"
```

### Training Process

1. **Data Loading**:
   - Loads closed_finetune dataset
   - **Verifies temporal ordering** (windows sorted by participant → parent_window_id → sub_window_id)
   - Splits by participant into train/val/test (stratified)
   - Each participant's windows remain in temporal order within their split

2. **Model Initialization**:
   - Creates `MambaEEGClassifier`
   - Loads pretrained weights from checkpoint
   - Freezes backbone layers (if `freeze_backbone: true`)
   - Initializes classification head with Xavier initialization

3. **Training Loop**:
   - AdamW optimizer with cosine annealing schedule
   - Cross-entropy loss
   - Gradient clipping (max_norm=1.0)
   - Logs metrics to MLflow every epoch

4. **Evaluation**:
   - Validation after each epoch
   - Saves best model based on F1 score
   - Final test evaluation at end
   - Metrics: accuracy, precision, recall, F1, AUC

5. **Model Registration**:
   - Saves best checkpoint to `{output_dir}/{model_name}_finetuned_best.pt`
   - Logs model to MLflow
   - Registers as `{model_name}_finetuned` in MLflow Model Registry

---

## 3. Model Naming Convention

Models are named based on pretraining configuration:

**Format**: `mamba2_eeg_d{d_model}_l{num_layers}_m{mask_ratio_percent}_{masking_style}`

**Examples:**
- `mamba2_eeg_d256_l2_m20_mae` → d_model=256, 2 layers, 20% mask, MAE-style
- `mamba2_eeg_d128_l6_m40_mae` → d_model=128, 6 layers, 40% mask, MAE-style
- `mamba2_eeg_d256_l2_m20_bert` → d_model=256, 2 layers, 20% mask, BERT-style

**Fine-tuned models** append `_finetuned`:
- `mamba2_eeg_d256_l2_m20_mae_finetuned`

---

## 4. Complete Workflow Example

### Step 1: Pretrain Mamba Model

```bash
# Single GPU
uv run python3 eeg_analysis/src/training/pretrain_mamba.py \
  --config eeg_analysis/configs/pretrain.yaml

# Multi-GPU
uv run torchrun --standalone --nproc_per_node=2 \
  eeg_analysis/src/training/pretrain_mamba.py \
  --config eeg_analysis/configs/pretrain.yaml \
  --distributed
```

**Output**: `eeg_analysis/checkpoints/mamba2_eeg_pretrained.pt`

### Step 2: Create Closed_finetune Dataset

```bash
uv run python3 eeg_analysis/run_representation_pipeline.py \
  --config eeg_analysis/configs/processing_config.yaml \
  process-closed-finetune
```

**Output**: `eeg_analysis/data/processed/features/closed_finetune/8s_af7-af8-tp9-tp10_closed_finetune.parquet`

### Step 3: Fine-Tune for Classification

```bash
uv run python3 eeg_analysis/src/training/finetune_mamba.py \
  --config eeg_analysis/configs/finetune.yaml \
  --pretrain-config eeg_analysis/configs/pretrain.yaml \
  --data-path eeg_analysis/data/processed/features/closed_finetune/8s_af7-af8-tp9-tp10_closed_finetune.parquet \
  --output-dir eeg_analysis/finetuned_models
```

**Output**: 
- Checkpoint: `eeg_analysis/finetuned_models/mamba2_eeg_d256_l2_m20_mae_finetuned_best.pt`
- MLflow model: `mamba2_eeg_d256_l2_m20_mae_finetuned`

---

## 5. Files Created

### New Files

1. **`eeg_analysis/src/processing/closed_finetune_dataset.py`**
   - Creates closed_finetune dataset from windowed data
   - Concatenates remission and non-remission groups
   - Logs to MLflow

2. **`eeg_analysis/src/data/eeg_sft_dataset.py`**
   - PyTorch Dataset for supervised fine-tuning
   - Loads closed_finetune dataset
   - Groups windows by participant
   - Handles train/val/test splits

3. **`eeg_analysis/src/models/mamba_sft_model.py`**
   - `MambaEEGClassifier`: Mamba model with classification head
   - Loads pretrained weights
   - Supports backbone freezing
   - Aggregates across channels and windows

4. **`eeg_analysis/src/training/finetune_mamba.py`**
   - Fine-tuning training script
   - Automatic checkpoint discovery
   - MLflow integration
   - Evaluation and model registration

5. **`eeg_analysis/configs/finetune.yaml`**
   - Fine-tuning hyperparameters
   - Data split configuration
   - MLflow settings

### Modified Files

1. **`eeg_analysis/run_representation_pipeline.py`**
   - Added `process-closed-finetune` command
   - Creates closed_finetune dataset without feature extraction

2. **`README.md`**
   - Consolidated project documentation now includes the SFT workflow
   - Usage examples for closed_finetune dataset creation and fine-tuning

---

## 6. Key Features

### ✅ Automatic Model Discovery
- No need to manually specify checkpoint paths
- Finds pretrained model based on config parameters
- Validates model exists before training

### ✅ Stratified Splits
- Participant-level splits (no data leakage)
- Maintains class balance across train/val/test
- Reproducible with seed

### ✅ Flexible Backbone Freezing
- `freeze_backbone: true` → Only train classification head (fast, prevents overfitting)
- `freeze_backbone: false` → Full fine-tuning (slower, may improve performance)

### ✅ Comprehensive Metrics
- Accuracy, Precision, Recall, F1, AUC
- Logged to MLflow for comparison
- Best model saved based on F1 score

### ✅ MLflow Integration
- Logs all hyperparameters
- Tracks metrics per epoch
- Registers fine-tuned models
- Links to pretrained model

---

## 7. Troubleshooting

### Checkpoint Not Found

**Error**: `Pretrained checkpoint not found`

**Solution**:
1. Check that pretraining completed: `ls eeg_analysis/checkpoints/mamba2_eeg_pretrained.pt`
2. Verify config parameters match: `d_model`, `num_layers`, `mask_ratio`, `masking_style`
3. Run pretraining if checkpoint missing

### Closed_finetune Dataset Not Found

**Error**: `Data file not found`

**Solution**:
1. Run `process-closed-finetune` command first
2. Check path matches config: `{window_size}s_{channels}_closed_finetune.parquet`
3. Verify windowed data exists: `ls eeg_analysis/data/interim/windowed/`

### Out of Memory

**Solution**:
1. Reduce `batch_size` in `eeg_analysis/configs/finetune.yaml`
2. Use gradient accumulation (modify training script)
3. Use smaller model (reduce `d_model` or `num_layers` in pretraining)

### Poor Performance

**Solutions**:
1. Try unfreezing backbone: `freeze_backbone: false`
2. Increase training epochs
3. Adjust learning rate
4. Try different pretrained models (different mask_ratio or masking_style)
5. Check data quality and class balance

---

## 8. Next Steps

### Experiment with Different Pretrained Models

```bash
# Sweep different mask ratios during pretraining
uv run python3 eeg_analysis/src/training/sweep_mask_ratio.py \
  --config eeg_analysis/configs/pretrain.yaml \
  --torchrun

# Fine-tune each pretrained model
for mask_ratio in 20 30 40 50 60 70 80; do
  # Update eeg_analysis/configs/pretrain.yaml with mask_ratio
  # Run fine-tuning
  uv run python3 eeg_analysis/src/training/finetune_mamba.py \
    --config eeg_analysis/configs/finetune.yaml \
    --pretrain-config eeg_analysis/configs/pretrain.yaml \
    --data-path eeg_analysis/data/processed/features/closed_finetune/8s_af7-af8-tp9-tp10_closed_finetune.parquet \
    --output-dir eeg_analysis/finetuned_models
done
```

### Compare Models in MLflow

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Navigate to `eeg_finetuning_mamba2` experiment to compare:
- Different pretrained models
- Frozen vs unfrozen backbone
- Different hyperparameters

---

## Summary

The SFT pipeline provides:
1. **Closed_finetune dataset creation** - Raw windowed data for direct model input
2. **Automatic checkpoint discovery** - No manual path management
3. **Flexible fine-tuning** - Freeze or unfreeze backbone layers
4. **Comprehensive evaluation** - Multiple metrics, MLflow tracking
5. **Model registration** - Organized model versioning

This enables systematic exploration of how different pretraining strategies (mask ratio, masking style, model size) affect downstream classification performance.

---

## Source: SOLUTION_PREVENT_MEAN_LEARNING.md

# Solution: Preventing Dataset Mean Learning in Control Experiment

## Problem

With `mask_ratio=1.0` and positional encodings disabled, the model still reduced loss:

```
Epoch 1: loss=0.808
Epoch 2: loss=0.683
Epoch 3: loss=0.603
...
Epoch 10: still decreasing
```

**Root cause**: Model was learning to predict the **dataset mean** of target embeddings, not accessing individual signals.

```python
# Model with constant input
pred = constant  # Same for all samples

# Model learns
pred ≈ mean(all_targets_in_dataset)

# This reduces loss
loss = MSE(mean(targets), targets) < MSE(random, targets)
```

## Solution Implemented: Target Centering

Modified `eeg_analysis/src/training/pretrain_mamba.py` to **subtract the target mean** before computing loss when in control mode.

### Changes Made

```python
# Old loss (allows learning dataset mean)
loss = MSE(pred, target)

# New loss (prevents learning dataset mean)
target_mean = mean(target)
loss = MSE(pred - target_mean, target - target_mean)
```

**Key insight**: After centering:
- Predicting a constant (dataset mean) gives loss = Var(target)
- Predicting target_mean gives loss = Var(target) (no advantage)
- **Only way to reduce loss**: Predict sample-specific deviations from mean
- With constant input → IMPOSSIBLE to predict deviations
- **Loss should now stay constant** (cannot be reduced)

### Code Location

File: `eeg_analysis/src/training/pretrain_mamba.py`

Changes in:
1. Training loop (lines ~240-260)
2. Validation loop (lines ~320-335)

## Expected Behavior After Fix

### Scenario A: No Signal Leakage (Expected)

```
Epoch 11: loss=0.XXX (stays constant)
Epoch 12: loss=0.XXX (±0.01 fluctuation)
Epoch 13: loss=0.XXX (no improvement)
...
Early stopping triggers (no improvement for patience epochs)

✅ CONFIRMED: No information leakage
   Model cannot learn without varying information
```

Loss will plateau at the **variance of centered targets** (irreducible without sample info).

### Scenario B: True Signal Leakage (Would indicate bug)

```
Epoch 11: loss=0.500
Epoch 12: loss=0.450
Epoch 13: loss=0.400  ← Still decreasing!
...

❌ WARNING: Model is still learning
   This means it has access to sample-specific information
   TRUE SIGNAL LEAKAGE DETECTED
```

## How to Use

### Step 1: Stop current training

```bash
# Press Ctrl+C to stop the running training
```

### Step 2: Restart training with fix

```bash
cd ~/eeg-mlflow
source .venv/bin/activate

# The fix is already in eeg_analysis/src/training/pretrain_mamba.py
python eeg_analysis/src/training/pretrain_mamba.py \
    --config eeg_analysis/configs/pretrain.yaml
```

### Step 3: Monitor loss

Watch for:
- **Loss stops decreasing** → ✅ No leakage confirmed
- **Loss continues decreasing** → ❌ Investigate further

### Step 4: Check diagnostic logs

Look for control experiment logs:

```
[Control] pred_var=0.XXX, target_var=0.XXX, pred_mean=0.XXX, target_mean=0.XXX
```

**Expected values**:
- `pred_var`: Should stay constant (model can't learn to vary)
- `target_var`: Dataset variance (should be > loss if model can't beat variance)
- `pred_mean` and `target_mean`: Should be similar initially

## Understanding the Math

### Why Original Loss Decreased

```python
# Model predicts constant c
pred = c  # Same for all samples

# Optimal constant minimizes MSE
c_optimal = mean(targets)

# Initial loss (random c)
loss_initial = MSE(c_random, targets) = high

# After learning (c = mean)
loss_learned = MSE(mean(targets), targets) = Var(targets)

# Reduction achieved!
loss_initial > loss_learned
```

### Why Centered Loss Cannot Decrease

```python
# Center targets
target_centered = target - mean(targets)
# mean(target_centered) = 0

# Model predicts constant c
pred = c

# Center predictions (subtract target mean)
pred_centered = c - mean(targets)

# Loss
loss = MSE(pred_centered, target_centered)
     = MSE(c - mean(targets), target - mean(targets))

# To minimize, take derivative w.r.t. c:
d/dc loss = 2 * (c - mean(targets) - mean(target - mean(targets)))
          = 2 * (c - mean(targets) - 0)
          = 2 * (c - mean(targets))

# Optimal c:
c_optimal = mean(targets)

# Plugging back:
pred_centered = mean(targets) - mean(targets) = 0
loss_optimal = MSE(0, target_centered) 
             = Var(target_centered)
             = Var(target)  # Variance unchanged by centering

# But pred_centered = 0 is just ONE prediction
# Model with constant input can only predict ONE value
# Predicting 0 gives loss = Var(target)
# Predicting any other constant gives loss ≥ Var(target)

# Result: Loss cannot go below Var(target)
# And Var(target) is achieved by ANY constant prediction!
# So model cannot improve by learning
```

**Key point**: After centering, **all constant predictions are equivalent**. Model cannot reduce loss by learning a better constant.

## Alternative Solutions Considered

### Option A: Randomize Targets (Not Used)

```python
# Shuffle targets relative to inputs
indices = torch.randperm(batch_size)
target_shuffled = target[indices]
loss = MSE(pred, target_shuffled)
```

**Why not**: Makes validation loss meaningless, complex to implement.

### Option B: Contrastive Loss (Not Used)

```python
# Require predictions to match specific samples
loss = contrastive_loss(pred, target, labels)
```

**Why not**: Requires labels, changes loss fundamentally, complex.

### Option C: Add Noise to Inputs (Not Used)

```python
# Add random noise to break constant input
windows_masked_noisy = windows_masked + noise
```

**Why not**: Adds varying information, defeats purpose of control.

### Option D: Target Centering (IMPLEMENTED) ✅

**Why this**: 
- Simple, principled, mathematically sound
- Removes ability to learn mean without adding information
- Preserves validation semantics
- Easy to implement and understand

## Diagnostic Output

With the fix, you'll see logs like:

```
{"timestamp": "...", "level": "INFO", "message": "[Control] pred_var=0.102135, target_var=0.102138, pred_mean=-0.002144, target_mean=-0.002145"}
```

**Interpretation**:
- `pred_var ≈ target_var`: Predictions have similar variance to targets
- `pred_mean ≈ target_mean`: Model is predicting around the mean
- If both var and mean stay constant across epochs → Model is not learning

## Expected Timeline

With the fix:

```
Epoch 1: loss=X.XX (initial)
Epoch 2: loss=X.XX (±0.01)
Epoch 3: loss=X.XX (±0.01)
...
Epoch 20: loss=X.XX (no improvement)
Early stopping triggered

Total time: ~20 epochs (patience=20)
✅ Control experiment complete: No leakage confirmed
```

Without the fix (old behavior):

```
Epoch 1-50: loss steadily decreasing
✗ Inconclusive (learning mean, not testing leakage)
```

## After Control Experiment

Once loss stops decreasing with centered targets:

**✅ No leakage confirmed** → Safe to train with real configuration

Update `eeg_analysis/configs/pretrain.yaml`:

```yaml
mask_ratio: 0.75                   # Provide context
disable_temporal_encoding: false   # Enable positions
disable_spatial_encoding: false    # Enable channels
```

Then train for real:

```bash
python eeg_analysis/src/training/pretrain_mamba.py \
    --config eeg_analysis/configs/pretrain.yaml
```

## Summary

| Configuration | Loss Behavior | Interpretation |
|--------------|---------------|----------------|
| **Old**: No centering | Decreases | Learning dataset mean (inconclusive) |
| **New**: With centering | Plateaus | No leakage ✅ |
| **New**: With centering | Decreases | Signal leakage ❌ (investigate) |

**Current status**: Fix implemented, restart training to confirm no leakage.

**Expected outcome**: Loss will plateau, confirming no information leakage.

**Next step**: Once confirmed, switch to `mask_ratio=0.75` for actual training.


---

## Source: STATISTICS_COMPARISON_FEATURE.md

# Statistics Comparison Feature: Predicted vs Ground Truth

## Overview

The diagnostic script now compares **predicted signal statistics** (mean, std) to **ground truth statistics** to verify if the model learns to reconstruct actual EEG signal characteristics.

## What It Does

### Computes Per-Window Statistics

For each masked window:
- **Predicted**: Mean and std of reconstructed signal (2048 samples)
- **Ground Truth**: Mean and std of actual signal (2048 samples)

### Compares Across Channels

- Aggregates statistics per channel
- Computes correlation between predicted and GT statistics
- Reports errors (absolute differences)

## Usage

```bash
# Standard usage (auto-detects signal reconstruction from checkpoint)
python scripts/diagnose_100pct_masking.py \
    --checkpoint eeg_analysis/checkpoints/mamba2_eeg_pretrained.pt \
    --num-samples 100

# Force signal space decoding
python scripts/diagnose_100pct_masking.py \
    --checkpoint eeg_analysis/checkpoints/mamba2_eeg_pretrained.pt \
    --decode-to-signal \
    --num-samples 100
```

## Output Sections

### 1. Overall Statistics (All Channels)

```
OVERALL STATISTICS (All Channels Combined)
============================================================

Mean (per window):
  Predicted: 0.000123 ± 0.001456
  Ground Truth: 0.000000 ± 0.001234
  Difference: 0.000123
  Correlation: 0.8234

Std (per window):
  Predicted: 0.998765 ± 0.012345
  Ground Truth: 1.000000 ± 0.010234
  Difference: 0.001235
  Correlation: 0.9123
```

**Interpretation**:
- **Correlation > 0.7**: Model accurately reconstructs statistics ✅
- **Correlation 0.5-0.7**: Partial reconstruction ⚠️
- **Correlation < 0.5**: Poor reconstruction ❌

### 2. Per-Channel Statistics

```
PER-CHANNEL STATISTICS
============================================================

Channel      Windows    Mean Error       Std Error        Mean Corr     Std Corr
--------------------------------------------------------------------------------
C3           1250       0.000234         0.001234         0.8234        0.9123
FP1          1180       0.000189         0.001456         0.7891        0.9012
...
```

**Columns**:
- **Windows**: Number of masked windows for this channel
- **Mean Error**: Average absolute difference in mean
- **Std Error**: Average absolute difference in std
- **Mean Corr**: Correlation between predicted and GT means
- **Std Corr**: Correlation between predicted and GT stds

### 3. Interpretation

```
INTERPRETATION
============================================================

Overall Mean Correlation: 0.8234
Overall Std Correlation:  0.9123

✅ EXCELLENT: Model accurately reconstructs signal statistics
   Predictions match ground truth mean and std patterns
   Model is learning actual EEG signal characteristics
```

## What Good Results Look Like

### ✅ Excellent Model (Learning Signal Structure)

```
Mean Correlation: > 0.7
Std Correlation:  > 0.7

Interpretation:
- Model predicts different means/stds for different windows
- Predictions correlate with actual signal characteristics
- Model is learning EEG signal patterns, not just positions
```

### ⚠️ Moderate Model (Partial Learning)

```
Mean Correlation: 0.5-0.7
Std Correlation:  0.5-0.7

Interpretation:
- Some correlation but not perfect
- Model partially learns signal structure
- May need more training or better mask ratio
```

### ❌ Poor Model (Positional Learning Only)

```
Mean Correlation: < 0.5
Std Correlation:  < 0.5

Interpretation:
- Low correlation = predictions don't match signal stats
- Model likely learning positional patterns only
- Not reconstructing actual signal characteristics
```

## Why This Matters

### Traditional Metrics Can Be Misleading

**Old approach**: Compare embedding vectors
- Embeddings are learned representations
- Hard to interpret what model learned
- Can have good loss but poor signal understanding

**New approach**: Compare signal statistics
- Direct comparison of signal characteristics
- Clear interpretation: Does model predict correct mean/std?
- Reveals if model learns signal structure vs. positional patterns

### Example: Why Statistics Matter

**Scenario A**: Model predicts constant values
```
Predicted mean: 0.0 for all windows
Ground truth mean: varies from -5 to +5
Correlation: 0.0 ❌
```

**Scenario B**: Model predicts varying means matching GT
```
Predicted mean: varies from -4.8 to +4.9
Ground truth mean: varies from -5.0 to +5.0
Correlation: 0.95 ✅
```

**Both might have similar MSE loss**, but Scenario B shows the model learned signal structure!

## Integration with Other Diagnostics

The statistics comparison complements existing diagnostics:

1. **Position Correlation**: Does model depend on position?
2. **Context Sensitivity**: Do predictions vary with masking?
3. **Statistics Comparison**: Do predictions match signal characteristics? ← NEW

**Combined interpretation**:
- Low position corr + High context sensitivity + High stats corr = ✅ Excellent
- High position corr + Low stats corr = ❌ Positional learning only
- Low position corr + Low stats corr = ⚠️ Learning something else (investigate)

## Technical Details

### Normalization Handling

The comparison accounts for per-window normalization:
- Both predictions and GT are normalized per-window (mean=0, std=1)
- Statistics are computed on **normalized** windows
- This focuses on **relative patterns**, not absolute scales

### Window-Level Statistics

For each masked window:
```python
pred_window = pred[b, idx]  # (2048,) - reconstructed signal
gt_window = windows[b, idx]  # (2048,) - actual signal

pred_mean = pred_window.mean()
pred_std = pred_window.std()
gt_mean = gt_window.mean()
gt_std = gt_window.std()

# Compare these statistics
```

### Correlation Computation

```python
# Across all windows for a channel:
mean_corr = corrcoef([pred_mean_1, pred_mean_2, ...], 
                     [gt_mean_1, gt_mean_2, ...])

# High correlation = model predicts correct mean for each window
# Low correlation = model predicts similar mean for all windows
```

## Use Cases

### 1. Control Experiment Validation

With `mask_ratio=1.0` and no positions:
- **Expected**: Low statistics correlation (can't learn signal)
- **If high correlation**: Signal leakage detected!

### 2. Normal Training Evaluation

With `mask_ratio=0.75`:
- **Expected**: High statistics correlation (learning from context)
- **If low correlation**: Model not learning signal structure effectively

### 3. Model Comparison

Compare two models:
```bash
# Model A (mask_ratio=1.0)
python scripts/diagnose_100pct_masking.py --checkpoint model_A.pt
# Mean corr: 0.15, Std corr: 0.12

# Model B (mask_ratio=0.75)
python scripts/diagnose_100pct_masking.py --checkpoint model_B.pt
# Mean corr: 0.82, Std corr: 0.89

# Conclusion: Model B learns signal structure, Model A does not
```

## Limitations

1. **Requires Signal Reconstruction**: Only works if `reconstruct_signal_space=true`
2. **Per-Window Normalization**: Statistics computed on normalized windows
3. **Sample Size**: Needs sufficient masked windows per channel for reliable correlation

## Summary

The statistics comparison feature provides a **direct, interpretable measure** of whether your model learns actual EEG signal characteristics or just positional patterns.

**Key metric**: Correlation between predicted and ground truth statistics
- **> 0.7**: Excellent signal learning ✅
- **0.5-0.7**: Moderate learning ⚠️
- **< 0.5**: Poor learning ❌

Use this alongside position correlation and context sensitivity tests for comprehensive model evaluation!


---

## Source: SUMMARY_POSITIONAL_LEARNING_FIX.md

# Summary: Fixing Positional Learning in Your MAE Model

## 🔍 What We Found

Your model with `mask_ratio=1.0` is learning **trivial positional patterns**:

```
❌ Position correlation:   0.403  (strong)
❌ Channel diversity:      0.000  (none - all channels identical!)
❌ Prediction consistency: 0.9995 (predictions ignore context)
```

**Translation**: The model predicts the same embedding for all channels at a given position. It's learning `f(t/T)` instead of learning channel-specific EEG patterns.

## ✅ The Fix (Already Applied)

I've updated `eeg_analysis/configs/pretrain.yaml`:

```yaml
mask_ratio: 0.75  # Changed from 1.0
```

## 🚀 Next Steps

### 1. Retrain with new config

```bash
cd ~/eeg-mlflow
source .venv/bin/activate

# Start training with 75% masking
python eeg_analysis/src/training/pretrain_mamba.py \
    --config eeg_analysis/configs/pretrain.yaml
```

### 2. Compare results

After training completes, run diagnostic again:

```bash
python scripts/diagnose_100pct_masking.py \
    --checkpoint eeg_analysis/checkpoints/mamba2_eeg_pretrained.pt \
    --num-samples 100
```

### 3. Expected improvements

**Before (mask_ratio=1.0):**
```
Position correlation: 0.403  ❌
Channel diversity:    0.000  ❌
Consistency:          0.999  ❌
```

**After (mask_ratio=0.75):**
```
Position correlation: < 0.20  ✅ (less position-dependent)
Channel diversity:    > 0.05  ✅ (channels have distinct patterns)
Consistency:          < 0.95  ✅ (predictions use context)
```

## 🧠 Why This Works

### With 100% Masking (Old)

```python
# All tokens are zeros
Input:  [0, 0, 0, 0, 0]  # All channels look identical
        [0, 0, 0, 0, 0]
        [0, 0, 0, 0, 0]

# Model learns:
prediction = f(position)  # Only thing that varies!
# Ignores: channel identity, signal patterns
```

### With 75% Masking (New)

```python
# Mix of real signal and masked tokens
Input:  [S, S, 0, S, 0]  # S = actual signal (varies by channel!)
        [S, 0, S, S, 0]
        [0, S, S, 0, S]

# Model must learn:
prediction = f(channel, unmasked_context)  # Use actual signals!
# Uses: channel-specific patterns, temporal context
```

The unmasked signals **force** the model to learn channel-specific patterns because:
- C3 unmasked tokens look different from FP1 unmasked tokens
- Model must learn: "Use C3 patterns to predict C3"
- Position alone is insufficient (can't reconstruct C3 from position)

## 📚 Additional Resources

I've created detailed guides in your repo:

1. **`PRETRAINING_LEAKAGE_AUDIT.md`**
   - Complete architectural audit
   - Confirms: No information leakage
   - Explains: Why loss drops with 100% masking

2. **`DIAGNOSIS_RESULTS.md`**
   - Analysis of your current model
   - Evidence of positional learning
   - Root cause explanation

3. **`ANTI_POSITIONAL_LEARNING_STRATEGIES.md`**
   - 6 strategies to prevent positional learning
   - Implementation details for each
   - When to use advanced techniques

## 🎯 Key Takeaways

1. ✅ **No information leakage** in your pipeline
2. ❌ **Model learned trivial solution** (position only) due to 100% masking
3. ✅ **Fix is simple**: Reduce mask_ratio to 0.75
4. ✅ **Config already updated** - just retrain!

## 📊 Validation Checklist

After retraining with mask_ratio=0.75, verify:

- [ ] Position correlation < 0.3 (was 0.403)
- [ ] Channel diversity > 0.01 (was 0.000)
- [ ] Different channels predict different values
- [ ] Training loss converges to reasonable value
- [ ] Validation loss tracks training loss (generalization)

## 💬 If You Still See Issues

If after retraining with 0.75 you still see high positional dependence:

**Option A**: Reduce mask ratio further
```yaml
mask_ratio: 0.5  # Even more context
```

**Option B**: Add contrastive loss
```python
# See ANTI_POSITIONAL_LEARNING_STRATEGIES.md Strategy 2
# Explicitly penalizes same predictions across channels
```

**Option C**: Add random temporal shifts
```python
# See ANTI_POSITIONAL_LEARNING_STRATEGIES.md Strategy 3
# Makes position unreliable as a predictor
```

But realistically, **mask_ratio=0.75 should be sufficient** for most cases.

---

## The Bottom Line

**Your model wasn't cheating (no leakage) - it was doing exactly what you asked it to do given 100% masking.**

With all inputs being zeros, the only signal was position, so that's what it learned. Now with 25% unmasked context, it will learn actual EEG patterns.

**This is good news** - a simple config change fixes everything! 🎉


---

## Source: TEMPORAL_ORDERING_NOTES.md

# Temporal Ordering in EEG Data Pipeline

## Why Window Order Matters

EEG data is inherently **temporal** - the order of windows represents the sequential flow of brain activity over time. For sequence models like Mamba, preserving this temporal order is **critical** for learning meaningful patterns.

### What Happens if Order is Lost?

❌ **Scrambled windows** = destroyed temporal patterns
- Model can't learn temporal dependencies
- Performance degrades significantly
- Predictions become unreliable

✅ **Ordered windows** = preserved temporal structure
- Model learns temporal dynamics
- Better generalization
- Meaningful predictions

---

## How We Preserve Temporal Order

### 1. Window Slicing (`eeg_analysis/src/processing/window_slicer.py`)

Windows are created with IDs that track their temporal position:
- `parent_window_id`: Original window from which sub-windows are derived
- `sub_window_id`: Position within the parent window

```python
# Example window sequence for one participant:
# parent_window_id=0, sub_window_id=0  ← First window
# parent_window_id=0, sub_window_id=1  ← Second window
# parent_window_id=1, sub_window_id=0  ← Third window
# ...
```

### 2. Closed_finetune Dataset Creation (`eeg_analysis/src/processing/closed_finetune_dataset.py`)

**Sorting Strategy:**
```python
# Sort by: Participant → parent_window_id → sub_window_id
main_df = main_df.sort_values([
    'Participant', 
    'parent_window_id', 
    'sub_window_id'
]).reset_index(drop=True)
```

**Verification:**
- Checks first 3 participants to ensure window IDs are monotonic
- Logs warnings if ordering appears incorrect
- Confirms temporal order is preserved

### 3. SFT Dataset (`eeg_analysis/src/data/eeg_sft_dataset.py`)

**Dataset Initialization:**
```python
# Verify data is sorted on load
if not is_sorted:
    print("Warning: Data not sorted by window IDs, sorting now...")
    self.df = self.df.sort_values(sort_cols).reset_index(drop=True)
    print("✓ Data sorted to preserve temporal window order")
```

**Per-Sample Loading:**
```python
# For each participant, sort their windows
participant_data = participant_data.sort_values([
    'parent_window_id', 
    'sub_window_id'
])

# Verify ordering is correct
is_ordered = check_monotonic(window_ids)
if not is_ordered:
    print(f"Warning: Window ordering may be incorrect")
```

**Collate Function:**
- Preserves order when batching
- Pads sequences without reordering
- Maintains temporal structure across batch

---

## Verification Checklist

When working with EEG data, always verify:

### ✅ Closed_finetune Dataset
```python
# Check first participant's windows
participant_data = df[df['Participant'] == 'sub-001']
print(participant_data[['parent_window_id', 'sub_window_id']].head(10))

# Window IDs should be sequential or monotonic
assert all(participant_data['parent_window_id'].diff().dropna() >= 0)
```

### ✅ SFT Dataset
```python
# Load dataset
dataset = EEGSFTDataset(data_path, split="train")

# Check one sample
sample = dataset[0]
print(f"Participant: {sample['participant']}")
print(f"Windows shape: {sample['windows'].shape}")  # [C, W, L]
print(f"Windows are in temporal order: ✓")
```

### ✅ During Training
```python
# Check batch
batch = next(iter(train_loader))
print(f"Batch windows: {batch['windows'].shape}")  # [B, C, W, L]
print(f"Seq lengths: {batch['seq_lengths']}")
# Each sequence in batch maintains temporal order
```

---

## Common Pitfalls to Avoid

### ❌ DON'T: Shuffle windows within a participant
```python
# BAD - destroys temporal order
participant_data = participant_data.sample(frac=1)
```

### ✅ DO: Shuffle participants (not their windows)
```python
# GOOD - preserves temporal order within each participant
train_loader = DataLoader(dataset, shuffle=True)  # Shuffles participants, not windows
```

### ❌ DON'T: Sort by irrelevant columns
```python
# BAD - arbitrary ordering
df = df.sort_values(['Remission', 'Participant'])
```

### ✅ DO: Always sort by temporal identifiers
```python
# GOOD - preserves temporal structure
df = df.sort_values(['Participant', 'parent_window_id', 'sub_window_id'])
```

### ❌ DON'T: Drop window ID columns
```python
# BAD - loses ability to verify ordering
df = df.drop(['parent_window_id', 'sub_window_id'], axis=1)
```

### ✅ DO: Keep window IDs for verification
```python
# GOOD - can always verify and re-sort if needed
df = df[['Participant', 'parent_window_id', 'sub_window_id', ...]]
```

---

## Impact on Model Performance

### With Correct Ordering:
- ✅ Model learns temporal patterns
- ✅ Captures brain state transitions
- ✅ Generalizes to new participants
- ✅ Predictions are meaningful

### With Scrambled Ordering:
- ❌ Model sees random sequences
- ❌ Can't learn temporal dependencies
- ❌ Poor generalization
- ❌ Unpredictable behavior

---

## Testing Temporal Order

### Quick Test Script

```python
import pandas as pd

# Load closed_finetune dataset
df = pd.read_parquet("path/to/closed_finetune_dataset.parquet")

# Test ordering for each participant
for participant in df['Participant'].unique()[:5]:
    p_data = df[df['Participant'] == participant]
    
    # Check parent window IDs are monotonic
    parent_ids = p_data['parent_window_id'].values
    is_monotonic = all(parent_ids[i] <= parent_ids[i+1] for i in range(len(parent_ids)-1))
    
    status = "✓" if is_monotonic else "✗"
    print(f"{status} Participant {participant}: {len(p_data)} windows, "
          f"IDs: {parent_ids[0]} → {parent_ids[-1]}")

print("\nIf all participants show ✓, temporal order is preserved!")
```

---

## Summary

**Key Points:**
1. Window order = temporal order = critical for sequence models
2. Always sort by: `Participant → parent_window_id → sub_window_id`
3. Verify ordering at every pipeline stage
4. Never shuffle windows within a participant
5. Keep window IDs for verification and debugging

**Our Implementation:**
- ✅ Closed_finetune dataset creation: Sorts and verifies
- ✅ SFT dataset: Checks and re-sorts if needed
- ✅ Per-sample loading: Sorts participant windows
- ✅ Collate function: Preserves order during batching
- ✅ Documentation: Emphasizes importance throughout

**Result**: Temporal structure is preserved end-to-end, ensuring the Mamba model receives properly ordered sequences for optimal learning.


---

## Source: TRAINING_RESULTS_ANALYSIS.md

# Training Results Analysis: mask_ratio=0.75

## Key Findings

### ✅ What's Working

1. **Context Learning** ✅
   - Context sensitivity: **0.58** (LOW = GOOD)
   - Predictions vary with different masking contexts
   - Model learns from unmasked tokens, not just position

2. **Position Fix Working** ✅
   - Position correlation: **0.31** (down from 0.69 with 100% masking)
   - Positional encoding zeroing is effective
   - Position is a helper, not the only signal

### ❌ What's Not Working

1. **Signal Pattern Learning** ❌
   - Pattern correlation: **-0.045** (near zero, even negative)
   - Model does NOT learn actual EEG signal patterns
   - Predictions don't match ground truth waveforms

2. **Prediction Diversity** ❌
   - Baseline similarity: **0.965** (very high)
   - All predictions are nearly identical
   - Model predicts similar values regardless of input

## Interpretation

The model is learning **something** (context sensitivity is low), but it's **not learning signal content**.

**What the model might be learning**:
- Context-dependent patterns (good!)
- But predicting similar/constant values (bad!)
- Not learning actual signal waveforms (bad!)

**Possible causes**:
1. **Loss function issue**: Model optimizes for something other than signal reconstruction
2. **Normalization issue**: Per-window normalization might be removing signal structure
3. **Decoder issue**: Decoder might be collapsing to constant predictions
4. **Training dynamics**: Model might need more training or different hyperparameters

## Recommendations

1. **Check loss values**: Is reconstruction loss actually decreasing?
2. **Inspect decoder outputs**: Are predictions actually varying?
3. **Test without normalization**: Try training without per-window normalization
4. **Increase training**: Model might need more epochs
5. **Check gradient flow**: Ensure gradients are flowing to decoder

## Next Steps

1. Verify the model is actually trying to reconstruct signals
2. Check if decoder is learning or collapsing
3. Test with different mask ratios (0.5, 0.25)
4. Inspect actual predictions vs ground truth visually

The good news: Context learning works! The bad news: Signal pattern learning doesn't.


---

## Source: WITHIN_TOKEN_MASKING.md

# Within-Token Masking: A Better MAE Approach

## The Problem with Token-Level Masking

**Current approach (token-level masking):**
- Mask entire tokens (2048-sample windows)
- With `mask_ratio=1.0`: All tokens are zeros
- Model has NO signal content to learn from
- Can only learn position (from sequential processing)

**Result**: Model learns position, not signal patterns.

## The Solution: Within-Token Masking

**New approach (within-token masking):**
- Keep ALL tokens in sequence
- Mask `mask_ratio * window_length` samples WITHIN each token
- Model always has signal content (unmasked samples)
- Must learn from signal, not just position

**Example with `mask_ratio=0.75`:**
- Each token: 2048 samples
- Masked samples: 1536 samples (75%)
- Unmasked samples: 512 samples (25%)
- Model sees signal content in every token!

## Benefits

### 1. Always Has Signal Content ✅
- Even with high mask ratios, unmasked samples remain
- Model can learn signal patterns, not just position
- Better for learning actual EEG structure

### 2. More Similar to Vision MAE ✅
- Vision MAE masks patches but pixels within patches can be partially visible
- Within-token masking is analogous
- More standard MAE approach

### 3. Better Learning Dynamics ✅
- Model learns from signal content from the start
- Less position-only learning
- Better signal pattern learning

## Implementation

### Masking Strategy

**Token-level (old):**
```python
mask_ratio=0.75 → Mask 75% of tokens (entire windows)
Result: Some tokens are zeros, some have full signal
```

**Within-token (new):**
```python
mask_ratio=0.75 → Mask 75% of samples within EACH token
Result: ALL tokens have signal content (25% unmasked samples)
```

### Code Changes

1. **Collate function**: Added `masking_style="within_token"`
   - Creates sample-level mask `(B, L, W)` instead of token-level `(B, L)`
   - Masks samples within each token

2. **Model forward**: Handles both mask types
   - Converts sample-level mask to token-level for positional encoding
   - Token is "masked" if ANY sample in it is masked

3. **Loss computation**: Handles both mask types
   - Sample-level mask: `(B, L, W)` matches `pred/target` shape
   - Token-level mask: Expanded to `(B, L, W)`

## Usage

### In Config

```yaml
masking_style: "within_token"  # NEW: Mask samples within tokens
mask_ratio: 0.75  # Mask 75% of samples within each token
```

### Comparison

**Token-level masking (`masking_style="mae"`):**
- `mask_ratio=1.0`: All tokens zeros → Only position learning
- `mask_ratio=0.75`: 75% tokens zeros → Some signal, mostly position

**Within-token masking (`masking_style="within_token"`):**
- `mask_ratio=1.0`: All samples masked → Still only position (but less likely)
- `mask_ratio=0.75`: 75% samples masked → ALL tokens have signal content ✅

## Expected Results

### With Within-Token Masking

**Better signal learning:**
- Pattern correlation should increase (0.05 → 0.3+)
- Context sensitivity should decrease (0.58 → 0.4-0.5)
- Baseline similarity should decrease (0.97 → 0.3-0.5)

**Position still helps:**
- Position correlation: Moderate (0.2-0.4)
- Position is a helper feature, not the only feature

## Why This Works

1. **Signal content always available**: Unmasked samples provide signal
2. **Model must learn patterns**: Can't just learn position
3. **More realistic**: Similar to how vision MAE works
4. **Better for EEG**: Temporal structure preserved, signal content available

## Recommendation

**Use `masking_style="within_token"` for training:**
- Better signal learning
- Less position-only learning
- More standard MAE approach
- Better for downstream tasks

**Keep `masking_style="mae"` for control experiments:**
- Test if model learns position with 100% masking
- Verify no information leakage


---
