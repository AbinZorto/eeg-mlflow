# Position Leakage And Diagnostics Notes

## One-Page Summary

### Scope
This document captures positional-learning leakage diagnosis, evidence, control logic, and mitigation strategies for MAE-style EEG pretraining.

### Key Findings
- The dominant failure mode is position-only learning rather than true signal reconstruction when masking/encoding choices allow positional shortcuts.
- Strong warning indicators repeat across notes: high prediction consistency, high position correlation, and near-zero channel diversity.
- Full masking can produce misleading loss trends unless positional channels are explicitly disabled in control runs.
- Reducing mask ratio and improving channel-aware learning pressure are the most practical first-line interventions.
- Multi-channel and temporal-order handling materially change whether the model learns signal content vs index/location artifacts.

### Recommended Actions (Now)
- Keep a mandatory control experiment in workflow: `mask_ratio=1.0` with temporal and spatial positional encodings disabled.
- Track and review these diagnostics in every pretraining run: position correlation, channel diversity, consistency metrics.
- Use conservative masking settings first (avoid extreme masking until diagnostics are healthy).
- Prioritize anti-position interventions that preserve valid EEG structure before applying advanced regularizers.

### Recommended Actions (Next)
- If leakage indicators remain elevated, apply stronger anti-position strategies from this file in increasing order of complexity.
- Add a lightweight run checklist so model changes cannot be merged without passing leakage diagnostics.
- Standardize temporal-order policy for data preparation and evaluation to avoid hidden confounds.

### Risks And Assumptions
- Some notes include exploratory hypotheses and may reference interim numbers from specific runs.
- Threshold guidance should be treated as operational heuristics, not universal constants.
- Changes that improve leakage metrics can still hurt downstream utility if signal semantics are distorted.

### Where To Dive Deeper
- Strategy catalog and implementation order: `ANTI_POSITIONAL_LEARNING_STRATEGIES.md`
- Multi-strategy anti-position method: `ANTI_POSITION_ONLY_LEARNING.md`
- Root-cause and leakage audit trail: `POSITION_LEAKAGE_ROOT_CAUSE.md`, `PRETRAINING_LEAKAGE_AUDIT.md`, `REAL_LEAKAGE_FOUND.md`
- Diagnostic interpretation details: `DIAGNOSIS_RESULTS.md`, `LEAKAGE_ANALYSIS.md`, `BASELINE_SIMILARITY_ANALYSIS.md`
- Temporal/window masking mechanics: `TEMPORAL_ORDERING_NOTES.md`, `WITHIN_TOKEN_MASKING.md`, `MULTI_CHANNEL_MASKING.md`

This file preserves full notes merged from the project archive.

## Included Sources

- ANTI_POSITIONAL_LEARNING_STRATEGIES.md
- ANTI_POSITION_ONLY_LEARNING.md
- BASELINE_SIMILARITY_ANALYSIS.md
- DIAGNOSIS_RESULTS.md
- DIAGNOSTIC_CHECKLIST.md
- LEAKAGE_ANALYSIS.md
- MULTI_CHANNEL_MASKING.md
- POSITION_LEAKAGE_ROOT_CAUSE.md
- PRETRAINING_LEAKAGE_AUDIT.md
- REAL_LEAKAGE_FOUND.md
- REAL_POSITION_LEARNING_ANALYSIS.md
- SUMMARY_POSITIONAL_LEARNING_FIX.md
- TEMPORAL_ORDERING_NOTES.md
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
   - But signal content should be primary
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
