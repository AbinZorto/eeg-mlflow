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

1. **Data Loading** (`eeg_pretraining_dataset.py:102-124`):
   ```python
   windows_t = torch.from_numpy(windows_np).to(torch.float32)  # (L, W=2048)
   return {"windows": windows_t, "channel_name": str(channel).upper(), "seq_len": int(windows_t.shape[0])}
   ```
   - Raw 2048-sample windows loaded from parquet files
   - No masking at this stage

2. **Collation with Masking** (`eeg_pretraining_dataset.py:164-177`):
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

3. **Model Forward Pass** (`mamba_eeg_model.py:259`):
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
**Location**: `window_slicer.py:48-71`

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
**Location**: `window_slicer.py:114-116`

```python
for i in range(num_complete_windows):
    start_idx = i * self.window_length  # NO overlap: 0, 2048, 4096, ...
    end_idx = start_idx + self.window_length
```

**Result**: Non-overlapping sequential windows

### Configuration
**Location**: `processing_config.yaml:60-65`

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

The windowing implementation used for pretraining data creation follows the non-overlapping sequential approach from `process_window()` (lines 114-116 in `window_slicer.py`):
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
**Location**: `eeg_pretraining_dataset.py:41-49`

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
**Location**: `eeg_pretraining_dataset.py:105`

```python
df = pd.read_parquet(str(fp), engine="pyarrow", columns=[channel])
```

Only **one channel** is loaded per sample.

### Model Input
**Location**: `pretrain_mamba.py:224`

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
**Location**: `pretrain_mamba.py:236-241`

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
**Location**: `pretrain_mamba.py:235`

```python
target = (model.module if isinstance(model, DDP) else model).encode_tokens_only(windows)
```

**Location**: `mamba_eeg_model.py:230-239`

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
**Location**: `eeg_pretraining_dataset.py:157-167`

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
**Location**: `pretrain_mamba.py:220-222`

```python
windows = batch["windows"].to(device, non_blocking=True)               # (B, L, W)
windows_masked = batch["windows_masked"].to(device, non_blocking=True) # (B, L, W)
mask_bool = batch["mask_bool"].to(device, non_blocking=True)           # (B, L)
```

Both `windows` (original) and `windows_masked` transferred to GPU.

### Model Forward
**Location**: `pretrain_mamba.py:229-235`

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
**Location**: `mamba_eeg_model.py:241-266`

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
**Location**: `mamba_eeg_model.py:106-127`

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
**Location**: `mamba_eeg_model.py:154-167`

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

**Spatial Encoding** (`mamba_eeg_model.py:63-72`):
```python
def forward(self, channel_names: List[str]) -> torch.Tensor:
    coords = torch.stack([self._coords_for(nm) for nm in channel_names], dim=0)  # (B, 3)
    return coords @ self.proj  # (B, 3) @ (3, D) → (B, D)
```
- Encodes electrode 3D position (e.g., "FP1" → [x, y, z])
- **Same for all tokens in a sequence** (single channel per sequence)
- Cannot leak signal content

**Temporal Encoding** (`mamba_eeg_model.py:83-103`):
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

**Location**: `eeg_pretraining_dataset.py:179-190`

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

**Location**: `pretrain_mamba.py:73-74`

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
python eeg_analysis/src/training/pretrain_mamba.py --config configs/pretrain.yaml

# Experiment 2: Standard MAE (context-based learning)  
# Edit pretrain.yaml: mask_ratio: 0.75
python eeg_analysis/src/training/pretrain_mamba.py --config configs/pretrain.yaml

# Experiment 3: Easy mode (strong context)
# Edit pretrain.yaml: mask_ratio: 0.5
python eeg_analysis/src/training/pretrain_mamba.py --config configs/pretrain.yaml

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

Update `configs/pretrain.yaml`:

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
| Dataset Loading | `src/data/eeg_pretraining_dataset.py` | 41-124 |
| Collate + Masking | `src/data/eeg_pretraining_dataset.py` | 127-200 |
| Training Loop | `src/training/pretrain_mamba.py` | 213-373 |
| Loss Computation | `src/training/pretrain_mamba.py` | 236-241 |
| Model Forward | `src/models/mamba_eeg_model.py` | 241-266 |
| Token Encoder | `src/models/mamba_eeg_model.py` | 106-127 |
| Windowing (overlap) | `src/processing/window_slicer.py` | 48-71 |
| Windowing (no overlap) | `src/processing/window_slicer.py` | 114-116 |
| Config | `configs/pretrain.yaml` | 1-30 |

---

## Diagnostic Tool

A diagnostic script has been provided to analyze what your model is learning:

```bash
python diagnose_100pct_masking.py \
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
python diagnose_100pct_masking.py --checkpoint your_model.pt

# Check downstream task performance
# Compare embeddings from mask_ratio=1.0 vs 0.75 vs random
```

**Option B: Safe Default**
```yaml
# Update pretrain.yaml
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

