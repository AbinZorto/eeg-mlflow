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

