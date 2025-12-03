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

