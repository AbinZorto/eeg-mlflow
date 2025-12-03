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

