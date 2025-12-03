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

1. **Model config** (from pretrain.yaml):
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

