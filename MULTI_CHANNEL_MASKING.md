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
python diagnose_100pct_masking.py \
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

