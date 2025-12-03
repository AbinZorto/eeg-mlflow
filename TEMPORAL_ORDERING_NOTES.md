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

### 1. Window Slicing (`src/processing/window_slicer.py`)

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

### 2. Primary Dataset Creation (`src/processing/primary_dataset.py`)

**Sorting Strategy:**
```python
# Sort by: Participant → parent_window_id → sub_window_id
primary_df = primary_df.sort_values([
    'Participant', 
    'parent_window_id', 
    'sub_window_id'
]).reset_index(drop=True)
```

**Verification:**
- Checks first 3 participants to ensure window IDs are monotonic
- Logs warnings if ordering appears incorrect
- Confirms temporal order is preserved

### 3. SFT Dataset (`src/data/eeg_sft_dataset.py`)

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

### ✅ Primary Dataset
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

# Load primary dataset
df = pd.read_parquet("path/to/primary_dataset.parquet")

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
- ✅ Primary dataset creation: Sorts and verifies
- ✅ SFT dataset: Checks and re-sorts if needed
- ✅ Per-sample loading: Sorts participant windows
- ✅ Collate function: Preserves order during batching
- ✅ Documentation: Emphasizes importance throughout

**Result**: Temporal structure is preserved end-to-end, ensuring the Mamba model receives properly ordered sequences for optimal learning.

