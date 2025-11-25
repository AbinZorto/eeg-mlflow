---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: .venv
    language: python
    name: python3
---

```python
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob, os

print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
```

```python
# Point to one .set file (MNE will automatically read the .fdt file)
set_file = "/home/abin/ds003478-download/sub-001/eeg/sub-001_task-Rest_run-01_eeg.set"

raw = mne.io.read_raw_eeglab(set_file, preload=True)
raw
```

```python
print("Channels:", raw.info["nchan"])
print("Sampling rate:", raw.info["sfreq"])
print("Duration (sec):", raw.n_times / raw.info["sfreq"])
print("Channel names:", raw.ch_names[:20])  # show first 20
```

```python
# raw._data is numpy array: shape = (channels, samples)
data = raw.get_data(units='uV')

print("Data shape:", data.shape)   # (66, ~250k samples)
print("First 10 values of channel 0:")
print(data[0, :10])

```

```python
plt.figure(figsize=(15, 4))
plt.plot(data[0, :2000])  # first 2000 samples of channel 0
plt.title(f"Example waveform (channel: {raw.ch_names[0]})")
plt.xlabel("Samples")
plt.ylabel("Amplitude (µV)")
plt.show()
```

```python
raw.compute_psd().plot()
```

```python
raw.plot()
```
