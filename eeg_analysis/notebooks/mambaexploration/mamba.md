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
# Discover EEG runs under /home/abin/ds* and load one per dataset
from pathlib import Path
import json

HOME = Path("/home/abin")
MAX_RUNS = None  # set to an int to cap the number of datasets

def discover_one_per_ds(root: Path):
    ds_dirs = sorted(p for p in root.glob("ds*") if p.is_dir())
    selected_files = []
    for ds in ds_dirs:
        eeg_files = list(ds.rglob("*.set")) + list(ds.rglob("*.vhdr"))
        if not eeg_files:
            continue
        eeg_files = sorted(eeg_files)
        selected_files.append(eeg_files[0])
    return ds_dirs, selected_files

ds_dirs, eeg_files = discover_one_per_ds(HOME)
print(f"Discovered {len(ds_dirs)} ds* folders under {HOME}")
print(f"Selected {len(eeg_files)} EEG files (one per dataset)")

if not eeg_files:
    raise ValueError(f"No .set or .vhdr files found under {HOME}")

if MAX_RUNS is not None:
    eeg_files = eeg_files[:MAX_RUNS]
    print(f"Using first {len(eeg_files)} datasets for detailed summary")

def _parse_bids_entities(path: Path) -> dict:
    entities = {}
    for part in path.stem.split("_"):
        if "-" in part:
            key, val = part.split("-", 1)
            entities[key] = val
    return entities

def _read_json_sidecar(path: Path) -> dict:
    json_path = path.with_suffix(".json")
    if not json_path.exists():
        return {}
    try:
        return json.loads(json_path.read_text())
    except Exception:
        return {}

def _channels_tsv_path(path: Path) -> Path:
    stem = path.stem
    if stem.endswith("_eeg"):
        stem = stem[:-4]
    return path.with_name(f"{stem}_channels.tsv")

def _events_tsv_path(path: Path) -> Path:
    stem = path.stem
    if stem.endswith("_eeg"):
        stem = stem[:-4]
    return path.with_name(f"{stem}_events.tsv")

def load_raw(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".set":
        return mne.io.read_raw_eeglab(str(path), preload=True)
    if suffix == ".vhdr":
        return mne.io.read_raw_brainvision(str(path), preload=True)
    if suffix == ".eeg":
        raise ValueError("BrainVision .eeg files require the .vhdr header file")
    raise ValueError(f"Unsupported EEG file type: {path.suffix}")

def describe_run(path: Path) -> dict:
    raw = load_raw(path)
    bids = _parse_bids_entities(path)
    sidecar = _read_json_sidecar(path)

    duration_s = raw.n_times / raw.info["sfreq"]
    channels_tsv = _channels_tsv_path(path)
    events_tsv = _events_tsv_path(path)

    return {
        "file": str(path),
        "subject": bids.get("sub"),
        "task": bids.get("task"),
        "acq": bids.get("acq"),
        "run": bids.get("run"),
        "n_channels": raw.info["nchan"],
        "sfreq_hz": float(raw.info["sfreq"]),
        "n_samples": raw.n_times,
        "duration_s": float(duration_s),
        "bads": len(raw.info.get("bads", [])),
        "channels_tsv": str(channels_tsv) if channels_tsv.exists() else None,
        "events_tsv": str(events_tsv) if events_tsv.exists() else None,
        "task_name": sidecar.get("TaskName"),
        "power_line_hz": sidecar.get("PowerLineFrequency"),
        "reference": sidecar.get("EEGReference"),
        "manufacturer": sidecar.get("Manufacturer"),
        "sampling_frequency": sidecar.get("SamplingFrequency"),
    }, raw

runs = []
raws = []
for eeg_path in eeg_files:
    info, raw = describe_run(eeg_path)
    runs.append(info)
    raws.append(raw)


# Use the first run for the single-run plots below
raw = raws[0]
```
```python
runs_df = pd.DataFrame(runs)
runs_df
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
