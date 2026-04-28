#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathlib import Path
import json
import numpy as np
from src.phase3_inference.nmt_preprocessor import NMTPreprocessor

EDF = "data/samples/0001082.edf"
NORM_PATH = "outputs/models/normalization_stats.json"  # adjust if needed

pre = NMTPreprocessor()

# load normalization stats if available
norm_stats = None
if Path(NORM_PATH).exists():
    with open(NORM_PATH, "r") as f:
        norm_stats = json.load(f)
else:
    print("Warning: normalization_stats.json not found. Running without z-score.")

# Step 1: load & inspect raw
raw = pre.load_edf(EDF)
print("raw sfreq:", raw.info["sfreq"])
print("raw channels (sample):", raw.ch_names[:5], "...", "total:", len(raw.ch_names))

# Step 2: normalize channel names and check
raw_norm = pre.normalize_channel_names(raw.copy())
print("normalized channels:", raw_norm.ch_names)
assert len(raw_norm.ch_names) == len(pre.target_channels), "Channel mismatch after normalize"

# Step 3: run the full pipeline (filter + optional ICA + segmentation + normalization)
out = pre.run(EDF, norm_stats=norm_stats, do_ica=False)  # set do_ica=False to speed up/fail-safe
pp = out["preprocessed_eeg"]
print("preprocessed shape:", pp.shape, "dtype:", pp.dtype)
print("n_epochs reported:", out["n_epochs"], "n_epoch_times:", len(out["epoch_times"]))

# Basic numeric checks
print("min/max preprocessed:", float(pp.min()), float(pp.max()))
print("finite check:", np.isfinite(pp).all())

# DC removal check (after per-window demean inside pipeline -> should be near zero)
# compute per-window per-channel mean (should be ~0)
means = pp.mean(axis=2)  # (n_epochs, n_channels)
print("max abs mean across windows/channels (should be small):", float(np.max(np.abs(means))))

# expected epochs formula
n_samples = int(raw.info["sfreq"] * raw.times[-1])
expected_epochs = (n_samples - pre.window_samples) // pre.step + 1
print("n_samples:", n_samples, "expected_epochs:", expected_epochs)

print("Test finished. If assertions pass and values look sane, preprocessor is OK.")