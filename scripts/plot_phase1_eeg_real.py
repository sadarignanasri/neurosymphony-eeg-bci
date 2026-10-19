from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np


DATASET = Path("/Users/gnanasrisadari/Desktop/ds004514")
SUBJECT = "sub-01"

EEG_FILE = (
    DATASET
    / SUBJECT
    / "eeg"
    / f"{SUBJECT}_task-eeg_eeg.vhdr"
)

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "visualization_outputs"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 72)
print("PHASE 1 — REAL EEG SIGNAL VISUALIZATION")
print("=" * 72)

print(f"Loading: {EEG_FILE}")

raw = mne.io.read_raw_brainvision(
    EEG_FILE,
    preload=False,
    verbose=False,
)

print(f"Channels: {len(raw.ch_names)}")
print(f"Sampling rate: {raw.info['sfreq']} Hz")
print(f"Duration: {raw.times[-1]:.2f} seconds")

# Select first EEG channel
channel_index = 0

# Plot only a short representative window
start_time = 380.0
duration = 5.0

data, times = raw.get_data(
    picks=[channel_index],
    start=int(start_time * raw.info["sfreq"]),
    stop=int((start_time + duration) * raw.info["sfreq"]),
    return_times=True,
)

signal = data[0] * 1e6

plt.figure(figsize=(12, 5))

plt.plot(times, signal)

plt.xlabel("Time (seconds)")
plt.ylabel("EEG amplitude (µV)")
plt.title(
    f"Phase 1 — Raw EEG Signal | {SUBJECT} | "
    f"{raw.ch_names[channel_index]}"
)

plt.tight_layout()

output = OUTPUT_DIR / "phase1_raw_eeg.png"
plt.savefig(output, dpi=200)
plt.show()

print()
print(f"Saved graph: {output}")
print("PHASE 1 EEG VISUALIZATION COMPLETED")
