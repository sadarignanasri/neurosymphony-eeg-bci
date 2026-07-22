import os
import mne
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# Dataset Information
# =====================================================

dataset_name = "ds004514"
subject_id = "sub-01"

# =====================================================
# Output Directory
# =====================================================

output_dir = rf"F:\experiments\{dataset_name}\{subject_id}\bimodal"

os.makedirs(output_dir, exist_ok=True)

# =====================================================
# File Paths
# =====================================================

eeg_file = (
    r"F:\ds004514_real\sub-01\eeg"
    r"\sub-01_task-eeg_eeg.bdf"
)

fnirs_file = (
    r"F:\ds004514_real\sub-01\nirs"
    r"\sub-01_task-nirs_nirs.snirf"
)

# =====================================================
# Load EEG
# =====================================================

print("Loading REAL EEG...")

eeg_raw = mne.io.read_raw_bdf(
    eeg_file,
    preload=False
)

# Use first 10 seconds only
eeg_raw.crop(
    tmin=0,
    tmax=10
)

# Load cropped data
eeg_raw.load_data()

# Downsample for synchronization
eeg_raw.resample(256)

print("EEG loaded successfully.")

# Extract first EEG channel
eeg_data, eeg_times = eeg_raw[:1]

# =====================================================
# Load fNIRS
# =====================================================

print("Loading REAL fNIRS...")

fnirs_raw = mne.io.read_raw_snirf(
    fnirs_file,
    preload=True
)

# Use same time window
fnirs_raw.crop(
    tmin=0,
    tmax=10
)

print("fNIRS loaded successfully.")

# Extract first fNIRS channel
fnirs_data, fnirs_times = fnirs_raw[:1]

# =====================================================
# Signal Processing
# =====================================================

# EEG normalization
eeg_signal = eeg_data[0]

eeg_signal = (
    eeg_signal - np.mean(eeg_signal)
) / np.std(eeg_signal)

# fNIRS normalization
fnirs_signal = fnirs_data[0]

fnirs_signal = (
    fnirs_signal - np.mean(fnirs_signal)
) / np.std(fnirs_signal)

# =====================================================
# Synchronization
# =====================================================

min_length = min(
    len(eeg_signal),
    len(fnirs_signal)
)

eeg_signal = eeg_signal[:min_length]

fnirs_signal = fnirs_signal[:min_length]

common_time = eeg_times[:min_length]

print("Signals synchronized successfully.")

# =====================================================
# Combined Bimodal Visualization
# =====================================================

plt.figure(figsize=(14, 6))

plt.plot(
    common_time,
    eeg_signal,
    label="EEG",
    linewidth=1
)

plt.plot(
    common_time,
    fnirs_signal,
    label="fNIRS",
    linewidth=1
)

plt.title(
    "Synchronized EEG + fNIRS Bimodal Signal"
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Normalized Amplitude"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

# =====================================================
# Save Plot
# =====================================================

save_path = os.path.join(
    output_dir,
    "synchronized_bimodal_signal.png"
)

plt.savefig(
    save_path,
    dpi=300
)

print(f"Bimodal plot saved at: {save_path}")

plt.show()

print("REAL bimodal acquisition completed successfully.")