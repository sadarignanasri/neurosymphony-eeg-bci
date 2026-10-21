from pathlib import Path
from scipy.io import loadmat
from scipy.signal import resample_poly
import numpy as np
import json

ROOTS = [
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults 3"),
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults-2 2"),
]

SUBJECT = "S01"
CLASS_LABEL = "FF"
ORIGINAL_SAMPLING_RATE = 500
TARGET_SAMPLING_RATE = 125
EXPECTED_CHANNELS = 31
EXPECTED_TIMEPOINTS = 6500

def locate_file(subject):
    relative = (
        Path(subject) / "EEG" / "preprocessed"
        / f"{subject}_preprocessed.mat"
    )
    for root in ROOTS:
        path = root / relative
        if path.exists():
            return path
    return None

path = locate_file(SUBJECT)
if path is None:
    raise FileNotFoundError(f"Preprocessed file not found for {SUBJECT}")

mat = loadmat(path)
if CLASS_LABEL not in mat:
    raise KeyError(f"Class {CLASS_LABEL} missing from {path}")

data = np.asarray(mat[CLASS_LABEL])
if data.shape[:2] != (EXPECTED_CHANNELS, EXPECTED_TIMEPOINTS):
    raise ValueError(f"Unexpected input shape: {data.shape}")

# Select one epoch only; no full-dataset tensor is created.
epoch = np.asarray(data[:, :, 0], dtype=np.float32)

# Reduce the sampling rate by a factor of four using
# polyphase resampling with anti-alias filtering.
resampled = resample_poly(epoch, up=1, down=4, axis=-1)

expected_output_points = EXPECTED_TIMEPOINTS // 4
if resampled.shape != (EXPECTED_CHANNELS, expected_output_points):
    raise ValueError(f"Unexpected output shape: {resampled.shape}")

if not np.isfinite(resampled).all():
    raise ValueError("Resampled signal contains non-finite values")

report = {
    "step": 185,
    "subject": SUBJECT,
    "class": CLASS_LABEL,
    "source_file": str(path),
    "input_shape": list(epoch.shape),
    "output_shape": list(resampled.shape),
    "original_sampling_rate_hz": ORIGINAL_SAMPLING_RATE,
    "target_sampling_rate_hz": TARGET_SAMPLING_RATE,
    "input_dtype": str(epoch.dtype),
    "output_dtype": str(resampled.dtype),
    "input_size_mb": round(epoch.nbytes / (1024 * 1024), 4),
    "output_size_mb": round(resampled.nbytes / (1024 * 1024), 4),
    "all_output_values_finite": bool(np.isfinite(resampled).all()),
    "method": "scipy.signal.resample_poly(up=1, down=4)",
    "full_dataset_transformed": False,
    "source_data_modified": False,
    "models_modified": False,
}

out = Path("results/eeg/visual_motivation/step185_downsampling_test.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("DOWNSAMPLING TEST: PASS")
print("Input shape:", epoch.shape)
print("Output shape:", resampled.shape)
print("Sampling rate:", ORIGINAL_SAMPLING_RATE, "->", TARGET_SAMPLING_RATE, "Hz")
print("Input memory:", report["input_size_mb"], "MB")
print("Output memory:", report["output_size_mb"], "MB")
print("All output values finite:", report["all_output_values_finite"])
print("Report saved:", out)
print("No transformed dataset or model saved.")
