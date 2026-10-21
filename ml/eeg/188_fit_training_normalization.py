from pathlib import Path
import numpy as np
import pandas as pd
import json

ROOT = Path("data/processed/visual_motivation_125hz")
MANIFEST = ROOT / "dataset_manifest.csv"
OUTPUT = Path("results/eeg/visual_motivation")
PARAMS_PATH = OUTPUT / "step188_training_channel_normalization.npz"
REPORT_PATH = OUTPUT / "step188_normalization_report.json"

if not MANIFEST.exists():
    raise FileNotFoundError(MANIFEST)

if PARAMS_PATH.exists() or REPORT_PATH.exists():
    raise FileExistsError(
        "Step 188 output already exists. Nothing was overwritten."
    )

df = pd.read_csv(MANIFEST)
train_rows = df[df["split"] == "train"]

if train_rows.empty:
    raise ValueError("No training arrays found.")

# Accumulate per-channel statistics without loading the full dataset
# into memory. Each epoch contributes equally.
count = 0
channel_sum = np.zeros(31, dtype=np.float64)
channel_sum_sq = np.zeros(31, dtype=np.float64)

for row in train_rows.itertuples(index=False):
    arr = np.load(row.output_file, mmap_mode="r", allow_pickle=False)

    if arr.ndim != 3 or arr.shape[1:] != (31, 1625):
        raise ValueError(f"Unexpected shape in {row.output_file}")

    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values in {row.output_file}")

    # Sum across epochs and time points, separately for each channel.
    x = np.asarray(arr, dtype=np.float64)
    channel_sum += x.sum(axis=(0, 2))
    channel_sum_sq += np.square(x).sum(axis=(0, 2))
    count += arr.shape[0] * arr.shape[2]

if count == 0:
    raise ValueError("No training samples accumulated.")

mean = channel_sum / count
variance = channel_sum_sq / count - np.square(mean)
variance = np.maximum(variance, 0.0)
std = np.sqrt(variance)

# Avoid division by zero for a constant channel.
safe_std = np.where(std < 1e-8, 1.0, std)

if not np.isfinite(mean).all() or not np.isfinite(safe_std).all():
    raise ValueError("Invalid normalization parameters.")

OUTPUT.mkdir(parents=True, exist_ok=True)
np.savez(
    PARAMS_PATH,
    mean=mean.astype(np.float32),
    std=safe_std.astype(np.float32),
    training_sample_count=np.int64(count),
    channels=np.int64(31),
    sampling_rate_hz=np.int64(125),
)

report = {
    "step": 188,
    "method": "per-channel standardization",
    "fit_split": "train only",
    "training_epochs": int(train_rows["epoch_count"].sum()),
    "training_values_per_channel": int(count),
    "channels": 31,
    "timepoints_per_epoch": 1625,
    "sampling_rate_hz": 125,
    "channel_mean_min": float(mean.min()),
    "channel_mean_max": float(mean.max()),
    "channel_std_min": float(std.min()),
    "channel_std_max": float(std.max()),
    "constant_or_near_constant_channels": [
        int(i) for i, value in enumerate(std) if value < 1e-8
    ],
    "parameters_file": str(PARAMS_PATH),
    "validation_used_for_fitting": False,
    "test_used_for_fitting": False,
    "source_data_modified": False,
    "existing_models_modified": False,
}

REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("TRAINING NORMALIZATION FIT: PASS")
print("Training epochs:", report["training_epochs"])
print("Values per channel:", count)
print("Channel mean range:", mean.min(), "to", mean.max())
print("Channel std range:", std.min(), "to", std.max())
print("Near-constant channels:", report["constant_or_near_constant_channels"])
print("Parameters saved:", PARAMS_PATH)
print("Report saved:", REPORT_PATH)
print("Validation/test data were not used to fit normalization.")
print("Source data and existing models were not modified.")
