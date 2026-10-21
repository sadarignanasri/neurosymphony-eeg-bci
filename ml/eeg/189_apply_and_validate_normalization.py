from pathlib import Path
import numpy as np
import pandas as pd
import json

SOURCE = Path("data/processed/visual_motivation_125hz")
OUTPUT = Path("data/processed/visual_motivation_125hz_normalized")
PARAMS = Path(
    "results/eeg/visual_motivation/"
    "step188_training_channel_normalization.npz"
)
REPORT = Path(
    "results/eeg/visual_motivation/"
    "step189_normalization_validation.json"
)

CLASSES = ["FF", "FR", "RF", "RR"]
EXPECTED_SPLITS = {"train": 1118, "validation": 343, "test": 413}

if not (SOURCE / "dataset_manifest.csv").exists():
    raise FileNotFoundError(SOURCE / "dataset_manifest.csv")
if not PARAMS.exists():
    raise FileNotFoundError(PARAMS)
if OUTPUT.exists() and any(OUTPUT.iterdir()):
    raise FileExistsError(
        f"Output directory is not empty: {OUTPUT}. Nothing overwritten."
    )
if REPORT.exists():
    raise FileExistsError(f"Report already exists: {REPORT}")

manifest = pd.read_csv(SOURCE / "dataset_manifest.csv")
params = np.load(PARAMS, allow_pickle=False)
mean = params["mean"].astype(np.float64)
std = params["std"].astype(np.float64)

if mean.shape != (31,) or std.shape != (31,):
    raise ValueError("Expected 31 normalization parameters.")
if not np.isfinite(mean).all() or not np.isfinite(std).all():
    raise ValueError("Normalization parameters contain non-finite values.")
if np.any(std <= 0):
    raise ValueError("Standard deviations must be positive.")

OUTPUT.mkdir(parents=True, exist_ok=True)
out_rows = []
split_counts = {key: 0 for key in EXPECTED_SPLITS}
train_sum = np.zeros(31, dtype=np.float64)
train_sum_sq = np.zeros(31, dtype=np.float64)
train_values = 0

for row in manifest.itertuples(index=False):
    if row.split not in EXPECTED_SPLITS:
        raise ValueError(f"Unexpected split: {row.split}")

    source_path = Path(row.output_file)
    x = np.load(source_path, allow_pickle=False, mmap_mode="r")

    if x.shape != (int(row.epoch_count), 31, 1625):
        raise ValueError(f"Unexpected shape: {source_path}")
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite source values: {source_path}")

    # Apply only the parameters fitted in Step 188.
    normalized = (
        np.asarray(x, dtype=np.float32)
        - mean.astype(np.float32)[None, :, None]
    ) / std.astype(np.float32)[None, :, None]

    if not np.isfinite(normalized).all():
        raise ValueError(f"Non-finite normalized values: {source_path}")

    relative = source_path.relative_to(SOURCE)
    destination = OUTPUT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")

    np.save(destination, normalized.astype(np.float32), allow_pickle=False)

    split_counts[row.split] += len(normalized)

    if row.split == "train":
        train_sum += normalized.astype(np.float64).sum(axis=(0, 2))
        train_sum_sq += np.square(normalized.astype(np.float64)).sum(axis=(0, 2))
        train_values += len(normalized) * 1625

    out_rows.append({
        "subject": row.subject,
        "split": row.split,
        "label": row.label,
        "label_id": int(row.label_id),
        "epoch_count": int(row.epoch_count),
        "channels": 31,
        "timepoints": 1625,
        "sampling_rate_hz": 125,
        "output_file": str(destination),
    })
    print(f"PASS {row.subject} {row.label} | {row.split} | {normalized.shape}")

if split_counts != EXPECTED_SPLITS:
    raise ValueError(f"Unexpected split counts: {split_counts}")

train_mean = train_sum / train_values
train_var = train_sum_sq / train_values
train_std = np.sqrt(np.maximum(train_var, 0))

if not np.allclose(train_mean, 0, atol=1e-4):
    raise ValueError(f"Training means not near zero: {train_mean}")
if not np.allclose(train_std, 1, atol=1e-3):
    raise ValueError(f"Training standard deviations not near one: {train_std}")

pd.DataFrame(out_rows).to_csv(OUTPUT / "dataset_manifest.csv", index=False)

report = {
    "step": 189,
    "total_epochs": int(sum(split_counts.values())),
    "split_epoch_counts": split_counts,
    "shape_per_epoch": [31, 1625],
    "sampling_rate_hz": 125,
    "normalization": "training-fitted per-channel standardization",
    "training_channel_mean_min": float(train_mean.min()),
    "training_channel_mean_max": float(train_mean.max()),
    "training_channel_std_min": float(train_std.min()),
    "training_channel_std_max": float(train_std.max()),
    "validation_or_test_used_for_fitting": False,
    "source_arrays_modified": False,
    "existing_models_modified": False,
    "output_directory": str(OUTPUT),
}

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\nNORMALIZATION VALIDATION: PASS")
print("Total epochs:", report["total_epochs"])
print("Split counts:", split_counts)
print("Training normalized mean range:",
      report["training_channel_mean_min"],
      "to", report["training_channel_mean_max"])
print("Training normalized std range:",
      report["training_channel_std_min"],
      "to", report["training_channel_std_max"])
print("Output:", OUTPUT)
print("Report:", REPORT)
print("Original arrays and existing models were not modified.")
