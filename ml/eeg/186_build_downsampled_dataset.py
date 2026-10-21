from pathlib import Path
from scipy.io import loadmat
from scipy.signal import resample_poly
import pandas as pd
import numpy as np
import json

ROOTS = [
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults 3"),
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults-2 2"),
]

BASE = Path("results/eeg/visual_motivation")
MANIFEST_PATH = BASE / "step174_split_manifest.csv"
OUTPUT_DIR = Path("data/processed/visual_motivation_125hz")

CLASSES = ["FF", "FR", "RF", "RR"]
EXPECTED_CHANNELS = 31
EXPECTED_TIMEPOINTS = 6500
EXPECTED_TOTAL_EPOCHS = 1874

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

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(MANIFEST_PATH)

manifest = pd.read_csv(MANIFEST_PATH)

if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
    raise FileExistsError(
        f"Output directory is not empty: {OUTPUT_DIR}\n"
        "Nothing was overwritten. Choose a new output directory "
        "or inspect the existing files first."
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
metadata_rows = []
total_epochs = 0

for subject in sorted(manifest["subject"].unique()):
    subject_rows = manifest[manifest["subject"] == subject]
    split_values = subject_rows["split"].unique()

    if len(split_values) != 1:
        raise ValueError(f"{subject} has multiple split assignments.")

    split = str(split_values[0])
    source_path = locate_file(subject)

    if source_path is None:
        raise FileNotFoundError(f"Missing preprocessed file: {subject}")

    mat = loadmat(source_path)
    subject_dir = OUTPUT_DIR / split / subject
    subject_dir.mkdir(parents=True, exist_ok=True)

    for label in CLASSES:
        row = subject_rows[subject_rows["label"] == label]
        if len(row) != 1:
            raise ValueError(f"Unexpected manifest rows for {subject}/{label}")

        expected_count = int(row.iloc[0]["epoch_count"])
        if label not in mat:
            raise KeyError(f"{subject}: missing class array {label}")

        source = np.asarray(mat[label])
        if source.ndim != 3:
            raise ValueError(f"{subject}/{label}: unexpected shape {source.shape}")
        if source.shape[:2] != (EXPECTED_CHANNELS, EXPECTED_TIMEPOINTS):
            raise ValueError(f"{subject}/{label}: unexpected shape {source.shape}")
        if source.shape[2] != expected_count:
            raise ValueError(
                f"{subject}/{label}: expected {expected_count} epochs, "
                f"found {source.shape[2]}"
            )
        if not np.isfinite(source).all():
            raise ValueError(f"{subject}/{label}: non-finite input values")

        # Result shape: epochs x channels x timepoints.
        downsampled = resample_poly(
            source.astype(np.float32, copy=False),
            up=1,
            down=4,
            axis=1,
        )
        downsampled = np.transpose(downsampled, (2, 0, 1))
        downsampled = np.ascontiguousarray(downsampled, dtype=np.float32)

        expected_shape = (expected_count, EXPECTED_CHANNELS, 1625)
        if downsampled.shape != expected_shape:
            raise ValueError(
                f"{subject}/{label}: output shape {downsampled.shape}, "
                f"expected {expected_shape}"
            )
        if not np.isfinite(downsampled).all():
            raise ValueError(f"{subject}/{label}: invalid resampled values")

        out_file = subject_dir / f"{label}.npy"
        if out_file.exists():
            raise FileExistsError(f"Refusing to overwrite {out_file}")

        np.save(out_file, downsampled, allow_pickle=False)

        metadata_rows.append({
            "subject": subject,
            "split": split,
            "label": label,
            "label_id": CLASSES.index(label),
            "epoch_count": expected_count,
            "channels": EXPECTED_CHANNELS,
            "timepoints": 1625,
            "sampling_rate_hz": 125,
            "source_file": str(source_path),
            "output_file": str(out_file),
        })
        total_epochs += expected_count

        print(
            f"{subject} {label}: {expected_count} epochs -> "
            f"{downsampled.shape} | {split}"
        )

if total_epochs != EXPECTED_TOTAL_EPOCHS:
    raise ValueError(
        f"Expected {EXPECTED_TOTAL_EPOCHS} epochs, processed {total_epochs}"
    )

metadata_path = OUTPUT_DIR / "dataset_manifest.csv"
pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)

summary = {
    "step": 186,
    "total_epochs": total_epochs,
    "subjects": int(manifest["subject"].nunique()),
    "classes": CLASSES,
    "shape_per_epoch": [EXPECTED_CHANNELS, 1625],
    "original_sampling_rate_hz": 500,
    "target_sampling_rate_hz": 125,
    "method": "scipy.signal.resample_poly(up=1, down=4)",
    "split_epoch_counts": {
        split: int(
            manifest.loc[manifest["split"] == split, "epoch_count"].sum()
        )
        for split in ["train", "validation", "test"]
    },
    "output_directory": str(OUTPUT_DIR),
    "metadata_file": str(metadata_path),
    "source_data_modified": False,
    "existing_models_modified": False,
}

summary_path = BASE / "step186_downsampled_dataset_summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\nDATASET BUILD: PASS")
print("Total epochs:", total_epochs)
print("Shape per epoch:", summary["shape_per_epoch"])
print("Split counts:", summary["split_epoch_counts"])
print("Output directory:", OUTPUT_DIR)
print("Manifest:", metadata_path)
print("Summary:", summary_path)
print("Original .mat files and existing models were not modified.")
