from pathlib import Path
from scipy.io import loadmat
from scipy.signal import welch
import numpy as np
import pandas as pd
import hashlib
import json

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
ROOTS = [
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults 3"),
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults-2 2"),
]

CLASSES = ("FF", "FR", "RF", "RR")
FS = 500.0
EXPECTED_CHANNELS = 31
EXPECTED_SAMPLES = 6500

# Conservative analysis bands, below the 50 Hz preprocessing cutoff.
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

OUT_DIR = Path("results/eeg/visual_motivation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def extract_features(epoch):
    """
    epoch shape: (channels, samples)
    Returns channel-wise statistical, absolute log-power,
    and relative band-power features.
    """
    epoch = np.asarray(epoch, dtype=np.float64)

    if epoch.shape != (EXPECTED_CHANNELS, EXPECTED_SAMPLES):
        raise ValueError(f"Unexpected epoch shape: {epoch.shape}")

    if not np.isfinite(epoch).all():
        raise ValueError("Epoch contains NaN or infinite values")

    # Remove each channel's mean before spectral analysis.
    centered = epoch - epoch.mean(axis=1, keepdims=True)

    features = {}

    # Time-domain features.
    for ch in range(EXPECTED_CHANNELS):
        x = epoch[ch]
        xc = centered[ch]
        prefix = f"ch{ch + 1:02d}"

        features[f"{prefix}_mean"] = float(np.mean(x))
        features[f"{prefix}_std"] = float(np.std(x))
        features[f"{prefix}_variance"] = float(np.var(x))
        features[f"{prefix}_rms"] = float(np.sqrt(np.mean(x ** 2)))

    # Welch power spectral density for each channel.
    freqs, psd = welch(
        centered,
        fs=FS,
        window="hann",
        nperseg=1024,
        noverlap=512,
        axis=1,
        detrend="constant",
        scaling="density",
    )

    total_mask = (freqs >= 1.0) & (freqs <= 45.0)
    total_power = np.trapezoid(psd[:, total_mask], freqs[total_mask], axis=1)
    total_power = np.maximum(total_power, np.finfo(float).tiny)

    for band_name, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs < high)

        if mask.sum() < 2:
            raise ValueError(f"Insufficient PSD bins for band {band_name}")

        power = np.trapezoid(psd[:, mask], freqs[mask], axis=1)
        power = np.maximum(power, np.finfo(float).tiny)
        relative_power = power / total_power

        for ch in range(EXPECTED_CHANNELS):
            prefix = f"ch{ch + 1:02d}"
            features[f"{prefix}_{band_name}_logpower"] = float(
                np.log10(power[ch])
            )
            features[f"{prefix}_{band_name}_relative"] = float(
                relative_power[ch]
            )

    return features

def main():
    rows = []
    seen_hashes = {}
    skipped_duplicates = []
    skipped_files = []
    accepted_files = []
    subject_sources = {}

    for root in ROOTS:
        if not root.exists():
            print(f"WARNING: Dataset folder not found: {root}")
            continue

        mat_files = sorted(
            root.glob("S*/EEG/preprocessed/*_preprocessed.mat")
        )

        for mat_path in mat_files:
            digest = sha256(mat_path)

            if digest in seen_hashes:
                skipped_duplicates.append({
                    "skipped_file": str(mat_path),
                    "same_as": seen_hashes[digest],
                    "sha256": digest,
                })
                print(f"SKIP EXACT DUPLICATE: {mat_path}")
                continue

            seen_hashes[digest] = str(mat_path)
            subject = mat_path.parents[2].name
            accepted_files.append(str(mat_path))
            subject_sources.setdefault(subject, []).append(str(mat_path))

            print(f"\nProcessing {subject}: {mat_path.parent.parent.parent.parent.name}")

            try:
                mat = loadmat(mat_path)
                subject_epoch_count = 0

                for class_name in CLASSES:
                    if class_name not in mat:
                        print(f"  WARNING: Missing class array {class_name}")
                        continue

                    data = np.asarray(mat[class_name])

                    if data.ndim != 3:
                        raise ValueError(
                            f"{class_name} must be 3D, got {data.shape}"
                        )

                    if data.shape[0] != EXPECTED_CHANNELS:
                        raise ValueError(
                            f"{class_name}: expected {EXPECTED_CHANNELS} channels, "
                            f"got {data.shape[0]}"
                        )

                    if data.shape[1] != EXPECTED_SAMPLES:
                        raise ValueError(
                            f"{class_name}: expected {EXPECTED_SAMPLES} samples, "
                            f"got {data.shape[1]}"
                        )

                    for epoch_idx in range(data.shape[2]):
                        epoch = data[:, :, epoch_idx]

                        try:
                            features = extract_features(epoch)
                        except Exception as exc:
                            skipped_files.append({
                                "file": str(mat_path),
                                "class": class_name,
                                "epoch_index": epoch_idx + 1,
                                "reason": str(exc),
                            })
                            continue

                        row = {
                            "subject_id": subject,
                            "label": class_name,
                            "epoch_index": epoch_idx + 1,
                            "source_file": str(mat_path),
                            "source_sha256": digest,
                        }
                        row.update(features)
                        rows.append(row)
                        subject_epoch_count += 1

                print(f"  Extracted valid epochs: {subject_epoch_count}")

            except Exception as exc:
                skipped_files.append({
                    "file": str(mat_path),
                    "reason": str(exc),
                })
                print(f"  ERROR: {exc}")

    if not rows:
        raise RuntimeError("No valid epochs were extracted. Check the errors above.")

    df = pd.DataFrame(rows)

    metadata_cols = [
        "subject_id", "label", "epoch_index",
        "source_file", "source_sha256",
    ]
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    csv_path = OUT_DIR / "eeg_features.csv"
    df.to_csv(csv_path, index=False)

    # Save feature names and extraction configuration for reproducibility.
    report = {
        "sampling_rate_hz": FS,
        "epoch_shape": [EXPECTED_CHANNELS, EXPECTED_SAMPLES],
        "time_range_ms": [-1000, 11998],
        "class_labels": list(CLASSES),
        "frequency_bands_hz": BANDS,
        "welch": {
            "window": "hann",
            "nperseg": 1024,
            "noverlap": 512,
            "scaling": "density",
        },
        "rows": int(len(df)),
        "unique_subjects": sorted(df["subject_id"].unique().tolist()),
        "class_counts": {
            str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()
        },
        "features_per_epoch": len(feature_cols),
        "feature_columns": feature_cols,
        "accepted_files": accepted_files,
        "duplicate_files_skipped": skipped_duplicates,
        "invalid_epochs_or_files": skipped_files,
        "output_csv": str(csv_path),
        "important_note": (
            "Rows from the same subject are not independent. Use subject-wise "
            "splits for evaluation. The two source folders are overlapping "
            "parts/copies of a dataset, not independent cohorts."
        ),
    }

    report_path = OUT_DIR / "step153_feature_extraction_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n========== STEP 153 SUMMARY ==========")
    print("Dataset shape:", df.shape)
    print("Unique subjects:", sorted(df["subject_id"].unique().tolist()))
    print("\nClass counts:")
    print(df["label"].value_counts().sort_index().to_string())
    print("\nFeature columns:", len(feature_cols))
    print("Exact duplicate files skipped:", len(skipped_duplicates))
    print("Invalid epochs/files:", len(skipped_files))
    print("CSV saved:", csv_path)
    print("Report saved:", report_path)
    print("Original data and existing models were not modified.")

if __name__ == "__main__":
    main()
