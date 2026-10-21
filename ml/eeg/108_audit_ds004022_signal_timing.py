
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch


# ============================================================
# EXPERIMENT 108: SIGNAL AND EPOCH TIMING AUDIT
# Dataset: ds004022
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "experiments" / "ds004022_verified_epochs" / "eeg_epochs.npz"
)
AUDIT_PATH = (
    ROOT / "experiments" / "ds004514_verified_epochs" / "trial_audit.csv"
)
# Dataset 2 extraction audit:
DS2_AUDIT_PATH = (
    ROOT / "experiments" / "ds004022_verified_epochs" / "trial_audit.csv"
)

OUT_DIR = ROOT / "experiments" / "ds004022_experiment108_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_FS = 500.0

BANDS = {
    "delta_1_4": (1.0, 4.0),
    "theta_4_8": (4.0, 8.0),
    "alpha_8_13": (8.0, 13.0),
    "beta_13_30": (13.0, 30.0),
    "gamma_30_45": (30.0, 45.0),
}


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Epoch dataset not found: {DATA_PATH}")

    with np.load(DATA_PATH, allow_pickle=True) as data:
        required = [
            "X", "y", "subjects", "runs", "onsets_seconds",
            "cue_markers", "class_names", "channel_names",
            "sampling_rate",
        ]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"Missing NPZ fields: {missing}")

        X = np.asarray(data["X"], dtype=np.float64)
        y = np.asarray(data["y"]).reshape(-1)
        subjects = np.asarray(data["subjects"]).reshape(-1).astype(str)
        runs = np.asarray(data["runs"]).reshape(-1).astype(str)
        onsets = np.asarray(data["onsets_seconds"], dtype=float).reshape(-1)
        cues = np.asarray(data["cue_markers"]).reshape(-1).astype(str)
        class_names = [str(v) for v in data["class_names"].tolist()]
        channel_names = [
            str(v) for v in data["channel_names"].tolist()
        ]
        sampling_rate = float(np.asarray(data["sampling_rate"]).squeeze())

    n_trials, n_channels, n_samples = X.shape

    if not all(
        len(a) == n_trials
        for a in [y, subjects, runs, onsets, cues]
    ):
        raise ValueError("Epoch metadata lengths do not match X.")

    if len(channel_names) != n_channels:
        raise ValueError("Channel name count does not match EEG channel count.")

    if not np.isfinite(X).all():
        raise ValueError("Non-finite EEG values found.")

    duration_seconds = n_samples / sampling_rate

    print("\nEXPERIMENT 108 — EEG SIGNAL AND TIMING AUDIT")
    print("=" * 60)
    print("Epoch tensor:", X.shape)
    print("Sampling rate:", sampling_rate)
    print("Epoch duration:", duration_seconds, "seconds")
    print("Channels:", channel_names)
    print("Subjects:", sorted(np.unique(subjects).tolist()))
    print("Cue markers:", sorted(np.unique(cues).tolist()))

    # --------------------------------------------------------
    # 1. Verify sampling rate, duration, and onset consistency
    # --------------------------------------------------------
    timing_rows = []

    for i in range(n_trials):
        timing_rows.append({
            "trial_index": i,
            "subject": subjects[i],
            "run": runs[i],
            "label": int(y[i]),
            "class_name": (
                class_names[int(y[i])]
                if 0 <= int(y[i]) < len(class_names)
                else "INVALID_LABEL"
            ),
            "cue_marker": cues[i],
            "imagery_onset_seconds": float(onsets[i]),
            "epoch_samples": int(n_samples),
            "sampling_rate_hz": sampling_rate,
            "epoch_duration_seconds": duration_seconds,
        })

    timing_df = pd.DataFrame(timing_rows)
    timing_df.to_csv(OUT_DIR / "epoch_timing_summary.csv", index=False)

    # --------------------------------------------------------
    # 2. Per-channel signal statistics
    # --------------------------------------------------------
    channel_rows = []

    for channel_idx, channel_name in enumerate(channel_names):
        channel_data = X[:, channel_idx, :]

        channel_rows.append({
            "channel_index": channel_idx,
            "channel_name": channel_name,
            "mean": float(np.mean(channel_data)),
            "std": float(np.std(channel_data)),
            "median_absolute_amplitude": float(
                np.median(np.abs(channel_data))
            ),
            "p99_absolute_amplitude": float(
                np.quantile(np.abs(channel_data), 0.99)
            ),
            "max_absolute_amplitude": float(
                np.max(np.abs(channel_data))
            ),
            "zero_fraction": float(np.mean(channel_data == 0)),
        })

    channel_df = pd.DataFrame(channel_rows)
    channel_df.to_csv(OUT_DIR / "channel_signal_statistics.csv", index=False)

    # --------------------------------------------------------
    # 3. Trial-level amplitude and variance
    # --------------------------------------------------------
    trial_rows = []

    for i in range(n_trials):
        epoch = X[i]

        trial_rows.append({
            "trial_index": i,
            "subject": subjects[i],
            "run": runs[i],
            "label": int(y[i]),
            "class_name": class_names[int(y[i])],
            "epoch_mean": float(np.mean(epoch)),
            "epoch_std": float(np.std(epoch)),
            "median_absolute_amplitude": float(
                np.median(np.abs(epoch))
            ),
            "p99_absolute_amplitude": float(
                np.quantile(np.abs(epoch), 0.99)
            ),
            "max_absolute_amplitude": float(np.max(np.abs(epoch))),
            "mean_channel_variance": float(
                np.mean(np.var(epoch, axis=1))
            ),
        })

    trial_df = pd.DataFrame(trial_rows)
    trial_df.to_csv(OUT_DIR / "trial_signal_statistics.csv", index=False)

    trial_group_summary = (
        trial_df.groupby(["subject", "class_name"])[
            [
                "epoch_std",
                "median_absolute_amplitude",
                "p99_absolute_amplitude",
                "mean_channel_variance",
            ]
        ]
        .agg(["mean", "median", "std"])
    )
    trial_group_summary.to_csv(
        OUT_DIR / "signal_statistics_by_subject_class.csv"
    )

    # --------------------------------------------------------
    # 4. Relative spectral power by trial and frequency band
    # --------------------------------------------------------
    spectral_rows = []

    for i in range(n_trials):
        freqs, psd = welch(
            X[i],
            fs=sampling_rate,
            nperseg=min(1000, n_samples),
            axis=-1,
        )

        total_mask = (freqs >= 1.0) & (freqs <= 45.0)
        total_power_by_channel = np.trapezoid(
            psd[:, total_mask],
            x=freqs[total_mask],
            axis=1,
        )

        row = {
            "trial_index": i,
            "subject": subjects[i],
            "run": runs[i],
            "label": int(y[i]),
            "class_name": class_names[int(y[i])],
        }

        for band_name, (low, high) in BANDS.items():
            mask = (freqs >= low) & (freqs < high)

            band_power_by_channel = np.trapezoid(
                psd[:, mask],
                x=freqs[mask],
                axis=1,
            )

            safe_total = np.maximum(total_power_by_channel, 1e-30)

            row[f"{band_name}_absolute_power"] = float(
                np.mean(band_power_by_channel)
            )
            row[f"{band_name}_relative_power"] = float(
                np.mean(band_power_by_channel / safe_total)
            )

        spectral_rows.append(row)

        if (i + 1) % 100 == 0:
            print(f"Analyzed spectra for {i + 1}/{n_trials} trials")

    spectral_df = pd.DataFrame(spectral_rows)
    spectral_df.to_csv(
        OUT_DIR / "trial_relative_spectral_power.csv",
        index=False,
    )

    relative_cols = [
        c for c in spectral_df.columns if c.endswith("_relative_power")
    ]

    relative_summary = (
        spectral_df.groupby(["subject", "class_name"])[relative_cols]
        .mean()
        .reset_index()
    )
    relative_summary.to_csv(
        OUT_DIR / "relative_power_by_subject_class.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 5. Audit timing metadata against extraction audit
    # --------------------------------------------------------
    audit_info = {
        "audit_file_exists": DS2_AUDIT_PATH.exists(),
        "audit_rows": None,
        "rows_match": None,
        "note": (
            "Matching row counts alone do not prove correct event timing. "
            "The epoch extraction code and original recording events must "
            "be inspected to establish the exact epoch start/end relative "
            "to the imagery onset."
        ),
    }

    if DS2_AUDIT_PATH.exists():
        audit = pd.read_csv(DS2_AUDIT_PATH)
        audit_info["audit_rows"] = int(len(audit))
        audit_info["rows_match"] = len(audit) == n_trials
        audit_info["audit_columns"] = audit.columns.tolist()

    # --------------------------------------------------------
    # 6. Summary report
    # --------------------------------------------------------
    summary = {
        "experiment": 108,
        "dataset": "ds004022",
        "n_trials": int(n_trials),
        "n_channels": int(n_channels),
        "n_samples_per_epoch": int(n_samples),
        "sampling_rate_hz": sampling_rate,
        "epoch_duration_seconds": duration_seconds,
        "expected_sampling_rate_hz": EXPECTED_FS,
        "sampling_rate_matches_expected": (
            abs(sampling_rate - EXPECTED_FS) < 1e-9
        ),
        "channel_names": channel_names,
        "subjects": sorted(np.unique(subjects).tolist()),
        "class_counts": {
            class_names[int(c)]: int(np.sum(y == c))
            for c in np.unique(y)
        },
        "all_values_finite": bool(np.isfinite(X).all()),
        "global_amplitude_quantiles": {
            str(q): float(np.quantile(np.abs(X), q))
            for q in [0.5, 0.9, 0.99, 0.999, 1.0]
        },
        "timing_audit": audit_info,
        "interpretation": (
            "This audit summarizes the supplied epochs. It cannot by itself "
            "establish whether the chosen time window is optimal or whether "
            "the labels are biologically separable."
        ),
    }

    with open(
        OUT_DIR / "experiment108_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    print("\nEXPERIMENT 108 COMPLETE")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print("\nOutputs:", OUT_DIR)


if __name__ == "__main__":
    main()
