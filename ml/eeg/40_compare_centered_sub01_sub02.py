"""
Step 40 — Baseline-centered cross-subject raw EEG comparison

Subjects:
    sub-01 vs sub-02

Channel:
    Fz = BioSemi B6

Processing:
    Original raw EEG
    Remove each trial's own mean
    No filtering
    No FFT
    No ML

Purpose:
    Determine whether cross-subject differences observed in
    Step 39 are mainly due to baseline/DC offset or whether
    temporal waveform differences remain after centering.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUBJECTS = ["sub-01", "sub-02"]

CHANNEL = "B6"
STANDARD_CHANNEL = "Fz"

SFREQ = 2048
TRIAL_SAMPLES = 6142
DURATION = TRIAL_SAMPLES / SFREQ

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "band_analysis"
    / "cross_subject"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EIGHT COMMON CLASS KEYS
# ============================================================

CLASS_NAMES = [
    "animal_auditory_imagery_task",
    "animal_silent",
    "animal_tactile_imagery_task",
    "animal_visual_imagery_task",
    "tool_auditory_imagery_task",
    "tool_silent",
    "tool_tactile_imagery_task",
    "tool_visual_imagery_task",
]


# ============================================================
# ACTUAL DATASET LABELS
# ============================================================

SUBJECT_LABELS = {

    "sub-01": {
        "animal_auditory_imagery_task":
            "animal_auditory_imagery_task",

        "animal_silent":
            "animal_silent_naming_task",

        "animal_tactile_imagery_task":
            "animal_tactile_imagery_task",

        "animal_visual_imagery_task":
            "animal_visual_imagery_task",

        "tool_auditory_imagery_task":
            "tool_auditory_imagery_task",

        "tool_silent":
            "tool_silent_naming_task",

        "tool_tactile_imagery_task":
            "tool_tactile_imagery_task",

        "tool_visual_imagery_task":
            "tool_visual_imagery_task",
    },

    "sub-02": {
        "animal_auditory_imagery_task":
            "animal_auditory_imagery_task",

        "animal_silent":
            "animal_silent_naming_task",

        "animal_tactile_imagery_task":
            "animal_tactile_imagery_task",

        "animal_visual_imagery_task":
            "animal_visual_imagery_task",

        "tool_auditory_imagery_task":
            "tool_auditory_imagery_task",

        "tool_silent":
            "tool_silent_naming_task",

        "tool_tactile_imagery_task":
            "tool_tactile_imagery_task",

        "tool_visual_imagery_task":
            "tool_visual_imagery_task",
    }
}


# ============================================================
# DISPLAY NAMES
# ============================================================

DISPLAY_NAMES = {
    "animal_auditory_imagery_task":
        "Animal - Auditory",

    "animal_silent":
        "Animal - Silent",

    "animal_tactile_imagery_task":
        "Animal - Tactile",

    "animal_visual_imagery_task":
        "Animal - Visual",

    "tool_auditory_imagery_task":
        "Tool - Auditory",

    "tool_silent":
        "Tool - Silent",

    "tool_tactile_imagery_task":
        "Tool - Tactile",

    "tool_visual_imagery_task":
        "Tool - Visual",
}


# ============================================================
# LOAD ONE SUBJECT
# ============================================================

def load_subject(subject):

    print("-" * 75)
    print(f"Loading {subject}")
    print("-" * 75)

    bdf_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / subject
        / f"{subject}_task-eeg_eeg.bdf"
    )

    events_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / subject
        / f"{subject}_task-eeg_events.tsv"
    )

    if not bdf_path.exists():
        raise FileNotFoundError(
            f"BDF not found:\n{bdf_path}"
        )

    if not events_path.exists():
        raise FileNotFoundError(
            f"Events file not found:\n{events_path}"
        )

    events = pd.read_csv(
        events_path,
        sep="\t"
    )

    label_mapping = SUBJECT_LABELS[subject]

    actual_labels = list(
        label_mapping.values()
    )

    task_df = events[
        events["trial_type"].isin(actual_labels)
    ].copy()

    task_df = task_df.reset_index(drop=True)

    print(
        f"Task trials found: {len(task_df)}"
    )

    if len(task_df) != 720:
        raise ValueError(
            f"{subject}: expected 720 task trials, "
            f"found {len(task_df)}"
        )

    raw = mne.io.read_raw_bdf(
        bdf_path,
        preload=False,
        verbose="ERROR"
    )

    if raw.info["sfreq"] != SFREQ:
        raise ValueError(
            f"{subject}: expected {SFREQ} Hz."
        )

    if CHANNEL not in raw.ch_names:
        raise ValueError(
            f"{subject}: {CHANNEL} not found."
        )

    channel_index = raw.ch_names.index(
        CHANNEL
    )

    results = {}

    for class_key in CLASS_NAMES:

        actual_label = label_mapping[
            class_key
        ]

        rows = task_df[
            task_df["trial_type"] == actual_label
        ]

        if len(rows) != 90:
            raise ValueError(
                f"{subject} / {actual_label}: "
                f"expected 90 trials, found {len(rows)}"
            )

        # Same deterministic rule as Step 39:
        # first occurrence of each class.
        first_row = rows.iloc[0]

        onset = float(
            first_row["onset"]
        )

        start = int(
            round(onset * SFREQ)
        )

        stop = start + TRIAL_SAMPLES

        data, _ = raw[
            [channel_index],
            start:stop
        ]

        signal = data[0].astype(
            np.float64
        )

        if len(signal) != TRIAL_SAMPLES:
            raise ValueError(
                f"{subject} / {class_key}: "
                "incorrect trial length."
            )

        # ====================================================
        # BASELINE CENTERING
        # ====================================================

        trial_mean = float(
            np.mean(signal)
        )

        centered_signal = (
            signal - trial_mean
        )

        # Verify centering
        centered_mean = float(
            np.mean(centered_signal)
        )

        if not np.isclose(
            centered_mean,
            0.0,
            atol=1e-15
        ):
            raise ValueError(
                f"{subject} / {class_key}: "
                "centering failed."
            )

        results[class_key] = {
            "raw_signal": signal,
            "centered_signal": centered_signal,
            "original_mean": trial_mean,
            "onset": onset,
            "start_sample": start,
            "event_row": int(first_row.name),
        }

    print(
        f"✓ {subject}: 8 centered trials extracted"
    )

    return results


# ============================================================
# LOAD BOTH SUBJECTS
# ============================================================

print("=" * 75)
print("STEP 40 — BASELINE-CENTERED CROSS-SUBJECT EEG")
print("=" * 75)

print("Subjects      : sub-01 vs sub-02")
print("Channel       : Fz (B6)")
print(f"Sampling rate : {SFREQ} Hz")
print(f"Samples/trial : {TRIAL_SAMPLES}")
print(f"Duration      : {DURATION:.6f} seconds")
print("Processing    : subtract trial mean")
print("Filtering     : NONE")
print("FFT           : NONE")
print("ML            : NONE")
print()

subject_data = {}

for subject in SUBJECTS:

    subject_data[subject] = load_subject(
        subject
    )


print("\n✓ Both subjects loaded successfully.")


# ============================================================
# TIME AXIS
# ============================================================

time = (
    np.arange(TRIAL_SAMPLES)
    / SFREQ
)


# ============================================================
# COMPARE CENTERED SIGNALS
# ============================================================

summary_rows = []

print("\n")
print("=" * 75)
print("BASELINE-CENTERED CROSS-SUBJECT STATISTICS")
print("=" * 75)


for class_key in CLASS_NAMES:

    signal_01 = subject_data[
        "sub-01"
    ][class_key]["centered_signal"]

    signal_02 = subject_data[
        "sub-02"
    ][class_key]["centered_signal"]

    display_name = DISPLAY_NAMES[
        class_key
    ]

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    std_01 = float(
        np.std(signal_01)
    )

    std_02 = float(
        np.std(signal_02)
    )

    min_01 = float(
        np.min(signal_01)
    )

    min_02 = float(
        np.min(signal_02)
    )

    max_01 = float(
        np.max(signal_01)
    )

    max_02 = float(
        np.max(signal_02)
    )

    correlation = float(
        np.corrcoef(
            signal_01,
            signal_02
        )[0, 1]
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (signal_01 - signal_02) ** 2
            )
        )
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"\n{display_name}"
    )

    print(
        f"  sub-01 centered std : {std_01:.6e}"
    )

    print(
        f"  sub-02 centered std : {std_02:.6e}"
    )

    print(
        f"  sub-01 min/max      : "
        f"{min_01:.6e} / {max_01:.6e}"
    )

    print(
        f"  sub-02 min/max      : "
        f"{min_02:.6e} / {max_02:.6e}"
    )

    print(
        f"  correlation         : {correlation:.4f}"
    )

    print(
        f"  RMSE                : {rmse:.6e}"
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(14, 5)
    )

    ax.plot(
        time,
        signal_01,
        linewidth=0.8,
        label="sub-01 centered"
    )

    ax.plot(
        time,
        signal_02,
        linewidth=0.8,
        label="sub-02 centered"
    )

    ax.axhline(
        0,
        linewidth=0.7,
        linestyle="--"
    )

    ax.set_title(
        f"{display_name}\n"
        f"Baseline-Centered Raw EEG — Fz"
    )

    ax.set_xlabel(
        "Time (seconds)"
    )

    ax.set_ylabel(
        "Centered Amplitude (V)"
    )

    ax.set_xlim(
        0,
        DURATION
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"step40_{class_key}_"
          f"centered_sub01_vs_sub02.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Save statistics
    # --------------------------------------------------------

    summary_rows.append(
        {
            "class": class_key,
            "display_name": display_name,

            "sub01_original_mean":
                subject_data[
                    "sub-01"
                ][class_key]["original_mean"],

            "sub02_original_mean":
                subject_data[
                    "sub-02"
                ][class_key]["original_mean"],

            "sub01_centered_mean":
                float(np.mean(signal_01)),

            "sub02_centered_mean":
                float(np.mean(signal_02)),

            "sub01_centered_std":
                std_01,

            "sub02_centered_std":
                std_02,

            "sub01_centered_min":
                min_01,

            "sub02_centered_min":
                min_02,

            "sub01_centered_max":
                max_01,

            "sub02_centered_max":
                max_02,

            "pearson_correlation":
                correlation,

            "rmse":
                rmse,
        }
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_path = (
    OUTPUT_DIR
    / "step40_centered_sub01_vs_sub02_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# CORRELATION COMPARISON
# ============================================================

fig, ax = plt.subplots(
    figsize=(14, 6)
)

x = np.arange(
    len(summary_df)
)

ax.bar(
    x,
    summary_df["pearson_correlation"]
)

ax.axhline(
    0,
    linewidth=0.8,
    linestyle="--"
)

ax.set_xticks(x)

ax.set_xticklabels(
    summary_df["display_name"],
    rotation=30,
    ha="right"
)

ax.set_ylabel(
    "Pearson Correlation"
)

ax.set_title(
    "Baseline-Centered Cross-Subject EEG Similarity\n"
    "sub-01 vs sub-02 — Fz"
)

ax.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

correlation_path = (
    OUTPUT_DIR
    / "step40_centered_correlation_comparison.png"
)

plt.savefig(
    correlation_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("STEP 40 COMPLETE")
print("=" * 75)

print("\nGenerated centered comparison plots:")
for path in sorted(
    OUTPUT_DIR.glob(
        "step40_*centered_sub01_vs_sub02.png"
    )
):
    print(f"  {path}")

print("\nSummary:")
print(f"  {summary_path}")

print("\nCorrelation comparison:")
print(f"  {correlation_path}")

print("\n✓ Original raw EEG preserved")
print("✓ Trial-wise mean removed only for this analysis")
print("✓ Same Fz channel")
print("✓ Same 6142-sample trial length")
print("✓ Same first-occurrence selection")
print("✓ No band-pass filtering")
print("✓ No FFT")
print("✓ No ML")