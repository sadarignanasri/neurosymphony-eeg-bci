from pathlib import Path

import mne
import numpy as np
import pandas as pd


# ============================================================
# STEP 44
# ALL 720 TRIALS -> 64 CHANNELS -> 5 BAND POWER FEATURES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUBJECT = "sub-01"

BDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / SUBJECT
    / f"{SUBJECT}_task-eeg_eeg.bdf"
)

EVENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / SUBJECT
    / f"{SUBJECT}_task-eeg_events.tsv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "eeg"
    / "band_analysis"
    / "step44"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EEG SETTINGS
# ============================================================

SFREQ = 2048.0

TRIAL_SAMPLES = 6142

TRIAL_DURATION = TRIAL_SAMPLES / SFREQ

EPSILON = 1e-20


# ============================================================
# FIVE EEG FREQUENCY BANDS
# ============================================================

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40),
}


# ============================================================
# VERIFIED DATASET CLASS NAMES
# ============================================================

CLASS_ORDER = [
    "animal_auditory_imagery_task",
    "animal_silent_naming_task",
    "animal_tactile_imagery_task",
    "animal_visual_imagery_task",
    "tool_auditory_imagery_task",
    "tool_silent_naming_task",
    "tool_tactile_imagery_task",
    "tool_visual_imagery_task",
]


DISPLAY_NAMES = {
    "animal_auditory_imagery_task":
        "Animal - Auditory",

    "animal_silent_naming_task":
        "Animal - Silent",

    "animal_tactile_imagery_task":
        "Animal - Tactile",

    "animal_visual_imagery_task":
        "Animal - Visual",

    "tool_auditory_imagery_task":
        "Tool - Auditory",

    "tool_silent_naming_task":
        "Tool - Silent",

    "tool_tactile_imagery_task":
        "Tool - Tactile",

    "tool_visual_imagery_task":
        "Tool - Visual",
}


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 44 — ALL TRIALS BAND-POWER FEATURE EXTRACTION")
print("=" * 70)

print(f"\nSubject : {SUBJECT}")

print(f"\nBDF:")
print(BDF_PATH)

print("\nEvents:")
print(EVENTS_PATH)


# ============================================================
# CHECK FILES
# ============================================================

if not BDF_PATH.exists():
    raise FileNotFoundError(
        f"BDF file not found:\n{BDF_PATH}"
    )

if not EVENTS_PATH.exists():
    raise FileNotFoundError(
        f"Events file not found:\n{EVENTS_PATH}"
    )


# ============================================================
# LOAD EVENTS.TSV
# ============================================================

print("\n" + "=" * 70)
print("LOADING EVENTS.TSV")
print("=" * 70)

events_df = pd.read_csv(
    EVENTS_PATH,
    sep="\t"
)

print(
    f"\nTotal event rows: "
    f"{len(events_df)}"
)

print(
    "\nEvent columns:"
)

print(
    events_df.columns.tolist()
)


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

if "trial_type" not in events_df.columns:
    raise ValueError(
        "'trial_type' column not found in events.tsv."
    )

if "sample" not in events_df.columns:
    raise ValueError(
        "'sample' column not found in events.tsv."
    )


# ============================================================
# SELECT ONLY THE 8 VERIFIED TASK CLASSES
# ============================================================

task_df = events_df[
    events_df["trial_type"].isin(CLASS_ORDER)
].copy()

task_df = task_df.reset_index(
    drop=True
)

print(
    f"\nTask trials found: "
    f"{len(task_df)}"
)


# ============================================================
# VALIDATE 720 TRIALS
# ============================================================

class_counts = (
    task_df["trial_type"]
    .value_counts()
)

print("\nClass counts:")

for class_name in CLASS_ORDER:

    count = int(
        class_counts.get(
            class_name,
            0
        )
    )

    print(
        f"  {class_name:<40} {count}"
    )


if len(task_df) != 720:

    raise ValueError(
        f"Expected 720 task trials, "
        f"found {len(task_df)}."
    )


for class_name in CLASS_ORDER:

    count = int(
        class_counts.get(
            class_name,
            0
        )
    )

    if count != 90:

        raise ValueError(
            f"{class_name}: "
            f"expected 90 trials, "
            f"found {count}."
        )


print(
    "\n✓ 720 task trials confirmed"
)

print(
    "✓ 8 classes confirmed"
)

print(
    "✓ 90 trials per class confirmed"
)


# ============================================================
# LOAD ORIGINAL BDF
# ============================================================

print("\n" + "=" * 70)
print("LOADING ORIGINAL BDF")
print("=" * 70)

print(
    "\nLoading BDF without preloading "
    "the entire recording into RAM..."
)

raw = mne.io.read_raw_bdf(
    BDF_PATH,
    preload=False,
    verbose="ERROR"
)

print(
    "\nOriginal recording:"
)

print(raw)

print(
    f"\nSampling frequency: "
    f"{raw.info['sfreq']} Hz"
)

print(
    f"Recording duration: "
    f"{raw.times[-1]:.3f} seconds"
)


# ============================================================
# VERIFY SAMPLING RATE
# ============================================================

if not np.isclose(
    raw.info["sfreq"],
    SFREQ
):

    raise ValueError(
        f"Expected sampling rate "
        f"{SFREQ} Hz, "
        f"found {raw.info['sfreq']} Hz."
    )


# ============================================================
# IDENTIFY THE 64 EEG CHANNELS
# ============================================================

eeg_channels = [
    ch
    for ch in raw.ch_names
    if ch.startswith("A")
    or ch.startswith("B")
]

if len(eeg_channels) != 64:

    raise ValueError(
        f"Expected 64 EEG channels, "
        f"found {len(eeg_channels)}."
    )


print(
    f"\n✓ EEG channels found: "
    f"{len(eeg_channels)}"
)


# ============================================================
# FEATURE STORAGE
# ============================================================

feature_rows = []


# ============================================================
# PROCESS ALL 720 TRIALS
# ============================================================

print("\n" + "=" * 70)
print("EXTRACTING BAND-POWER FEATURES")
print("=" * 70)

print(
    "\nFeatures per trial: "
    "64 channels × 5 bands = 320"
)

print(
    f"Trial duration: "
    f"{TRIAL_DURATION:.6f} seconds"
)


for trial_number, row in task_df.iterrows():

    trial_number = trial_number + 1

    class_name = row["trial_type"]

    display_name = DISPLAY_NAMES[
        class_name
    ]

    # --------------------------------------------------------
    # IMPORTANT:
    # Use the verified sample column from events.tsv.
    # --------------------------------------------------------

    start_sample = int(
        row["sample"]
    )

    stop_sample = (
        start_sample
        + TRIAL_SAMPLES
    )

    # --------------------------------------------------------
    # Recording boundary check
    # --------------------------------------------------------

    if stop_sample > raw.n_times:

        raise ValueError(
            f"Trial {trial_number} "
            f"({display_name}) exceeds "
            f"recording boundary."
        )

    # --------------------------------------------------------
    # Read ONLY this trial from BDF
    # --------------------------------------------------------

    trial_data = raw.get_data(
        picks=eeg_channels,
        start=start_sample,
        stop=stop_sample
    )

    # --------------------------------------------------------
    # Validate trial shape
    # --------------------------------------------------------

    if trial_data.shape != (
        64,
        TRIAL_SAMPLES
    ):

        raise ValueError(
            f"Trial {trial_number}: "
            f"unexpected shape "
            f"{trial_data.shape}"
        )

    # --------------------------------------------------------
    # Create temporary RawArray
    # --------------------------------------------------------

    info = mne.create_info(
        ch_names=eeg_channels,
        sfreq=SFREQ,
        ch_types=["eeg"] * 64
    )

    trial_raw = mne.io.RawArray(
        trial_data,
        info,
        verbose=False
    )

    # --------------------------------------------------------
    # Welch PSD
    #
    # We calculate spectral power from the raw trial.
    # No incorrect Status-code mapping is used.
    # --------------------------------------------------------

    spectrum = trial_raw.compute_psd(
        method="welch",
        fmin=1,
        fmax=40,
        n_fft=4096,
        n_overlap=2048,
        verbose=False
    )

    psd = spectrum.get_data()

    frequencies = spectrum.freqs

    # --------------------------------------------------------
    # Create metadata
    # --------------------------------------------------------

    feature_row = {

        "subject":
            SUBJECT,

        "trial":
            trial_number,

        "label":
            class_name,

        "label_name":
            display_name,

        "event_row":
            int(row.name),

        "start_sample":
            start_sample,

        "duration_seconds":
            TRIAL_DURATION,
    }

    # --------------------------------------------------------
    # Extract 5-band power from all 64 channels
    # --------------------------------------------------------

    for channel_index, channel_name in enumerate(
        eeg_channels
    ):

        channel_psd = psd[
            channel_index
        ]

        for band_name, (
            low_freq,
            high_freq
        ) in BANDS.items():

            # --------------------------------------------
            # Frequency mask
            # --------------------------------------------

            band_mask = (
                (frequencies >= low_freq)
                &
                (frequencies < high_freq)
            )

            if not np.any(
                band_mask
            ):

                power = 0.0

            else:

                power = np.trapezoid(
                    channel_psd[
                        band_mask
                    ],
                    frequencies[
                        band_mask
                    ]
                )

            # --------------------------------------------
            # Feature names
            # --------------------------------------------

            power_name = (
                f"{channel_name}_"
                f"{band_name}_power"
            )

            logpower_name = (
                f"{channel_name}_"
                f"{band_name}_logpower"
            )

            # --------------------------------------------
            # Store features
            # --------------------------------------------

            feature_row[
                power_name
            ] = power

            feature_row[
                logpower_name
            ] = np.log(
                power + EPSILON
            )

    feature_rows.append(
        feature_row
    )

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if trial_number % 25 == 0:

        print(
            f"Processed "
            f"{trial_number}/720 trials"
        )


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    feature_rows
)

print(
    "\nFinal dataframe shape:"
)

print(
    df.shape
)


# ============================================================
# IDENTIFY FEATURE COLUMNS
# ============================================================

metadata_columns = [
    "subject",
    "trial",
    "label",
    "label_name",
    "event_row",
    "start_sample",
    "duration_seconds",
]

power_columns = [
    column
    for column in df.columns
    if column.endswith("_power")
    and not column.endswith("_logpower")
]

logpower_columns = [
    column
    for column in df.columns
    if column.endswith("_logpower")
]


# ============================================================
# VALIDATE FEATURE COUNT
# ============================================================

print(
    f"\nPower features: "
    f"{len(power_columns)}"
)

print(
    f"Log-power features: "
    f"{len(logpower_columns)}"
)


if len(power_columns) != 320:

    raise ValueError(
        f"Expected 320 power features, "
        f"found {len(power_columns)}."
    )


if len(logpower_columns) != 320:

    raise ValueError(
        f"Expected 320 log-power features, "
        f"found {len(logpower_columns)}."
    )


# ============================================================
# CHECK FOR NaN / INFINITE VALUES
# ============================================================

feature_values = df[
    power_columns
    + logpower_columns
].to_numpy()


if np.isnan(
    feature_values
).any():

    raise ValueError(
        "NaN values found "
        "in feature matrix."
    )


if np.isinf(
    feature_values
).any():

    raise ValueError(
        "Infinite values found "
        "in feature matrix."
    )


print(
    "\n✓ No NaN values"
)

print(
    "✓ No infinite values"
)


# ============================================================
# CLASS DISTRIBUTION CHECK
# ============================================================

print(
    "\nFinal class distribution:"
)

print(
    df["label_name"]
    .value_counts()
)


# ============================================================
# SAVE COMPLETE MATRIX
# ============================================================

complete_path = (
    OUTPUT_DIR
    / "step44_sub01_band_power_features.csv"
)

df.to_csv(
    complete_path,
    index=False
)


# ============================================================
# SAVE POWER-ONLY MATRIX
# ============================================================

power_df = df[
    metadata_columns
    + power_columns
]

power_path = (
    OUTPUT_DIR
    / "step44_sub01_band_power_only.csv"
)

power_df.to_csv(
    power_path,
    index=False
)


# ============================================================
# SAVE LOG-POWER MATRIX
# ============================================================

logpower_df = df[
    metadata_columns
    + logpower_columns
]

logpower_path = (
    OUTPUT_DIR
    / "step44_sub01_band_logpower.csv"
)

logpower_df.to_csv(
    logpower_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 44 COMPLETE")
print("=" * 70)

print(
    f"\nTrials processed : {len(df)}"
)

print(
    f"Power features   : "
    f"{len(power_columns)}"
)

print(
    f"Log-power features : "
    f"{len(logpower_columns)}"
)

print(
    "\nExpected:"
)

print(
    "720 trials × 320 power features"
)

print(
    "720 trials × 320 log-power features"
)

print(
    "\nSaved files:"
)

print(complete_path)

print(power_path)

print(logpower_path)

print(
    "\n✓ Events taken from events.tsv"
)

print(
    "✓ Actual trial_type labels preserved"
)

print(
    "✓ 720 trials verified"
)

print(
    "✓ 90 trials per class verified"
)

print(
    "✓ 64 EEG channels verified"
)

print(
    "✓ 320 power features verified"
)