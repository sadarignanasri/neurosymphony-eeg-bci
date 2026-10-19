"""
===============================================================================
Neuro-Symphony
Step 31: All 12 Subjects — Memory-Safe Temporal EEG Preprocessing
===============================================================================

Purpose
-------
Create clean temporal EEG trials from the original DS004514 BDF recordings.

Pipeline
--------
Original BDF
    ↓
Original task events
    ↓
Memory-safe temporal chunks
    ↓
Existing project EEG preprocessing
    ↓
64 EEG channels
    ↓
720 labelled temporal trials
    ↓
6142 samples/trial
    ↓
float32 temporal EEG arrays

Important
---------
- No FFT is performed here.
- No machine-learning model is trained here.
- No train/test split is performed here.
- The existing project preprocessing pipeline is used.
- BDF files are NOT loaded completely into RAM.
- EEG preprocessing is performed on manageable overlapping chunks.
- 15 seconds of context is used around each 120-second core chunk.
- Temporal output is written directly to a disk-backed .npy file.
- One subject is processed at a time.
- Already completed valid subjects are skipped.
===============================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================

from pathlib import Path
import gc
import sys
import shutil
import traceback

import numpy as np
import pandas as pd
import mne


# ============================================================================
# PROJECT ROOT
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.preprocessing.eeg import purify_eeg


# ============================================================================
# CONFIGURATION
# ============================================================================

RAW_ROOT = PROJECT_ROOT / "data" / "raw"
FEATURE_ROOT = PROJECT_ROOT / "data" / "features"


SUBJECTS = [
    f"sub-{i:02d}"
    for i in range(1, 13)
]


EXPECTED_TRIALS = 720
EXPECTED_CLASSES = 8
TRIALS_PER_CLASS = 90

SAMPLING_RATE = 2048.0


# ============================================================================
# TRIAL CONFIGURATION
# ============================================================================

TRIAL_SAMPLES = 6142

TRIAL_DURATION_SECONDS = (
    TRIAL_SAMPLES / SAMPLING_RATE
)


# ============================================================================
# MEMORY-SAFE CHUNK CONFIGURATION
# ============================================================================

# The central/core portion of each chunk.
CHUNK_DURATION_SECONDS = 120.0

# Extra data before and after each core chunk.
#
# This provides enough context for the project's FIR filtering.
CONTEXT_SECONDS = 15.0


# ============================================================================
# EXPECTED TASK LABELS
# ============================================================================

EXPECTED_LABELS = [
    "animal_tactile_imagery_task",
    "animal_visual_imagery_task",
    "animal_auditory_imagery_task",
    "animal_silent_naming_task",
    "tool_tactile_imagery_task",
    "tool_visual_imagery_task",
    "tool_auditory_imagery_task",
    "tool_silent_naming_task",
]


LABEL_TO_ID = {
    label: index
    for index, label in enumerate(EXPECTED_LABELS)
}


# ============================================================================
# EXPECTED EEG CHANNELS
# ============================================================================

BIOSEMI_EEG_CHANNELS = (
    [f"A{i}" for i in range(1, 33)]
    + [f"B{i}" for i in range(1, 33)]
)

EXPECTED_EEG_COUNT = 64


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(title):
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def print_subheader(title):
    print()
    print("-" * 90)
    print(title)
    print("-" * 90)


# ============================================================================
# EVENT LOADING
# ============================================================================

def load_task_events(events_path):
    """
    Load events.tsv and return the 720 task events.
    """

    print_subheader("LOADING EVENTS")

    events = pd.read_csv(
        events_path,
        sep="\t"
    )

    print(
        f"Total event rows: {len(events)}"
    )

    if "trial_type" not in events.columns:
        raise ValueError(
            "events.tsv does not contain the required "
            "'trial_type' column."
        )

    task_events = events[
        events["trial_type"].isin(EXPECTED_LABELS)
    ].copy()

    task_events = task_events.reset_index(
        drop=True
    )

    print(
        f"Task trials found: {len(task_events)}"
    )

    if len(task_events) != EXPECTED_TRIALS:
        raise ValueError(
            f"Expected {EXPECTED_TRIALS} task trials, "
            f"but found {len(task_events)}."
        )

    print(
        f"✓ Exactly {EXPECTED_TRIALS} task trials."
    )

    # ------------------------------------------------------------------------
    # Validate classes
    # ------------------------------------------------------------------------

    print_subheader("CLASS COUNTS")

    counts = task_events[
        "trial_type"
    ].value_counts()

    for label in EXPECTED_LABELS:

        count = int(
            counts.get(label, 0)
        )

        print(
            f"{label}: {count}"
        )

        if count != TRIALS_PER_CLASS:
            raise ValueError(
                f"Class '{label}' contains {count} trials. "
                f"Expected {TRIALS_PER_CLASS}."
            )

    if len(counts) != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_CLASSES} classes, "
            f"but found {len(counts)}."
        )

    print()
    print(
        f"✓ All {EXPECTED_CLASSES} classes contain "
        f"{TRIALS_PER_CLASS} trials."
    )

    return task_events


# ============================================================================
# EVENT SAMPLE
# ============================================================================

def get_event_sample(event_row, sfreq):
    """
    Convert event onset into a sample index.

    The original 'sample' column is preferred.
    """

    if "sample" in event_row.index:

        sample_value = event_row["sample"]

        if pd.notna(sample_value):
            return int(
                round(
                    float(sample_value)
                )
            )

    if "onset" not in event_row.index:
        raise ValueError(
            "events.tsv must contain either "
            "'sample' or 'onset'."
        )

    onset = float(
        event_row["onset"]
    )

    return int(
        round(
            onset * sfreq
        )
    )


# ============================================================================
# VALIDATE ORIGINAL BDF
# ============================================================================

def validate_raw_structure(raw, subject):
    """
    Validate the original BDF structure.
    """

    print_subheader(
        "VALIDATING ORIGINAL BDF"
    )

    print(
        f"Original channels: {len(raw.ch_names)}"
    )

    print(
        f"Original sampling rate: "
        f"{raw.info['sfreq']:.6f} Hz"
    )

    print(
        f"Original samples: {raw.n_times}"
    )

    if len(raw.ch_names) != 80:
        raise ValueError(
            f"{subject}: expected 80 original channels, "
            f"found {len(raw.ch_names)}."
        )

    if not np.isclose(
        raw.info["sfreq"],
        SAMPLING_RATE
    ):
        raise ValueError(
            f"{subject}: expected sampling rate "
            f"{SAMPLING_RATE}, "
            f"found {raw.info['sfreq']}."
        )

    missing_eeg = [
        channel
        for channel in BIOSEMI_EEG_CHANNELS
        if channel not in raw.ch_names
    ]

    if missing_eeg:
        raise ValueError(
            f"{subject}: missing BioSemi EEG channels: "
            f"{missing_eeg}"
        )

    if "Status" not in raw.ch_names:
        raise ValueError(
            f"{subject}: Status channel not found."
        )

    print(
        "✓ Original BDF structure verified."
    )


# ============================================================================
# OUTPUT PATHS
# ============================================================================

def get_output_paths(subject):

    output_dir = (
        FEATURE_ROOT
        / f"temporal_eeg_{subject}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    final_temporal_path = (
        output_dir
        / f"{subject}_temporal_eeg_preprocessed.npy"
    )

    partial_temporal_path = (
        output_dir
        / f"{subject}_temporal_eeg_preprocessed.partial.npy"
    )

    metadata_path = (
        output_dir
        / f"{subject}_temporal_eeg_preprocessed_metadata.csv"
    )

    channel_path = (
        output_dir
        / "preprocessed_channel_names.txt"
    )

    return (
        output_dir,
        final_temporal_path,
        partial_temporal_path,
        metadata_path,
        channel_path
    )


# ============================================================================
# CHECK COMPLETED SUBJECT
# ============================================================================

def check_existing_subject(subject):

    (
        output_dir,
        temporal_path,
        partial_path,
        metadata_path,
        channel_path
    ) = get_output_paths(subject)

    if not temporal_path.exists():
        return False

    if not metadata_path.exists():
        return False

    if not channel_path.exists():
        return False

    try:

        data = np.load(
            temporal_path,
            mmap_mode="r"
        )

        expected_shape = (
            EXPECTED_TRIALS,
            EXPECTED_EEG_COUNT,
            TRIAL_SAMPLES
        )

        if data.shape != expected_shape:
            print(
                f"Existing {subject} output has "
                f"unexpected shape: {data.shape}"
            )
            return False

        if data.dtype != np.float32:
            print(
                f"Existing {subject} output has "
                f"unexpected dtype: {data.dtype}"
            )
            return False

        metadata = pd.read_csv(
            metadata_path
        )

        if len(metadata) != EXPECTED_TRIALS:
            print(
                f"Existing {subject} metadata has "
                f"{len(metadata)} rows."
            )
            return False

        del data
        gc.collect()

        print()
        print(
            f"✓ {subject} already has a valid "
            f"temporal EEG dataset."
        )

        print(
            "  Skipping reprocessing."
        )

        return True

    except Exception as error:

        print(
            f"Existing {subject} output could not be "
            f"validated: {error}"
        )

        return False


# ============================================================================
# CREATE DISK-BACKED OUTPUT
# ============================================================================

def create_temporal_output(partial_path):

    expected_shape = (
        EXPECTED_TRIALS,
        EXPECTED_EEG_COUNT,
        TRIAL_SAMPLES
    )

    if partial_path.exists():
        partial_path.unlink()

    print()
    print(
        "Creating disk-backed temporal output:"
    )

    print(
        f"  Shape: {expected_shape}"
    )

    print(
        "  Dtype: float32"
    )

    temporal_data = np.lib.format.open_memmap(
        partial_path,
        mode="w+",
        dtype=np.float32,
        shape=expected_shape
    )

    return temporal_data


# ============================================================================
# PROCESS ONE SUBJECT
# ============================================================================

def process_subject(subject):

    print_header(
        f"PROCESSING {subject}"
    )

    subject_dir = (
        RAW_ROOT
        / subject
    )

    bdf_path = (
        subject_dir
        / f"{subject}_task-eeg_eeg.bdf"
    )

    events_path = (
        subject_dir
        / f"{subject}_task-eeg_events.tsv"
    )

    (
        output_dir,
        final_temporal_path,
        partial_temporal_path,
        metadata_path,
        channel_path
    ) = get_output_paths(subject)

    # ------------------------------------------------------------------------
    # Skip already completed valid subject
    # ------------------------------------------------------------------------

    if check_existing_subject(subject):
        return True

    # ------------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------------

    print_subheader(
        "CHECKING INPUT FILES"
    )

    if not bdf_path.exists():
        raise FileNotFoundError(
            f"BDF not found:\n{bdf_path}"
        )

    if not events_path.exists():
        raise FileNotFoundError(
            f"Events TSV not found:\n{events_path}"
        )

    print(
        f"✓ BDF found: {bdf_path.name}"
    )

    print(
        f"✓ Events TSV found: {events_path.name}"
    )

    # ------------------------------------------------------------------------
    # Load events
    # ------------------------------------------------------------------------

    task_events = load_task_events(
        events_path
    )

    # ------------------------------------------------------------------------
    # Open BDF WITHOUT preloading the entire recording
    # ------------------------------------------------------------------------

    print_subheader(
        "OPENING ORIGINAL BDF"
    )

    print(
        "Mode: preload=False"
    )

    print(
        "The complete BDF will NOT be loaded into RAM."
    )

    raw = mne.io.read_raw_bdf(
        bdf_path,
        preload=False,
        verbose=True
    )

    # ------------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------------

    validate_raw_structure(
        raw,
        subject
    )

    sfreq = float(
        raw.info["sfreq"]
    )

    recording_samples = int(
        raw.n_times
    )

    recording_duration = (
        recording_samples / sfreq
    )

    print()
    print(
        f"Recording duration: "
        f"{recording_duration:.2f} seconds"
    )

    # ------------------------------------------------------------------------
    # Create output memmap
    # ------------------------------------------------------------------------

    temporal_data = create_temporal_output(
        partial_temporal_path
    )

    metadata_rows = []

    saved_trial_indices = set()

    channel_names = None

    # ------------------------------------------------------------------------
    # Determine chunk count
    # ------------------------------------------------------------------------

    chunk_samples = int(
        round(
            CHUNK_DURATION_SECONDS
            * sfreq
        )
    )

    context_samples = int(
        round(
            CONTEXT_SECONDS
            * sfreq
        )
    )

    total_chunks = int(
        np.ceil(
            recording_samples
            / chunk_samples
        )
    )

    print()
    print(
        "Chunk configuration:"
    )

    print(
        f"  Core duration: "
        f"{CHUNK_DURATION_SECONDS:.1f} sec"
    )

    print(
        f"  Context: "
        f"{CONTEXT_SECONDS:.1f} sec on each side"
    )

    print(
        f"  Total chunks: "
        f"{total_chunks}"
    )

    # =========================================================================
    # PROCESS CHUNKS
    # =========================================================================

    try:

        for chunk_number in range(
            total_chunks
        ):

            core_start_sample = (
                chunk_number
                * chunk_samples
            )

            core_end_sample = min(
                core_start_sample
                + chunk_samples,
                recording_samples
            )

            # ---------------------------------------------------------------
            # Trials assigned to this chunk
            #
            # Assignment is based on trial START sample.
            # ---------------------------------------------------------------

            chunk_trial_indices = []

            for trial_index, (_, event_row) in enumerate(
                task_events.iterrows()
            ):

                event_sample = get_event_sample(
                    event_row,
                    sfreq
                )

                if (
                    event_sample >= core_start_sample
                    and event_sample < core_end_sample
                ):

                    chunk_trial_indices.append(
                        trial_index
                    )

            if not chunk_trial_indices:

                print()
                print(
                    f"Chunk "
                    f"{chunk_number + 1}/{total_chunks}: "
                    f"no task trials"
                )

                continue

            # ---------------------------------------------------------------
            # Add context around the core chunk
            # ---------------------------------------------------------------

            load_start_sample = max(
                0,
                core_start_sample
                - context_samples
            )

            load_end_sample = min(
                recording_samples,
                core_end_sample
                + context_samples
            )

            load_start_time = (
                load_start_sample
                / sfreq
            )

            load_end_time = (
                load_end_sample
                / sfreq
            )

            print()
            print("=" * 90)
            print(
                f"CHUNK "
                f"{chunk_number + 1}/{total_chunks}"
            )
            print("=" * 90)

            print(
                f"Core samples: "
                f"{core_start_sample} → "
                f"{core_end_sample}"
            )

            print(
                f"Load samples: "
                f"{load_start_sample} → "
                f"{load_end_sample}"
            )

            print(
                f"Trials in this chunk: "
                f"{len(chunk_trial_indices)}"
            )

            print(
                f"Load duration: "
                f"{load_end_time - load_start_time:.2f} sec"
            )

            # ---------------------------------------------------------------
            # Load ONLY this chunk from BDF
            # ---------------------------------------------------------------

            chunk_raw = None
            processed_chunk = None

            try:

                print()
                print(
                    "Loading chunk into RAM..."
                )

                chunk_raw = (
                    raw.copy()
                    .crop(
                        tmin=load_start_time,
                        tmax=load_end_time,
                        include_tmax=False
                    )
                )

                chunk_raw.load_data()

                print(
                    "✓ Chunk loaded."
                )

                # -----------------------------------------------------------
                # Existing project preprocessing
                # -----------------------------------------------------------

                print()
                print(
                    "Applying existing project EEG preprocessing..."
                )

                processed_chunk = purify_eeg(
                    chunk_raw
                )

                print(
                    "✓ Chunk preprocessing completed."
                )

                # -----------------------------------------------------------
                # Identify 64 EEG channels
                # -----------------------------------------------------------

                eeg_picks = mne.pick_types(
                    processed_chunk.info,
                    eeg=True,
                    exclude=[]
                )

                if len(eeg_picks) != EXPECTED_EEG_COUNT:
                    raise ValueError(
                        f"{subject}: expected "
                        f"{EXPECTED_EEG_COUNT} EEG channels "
                        f"after preprocessing, found "
                        f"{len(eeg_picks)}."
                    )

                if channel_names is None:

                    channel_names = [
                        processed_chunk.ch_names[index]
                        for index in eeg_picks
                    ]

                    print()
                    print(
                        "✓ Exactly 64 EEG channels identified."
                    )

                # -----------------------------------------------------------
                # Extract trials from this processed chunk
                # -----------------------------------------------------------

                for trial_index in chunk_trial_indices:

                    event_row = task_events.iloc[
                        trial_index
                    ]

                    label_name = str(
                        event_row["trial_type"]
                    )

                    label_id = LABEL_TO_ID[
                        label_name
                    ]

                    absolute_start_sample = (
                        get_event_sample(
                            event_row,
                            sfreq
                        )
                    )

                    absolute_stop_sample = (
                        absolute_start_sample
                        + TRIAL_SAMPLES
                    )

                    if absolute_stop_sample > recording_samples:

                        raise ValueError(
                            f"{subject}: trial "
                            f"{trial_index} exceeds "
                            f"recording length."
                        )

                    # -------------------------------------------------------
                    # Convert absolute sample to local chunk index
                    # -------------------------------------------------------

                    local_start_sample = (
                        absolute_start_sample
                        - load_start_sample
                    )

                    local_stop_sample = (
                        local_start_sample
                        + TRIAL_SAMPLES
                    )

                    if local_start_sample < 0:
                        raise ValueError(
                            f"{subject}: negative local "
                            f"trial start."
                        )

                    if (
                        local_stop_sample
                        > processed_chunk.n_times
                    ):
                        raise ValueError(
                            f"{subject}: trial "
                            f"{trial_index} does not fit "
                            f"inside processed chunk."
                        )

                    # -------------------------------------------------------
                    # Read trial
                    # -------------------------------------------------------

                    trial_data = (
                        processed_chunk.get_data(
                            picks=eeg_picks,
                            start=local_start_sample,
                            stop=local_stop_sample
                        )
                    )

                    expected_trial_shape = (
                        EXPECTED_EEG_COUNT,
                        TRIAL_SAMPLES
                    )

                    if trial_data.shape != expected_trial_shape:

                        raise ValueError(
                            f"{subject}: trial "
                            f"{trial_index} has unexpected "
                            f"shape {trial_data.shape}."
                        )

                    # -------------------------------------------------------
                    # Save directly to disk-backed output
                    # -------------------------------------------------------

                    temporal_data[
                        trial_index
                    ] = np.asarray(
                        trial_data,
                        dtype=np.float32
                    )

                    metadata_rows.append(
                        {
                            "subject_id": subject,
                            "trial_index": trial_index,
                            "sample_start":
                                absolute_start_sample,
                            "sample_end":
                                absolute_stop_sample,
                            "onset_seconds":
                                absolute_start_sample / sfreq,
                            "label_id":
                                label_id,
                            "label_name":
                                label_name,
                            "duration_seconds":
                                TRIAL_DURATION_SECONDS
                        }
                    )

                    saved_trial_indices.add(
                        trial_index
                    )

                    del trial_data

                    print(
                        f"  ✓ Trial "
                        f"{trial_index + 1:03d}/"
                        f"{EXPECTED_TRIALS} saved"
                    )

            finally:

                # -----------------------------------------------------------
                # Release chunk memory
                # -----------------------------------------------------------

                processed_chunk = None
                chunk_raw = None

                gc.collect()

                print()
                print(
                    "✓ Chunk memory released."
                )

    except Exception:

        # ---------------------------------------------------------------
        # Close output before raising
        # ---------------------------------------------------------------

        temporal_data.flush()

        raise

    # =========================================================================
    # FINAL VALIDATION
    # =========================================================================

    print_header(
        f"{subject} — FINAL VALIDATION"
    )

    temporal_data.flush()

    print(
        f"Trials extracted: "
        f"{len(saved_trial_indices)}"
    )

    if len(saved_trial_indices) != EXPECTED_TRIALS:

        missing_trials = sorted(
            set(range(EXPECTED_TRIALS))
            - saved_trial_indices
        )

        raise RuntimeError(
            f"{subject}: expected "
            f"{EXPECTED_TRIALS} trials but extracted "
            f"{len(saved_trial_indices)}.\n"
            f"Missing trial indices: "
            f"{missing_trials}"
        )

    # ------------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------------

    metadata = pd.DataFrame(
        metadata_rows
    )

    metadata = metadata.sort_values(
        "trial_index"
    ).reset_index(
        drop=True
    )

    if len(metadata) != EXPECTED_TRIALS:
        raise RuntimeError(
            f"{subject}: metadata contains "
            f"{len(metadata)} rows instead of "
            f"{EXPECTED_TRIALS}."
        )

    # ------------------------------------------------------------------------
    # Check labels
    # ------------------------------------------------------------------------

    class_counts = (
        metadata[
            "label_name"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "Class counts:"
    )

    print(
        class_counts
    )

    if len(class_counts) != EXPECTED_CLASSES:
        raise RuntimeError(
            f"{subject}: expected "
            f"{EXPECTED_CLASSES} classes."
        )

    for label in EXPECTED_LABELS:

        count = int(
            class_counts.get(
                label,
                0
            )
        )

        if count != TRIALS_PER_CLASS:
            raise RuntimeError(
                f"{subject}: class "
                f"{label} has {count} trials."
            )

    # ------------------------------------------------------------------------
    # Numerical validation
    #
    # Use slices rather than np.min over the complete file to avoid
    # unnecessarily loading the entire 1+ GB array into RAM.
    # ------------------------------------------------------------------------

    print()
    print(
        "Checking temporal data..."
    )

    sample_indices = [
        0,
        EXPECTED_TRIALS // 2,
        EXPECTED_TRIALS - 1
    ]

    for index in sample_indices:

        sample = np.asarray(
            temporal_data[index]
        )

        if np.isnan(sample).any():
            raise RuntimeError(
                f"{subject}: NaN detected "
                f"in trial {index}."
            )

        if np.isinf(sample).any():
            raise RuntimeError(
                f"{subject}: Inf detected "
                f"in trial {index}."
            )

        del sample

    print(
        "✓ Sampled temporal data contains "
        "no NaN or Inf values."
    )

    # ------------------------------------------------------------------------
    # Save metadata
    # ------------------------------------------------------------------------

    metadata.to_csv(
        metadata_path,
        index=False
    )

    # ------------------------------------------------------------------------
    # Save channel names
    # ------------------------------------------------------------------------

    if channel_names is None:
        raise RuntimeError(
            f"{subject}: EEG channel names were "
            f"not captured."
        )

    if len(channel_names) != EXPECTED_EEG_COUNT:
        raise RuntimeError(
            f"{subject}: expected 64 channel names, "
            f"found {len(channel_names)}."
        )

    with open(
        channel_path,
        "w",
        encoding="utf-8"
    ) as file:

        for channel in channel_names:
            file.write(
                f"{channel}\n"
            )

    # ------------------------------------------------------------------------
    # Close temporal memmap
    # ------------------------------------------------------------------------

    temporal_data.flush()

    del temporal_data

    gc.collect()

    # ------------------------------------------------------------------------
    # Move partial file to final file
    # ------------------------------------------------------------------------

    if final_temporal_path.exists():
        final_temporal_path.unlink()

    partial_temporal_path.replace(
        final_temporal_path
    )

    # ------------------------------------------------------------------------
    # Final output verification
    # ------------------------------------------------------------------------

    final_data = np.load(
        final_temporal_path,
        mmap_mode="r"
    )

    expected_shape = (
        EXPECTED_TRIALS,
        EXPECTED_EEG_COUNT,
        TRIAL_SAMPLES
    )

    if final_data.shape != expected_shape:
        raise RuntimeError(
            f"{subject}: final output shape "
            f"{final_data.shape} does not match "
            f"{expected_shape}."
        )

    if final_data.dtype != np.float32:
        raise RuntimeError(
            f"{subject}: final output dtype "
            f"{final_data.dtype} is not float32."
        )

    print()
    print(
        "✓ Final temporal EEG file verified."
    )

    print(
        f"  Shape: {final_data.shape}"
    )

    print(
        f"  Dtype: {final_data.dtype}"
    )

    del final_data

    gc.collect()

    # =========================================================================
    # COMPLETE
    # =========================================================================

    print_header(
        f"{subject} COMPLETE"
    )

    print()
    print(
        f"Trials   : {EXPECTED_TRIALS}"
    )

    print(
        f"Channels : {EXPECTED_EEG_COUNT}"
    )

    print(
        f"Samples  : {TRIAL_SAMPLES}"
    )

    print(
        f"Duration : "
        f"{TRIAL_DURATION_SECONDS:.6f} sec"
    )

    print()
    print(
        "Output:"
    )

    print(
        final_temporal_path
    )

    print()
    print(
        "✓ Temporal EEG dataset ready."
    )

    return True


# ============================================================================
# MAIN
# ============================================================================

def main():

    print_header(
        "NEURO-SYMPHONY — STEP 31\n"
        "ALL 12 SUBJECTS — MEMORY-SAFE TEMPORAL EEG"
    )

    print()
    print(
        f"Project root:\n{PROJECT_ROOT}"
    )

    print()
    print(
        "Input:"
    )

    print(
        "  Original DS004514 BDF recordings"
    )

    print()
    print(
        "Pipeline:"
    )

    print(
        "  Original BDF"
    )

    print(
        "      ↓"
    )

    print(
        "  120-second core chunks"
    )

    print(
        "      ↓"
    )

    print(
        "  ±15-second context"
    )

    print(
        "      ↓"
    )

    print(
        "  Existing project EEG preprocessing"
    )

    print(
        "      ↓"
    )

    print(
        "  64 EEG channels"
    )

    print(
        "      ↓"
    )

    print(
        "  720 temporal trials"
    )

    print()
    print(
        f"Trial length: "
        f"{TRIAL_SAMPLES} samples "
        f"({TRIAL_DURATION_SECONDS:.6f} seconds)"
    )

    print()
    print(
        "Memory strategy:"
    )

    print(
        "  ✓ BDF opened with preload=False"
    )

    print(
        "  ✓ Only manageable chunks loaded into RAM"
    )

    print(
        "  ✓ 15-second context around each chunk"
    )

    print(
        "  ✓ Temporal output stored as disk-backed .npy"
    )

    print(
        "  ✓ One subject processed at a time"
    )

    print()
    print(
        "Already completed valid subjects will be skipped."
    )

    print(
        "No raw BDF files are modified."
    )

    print()

    successful_subjects = []
    failed_subjects = []

    # =========================================================================
    # PROCESS ALL SUBJECTS
    # =========================================================================

    for subject in SUBJECTS:

        try:

            result = process_subject(
                subject
            )

            if result:
                successful_subjects.append(
                    subject
                )

        except KeyboardInterrupt:

            print()
            print(
                "PROCESSING INTERRUPTED BY USER."
            )

            raise

        except Exception as error:

            failed_subjects.append(
                subject
            )

            print_header(
                f"✗ {subject} FAILED"
            )

            print(
                str(error)
            )

            print()
            print(
                "FULL TRACEBACK"
            )

            traceback.print_exc()

            print()
            print(
                "Processing stopped."
            )

            break

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print_header(
        "STEP 31 — FINAL SUMMARY"
    )

    print()
    print(
        f"Subjects requested : "
        f"{len(SUBJECTS)}"
    )

    print(
        f"Subjects completed : "
        f"{len(successful_subjects)}"
    )

    print(
        f"Subjects failed    : "
        f"{len(failed_subjects)}"
    )

    print()

    if successful_subjects:

        print(
            "COMPLETED:"
        )

        for subject in successful_subjects:

            print(
                f"  ✓ {subject}"
            )

    if failed_subjects:

        print()

        print(
            "FAILED:"
        )

        for subject in failed_subjects:

            print(
                f"  ✗ {subject}"
            )

    print()

    if len(successful_subjects) == len(SUBJECTS):

        print(
            "=" * 90
        )

        print(
            "✓ ALL 12 SUBJECTS COMPLETED SUCCESSFULLY"
        )

        print(
            "=" * 90
        )

        print()
        print(
            "All temporal EEG datasets are ready."
        )

        print(
            "Next step: true temporal FFT analysis."
        )

    else:

        print(
            "=" * 90
        )

        print(
            "⚠ STEP 31 DID NOT COMPLETE FOR ALL SUBJECTS"
        )

        print(
            "=" * 90
        )

        print()
        print(
            "Do NOT proceed to FFT until the failed "
            "subject is resolved."
        )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()