import os
import sys
from dataclasses import replace

import numpy as np

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)

# ============================================================
# PHASE 1 IMPORTS
# ============================================================

from src.acquisition.dataset_manager import (
    DatasetManager,
)

from src.acquisition.subject_manager import (
    SubjectManager,
)

from src.acquisition.recording import (
    RecordingManager,
    RecordingModality,
)

from src.acquisition.eeg_loader import (
    EEGLoader,
    EEGData,
)

# ============================================================
# PHASE 2 IMPORTS
# ============================================================

from src.preprocessing.phase1_bridge import (
    preprocess_phase1_eeg,
)

# ============================================================
# TEST SETTINGS
# ============================================================

TEST_DURATION_SECONDS = 60.0

print("=" * 75)
print("PHASE 2.14 — REAL PHASE 1 -> PHASE 2 EEG HANDOFF")
print("=" * 75)


# ============================================================
# PHASE 1 — DATASET DISCOVERY
# ============================================================

print("\n[PHASE 1] Discovering datasets...")

dataset_manager = DatasetManager()

dataset_map = (
    dataset_manager.discover_datasets()
)

datasets = tuple(
    sorted(
        dataset_map.values(),
        key=lambda item: item.dataset_name.lower(),
    )
)

print(
    "Datasets discovered:",
    len(datasets),
)

for dataset in datasets:
    print(
        f"  - {dataset.dataset_name}"
    )


assert datasets, (
    "Phase 1 did not discover any datasets."
)

print(
    "PASS: Phase 1 dataset discovery works."
)


# ============================================================
# PHASE 1 — SUBJECT DISCOVERY
# ============================================================

print("\n[PHASE 1] Discovering subjects...")

subject_manager = SubjectManager(
    dataset_manager=dataset_manager,
)

subject_map = (
    subject_manager.discover_subjects()
)

subjects = tuple(
    sorted(
        subject_map.values(),
        key=lambda item: (
            item.dataset_name.lower(),
            item.subject_id.lower(),
        ),
    )
)

print(
    "Subjects discovered:",
    len(subjects),
)

assert subjects, (
    "Phase 1 did not discover any subjects."
)

print(
    "PASS: Phase 1 subject discovery works."
)


# ============================================================
# PHASE 1 — RECORDING DISCOVERY
# ============================================================

print("\n[PHASE 1] Discovering recordings...")

recording_manager = RecordingManager()

all_recordings = []

for subject in subjects:

    discovered = (
        recording_manager.discover_recordings(
            subject
        )
    )

    all_recordings.extend(
        discovered
    )

print(
    "Recording groups discovered:",
    len(all_recordings),
)

assert all_recordings, (
    "Phase 1 did not discover any recordings."
)

# ============================================================
# SELECT FIRST EEG-COMPATIBLE RECORDING
# ============================================================

eeg_recording_info = None

for recording_info in all_recordings:

    if recording_info.modality in {
        RecordingModality.EEG,
        RecordingModality.MULTIMODAL,
        RecordingModality.UNKNOWN,
    }:

        eeg_recording_info = (
            recording_info
        )

        break


assert eeg_recording_info is not None, (
    "No EEG-compatible recording was discovered."
)

print(
    "\nSelected recording:"
)

print(
    "  Dataset :",
    eeg_recording_info.dataset_name,
)

print(
    "  Subject :",
    eeg_recording_info.subject_id,
)

print(
    "  Recording:",
    eeg_recording_info.recording_id,
)

print(
    "  Modality:",
    eeg_recording_info.modality.value,
)


# ============================================================
# BUILD UNIVERSAL RECORDING
# ============================================================

# Use the same Universal Recording construction logic
# already implemented by Phase 1.

from src.acquisition.acquisition_engine import (
    Phase1IntegrationEngine,
)

engine = Phase1IntegrationEngine(
    config=None
)

universal_recording = (
    engine._build_universal_recordings(
        [eeg_recording_info]
    )[0][0]
)

print(
    "\nUniversal Recording:"
)

print(
    "  ID       :",
    universal_recording.recording_id,
)

print(
    "  Subject  :",
    universal_recording.subject_id,
)

print(
    "  Modality :",
    universal_recording.modality.value,
)

print(
    "  Path     :",
    universal_recording.recording_path,
)


# ============================================================
# PHASE 1 — REAL EEG LOADER
# ============================================================

print(
    "\n[PHASE 1] Loading through real EEGLoader..."
)

eeg_loader = EEGLoader()

eeg_data = eeg_loader.load(
    universal_recording
)

print(
    "Phase 1 EEGData received."
)

print(
    "  Type        :",
    type(eeg_data).__name__,
)

print(
    "  Channels    :",
    eeg_data.n_channels,
)

print(
    "  Samples     :",
    eeg_data.n_samples,
)

print(
    "  Sampling Hz :",
    eeg_data.sampling_rate,
)


assert isinstance(
    eeg_data,
    EEGData,
)

print(
    "PASS: Real Phase 1 EEGLoader returned EEGData."
)


# ============================================================
# TAKE SHORT DEVELOPMENT SEGMENT
# ============================================================

max_samples = int(
    TEST_DURATION_SECONDS
    * eeg_data.sampling_rate
)

if eeg_data.n_samples > max_samples:

    short_data = (
        eeg_data.data[
            :,
            :max_samples
        ]
    )

else:

    short_data = eeg_data.data


short_eeg_data = replace(
    eeg_data,
    data=np.asarray(
        short_data
    ),
)

print(
    "\nDevelopment segment:"
)

print(
    "  Duration:",
    f"{short_eeg_data.duration_seconds:.2f} seconds",
)

print(
    "  Samples :",
    short_eeg_data.n_samples,
)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

# The Phase 2 engine must NOT assume that every dataset
# is DS004514.
#
# For this first real integration test, we intentionally
# pass no dataset-specific configuration.
#
# This tests whether Phase 2 can consume the REAL
# Phase 1 EEGData representation independently.
# ============================================================

dataset_config = None


# ============================================================
# PHASE 2
# ============================================================

print(
    "\n[PHASE 2] Preprocessing real Phase 1 EEGData..."
)

processed_eeg_data, quality_report = (
    preprocess_phase1_eeg(
        short_eeg_data,
        dataset_config=dataset_config,
    )
)

print(
    "Phase 2 returned processed EEGData."
)

print(
    "  Type     :",
    type(processed_eeg_data).__name__,
)

print(
    "  Channels :",
    processed_eeg_data.n_channels,
)

print(
    "  Samples  :",
    processed_eeg_data.n_samples,
)

print(
    "  Sampling :",
    processed_eeg_data.sampling_rate,
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print(
    "\nFinal validation:"
)

assert isinstance(
    processed_eeg_data,
    EEGData,
)

assert (
    processed_eeg_data.n_channels
    == short_eeg_data.n_channels
)

assert (
    processed_eeg_data.n_samples
    == short_eeg_data.n_samples
)

nan_count = int(
    np.isnan(
        processed_eeg_data.data
    ).sum()
)

inf_count = int(
    np.isinf(
        processed_eeg_data.data
    ).sum()
)

print(
    "  NaNs      :",
    nan_count,
)

print(
    "  Infinities:",
    inf_count,
)

assert nan_count == 0
assert inf_count == 0

# ------------------------------------------------------------
# Metadata preservation
# ------------------------------------------------------------

assert (
    processed_eeg_data.recording_id
    == short_eeg_data.recording_id
)

assert (
    processed_eeg_data.subject_id
    == short_eeg_data.subject_id
)

assert (
    processed_eeg_data.sampling_rate
    == short_eeg_data.sampling_rate
)

assert (
    "phase2"
    in processed_eeg_data.metadata
)


# ============================================================
# SUCCESS
# ============================================================

print(
    "\nPASS: Real Phase 1 Universal Recording "
    "reached EEGLoader."
)

print(
    "PASS: Real Phase 1 EEGData reached Phase 2."
)

print(
    "PASS: Phase 2 returned EEGData."
)

print(
    "PASS: Signal dimensions were preserved."
)

print(
    "PASS: Recording metadata was preserved."
)

print(
    "PASS: No NaN or infinite values."
)

print("\n" + "=" * 75)
print("PHASE 2.14 COMPLETE")
print("=" * 75)

print(
    "\nREAL PHASE 1 -> PHASE 2 EEG HANDOFF WORKS."
)