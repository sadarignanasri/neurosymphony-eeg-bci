import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

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

from src.acquisition.fnirs_loader import (
    FNIRSLoader,
    FNIRSData,
)

# ============================================================
# PHASE 2 IMPORTS
# ============================================================

from src.preprocessing.fnirs_bridge import (
    preprocess_phase1_fnirs,
)

# ============================================================
# TEST SETTINGS
# ============================================================

TEST_DURATION_SECONDS = 60.0

print("=" * 75)
print("PHASE 2 — REAL PHASE 1 -> PHASE 2 fNIRS HANDOFF")
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
# SELECT FIRST REAL fNIRS RECORDING
# ============================================================

fnirs_recording_info = None

for recording_info in all_recordings:

    if recording_info.modality in {
        RecordingModality.FNIRS,
        RecordingModality.MULTIMODAL,
    }:

        fnirs_recording_info = (
            recording_info
        )

        break


assert fnirs_recording_info is not None, (
    "No fNIRS-compatible recording was discovered."
)

print(
    "\nSelected fNIRS recording:"
)

print(
    "  Dataset  :",
    fnirs_recording_info.dataset_name,
)

print(
    "  Subject  :",
    fnirs_recording_info.subject_id,
)

print(
    "  Recording:",
    fnirs_recording_info.recording_id,
)

print(
    "  Modality :",
    fnirs_recording_info.modality.value,
)

print(
    "  Path     :",
    fnirs_recording_info.recording_path,
)


# ============================================================
# BUILD UNIVERSAL RECORDING
# ============================================================

print(
    "\nBuilding Universal Recording..."
)

from src.acquisition.acquisition_engine import (
    Phase1IntegrationEngine,
)

engine = Phase1IntegrationEngine(
    config=None
)

universal_recording = (
    engine._build_universal_recordings(
        [fnirs_recording_info]
    )[0][0]
)


# ============================================================
# SAFETY CHECK:
# UNIVERSAL RECORDING MUST POINT TO REAL SIGNAL FILE
# ============================================================

recording_path = Path(
    universal_recording.recording_path
)

if recording_path.suffix.lower() not in {
    ".snirf",
    ".nirs",
    ".oxy3",
    ".wl1",
    ".wl2",
}:

    recording_dir = recording_path.parent
    base_name = recording_path.name

    # --------------------------------------------------------
    # Convert channels.tsv-style discovery path into the
    # actual fNIRS signal file.
    # --------------------------------------------------------

    candidates = []

    if "_channels.tsv" in base_name:
        signal_base = base_name.replace(
            "_channels.tsv",
            "",
        )

        candidates.append(
            recording_dir / (signal_base + "_nirs.snirf")
        )

    # --------------------------------------------------------
    # Search the recording directory for supported fNIRS
    # signal files.
    # --------------------------------------------------------

    if os.path.isdir(recording_dir):

        for filename in sorted(
            os.listdir(recording_dir)
        ):

            if filename.lower().endswith(
                (
                    ".snirf",
                    ".nirs",
                    ".oxy3",
                    ".wl1",
                    ".wl2",
                )
            ):

                candidates.append(
                    recording_dir / filename
                )

    signal_path = None

    for candidate in candidates:

        if os.path.isfile(candidate):

            signal_path = candidate
            break

    if signal_path is None:

        raise FileNotFoundError(
            "Could not resolve a real fNIRS signal file "
            f"from discovered path:\n{recording_path}"
        )

    print(
        "\nCorrecting fNIRS signal path:"
    )

    print(
        "  Discovered path:",
        recording_path,
    )

    print(
        "  Actual signal :",
        signal_path,
    )

    recording_path = Path(signal_path)

    # --------------------------------------------------------
    # Universal Recording is immutable in the project model,
    # therefore rebuild it with a pathlib.Path.
    # --------------------------------------------------------

    universal_recording = replace(
        universal_recording,
        recording_path=recording_path,
    )


print(
    "Universal Recording:"
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
# VERIFY SIGNAL FILE EXISTS
# ============================================================

assert Path(
    universal_recording.recording_path
).is_file(), (
    "Resolved fNIRS signal file does not exist:\n"
    f"{universal_recording.recording_path}"
)

print(
    "PASS: Real fNIRS signal file exists."
)


# ============================================================
# PHASE 1 — REAL FNIRS LOADER
# ============================================================

print(
    "\n[PHASE 1] Loading through real FNIRSLoader..."
)

fnirs_loader = FNIRSLoader()

fnirs_data = fnirs_loader.load(
    universal_recording
)

print(
    "Phase 1 FNIRSData received."
)

print(
    "  Type        :",
    type(fnirs_data).__name__,
)

print(
    "  Channels    :",
    fnirs_data.n_channels,
)

print(
    "  Samples     :",
    fnirs_data.n_samples,
)

print(
    "  Sampling Hz :",
    fnirs_data.sampling_rate,
)

print(
    "  Duration    :",
    fnirs_data.duration_seconds,
)


# ============================================================
# PHASE 1 VALIDATION
# ============================================================

assert isinstance(
    fnirs_data,
    FNIRSData,
)

print(
    "PASS: Real Phase 1 FNIRSLoader returned FNIRSData."
)


assert isinstance(
    fnirs_data.data,
    np.ndarray,
)

assert fnirs_data.data.ndim == 2

assert fnirs_data.data.shape[0] > 0

assert fnirs_data.data.shape[1] > 0

assert fnirs_data.sampling_rate > 0

assert np.isfinite(
    fnirs_data.data
).all()

print(
    "PASS: Phase 1 fNIRS data is finite and structurally valid."
)


# ============================================================
# SAVE ORIGINAL DATA FOR IMMUTABILITY TEST
# ============================================================

original_data = (
    fnirs_data.data.copy()
)


original_channel_names = (
    fnirs_data.channel_names
)

original_channel_types = (
    fnirs_data.channel_types
)

original_metadata = (
    dict(fnirs_data.metadata)
)

original_fnirs_metadata = (
    fnirs_data.fnirs_metadata
)


# ============================================================
# OPTIONAL DEVELOPMENT CROP
#
# We use only a short segment for Phase 2 development
# validation so that the complete real pipeline can be tested
# without unnecessarily processing the full 4416-second file.
# ============================================================

max_samples = int(
    TEST_DURATION_SECONDS
    * fnirs_data.sampling_rate
)

if (
    fnirs_data.n_samples
    > max_samples
):

    short_data = (
        fnirs_data.data[
            :,
            :max_samples
        ].copy()
    )

    fnirs_data_for_phase2 = replace(
        fnirs_data,
        data=short_data,
    )

    print(
        "\nUsing development segment:"
    )

    print(
        "  Duration:",
        TEST_DURATION_SECONDS,
        "seconds",
    )

    print(
        "  Samples :",
        fnirs_data_for_phase2.n_samples,
    )

else:

    fnirs_data_for_phase2 = fnirs_data

    print(
        "\nUsing complete recording:"
    )

    print(
        "  Samples:",
        fnirs_data_for_phase2.n_samples,
    )


# ============================================================
# PHASE 2 — REAL fNIRS PREPROCESSING
# ============================================================

print(
    "\n[PHASE 2] Running fNIRS preprocessing..."
)

result = preprocess_phase1_fnirs(
    fnirs_data_for_phase2
)

processed_fnirs = result.data

quality_report = result.quality_report

print(
    "PASS: Phase 2 preprocessing completed."
)


# ============================================================
# PHASE 2 OUTPUT VALIDATION
# ============================================================

print(
    "\nPhase 2 output:"
)

print(
    "  Type        :",
    type(processed_fnirs).__name__,
)

print(
    "  Channels    :",
    processed_fnirs.n_channels,
)

print(
    "  Samples     :",
    processed_fnirs.n_samples,
)

print(
    "  Sampling Hz :",
    processed_fnirs.sampling_rate,
)

print(
    "  Duration    :",
    processed_fnirs.duration_seconds,
)


assert isinstance(
    processed_fnirs,
    FNIRSData,
)

print(
    "PASS: Phase 2 returned FNIRSData."
)


assert processed_fnirs.data.ndim == 2

assert processed_fnirs.data.shape == (
    fnirs_data_for_phase2.data.shape
)

print(
    "PASS: Signal shape preserved."
)


assert (
    processed_fnirs.channel_names
    == fnirs_data_for_phase2.channel_names
)

print(
    "PASS: Channel names preserved."
)


assert (
    processed_fnirs.channel_types
    == fnirs_data_for_phase2.channel_types
)

print(
    "PASS: Channel types preserved."
)


assert (
    processed_fnirs.n_channels
    == fnirs_data_for_phase2.n_channels
)

print(
    "PASS: Channel count preserved."
)


assert (
    processed_fnirs.sampling_rate
    == fnirs_data_for_phase2.sampling_rate
)

print(
    "PASS: Sampling rate preserved."
)


assert (
    processed_fnirs.metadata
    == fnirs_data_for_phase2.metadata
)

assert (
    processed_fnirs.fnirs_metadata
    == fnirs_data_for_phase2.fnirs_metadata
)

print(
    "PASS: Metadata preserved."
)


assert np.isfinite(
    processed_fnirs.data
).all()

print(
    "PASS: Processed fNIRS contains only finite values."
)


# ============================================================
# IMMUTABILITY VALIDATION
# ============================================================

assert np.array_equal(
    fnirs_data.data,
    original_data,
)

print(
    "PASS: Original Phase 1 fNIRS data remains unchanged."
)


assert (
    fnirs_data.channel_names
    == original_channel_names
)

assert (
    fnirs_data.channel_types
    == original_channel_types
)

assert (
    fnirs_data.metadata
    == original_metadata
)

assert (
    fnirs_data.fnirs_metadata
    == original_fnirs_metadata
)

print(
    "PASS: Phase 1 metadata remains unchanged."
)


# ============================================================
# QUALITY REPORT
# ============================================================

print(
    "\nQuality report:"
)

quality_report_dict = asdict(
    quality_report
)

for key, value in quality_report_dict.items():

    print(
        f"  {key}: {value}"
    )


# ============================================================
# QUALITY REPORT ASSERTIONS
# ============================================================

assert (
    quality_report.processing_status
    == "PASS"
)

assert quality_report.finite_before

assert quality_report.finite_after

assert quality_report.nan_count_before == 0

assert quality_report.inf_count_before == 0

assert quality_report.nan_count_after == 0

assert quality_report.inf_count_after == 0

assert quality_report.shape_preserved

assert quality_report.channel_count_preserved

assert quality_report.channel_names_preserved

assert quality_report.channel_types_preserved

assert quality_report.metadata_preserved

print(
    "PASS: Phase 2 quality report validation passed."
)


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 75)
print(
    "REAL PHASE 1 -> PHASE 2 fNIRS VALIDATION PASSED"
)
print("=" * 75)

print(
    "\nReal dataset successfully passed:"
)

print(
    "  Phase 1 dataset discovery       : PASS"
)

print(
    "  Phase 1 fNIRS recording discovery: PASS"
)

print(
    "  Phase 1 real SNIRF loading      : PASS"
)

print(
    "  FNIRSData validation            : PASS"
)

print(
    "  Phase 1 -> Phase 2 bridge       : PASS"
)

print(
    "  Phase 2 preprocessing           : PASS"
)

print(
    "  Output validation               : PASS"
)

print(
    "  Data immutability               : PASS"
)

print(
    "  Quality report                  : PASS"
)

print(
    "\nPHASE 2 fNIRS REAL-DATA VALIDATION COMPLETE."
)
