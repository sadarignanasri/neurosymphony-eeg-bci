
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from src.acquisition.fnirs_loader import FNIRSData
from src.preprocessing.fnirs import preprocess_fnirs


print("=" * 75)
print("PHASE 2 — fNIRS PREPROCESSING VALIDATION")
print("=" * 75)


# ------------------------------------------------------------
# CREATE SYNTHETIC PHASE 1 OUTPUT
# ------------------------------------------------------------

print("\nCreating synthetic Phase 1 FNIRSData...")

rng = np.random.default_rng(42)

original_data = rng.normal(
    0.0,
    1.0,
    size=(8, 1000)
)

fnirs = FNIRSData(
    recording_id="test-fnirs-001",
    recording_path=PROJECT_ROOT,
    modality=None,
    layout=None,
    subject_id="sub-test",
    session_id=None,
    task_id="test",
    run_id="01",
    source_kind="FILE",
    adapter_name="test_adapter",
    sampling_rate=10.0,
    channel_names=tuple(
        f"S{i+1}_D{i+1}"
        for i in range(8)
    ),
    data=original_data,
    channel_types=tuple(
        "fnirs"
        for _ in range(8)
    ),
    fnirs_metadata=None,
    metadata={
        "test": True,
        "phase": 1,
    },
)

print("PASS: Synthetic FNIRSData created.")

print(
    "Channels:",
    fnirs.n_channels
)

print(
    "Samples:",
    fnirs.n_samples
)

print(
    "Sampling rate:",
    fnirs.sampling_rate
)

print(
    "Duration:",
    fnirs.duration_seconds,
    "seconds"
)


# ------------------------------------------------------------
# ORIGINAL DATA BACKUP
# ------------------------------------------------------------

original_backup = fnirs.data.copy()


# ------------------------------------------------------------
# RUN PHASE 2
# ------------------------------------------------------------

print("\nRunning Phase 2 fNIRS preprocessing...")

result = preprocess_fnirs(
    fnirs
)

processed = result.data
quality = result.quality_report

print("PASS: Phase 2 preprocessing completed.")


# ------------------------------------------------------------
# OUTPUT CHECK
# ------------------------------------------------------------

print("\nOutput validation:")

assert processed is not fnirs

print(
    "PASS: Phase 2 returned a new FNIRSData object."
)

assert processed.data is not fnirs.data

print(
    "PASS: Phase 1 data array was not reused."
)


# ------------------------------------------------------------
# SHAPE CHECK
# ------------------------------------------------------------

assert processed.data.shape == original_backup.shape

print(
    "PASS: Signal shape preserved."
)


# ------------------------------------------------------------
# CHANNEL CHECK
# ------------------------------------------------------------

assert processed.channel_names == fnirs.channel_names

print(
    "PASS: Channel names preserved."
)

assert processed.channel_types == fnirs.channel_types

print(
    "PASS: Channel types preserved."
)

assert processed.n_channels == fnirs.n_channels

print(
    "PASS: Channel count preserved."
)


# ------------------------------------------------------------
# SAMPLING RATE CHECK
# ------------------------------------------------------------

assert processed.sampling_rate == fnirs.sampling_rate

print(
    "PASS: Sampling rate preserved."
)


# ------------------------------------------------------------
# METADATA CHECK
# ------------------------------------------------------------

assert processed.metadata == fnirs.metadata

print(
    "PASS: Metadata preserved."
)


# ------------------------------------------------------------
# FINITE DATA CHECK
# ------------------------------------------------------------

assert np.isfinite(
    processed.data
).all()

print(
    "PASS: Processed fNIRS contains only finite values."
)


# ------------------------------------------------------------
# ORIGINAL DATA INTEGRITY
# ------------------------------------------------------------

assert np.array_equal(
    fnirs.data,
    original_backup
)

print(
    "PASS: Original Phase 1 fNIRS data remains unchanged."
)


# ------------------------------------------------------------
# QUALITY REPORT
# ------------------------------------------------------------

print("\nQuality report:")

print(
    "Total channels:",
    quality.total_channels
)

print(
    "Total samples:",
    quality.total_samples
)

print(
    "Sampling rate:",
    quality.sampling_rate
)

print(
    "Duration:",
    quality.duration_seconds
)

print(
    "NaNs before:",
    quality.nan_count_before
)

print(
    "Infinities before:",
    quality.inf_count_before
)

print(
    "NaNs after:",
    quality.nan_count_after
)

print(
    "Infinities after:",
    quality.inf_count_after
)

print(
    "Finite before:",
    quality.finite_before
)

print(
    "Finite after:",
    quality.finite_after
)

print(
    "Shape preserved:",
    quality.shape_preserved
)

print(
    "Channel count preserved:",
    quality.channel_count_preserved
)

print(
    "Channel names preserved:",
    quality.channel_names_preserved
)

print(
    "Channel types preserved:",
    quality.channel_types_preserved
)

print(
    "Metadata preserved:",
    quality.metadata_preserved
)


# ------------------------------------------------------------
# FINAL ASSERTIONS
# ------------------------------------------------------------

assert quality.finite_before
assert quality.finite_after
assert quality.shape_preserved
assert quality.channel_count_preserved
assert quality.channel_names_preserved
assert quality.channel_types_preserved
assert quality.metadata_preserved
assert quality.processing_status == "PASS"


print("\n" + "=" * 75)
print("PHASE 2 fNIRS BASELINE VALIDATION PASSED")
print("=" * 75)