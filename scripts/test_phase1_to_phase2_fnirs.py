
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
from src.preprocessing.fnirs_bridge import preprocess_phase1_fnirs


print("=" * 75)
print("PHASE 1 → PHASE 2 fNIRS HANDOFF TEST")
print("=" * 75)


# ------------------------------------------------------------
# 1. Create representative Phase 1 FNIRSData
# ------------------------------------------------------------

print("\nCreating Phase 1 FNIRSData...")

rng = np.random.default_rng(42)

data = rng.normal(
    0.0,
    1.0,
    size=(8, 1000)
)

channel_names = tuple(
    f"S{i + 1}_D{i + 1}"
    for i in range(8)
)

channel_types = tuple(
    "fnirs"
    for _ in range(8)
)

fnirs_data = FNIRSData(
    recording_id="phase1-fnirs-test",
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
    channel_names=channel_names,
    data=data,
    channel_types=channel_types,
    fnirs_metadata=None,
    metadata={
        "source": "phase1_test",
        "universal": True,
    },
)

print("PASS: Phase 1 FNIRSData created.")


# ------------------------------------------------------------
# 2. Preserve original data
# ------------------------------------------------------------

original_data = fnirs_data.data.copy()


# ------------------------------------------------------------
# 3. Run Phase 1 → Phase 2 bridge
# ------------------------------------------------------------

print("\nRunning Phase 1 → Phase 2 fNIRS bridge...")

result = preprocess_phase1_fnirs(
    fnirs_data
)

processed = result.data
quality = result.quality_report

print("PASS: Phase 1 → Phase 2 handoff completed.")


# ------------------------------------------------------------
# 4. Validate output type
# ------------------------------------------------------------

assert isinstance(
    processed,
    FNIRSData
)

print(
    "PASS: Phase 2 output is FNIRSData."
)


# ------------------------------------------------------------
# 5. Validate dimensions
# ------------------------------------------------------------

assert processed.data.shape == original_data.shape

print(
    "PASS: Signal dimensions preserved."
)


assert processed.n_channels == fnirs_data.n_channels

print(
    "PASS: Channel count preserved."
)


assert processed.n_samples == fnirs_data.n_samples

print(
    "PASS: Sample count preserved."
)


# ------------------------------------------------------------
# 6. Validate channel information
# ------------------------------------------------------------

assert (
    processed.channel_names
    == fnirs_data.channel_names
)

print(
    "PASS: Channel names preserved."
)


assert (
    processed.channel_types
    == fnirs_data.channel_types
)

print(
    "PASS: Channel types preserved."
)


# ------------------------------------------------------------
# 7. Validate sampling rate
# ------------------------------------------------------------

assert (
    processed.sampling_rate
    == fnirs_data.sampling_rate
)

print(
    "PASS: Sampling rate preserved."
)


# ------------------------------------------------------------
# 8. Validate metadata
# ------------------------------------------------------------

assert (
    processed.metadata
    == fnirs_data.metadata
)

print(
    "PASS: Metadata preserved."
)


# ------------------------------------------------------------
# 9. Validate finite output
# ------------------------------------------------------------

assert np.isfinite(
    processed.data
).all()

print(
    "PASS: Phase 2 fNIRS output contains only finite values."
)


# ------------------------------------------------------------
# 10. Validate Phase 1 immutability
# ------------------------------------------------------------

assert np.array_equal(
    fnirs_data.data,
    original_data
)

print(
    "PASS: Phase 1 fNIRS input remains unchanged."
)


# ------------------------------------------------------------
# 11. Validate quality report
# ------------------------------------------------------------

assert quality.processing_status == "PASS"

assert quality.finite_before

assert quality.finite_after

assert quality.shape_preserved

assert quality.channel_count_preserved

assert quality.channel_names_preserved

assert quality.channel_types_preserved

assert quality.metadata_preserved

print(
    "PASS: Quality report validation passed."
)


# ------------------------------------------------------------
# FINAL
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("PHASE 1 → PHASE 2 fNIRS HANDOFF PASSED")
print("=" * 75)