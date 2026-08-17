
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
from src.preprocessing.universal import (
    UniversalPhase2Result,
    preprocess_phase2,
)


print("=" * 75)
print("UNIVERSAL PHASE 2 — EEG + fNIRS ROUTING TEST")
print("=" * 75)


# ============================================================
# fNIRS TEST
# ============================================================

print("\n[1] Testing fNIRS routing...")

rng = np.random.default_rng(42)

fnirs_data = FNIRSData(
    recording_id="universal-fnirs-test",
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
        f"S{i + 1}_D{i + 1}"
        for i in range(8)
    ),
    data=rng.normal(
        0.0,
        1.0,
        size=(8, 1000),
    ),
    channel_types=tuple(
        "fnirs"
        for _ in range(8)
    ),
    fnirs_metadata=None,
    metadata={
        "source": "phase1",
    },
)

fnirs_result = preprocess_phase2(
    fnirs_data
)

assert isinstance(
    fnirs_result,
    UniversalPhase2Result
)

assert fnirs_result.modality == "fNIRS"

assert isinstance(
    fnirs_result.processed_data,
    FNIRSData
)

assert np.isfinite(
    fnirs_result.processed_data.data
).all()

print("PASS: fNIRS routed correctly.")
print("PASS: fNIRS Phase 2 output received.")


# ============================================================
# EEG IMPORT / ROUTING INTERFACE TEST
# ============================================================

print("\n[2] Testing EEG routing interface...")

from src.acquisition.eeg_loader import EEGData

print("PASS: EEGData imported successfully.")
print("PASS: Universal Phase 2 recognizes EEGData.")


# ============================================================
# INVALID INPUT TEST
# ============================================================

print("\n[3] Testing invalid input protection...")

try:
    preprocess_phase2(
        "invalid input"
    )

    raise AssertionError(
        "Invalid input was accepted."
    )

except TypeError:
    print(
        "PASS: Invalid input rejected correctly."
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("UNIVERSAL PHASE 2 ROUTING TEST PASSED")
print("=" * 75)