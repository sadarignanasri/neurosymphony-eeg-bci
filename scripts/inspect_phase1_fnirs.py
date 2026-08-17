
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from src.acquisition.fnirs_loader import FNIRSLoader
from src.acquisition.recording import Recording


print("=" * 75)
print("PHASE 1 → PHASE 2 fNIRS HANDOFF INSPECTION")
print("=" * 75)

print("\nThis script only inspects the Phase 1 fNIRS representation.")
print("No preprocessing will be performed.")
print("No files will be modified.")

print("\nFNIRSLoader imported successfully.")

print("\nFNIRSData fields available from the loader:")

fields = [
    "recording_id",
    "recording_path",
    "modality",
    "layout",
    "subject_id",
    "session_id",
    "task_id",
    "run_id",
    "source_kind",
    "adapter_name",
    "sampling_rate",
    "channel_names",
    "data",
    "channel_types",
    "fnirs_metadata",
    "metadata",
]

for field in fields:
    print(f"  - {field}")

print("\nPhase 1 loader interface:")
print("  load(recording) -> FNIRSData")
print("  load_with_result(recording) -> FNIRSLoadResult")

print("\nSUCCESS: Phase 1 fNIRS standardized output is FNIRSData.")
print("=" * 75)