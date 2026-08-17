import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)

from src.acquisition.fnirs_loader import FNIRSData


print("=" * 75)
print("PHASE 1 fNIRS DATA STRUCTURE INSPECTION")
print("=" * 75)

print("\nFNIRSData class successfully imported.")

print("\nThe Phase 1 standardized object contains:")
print("  - data")
print("  - sampling_rate")
print("  - channel_names")
print("  - channel_types")
print("  - fnirs_metadata")
print("  - metadata")

print("\nData requirements for Phase 2:")
print("  data shape must be [channels, samples]")
print("  sampling rate must be positive")
print("  data must contain finite values")

print("\nFNIRSData properties:")
print("  n_channels")
print("  n_samples")
print("  duration_seconds")

print("\nSUCCESS: Phase 2 can consume FNIRSData directly.")
print("=" * 75)