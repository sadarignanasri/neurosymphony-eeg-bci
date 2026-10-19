"""
Neuro-Symphony
Step 28: Verify original sub-01 EEG BDF and events

This script ONLY inspects the original BDF and event files.
It does not modify the data or train a model.
"""

from pathlib import Path
import json

import mne
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUBJECT_DIR = PROJECT_ROOT / "data" / "raw" / "sub-01"

BDF_FILE = SUBJECT_DIR / "sub-01_task-eeg_eeg.bdf"
EVENTS_TSV = SUBJECT_DIR / "sub-01_task-eeg_events.tsv"
EVENTS_JSON = SUBJECT_DIR / "sub-01_task-eeg_events.json"


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 80)
print("NEURO-SYMPHONY — SUB-01 ORIGINAL BDF VERIFICATION")
print("=" * 80)

print("\nChecking files...")

for file_path in [BDF_FILE, EVENTS_TSV, EVENTS_JSON]:
    if file_path.exists():
        size_mb = file_path.stat().st_size / (1024 ** 2)
        print(f"✓ {file_path.name} ({size_mb:.2f} MB)")
    else:
        print(f"✗ MISSING: {file_path}")


# ============================================================
# LOAD BDF
# ============================================================

print("\n" + "=" * 80)
print("LOADING ORIGINAL BDF")
print("=" * 80)

print(f"\nFile: {BDF_FILE}")

raw = mne.io.read_raw_bdf(
    BDF_FILE,
    preload=False,
    verbose=True
)


# ============================================================
# BASIC RECORDING INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("BDF INFORMATION")
print("=" * 80)

print(f"\nNumber of channels : {len(raw.ch_names)}")
print(f"Sampling frequency : {raw.info['sfreq']} Hz")
print(f"Number of samples  : {raw.n_times}")
print(f"Duration           : {raw.times[-1]:.3f} seconds")
print(f"Duration           : {raw.times[-1] / 60:.3f} minutes")

print("\nChannel names:")
print(raw.ch_names)


# ============================================================
# CHANNEL TYPES
# ============================================================

print("\n" + "=" * 80)
print("CHANNEL TYPES")
print("=" * 80)

channel_types = raw.get_channel_types()

type_counts = pd.Series(channel_types).value_counts()

print(type_counts)


# ============================================================
# EVENTS TSV
# ============================================================

print("\n" + "=" * 80)
print("EVENTS.TSV")
print("=" * 80)

events = pd.read_csv(EVENTS_TSV, sep="\t")

print("\nColumns:")
print(events.columns.tolist())

print("\nNumber of event rows:")
print(len(events))

print("\nFirst 10 rows:")
print(events.head(10).to_string(index=False))


# ============================================================
# EVENT COLUMNS
# ============================================================

print("\n" + "=" * 80)
print("EVENT COLUMN INFORMATION")
print("=" * 80)

for column in events.columns:
    print(f"\n--- {column} ---")
    print(events[column].value_counts(dropna=False).head(30))


# ============================================================
# EVENT JSON
# ============================================================

print("\n" + "=" * 80)
print("EVENTS.JSON")
print("=" * 80)

if EVENTS_JSON.exists():

    with open(EVENTS_JSON, "r", encoding="utf-8") as f:
        events_json = json.load(f)

    print(json.dumps(events_json, indent=2))


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)

print("\nNo preprocessing was performed.")
print("No data was modified.")
print("No model was trained.")

print("\nNext step will be decided from the actual event structure above.")