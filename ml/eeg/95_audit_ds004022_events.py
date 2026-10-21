
from pathlib import Path

import mne
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT / "major datasets" / "ds004022"
OUTPUT_DIR = PROJECT / "experiments" / "ds004022_inspection"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

for subject in ["sub-02", "sub-03", "sub-04", "sub-05"]:
    eeg_dir = DATA_ROOT / subject / "eeg"
    bdf_candidates = sorted(eeg_dir.glob("*run-1*.set"))

    if not bdf_candidates:
        print(f"No run-1 .set file found for {subject}")
        continue

    file_path = bdf_candidates[0]

    print(f"\nInspecting {subject}: {file_path.name}")

    raw = mne.io.read_raw_eeglab(
        file_path,
        preload=False,
        verbose="ERROR",
    )

    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    print(f"Recording duration: {raw.times[-1]:.2f} seconds")
    print(f"Annotations: {len(raw.annotations)}")

    for index, annotation in enumerate(raw.annotations):
        rows.append({
            "subject": subject,
            "recording": file_path.name,
            "annotation_index": index,
            "onset_seconds": float(annotation["onset"]),
            "duration_seconds": float(annotation["duration"]),
            "description": str(annotation["description"]),
        })

    raw.close()

events_df = pd.DataFrame(rows)

if events_df.empty:
    raise RuntimeError("No annotations were collected.")

output_path = OUTPUT_DIR / "event_timing_audit.csv"
events_df.to_csv(output_path, index=False)

print("\nAnnotation counts by subject and label:")
print(
    events_df.groupby(
        ["subject", "description"]
    ).size().to_string()
)

print("\nFirst 35 annotations:")
print(
    events_df[
        [
            "subject",
            "annotation_index",
            "onset_seconds",
            "duration_seconds",
            "description",
        ]
    ].head(35).to_string(index=False)
)

print(f"\nSaved audit: {output_path}")
