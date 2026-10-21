
from pathlib import Path

import mne
import pandas as pd


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

PROJECT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT / "major datasets" / "ds004022"
OUTPUT_DIR = PROJECT / "experiments" / "ds004022_inspection"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "candidate_trial_sequences.csv"

# Normalize repeated spaces in annotation labels.
CANDIDATE_TRIAL_MARKER = "S 2"
WINDOW_SECONDS = 16.0
MAX_PRINTED_TRIALS_PER_SUBJECT = 5

rows = []
summary_rows = []


# --------------------------------------------------
# 2. Validate dataset directory
# --------------------------------------------------

if not DATA_ROOT.exists():
    raise FileNotFoundError(
        f"Dataset directory not found: {DATA_ROOT}"
    )

set_files = sorted(DATA_ROOT.rglob("*run-1*.set"))

if not set_files:
    raise FileNotFoundError(
        f"No run-1 .set files found inside {DATA_ROOT}"
    )

print("=" * 75)
print("DS004022 EVENT SEQUENCE AUDIT")
print("=" * 75)
print(f"Run-1 recordings found: {len(set_files)}")


# --------------------------------------------------
# 3. Inspect event sequences
# --------------------------------------------------

for set_path in set_files:
    subject = next(
        (
            part
            for part in set_path.parts
            if part.startswith("sub-")
        ),
        "unknown",
    )

    print(f"\nInspecting {subject}: {set_path.name}")

    raw = mne.io.read_raw_eeglab(
        set_path,
        preload=False,
        verbose="ERROR",
    )

    try:
        # Collapse repeated internal spaces:
        # "S  2" becomes "S 2".
        annotations = [
            {
                "onset": float(annotation["onset"]),
                "duration": float(annotation["duration"]),
                "label": " ".join(
                    str(annotation["description"]).split()
                ),
            }
            for annotation in raw.annotations
        ]
    finally:
        raw.close()

    annotations.sort(key=lambda item: item["onset"])

    label_counts = pd.Series(
        [event["label"] for event in annotations]
    ).value_counts().to_dict()

    print(f"Total annotations: {len(annotations)}")
    print(f"Normalized labels: {label_counts}")

    candidate_events = [
        event
        for event in annotations
        if event["label"] == CANDIDATE_TRIAL_MARKER
    ]

    print(
        f"Occurrences of {CANDIDATE_TRIAL_MARKER!r}: "
        f"{len(candidate_events)}"
    )

    summary_rows.append({
        "subject": subject,
        "recording": set_path.name,
        "annotation_count": len(annotations),
        "candidate_marker": CANDIDATE_TRIAL_MARKER,
        "candidate_marker_count": len(candidate_events),
        "normalized_label_counts": str(label_counts),
    })

    printed_count = 0

    for event_index, event in enumerate(annotations):
        if event["label"] != CANDIDATE_TRIAL_MARKER:
            continue

        window_end = event["onset"] + WINDOW_SECONDS

        nearby = [
            item
            for item in annotations
            if event["onset"] <= item["onset"] <= window_end
        ]

        sequence = " -> ".join(
            f"{item['label']}@{item['onset']:.3f}s"
            for item in nearby
        )

        rows.append({
            "subject": subject,
            "recording": set_path.name,
            "annotation_index": event_index,
            "trial_start_onset": event["onset"],
            "window_end_seconds": window_end,
            "events_within_window": len(nearby),
            "events_within_16_seconds": sequence,
        })

        if printed_count < MAX_PRINTED_TRIALS_PER_SUBJECT:
            print(f"\nCandidate sequence {printed_count + 1}:")
            print(sequence)
            printed_count += 1


# --------------------------------------------------
# 4. Save reports
# --------------------------------------------------

sequence_columns = [
    "subject",
    "recording",
    "annotation_index",
    "trial_start_onset",
    "window_end_seconds",
    "events_within_window",
    "events_within_16_seconds",
]

summary_columns = [
    "subject",
    "recording",
    "annotation_count",
    "candidate_marker",
    "candidate_marker_count",
    "normalized_label_counts",
]

pd.DataFrame(
    rows,
    columns=sequence_columns,
).to_csv(
    OUTPUT_CSV,
    index=False,
)

summary_path = OUTPUT_DIR / "event_mapping_summary.csv"

pd.DataFrame(
    summary_rows,
    columns=summary_columns,
).to_csv(
    summary_path,
    index=False,
)


# --------------------------------------------------
# 5. Display final summary
# --------------------------------------------------

print("\n" + "=" * 75)
print("EVENT SEQUENCE AUDIT SUMMARY")
print("=" * 75)

print(f"Recordings inspected: {len(set_files)}")
print(f"Candidate sequences found: {len(rows)}")

for summary in summary_rows:
    print(
        f"{summary['subject']}: "
        f"{summary['candidate_marker_count']} candidate markers"
    )

print("\nReports saved:")
print(OUTPUT_CSV)
print(summary_path)

if not rows:
    print(
        "\nNo candidate markers were found after normalization. "
        "Inspect event_mapping_summary.csv for the exact normalized "
        "labels before changing the candidate marker."
    )

print(
    "\nThis is an event inspection script only. "
    "It does not assign motor-imagery classes or extract EEG epochs."
)
