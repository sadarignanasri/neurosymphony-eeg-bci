
from pathlib import Path
import json
import traceback

import mne
import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

PROJECT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT / "major datasets" / "ds004022"
OUTPUT_DIR = PROJECT / "experiments" / "ds004022_inspection"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("DS004022 EEG DATASET INSPECTION")
print("=" * 70)
print(f"Project directory: {PROJECT}")
print(f"Dataset directory: {DATA_ROOT}")
print(f"Output directory: {OUTPUT_DIR}")


if not DATA_ROOT.exists():
    raise FileNotFoundError(
        f"Dataset directory not found: {DATA_ROOT}"
    )


# --------------------------------------------------
# 2. Discover EEGLAB recordings
# --------------------------------------------------

set_files = sorted(DATA_ROOT.rglob("*.set"))

if not set_files:
    raise FileNotFoundError(
        f"No .set recordings found inside {DATA_ROOT}"
    )

print(f"\nEEGLAB recordings found: {len(set_files)}")

recording_rows = []
event_rows = []
errors = []


# --------------------------------------------------
# 3. Inspect each recording
# --------------------------------------------------

for file_index, set_path in enumerate(set_files, start=1):

    print("\n" + "-" * 70)
    print(f"Recording {file_index}/{len(set_files)}")
    print(f"File: {set_path.relative_to(DATA_ROOT)}")

    relative_path = str(set_path.relative_to(DATA_ROOT))

    record = {
        "file": relative_path,
        "subject_folder": set_path.parent.name,
        "file_size_mb": round(
            set_path.stat().st_size / (1024 * 1024), 3
        ),
        "data_type": "unknown",
        "sampling_rate_hz": None,
        "channel_count": None,
        "channel_names": None,
        "channel_types": None,
        "duration_seconds": None,
        "sample_count": None,
        "epoch_count": None,
        "event_count": None,
        "event_labels": None,
        "events_tsv": None,
        "status": "failed",
        "error": None,
    }

    try:
        # First try reading the file as continuous EEG.
        try:
            raw = mne.io.read_raw_eeglab(
                set_path,
                preload=False,
                verbose="ERROR",
            )

            record["data_type"] = "continuous"
            record["sampling_rate_hz"] = float(
                raw.info["sfreq"]
            )
            record["channel_count"] = len(raw.ch_names)
            record["channel_names"] = json.dumps(
                raw.ch_names
            )
            record["channel_types"] = json.dumps(
                raw.get_channel_types()
            )
            record["duration_seconds"] = float(
                raw.n_times / raw.info["sfreq"]
            )
            record["sample_count"] = int(raw.n_times)

            print("Data type: continuous EEG")
            print(
                f"Sampling rate: "
                f"{record['sampling_rate_hz']} Hz"
            )
            print(
                f"Channels: {record['channel_count']}"
            )
            print(
                f"Duration: "
                f"{record['duration_seconds']:.2f} seconds"
            )

            # Extract events from MNE annotations.
            if len(raw.annotations) > 0:
                try:
                    events, event_id = (
                        mne.events_from_annotations(
                            raw,
                            verbose="ERROR",
                        )
                    )

                    record["event_count"] = int(
                        len(events)
                    )
                    record["event_labels"] = json.dumps(
                        event_id
                    )

                    for label, event_code in event_id.items():
                        event_rows.append({
                            "file": relative_path,
                            "source": "MNE annotations",
                            "event_label": label,
                            "event_code": int(event_code),
                            "event_count": int(
                                np.sum(
                                    events[:, 2] == event_code
                                )
                            ),
                        })

                    print(
                        f"Annotation events: {len(events)}"
                    )
                    print(
                        f"Event labels: {list(event_id.keys())}"
                    )

                except Exception as event_error:
                    print(
                        "Could not convert annotations "
                        f"into events: {event_error}"
                    )

            # Close the file without loading all EEG into memory.
            raw.close()

        except Exception as raw_error:

            # If continuous loading fails, try an epoched EEGLAB file.
            print(
                "Continuous loading failed. "
                "Trying epoched EEGLAB format."
            )

            try:
                epochs = mne.read_epochs_eeglab(
                    set_path,
                    verbose="ERROR",
                )

                record["data_type"] = "epoched"
                record["sampling_rate_hz"] = float(
                    epochs.info["sfreq"]
                )
                record["channel_count"] = len(
                    epochs.ch_names
                )
                record["channel_names"] = json.dumps(
                    epochs.ch_names
                )
                record["channel_types"] = json.dumps(
                    epochs.get_channel_types()
                )
                record["duration_seconds"] = float(
                    epochs.times[-1] - epochs.times[0]
                    + 1.0 / epochs.info["sfreq"]
                )
                record["sample_count"] = int(
                    len(epochs.times)
                )
                record["epoch_count"] = len(epochs)

                if epochs.events is not None:
                    record["event_count"] = len(
                        epochs.events
                    )

                if epochs.event_id:
                    record["event_labels"] = json.dumps(
                        epochs.event_id
                    )

                    for label, event_code in (
                        epochs.event_id.items()
                    ):
                        event_rows.append({
                            "file": relative_path,
                            "source": "EEGLAB epochs",
                            "event_label": label,
                            "event_code": int(event_code),
                            "event_count": int(
                                np.sum(
                                    epochs.events[:, 2]
                                    == event_code
                                )
                            ),
                        })

                print("Data type: epoched EEG")
                print(
                    f"Sampling rate: "
                    f"{record['sampling_rate_hz']} Hz"
                )
                print(
                    f"Channels: {record['channel_count']}"
                )
                print(
                    f"Epochs: {record['epoch_count']}"
                )
                print(
                    f"Event labels: {record['event_labels']}"
                )

            except Exception as epoch_error:
                raise RuntimeError(
                    "Could not read as continuous or epoched EEG. "
                    f"Raw error: {raw_error}. "
                    f"Epoch error: {epoch_error}"
                ) from epoch_error

        # Find a matching events.tsv file, if available.
        possible_tsv_files = list(
            set_path.parent.glob("*events.tsv")
        )

        if possible_tsv_files:
            tsv_path = possible_tsv_files[0]
            record["events_tsv"] = str(
                tsv_path.relative_to(DATA_ROOT)
            )

            try:
                events_df = pd.read_csv(
                    tsv_path,
                    sep="\t",
                )

                print(
                    f"Events TSV: "
                    f"{tsv_path.name}"
                )
                print(
                    f"TSV columns: "
                    f"{list(events_df.columns)}"
                )

                # Preserve the original event labels for inspection.
                if "trial_type" in events_df.columns:
                    counts = (
                        events_df["trial_type"]
                        .fillna("MISSING")
                        .astype(str)
                        .value_counts()
                    )

                    for label, count in counts.items():
                        event_rows.append({
                            "file": relative_path,
                            "source": "events.tsv",
                            "event_label": label,
                            "event_code": None,
                            "event_count": int(count),
                        })

                    print("TSV event counts:")
                    print(counts.to_string())

            except Exception as tsv_error:
                print(
                    f"Could not read events TSV: {tsv_error}"
                )

        record["status"] = "success"

    except Exception as error:
        record["error"] = str(error)

        errors.append({
            "file": relative_path,
            "error": str(error),
        })

        print(f"ERROR: {error}")

    recording_rows.append(record)


# --------------------------------------------------
# 4. Save inspection reports
# --------------------------------------------------

recordings_df = pd.DataFrame(recording_rows)
recordings_df.to_csv(
    OUTPUT_DIR / "recording_inventory.csv",
    index=False,
)

events_df = pd.DataFrame(event_rows)
events_df.to_csv(
    OUTPUT_DIR / "event_inventory.csv",
    index=False,
)

errors_df = pd.DataFrame(
    errors,
    columns=["file", "error"],
)
errors_df.to_csv(
    OUTPUT_DIR / "inspection_errors.csv",
    index=False,
)


# --------------------------------------------------
# 5. Generate summary
# --------------------------------------------------

successful = recordings_df[
    recordings_df["status"] == "success"
]

summary = {
    "dataset_directory": str(DATA_ROOT),
    "recordings_discovered": len(set_files),
    "recordings_inspected_successfully": len(successful),
    "recordings_failed": len(errors),
    "data_types": (
        successful["data_type"]
        .value_counts()
        .to_dict()
    ),
    "sampling_rates_hz": (
        successful["sampling_rate_hz"]
        .dropna()
        .value_counts()
        .to_dict()
    ),
    "channel_counts": (
        successful["channel_count"]
        .dropna()
        .value_counts()
        .to_dict()
    ),
    "files_with_event_metadata": int(
        successful["events_tsv"].notna().sum()
    ),
    "errors": errors,
}

(OUTPUT_DIR / "summary.json").write_text(
    json.dumps(summary, indent=2, default=str),
    encoding="utf-8",
)


# --------------------------------------------------
# 6. Display final results
# --------------------------------------------------

print("\n" + "=" * 70)
print("INSPECTION SUMMARY")
print("=" * 70)

print(
    f"Recordings discovered: {len(set_files)}"
)
print(
    f"Successfully inspected: {len(successful)}"
)
print(
    f"Failed recordings: {len(errors)}"
)

print("\nData types:")
print(
    successful["data_type"].value_counts().to_string()
    if not successful.empty
    else "No successful recordings"
)

print("\nSampling rates:")
print(
    successful["sampling_rate_hz"].value_counts().to_string()
    if not successful.empty
    else "No sampling rates available"
)

print("\nChannel counts:")
print(
    successful["channel_count"].value_counts().to_string()
    if not successful.empty
    else "No channel counts available"
)

print("\nReports saved to:")
print(OUTPUT_DIR)
print("  recording_inventory.csv")
print("  event_inventory.csv")
print("  inspection_errors.csv")
print("  summary.json")

if errors:
    print(
        "\nSome recordings failed inspection. "
        "Review inspection_errors.csv before proceeding."
    )
