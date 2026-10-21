
from pathlib import Path
import json

import mne
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT / "major datasets" / "ds004022"
EPOCH_DIR = PROJECT / "experiments" / "ds004022_verified_epochs"
OUT_DIR = PROJECT / "experiments" / "ds004022_experiment109_results"

OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGERY_MARKER = "S 8"
CLASS_CUES = {"S 3", "S 4", "S 5", "S 6"}
EXPECTED_FS = 500.0


def normalize_marker(value):
    return " ".join(str(value).strip().upper().split())


def main():
    npz_path = EPOCH_DIR / "eeg_epochs.npz"
    audit_path = EPOCH_DIR / "trial_audit.csv"

    if not npz_path.exists() or not audit_path.exists():
        raise FileNotFoundError(
            "Verified epochs or trial audit not found."
        )

    set_files = sorted(DATA_ROOT.rglob("*run-*_eeg.set"))

    if not set_files:
        raise FileNotFoundError(
            f"No original EEGLAB files found in {DATA_ROOT}"
        )

    with np.load(npz_path, allow_pickle=True) as d:
        X = d["X"]
        y = d["y"]
        subjects = d["subjects"].astype(str)
        runs = d["runs"].astype(str)
        onsets = d["onsets_seconds"].astype(float)
        cues = d["cue_markers"].astype(str)

        fs = float(np.asarray(d["sampling_rate"]).squeeze())

    audit = pd.read_csv(audit_path)

    if fs != EXPECTED_FS:
        raise ValueError(f"Expected {EXPECTED_FS} Hz, found {fs}")

    if len(X) != len(audit):
        raise ValueError("Epoch count and audit count differ.")

    # Confirm the extracted data still corresponds to the audit.
    normalized_cues = np.array([normalize_marker(c) for c in cues])
    audit_cues = np.array([
        normalize_marker(c) for c in audit["cue_marker"]
    ])

    metadata_checks = {
        "trial_count_matches": len(X) == len(audit),
        "subjects_match": np.array_equal(
            subjects, audit["subject"].astype(str).to_numpy()
        ),
        "runs_match": np.array_equal(
            runs, audit["run"].astype(str).to_numpy()
        ),
        "onsets_match": np.allclose(
            onsets,
            audit["imagery_onset_seconds"].to_numpy(float),
            atol=1e-6,
        ),
        "cues_match": np.array_equal(normalized_cues, audit_cues),
        "labels_match": np.array_equal(
            y.astype(int), audit["class_id"].to_numpy(int)
        ),
    }

    print("\nEXPERIMENT 109 — ORIGINAL EVENT TIMING")
    print("=" * 65)
    print("Original recordings:", len(set_files))
    print("Epoch tensor:", X.shape)
    print("Metadata checks:", metadata_checks)

    if not all(metadata_checks.values()):
        raise ValueError("An existing epoch/audit alignment check failed.")

    event_rows = []
    recording_rows = []

    for set_path in set_files:
        subject = next(
            part for part in set_path.parts
            if part.startswith("sub-")
        )
        run_token = set_path.stem.split("_run-")[-1]
        run_number = int(run_token.split("_")[0])

        print(f"Inspecting {subject}, run {run_number}")

        raw = mne.io.read_raw_eeglab(
            set_path,
            preload=False,
            verbose="ERROR",
        )

        try:
            raw.rename_channels({
                name: name.strip() for name in raw.ch_names
            })

            recording_duration = float(raw.times[-1])
            sfreq = float(raw.info["sfreq"])

            annotations = sorted(
                [
                    {
                        "onset": float(a["onset"]),
                        "marker": normalize_marker(a["description"]),
                        "duration": float(a["duration"]),
                    }
                    for a in raw.annotations
                ],
                key=lambda a: a["onset"],
            )

            imagery_events = [
                a for a in annotations
                if a["marker"] == IMAGERY_MARKER
            ]

            cues = [
                a for a in annotations
                if a["marker"] in CLASS_CUES
            ]

            # Record every relevant event and its neighbors.
            for i, event in enumerate(annotations):
                if event["marker"] not in CLASS_CUES | {IMAGERY_MARKER}:
                    continue

                previous_event = annotations[i - 1] if i > 0 else None
                next_event = (
                    annotations[i + 1]
                    if i + 1 < len(annotations)
                    else None
                )

                event_rows.append({
                    "subject": subject,
                    "run": run_number,
                    "event_index": i,
                    "marker": event["marker"],
                    "onset_seconds": event["onset"],
                    "duration_seconds": event["duration"],
                    "previous_marker": (
                        previous_event["marker"]
                        if previous_event else None
                    ),
                    "previous_onset_seconds": (
                        previous_event["onset"]
                        if previous_event else np.nan
                    ),
                    "next_marker": (
                        next_event["marker"]
                        if next_event else None
                    ),
                    "next_onset_seconds": (
                        next_event["onset"]
                        if next_event else np.nan
                    ),
                    "seconds_to_next_event": (
                        next_event["onset"] - event["onset"]
                        if next_event else np.nan
                    ),
                })

            onset_diffs = np.diff([
                event["onset"] for event in imagery_events
            ])

            recording_rows.append({
                "subject": subject,
                "run": run_number,
                "sampling_rate_hz": sfreq,
                "recording_duration_seconds": recording_duration,
                "annotation_count": len(annotations),
                "class_cue_count": len(cues),
                "imagery_marker_count": len(imagery_events),
                "imagery_intervals_median_seconds": (
                    float(np.median(onset_diffs))
                    if len(onset_diffs) else np.nan
                ),
                "imagery_intervals_min_seconds": (
                    float(np.min(onset_diffs))
                    if len(onset_diffs) else np.nan
                ),
                "imagery_intervals_max_seconds": (
                    float(np.max(onset_diffs))
                    if len(onset_diffs) else np.nan
                ),
                "first_imagery_onset_seconds": (
                    imagery_events[0]["onset"]
                    if imagery_events else np.nan
                ),
                "last_imagery_onset_seconds": (
                    imagery_events[-1]["onset"]
                    if imagery_events else np.nan
                ),
            })

        finally:
            raw.close()

    events_df = pd.DataFrame(event_rows)
    recordings_df = pd.DataFrame(recording_rows)

    events_df.to_csv(OUT_DIR / "original_event_timeline.csv", index=False)
    recordings_df.to_csv(
        OUT_DIR / "recording_timing_summary.csv", index=False
    )

    # Compare every stored imagery onset with the corresponding S8 event.
    comparison_rows = []

    for _, row in audit.iterrows():
        matching = events_df[
            (events_df["subject"] == str(row["subject"]))
            & (events_df["run"].astype(int) == int(row["run"]))
            & (events_df["marker"] == IMAGERY_MARKER)
            & np.isclose(
                events_df["onset_seconds"].to_numpy(float),
                float(row["imagery_onset_seconds"]),
                atol=1e-6,
            )
        ]

        comparison_rows.append({
            "subject": str(row["subject"]),
            "run": int(row["run"]),
            "trial_index": int(row["trial_index"]),
            "stored_onset_seconds": float(
                row["imagery_onset_seconds"]
            ),
            "matching_original_S8_events": len(matching),
            "exactly_one_original_S8_match": len(matching) == 1,
            "stored_cue_marker": normalize_marker(row["cue_marker"]),
            "stored_class_name": row["class_name"],
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(
        OUT_DIR / "epoch_to_original_event_comparison.csv",
        index=False,
    )

    report = {
        "experiment": 109,
        "dataset": "ds004022",
        "original_recordings_inspected": len(set_files),
        "epochs": int(len(X)),
        "metadata_checks": metadata_checks,
        "original_S8_event_matches": int(
            comparison_df["exactly_one_original_S8_match"].sum()
        ),
        "epochs_with_exactly_one_original_S8_match": int(len(X)),
        "all_epochs_have_exactly_one_S8_match": bool(
            comparison_df["exactly_one_original_S8_match"].all()
        ),
        "all_recordings_at_expected_sampling_rate": bool(
            np.allclose(
                recordings_df["sampling_rate_hz"],
                EXPECTED_FS,
            )
        ),
        "important_limitation": (
            "Event alignment confirms that stored onset timestamps "
            "match original S8 annotations. It does not establish the "
            "biological start/end of motor imagery or prove that five "
            "seconds is the optimal decoding window."
        ),
    }

    with open(
        OUT_DIR / "experiment109_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(report, f, indent=2)

    print("\nEXPERIMENT 109 COMPLETE")
    print("=" * 65)
    print(json.dumps(report, indent=2))
    print("\nRecording summary:")
    print(recordings_df.to_string(index=False))
    print("\nOutputs:", OUT_DIR)


if __name__ == "__main__":
    main()
