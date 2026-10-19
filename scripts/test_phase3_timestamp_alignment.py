"""
Phase 3 — Real EEG/fNIRS timestamp alignment validation.

Validates the reusable timestamp alignment component against
the real ds004514 subject-01 event markers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.synchronization.timestamp_alignment import (
    TimestampAlignment,
    calculate_alignment_error,
    estimate_offset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = Path.home() / "Desktop" / "ds004514"

EEG_EVENTS = (
    DATASET_ROOT
    / "sub-01"
    / "eeg"
    / "sub-01_task-eeg_events.tsv"
)

FNIRS_EVENTS = (
    DATASET_ROOT
    / "sub-01"
    / "nirs"
    / "sub-01_task-nirs_events.tsv"
)


def load_events(path: Path) -> pd.DataFrame:
    """Load and validate one BIDS events TSV file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Event file not found: {path}"
        )

    events = pd.read_csv(
        path,
        sep="\t",
    )

    required_columns = {
        "onset",
        "trial_type",
        "value",
    }

    missing = required_columns - set(events.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in {path.name}: "
            f"{sorted(missing)}"
        )

    events = events.copy()

    events["onset"] = pd.to_numeric(
        events["onset"],
        errors="coerce",
    )

    events = events.dropna(
        subset=["onset"]
    ).reset_index(drop=True)

    return events


def match_events(
    eeg_events: pd.DataFrame,
    fnirs_events: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match corresponding events using trial type and value.

    The matching is occurrence-aware: the first EEG event of a given
    (trial_type, value) pair is matched with the first fNIRS event,
    the second with the second, and so on.
    """

    eeg_groups: dict[tuple[str, str], list[float]] = {}
    fnirs_groups: dict[tuple[str, str], list[float]] = {}

    for _, row in eeg_events.iterrows():
        key = (
            str(row["trial_type"]),
            str(row["value"]),
        )

        eeg_groups.setdefault(
            key,
            [],
        ).append(float(row["onset"]))

    for _, row in fnirs_events.iterrows():
        key = (
            str(row["trial_type"]),
            str(row["value"]),
        )

        fnirs_groups.setdefault(
            key,
            [],
        ).append(float(row["onset"]))

    eeg_matched: list[float] = []
    fnirs_matched: list[float] = []

    common_keys = sorted(
        set(eeg_groups) & set(fnirs_groups)
    )

    for key in common_keys:
        eeg_times = eeg_groups[key]
        fnirs_times = fnirs_groups[key]

        count = min(
            len(eeg_times),
            len(fnirs_times),
        )

        eeg_matched.extend(
            eeg_times[:count]
        )

        fnirs_matched.extend(
            fnirs_times[:count]
        )

    if not eeg_matched:
        raise RuntimeError(
            "No corresponding EEG/fNIRS events were found."
        )

    return (
        np.asarray(eeg_matched),
        np.asarray(fnirs_matched),
    )


def main() -> None:
    print("=" * 72)
    print("PHASE 3 — REAL EEG–fNIRS TIMESTAMP ALIGNMENT")
    print("=" * 72)

    print()
    print("[1] Loading real EEG events...")
    eeg_events = load_events(EEG_EVENTS)

    print(
        f"EEG events loaded: {len(eeg_events)}"
    )

    print()
    print("[2] Loading real fNIRS events...")
    fnirs_events = load_events(FNIRS_EVENTS)

    print(
        f"fNIRS events loaded: {len(fnirs_events)}"
    )

    print()
    print("[3] Matching corresponding events...")

    eeg_times, fnirs_times = match_events(
        eeg_events,
        fnirs_events,
    )

    print(
        f"Matched events: {len(eeg_times)}"
    )

    if len(eeg_times) != len(fnirs_times):
        raise RuntimeError(
            "Matched EEG and fNIRS arrays have different lengths."
        )

    print()
    print("[4] Estimating temporal offset...")

    alignment = estimate_offset(
        eeg_times,
        fnirs_times,
    )

    print(
        f"Estimated offset: "
        f"{alignment.offset_seconds:.6f} s"
    )

    print()
    print("[5] Calculating alignment error...")

    metrics = calculate_alignment_error(
        eeg_times,
        fnirs_times,
        alignment,
    )

    print(
        f"Mean absolute error:   "
        f"{metrics['mean_error_seconds']:.6f} s"
    )

    print(
        f"Median absolute error: "
        f"{metrics['median_error_seconds']:.6f} s"
    )

    print(
        f"Standard deviation:    "
        f"{metrics['std_error_seconds']:.6f} s"
    )

    print(
        f"Maximum absolute error:"
        f" {metrics['max_error_seconds']:.6f} s"
    )

    print()
    print("[6] Validating synchronization quality...")

    if not np.isclose(
        alignment.offset_seconds,
        51.89,
        atol=0.20,
    ):
        raise AssertionError(
            "Estimated EEG–fNIRS offset is outside "
            "the expected ~51.89 second range."
        )

    if metrics["std_error_seconds"] > 0.20:
        raise AssertionError(
            "Synchronization error is too large."
        )

    print("PASS: Offset is approximately 51.89 seconds.")
    print("PASS: Synchronization variation is below 0.20 seconds.")

    print()
    print("[7] Testing timestamp conversion...")

    test_eeg_time = eeg_times[:5]

    converted_fnirs = alignment.eeg_to_fnirs(
        test_eeg_time
    )

    reconstructed_eeg = alignment.fnirs_to_eeg(
        converted_fnirs
    )

    if not np.allclose(
        test_eeg_time,
        reconstructed_eeg,
    ):
        raise AssertionError(
            "Timestamp conversion round-trip failed."
        )

    print(
        "PASS: EEG → fNIRS → EEG round-trip works."
    )

    print()
    print("=" * 72)
    print("PHASE 3 TIMESTAMP ALIGNMENT VALIDATION PASSED")
    print("=" * 72)

    print()
    print("Summary:")
    print(
        f"  EEG events       : {len(eeg_events)}"
    )
    print(
        f"  fNIRS events     : {len(fnirs_events)}"
    )
    print(
        f"  Matched events   : {len(eeg_times)}"
    )
    print(
        f"  Estimated offset : "
        f"{alignment.offset_seconds:.6f} s"
    )
    print(
        f"  Alignment std    : "
        f"{metrics['std_error_seconds']:.6f} s"
    )


if __name__ == "__main__":
    main()