
from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# EXPERIMENT 105: EEG EPOCH-TO-LABEL ALIGNMENT VERIFICATION
# Dataset: ds004022
# Read-only verification; does not modify the original dataset.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "experiments" / "ds004022_verified_epochs" / "eeg_epochs.npz"
)
AUDIT_PATH = (
    ROOT / "experiments" / "ds004022_verified_epochs" / "trial_audit.csv"
)
OUT_DIR = ROOT / "experiments" / "ds004022_experiment105_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Official cue-to-class mapping verified in Experiment 101.
CUE_TO_CLASS = {
    "S3": (0, "reaching"),
    "S4": (1, "grasping"),
    "S5": (2, "lifting"),
    "S6": (3, "twisting"),
}


def normalize_text(value):
    """Normalize text for case-insensitive comparison."""
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def normalize_cue(value):
    """
    Normalize cue markers:
        'S 3' -> 'S3'
        's 4' -> 'S4'
        'S5'  -> 'S5'
        '6'   -> 'S6'
    """
    text = str(value).strip().upper()
    text = "".join(text.split())

    if text.startswith("S"):
        return text

    if text.isdigit():
        return "S" + text

    return text


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not AUDIT_PATH.exists():
        raise FileNotFoundError(f"Trial audit not found: {AUDIT_PATH}")

    # --------------------------------------------------------
    # 1. Load original epoch data and metadata
    # --------------------------------------------------------
    with np.load(DATA_PATH, allow_pickle=True) as data:
        required = [
            "X",
            "y",
            "subjects",
            "runs",
            "onsets_seconds",
            "cue_markers",
            "class_names",
        ]

        missing = [key for key in required if key not in data.files]

        if missing:
            raise KeyError(
                f"Missing NPZ fields: {missing}. "
                f"Available fields: {data.files}"
            )

        X = data["X"]
        y = data["y"].reshape(-1)
        subjects = data["subjects"].reshape(-1)
        runs = data["runs"].reshape(-1)
        onsets = data["onsets_seconds"].reshape(-1)
        cues = data["cue_markers"].reshape(-1)
        class_names = [
            str(value).strip()
            for value in data["class_names"].tolist()
        ]

    audit = pd.read_csv(AUDIT_PATH)

    n = len(y)

    arrays = [X, subjects, runs, onsets, cues]

    if any(len(array) != n for array in arrays):
        raise ValueError(
            "NPZ epoch data and metadata arrays have inconsistent lengths."
        )

    if len(audit) != n:
        raise ValueError(
            f"Audit contains {len(audit)} rows, but NPZ contains {n} trials."
        )

    expected_columns = [
        "subject",
        "run",
        "imagery_onset_seconds",
        "cue_marker",
        "class_name",
        "class_id",
    ]

    missing_columns = [
        column for column in expected_columns
        if column not in audit.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing columns in trial audit: {missing_columns}"
        )

    if not np.issubdtype(y.dtype, np.integer):
        if not np.all(np.equal(y, np.floor(y))):
            raise ValueError("Labels contain non-integer class IDs.")
        y = y.astype(int)

    findings = []

    # --------------------------------------------------------
    # 2. Compare each NPZ trial against the corresponding
    #    row in the extraction audit
    # --------------------------------------------------------
    for i in range(n):
        row = audit.iloc[i]

        npz_subject = normalize_text(subjects[i])
        audit_subject = normalize_text(row["subject"])

        npz_run = normalize_text(runs[i])
        audit_run = normalize_text(row["run"])

        npz_cue = normalize_cue(cues[i])
        audit_cue = normalize_cue(row["cue_marker"])

        npz_class_id = int(y[i])
        audit_class_id = int(row["class_id"])

        if 0 <= npz_class_id < len(class_names):
            npz_class_name = class_names[npz_class_id]
        else:
            npz_class_name = "INVALID_CLASS_ID"

        audit_class_name = str(row["class_name"]).strip()

        npz_class_name_norm = normalize_text(npz_class_name)
        audit_class_name_norm = normalize_text(audit_class_name)

        npz_onset = float(onsets[i])
        audit_onset = float(row["imagery_onset_seconds"])

        onset_difference = abs(npz_onset - audit_onset)

        # Verify cue marker against the official mapping.
        cue_mapping = CUE_TO_CLASS.get(npz_cue)

        cue_mapping_ok = (
            cue_mapping is not None
            and cue_mapping[0] == npz_class_id
            and normalize_text(cue_mapping[1]) == npz_class_name_norm
        )

        # Also verify the audit's cue marker against its class ID/name.
        audit_cue_mapping = CUE_TO_CLASS.get(audit_cue)

        audit_cue_mapping_ok = (
            audit_cue_mapping is not None
            and audit_cue_mapping[0] == audit_class_id
            and normalize_text(audit_cue_mapping[1])
            == audit_class_name_norm
        )

        findings.append({
            "epoch_index": i,

            "subject_match": npz_subject == audit_subject,
            "run_match": npz_run == audit_run,
            "onset_match": onset_difference <= 1e-5,
            "onset_difference_seconds": onset_difference,

            "cue_match": npz_cue == audit_cue,
            "class_id_match": npz_class_id == audit_class_id,
            "class_name_match": (
                npz_class_name_norm == audit_class_name_norm
            ),

            "cue_to_class_mapping_ok": cue_mapping_ok,
            "audit_cue_to_class_mapping_ok": audit_cue_mapping_ok,

            "npz_subject": str(subjects[i]),
            "audit_subject": str(row["subject"]),

            "npz_run": str(runs[i]),
            "audit_run": str(row["run"]),

            "npz_cue": npz_cue,
            "audit_cue": audit_cue,

            "npz_class_id": npz_class_id,
            "audit_class_id": audit_class_id,

            "npz_class_name": npz_class_name,
            "audit_class_name": audit_class_name,
        })

    result = pd.DataFrame(findings)

    # --------------------------------------------------------
    # 3. Calculate verification summary
    # --------------------------------------------------------
    checks = [
        "subject_match",
        "run_match",
        "onset_match",
        "cue_match",
        "class_id_match",
        "class_name_match",
        "cue_to_class_mapping_ok",
        "audit_cue_to_class_mapping_ok",
    ]

    result["all_checks_pass"] = result[checks].all(axis=1)

    result.to_csv(
        OUT_DIR / "epoch_alignment_details.csv",
        index=False,
    )

    summary = {
        "experiment": 105,
        "dataset": "ds004022",
        "trials_checked": int(n),
        "checks": {
            column: {
                "passed": int(result[column].sum()),
                "failed": int((~result[column]).sum()),
            }
            for column in checks
        },
        "trials_passing_all_checks": int(
            result["all_checks_pass"].sum()
        ),
        "trials_failing_any_check": int(
            (~result["all_checks_pass"]).sum()
        ),
        "cue_to_class_mapping": {
            cue: {
                "class_id": value[0],
                "class_name": value[1],
            }
            for cue, value in CUE_TO_CLASS.items()
        },
        "note": (
            "Cue whitespace is normalized before comparison. "
            "Metadata is compared row-by-row against the extraction audit. "
            "Passing these checks supports metadata alignment but does not "
            "by itself prove that the EEG signal contains useful features."
        ),
    }

    with open(
        OUT_DIR / "experiment105_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    # --------------------------------------------------------
    # 4. Print results
    # --------------------------------------------------------
    print("\nEXPERIMENT 105 - ALIGNMENT VERIFICATION")
    print("=" * 55)
    print(json.dumps(summary, indent=2))

    print("\nDetailed results:")
    print(OUT_DIR / "epoch_alignment_details.csv")

    print("\nSummary report:")
    print(OUT_DIR / "experiment105_report.json")

    if summary["trials_failing_any_check"] == 0:
        print("\nSUCCESS: All trials passed all alignment checks.")
    else:
        print(
            "\nWARNING: Some checks failed. "
            "Inspect epoch_alignment_details.csv."
        )


if __name__ == "__main__":
    main()
