
from pathlib import Path
import csv
import json

PROJECT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT / "major datasets"
OUTPUT = PROJECT / "experiments" / "dataset_compatibility_report.csv"

DATASETS = {
    "ds004514_real": {
        "task": "semantic_imagery",
        "classes": 8,
        "channels": 64,
        "sampling_rate": 2048,
        "format": "BDF",
    },
    "ds004022": {
        "task": "motor_imagery",
        "classes": 4,
        "channels": 18,
        "sampling_rate": 500,
        "format": "EEGLAB SET",
    },
    "A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults": {
        "task": "cognitive_motivation",
        "classes": None,
        "channels": None,
        "sampling_rate": None,
        "format": "EDF",
    },
    "Simultaneous EEG-fNIRS Data on Learning Capability": {
        "task": "implicit_learning",
        "classes": None,
        "channels": None,
        "sampling_rate": None,
        "format": "EDF",
    },
}

def count_recordings(folder):
    extensions = {".bdf", ".edf", ".set", ".vhdr", ".fif"}
    return sum(
        1 for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in extensions
        and not p.name.endswith(".fdt")
        and ".git" not in p.parts
        and ".datalad" not in p.parts
    )

def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for name, info in DATASETS.items():
        folder = DATA_ROOT / name
        row = {
            "dataset": name,
            **info,
            "folder_exists": folder.is_dir(),
            "recording_count": (
                count_recordings(folder) if folder.is_dir() else 0
            ),
            "same_task_as_primary": (
                info["task"] == "semantic_imagery"
            ),
            "merge_decision": (
                "Primary benchmark"
                if info["task"] == "semantic_imagery"
                else "Separate task-specific experiment"
            ),
        }
        rows.append(row)

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("Dataset audit saved to:")
    print(OUTPUT)
    print("\nDataset summary:")

    for row in rows:
        print(
            f'{row["dataset"]}: '
            f'{row["recording_count"]} recordings, '
            f'{row["merge_decision"]}'
        )

    print("\nOriginal datasets were not modified.")

if __name__ == "__main__":
    main()
