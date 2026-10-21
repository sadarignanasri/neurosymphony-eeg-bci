
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT / "major datasets"

DATASETS = [
    "ds004514_real",
    "ds004022",
    "Simultaneous EEG-fNIRS Data on Learning Capability",
    "A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults",
]

EEG_EXTENSIONS = {".bdf", ".edf", ".set", ".fif", ".vhdr"}

def inspect_dataset(name):
    folder = ROOT / name

    print("\n" + "=" * 75)
    print(f"DATASET: {name}")
    print("=" * 75)

    if not folder.exists():
        print("Folder not found.")
        return

    files = [
        p for p in folder.rglob("*")
        if p.is_file()
        and ".git" not in p.parts
        and ".datalad" not in p.parts
    ]

    recordings = [
        p for p in files
        if p.suffix.lower() in EEG_EXTENSIONS
        and not p.name.endswith(".fdt")
    ]

    print(f"Total files: {len(files)}")
    print(f"EEG recording files: {len(recordings)}")

    for path in recordings[:12]:
        print("EEG:", path.relative_to(folder))

    for filename in (
        "README", "README.md", "readme.txt",
        "dataset_description.json",
        "task_motorimagery_eeg.json",
    ):
        matches = [
            p for p in files
            if p.parent == folder and p.name.lower() == filename.lower()
        ]

        for path in matches:
            print(f"\n--- {path.name} ---")
            try:
                content = path.read_text(
                    encoding="utf-8", errors="replace"
                )
                print(content[:3500])
            except OSError as error:
                print("Read error:", error)

    event_files = [
        p for p in files
        if p.name.endswith("_events.tsv")
    ]
    print(f"\nEvent metadata files: {len(event_files)}")

    for path in event_files[:3]:
        print(f"\n--- {path.relative_to(folder)} ---")
        try:
            print(path.read_text(
                encoding="utf-8", errors="replace"
            )[:1500])
        except OSError as error:
            print("Read error:", error)

if __name__ == "__main__":
    for dataset in DATASETS:
        inspect_dataset(dataset)

    print("\nInspection complete. No dataset files were modified.")
