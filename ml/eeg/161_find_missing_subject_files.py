from pathlib import Path

ROOTS = [
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults 3"),
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults-2 2"),
]

MISSING = ["S02", "S04", "S06", "S11", "S12", "S15"]

for subject in MISSING:
    print(f"\n{'=' * 75}")
    print(f"{subject}")
    print("=" * 75)

    found = False
    for root in ROOTS:
        subject_dir = root / subject
        if not subject_dir.exists():
            continue

        for path in sorted(subject_dir.rglob("*")):
            if path.is_file() and (
                path.suffix.lower() in {".mat", ".edf", ".fif", ".set", ".csv"}
                or "epoch" in path.name.lower()
            ):
                print(path)
                found = True

    if not found:
        print("No matching EEG/data files found in either dataset folder.")

print("\nInspection complete. No files modified.")
