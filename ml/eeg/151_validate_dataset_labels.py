
"""Step 151: Validate EEG label spreadsheets and event annotations."""

from pathlib import Path
import hashlib
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "dataset_1": ROOT / "data" / (
        "A simultaneous EEG-fNIRS dataset of the visual cognitive "
        "motivation study in healthy adults 3"
    ),
    "dataset_2": ROOT / "data" / (
        "A simultaneous EEG-fNIRS dataset of the visual cognitive "
        "motivation study in healthy adults-2 2"
    ),
}

OUT = ROOT / "results" / "eeg" / "dataset_inventory"
OUT.mkdir(parents=True, exist_ok=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_excel(path):
    print(f"\nEXCEL: {path.relative_to(ROOT)}")
    try:
        book = pd.ExcelFile(path)
        for sheet in book.sheet_names:
            frame = pd.read_excel(path, sheet_name=sheet, header=None)
            print(f"  Sheet: {sheet!r}; shape={frame.shape}")
            print(frame.head(12).to_string(index=False, header=False))
    except Exception as exc:
        print(f"  ERROR: {type(exc).__name__}: {exc}")


def inspect_annotation(path):
    print(f"\nANNOTATION: {path.relative_to(ROOT)}")
    try:
        lines = path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        print(f"  Total lines: {len(lines)}")
        for line in lines[:12]:
            print(f"  {line}")
    except Exception as exc:
        print(f"  ERROR: {type(exc).__name__}: {exc}")


def main():
    report = {}

    for name, root in DATASETS.items():
        print("\n" + "=" * 78)
        print(name)
        print("=" * 78)

        if not root.is_dir():
            print(f"Missing dataset directory: {root}")
            continue

        excel_files = sorted(root.glob("S*/S*_type.xlsx"))
        annotation_files = sorted(
            root.glob("S*/EEG/raw/*annotations.txt")
        )
        preprocessed_files = sorted(
            root.glob("S*/EEG/preprocessed/*_preprocessed.mat")
        )

        print(f"Label spreadsheets: {len(excel_files)}")
        print(f"Annotation files: {len(annotation_files)}")
        print(f"Preprocessed EEG files: {len(preprocessed_files)}")

        for path in excel_files:
            inspect_excel(path)

        for path in annotation_files:
            inspect_annotation(path)

        report[name] = {
            "label_files": [
                str(p.relative_to(ROOT)) for p in excel_files
            ],
            "annotation_files": [
                str(p.relative_to(ROOT)) for p in annotation_files
            ],
            "preprocessed_files": [
                str(p.relative_to(ROOT)) for p in preprocessed_files
            ],
        }

    # Compare hashes for identically named preprocessed files
    # in the two dataset folders. This checks exact byte equality.
    print("\n" + "=" * 78)
    print("CROSS-DATASET PREPROCESSED FILE COMPARISON")
    print("=" * 78)

    root1 = DATASETS["dataset_1"]
    root2 = DATASETS["dataset_2"]

    if root1.is_dir() and root2.is_dir():
        files1 = {
            p.relative_to(root1).as_posix(): p
            for p in root1.glob("S*/EEG/preprocessed/*_preprocessed.mat")
        }
        files2 = {
            p.relative_to(root2).as_posix(): p
            for p in root2.glob("S*/EEG/preprocessed/*_preprocessed.mat")
        }

        common = sorted(set(files1) & set(files2))
        if not common:
            print("No matching relative preprocessed file paths.")
        for relative in common:
            same = sha256(files1[relative]) == sha256(files2[relative])
            print(f"{relative}: {'IDENTICAL' if same else 'DIFFERENT'}")

    report_path = OUT / "step151_label_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
