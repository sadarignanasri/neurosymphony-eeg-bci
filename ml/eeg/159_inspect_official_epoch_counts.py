from pathlib import Path
import pandas as pd

data_roots = [
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults 3"),
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults-2 2"),
]

files = []
for root in data_roots:
    for path in root.rglob("*"):
        if path.is_file() and path.name.lower().startswith("total_epoch"):
            files.append(path)

if not files:
    print("Total_epoch.xlsx was not found under either dataset folder.")
    print("Searching the data directory for similarly named files:")
    for path in Path("data").rglob("*"):
        if path.is_file() and "total_epoch" in path.name.lower():
            print(path)
    raise SystemExit(1)

seen = set()

for path in sorted(files):
    print(f"\n{'=' * 80}")
    print(f"FILE: {path}")
    print("=" * 80)

    try:
        digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            print("Duplicate file content; skipping.")
            continue
        seen.add(digest)

        workbook = pd.ExcelFile(path)
        print("Sheets:", workbook.sheet_names)

        for sheet in workbook.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet, header=None)
            print(f"\n--- Sheet: {sheet} | shape={df.shape} ---")
            print(df.head(25).to_string(index=False, header=False))
    except Exception as exc:
        print("ERROR:", exc)

print("\nInspection complete. No dataset files were modified.")
