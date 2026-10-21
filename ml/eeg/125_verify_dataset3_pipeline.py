
from pathlib import Path
import json

ROOT = Path(r"D:\neurosymphony-eeg-bci\major datasets\28955456")
OUT = Path(r"D:\neurosymphony-eeg-bci\experiments\dataset3_28955456_audit")
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = {
    "epoch_creation": (
        ROOT / "code" / "EpoDataMaking" / "MakeDatasetFromRaw.m",
        1,
        75,
    ),
    "classification_preprocessing": (
        ROOT / "code" / "CLFResult" / "FeaturesTrainFold.m",
        1,
        45,
    ),
}

report = {}

for name, (path, start, end) in TARGETS.items():
    print("\n" + "=" * 75)
    print(name.upper())
    print("FILE:", path)

    if not path.exists():
        print("ERROR: File not found")
        report[name] = {"file": str(path), "error": "File not found"}
        continue

    lines = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    selected = lines[start - 1:end]

    print(f"LINES {start}-{min(end, len(lines))}")
    for number, line in enumerate(selected, start=start):
        print(f"{number:4}: {line}")

    report[name] = {
        "file": str(path),
        "start_line": start,
        "end_line": min(end, len(lines)),
        "source": [
            {"line": i, "text": line}
            for i, line in enumerate(selected, start=start)
        ],
    }

output = OUT / "verified_pipeline_source.json"
output.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\nSaved report:", output)
print("Only source files were read; no dataset files were modified.")
