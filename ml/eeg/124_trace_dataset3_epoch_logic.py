
from pathlib import Path
import re
import json

ROOT = Path(r"D:\neurosymphony-eeg-bci\major datasets\28955456")
OUT = Path(r"D:\neurosymphony-eeg-bci\experiments\dataset3_28955456_audit")
OUT.mkdir(parents=True, exist_ok=True)

CODE_ROOT = ROOT / "code"

# Search the preparation and classification code for evidence
# about epoch dimensions, labels, channels, and sampling frequency.
PATTERNS = re.compile(
    r"EpoDataMaking|className|channelsName|"
    r"epo\.X|epo\.y|epo\.fs|"
    r"epoch|label|event|trigger|"
    r"sampling|sampleRate|sample_rate|"
    r"9531|256|EO|EC",
    re.IGNORECASE,
)

results = []

for path in sorted(CODE_ROOT.rglob("*")):
    if not path.is_file() or path.suffix.lower() not in {
        ".m", ".py", ".txt"
    }:
        continue

    try:
        lines = path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError:
        continue

    # Focus on epoch preparation and code that directly reads
    # the epoched MATLAB structure.
    relative = str(path.relative_to(CODE_ROOT)).lower()
    relevant_file = (
        "epodatamaking" in relative
        or "featurestrainfold" in relative
        or "csptrainfold" in relative
        or "fbcsptrainfold" in relative
        or path.name.lower() in {"test.m", "main.m"}
    )

    if not relevant_file:
        continue

    hits = []
    for i, line in enumerate(lines):
        if PATTERNS.search(line):
            start = max(0, i - 2)
            end = min(len(lines), i + 3)

            hits.append({
                "line": i + 1,
                "context": [
                    f"{j + 1}: {lines[j]}"
                    for j in range(start, end)
                ],
            })

    if hits:
        results.append({
            "file": str(path),
            "matches": hits,
        })

output = OUT / "epoch_logic_source_report.json"
output.write_text(
    json.dumps(results, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("=== DATASET 3 EPOCH LOGIC SEARCH ===")
print("Relevant source files:", len(results))

for item in results:
    print("\nFILE:", item["file"])

    for hit in item["matches"][:35]:
        print("\n".join(hit["context"]))
        print("-" * 40)

print("\nFull report:", output)
print("Original source and dataset files were only read.")
