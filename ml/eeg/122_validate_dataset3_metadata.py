
from pathlib import Path
import json
import re
import numpy as np
from scipy.io import loadmat

ROOT = Path(r"D:\neurosymphony-eeg-bci\major datasets\28955456")
OUT = Path(r"D:\neurosymphony-eeg-bci\experiments\dataset3_28955456_audit")
OUT.mkdir(parents=True, exist_ok=True)

KEYWORDS = re.compile(
    r"className|channelsName|epo\.X|epo\.y|"
    r"\bfs\b|epoch|label|event|trigger|EO|EC",
    re.IGNORECASE,
)


def unwrap(value):
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    return value


def field(obj, name):
    obj = unwrap(obj)

    if hasattr(obj, "_fieldnames") and name in obj._fieldnames:
        return unwrap(getattr(obj, name))

    if isinstance(obj, np.void) and obj.dtype.names:
        if name in obj.dtype.names:
            return unwrap(obj[name])

    if isinstance(obj, dict):
        return unwrap(obj.get(name))

    return None


def describe_labels(value):
    if value is None:
        return {"present": False}

    arr = np.asarray(unwrap(value))
    flat = arr.ravel()
    values = []

    for item in flat[:40]:
        item = unwrap(item)

        if isinstance(item, bytes):
            item = item.decode("utf-8", errors="replace")
        elif isinstance(item, np.generic):
            item = item.item()

        values.append(str(item))

    return {
        "present": True,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "unique_values": sorted(set(values)),
        "sample_values": values[:20],
    }


def scalar_description(value):
    if value is None:
        return None

    arr = np.asarray(unwrap(value))

    if arr.size <= 20:
        return arr.tolist()

    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


# ------------------------------------------------------------
# 1. Inspect all epoched EEG files
# ------------------------------------------------------------

records = []

for group in ["Normals", "Patients"]:
    eeg_root = ROOT / "epo_data" / group / "EEG"

    if not eeg_root.exists():
        continue

    for path in sorted(eeg_root.rglob("*.mat")):
        rec = {
            "group": group,
            "subject": path.parent.name,
            "file": str(path),
        }

        try:
            mat = loadmat(
                path,
                squeeze_me=True,
                struct_as_record=False,
            )

            if "epo" not in mat:
                raise ValueError("Missing epo variable")

            epo = unwrap(mat["epo"])
            X = field(epo, "X")
            y = field(epo, "y")
            fs = field(epo, "fs")
            channels = field(epo, "channelsName")
            classes = field(epo, "className")

            X = np.asarray(X) if X is not None else None
            y_arr = np.asarray(unwrap(y)).ravel() if y is not None else None

            rec.update({
                "X_shape": list(X.shape) if X is not None else None,
                "X_dtype": str(X.dtype) if X is not None else None,
                "y_shape": list(y_arr.shape) if y_arr is not None else None,
                "y_dtype": str(y_arr.dtype) if y_arr is not None else None,
                "y_values": describe_labels(y),
                "fs": scalar_description(fs),
                "channels_shape": (
                    list(np.asarray(channels).shape)
                    if channels is not None else None
                ),
                "channels_dtype": (
                    str(np.asarray(channels).dtype)
                    if channels is not None else None
                ),
                "channels_repr": repr(channels)[:1500],
                "classes_shape": (
                    list(np.asarray(classes).shape)
                    if classes is not None else None
                ),
                "classes_dtype": (
                    str(np.asarray(classes).dtype)
                    if classes is not None else None
                ),
                "classes_repr": repr(classes)[:1500],
                "error": None,
            })

        except Exception as exc:
            rec["error"] = repr(exc)

        records.append(rec)

report_path = OUT / "metadata_validation.json"
report_path.write_text(
    json.dumps(records, indent=2, default=str),
    encoding="utf-8",
)

# ------------------------------------------------------------
# 2. Summarize dimensions, sampling rates, and labels
# ------------------------------------------------------------

valid = [r for r in records if not r.get("error")]
failed = [r for r in records if r.get("error")]

def frequency_summary(rows):
    return sorted({
        json.dumps(r.get("fs"), sort_keys=True)
        for r in rows
    })

print("\n=== DATASET 3 METADATA VALIDATION ===")
print("Files discovered:", len(records))
print("Successful:", len(valid))
print("Failed:", len(failed))

for group in ["Normals", "Patients"]:
    subset = [r for r in valid if r["group"] == group]

    print(f"\n{group}: {len(subset)} files")
    print("X shapes:", sorted({
        str(r.get("X_shape")) for r in subset
    }))
    print("Sampling rate representations:", frequency_summary(subset))
    print("Label representations:", sorted({
        json.dumps(r.get("y_values"), sort_keys=True)
        for r in subset
    }))

print("\n=== FIRST 4 SUCCESSFUL FILES ===")
for rec in valid[:4]:
    print(json.dumps(rec, indent=2, default=str))

if failed:
    print("\n=== FIRST ERRORS ===")
    for rec in failed[:5]:
        print(rec["file"], rec["error"])

# ------------------------------------------------------------
# 3. Search dataset preparation source code
# ------------------------------------------------------------

code_root = ROOT / "code"
source_hits = []

if code_root.exists():
    extensions = {".m", ".py", ".txt"}

    for path in sorted(code_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            continue

        lines = text.splitlines()

        for index, line in enumerate(lines):
            if KEYWORDS.search(line):
                source_hits.append({
                    "file": str(path),
                    "line": index + 1,
                    "text": line.strip()[:350],
                })

source_path = OUT / "preparation_code_matches.json"
source_path.write_text(
    json.dumps(source_hits, indent=2),
    encoding="utf-8",
)

print("\n=== PREPARATION CODE MATCHES ===")
print("Matching lines:", len(source_hits))

for hit in source_hits[:100]:
    print(
        f"{hit['file']}:{hit['line']}: "
        f"{hit['text']}"
    )

print("\nSaved metadata report:", report_path)
print("Saved source-code matches:", source_path)
print("Original dataset files were only read.")
