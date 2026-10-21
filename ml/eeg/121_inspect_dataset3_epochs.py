
from pathlib import Path
import csv
import json
import numpy as np
from scipy.io import loadmat

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(
    r"D:\neurosymphony-eeg-bci\major datasets\28955456"
)

OUT = Path(
    r"D:\neurosymphony-eeg-bci\experiments\dataset3_28955456_audit"
)

OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# MATLAB STRUCTURE HELPERS
# ============================================================

def unwrap(value):
    """Unwrap single-element NumPy arrays."""
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    return value


def get_field(obj, name):
    """Read a field from a MATLAB structure."""
    obj = unwrap(obj)

    if hasattr(obj, "_fieldnames") and name in obj._fieldnames:
        return unwrap(getattr(obj, name))

    if isinstance(obj, np.void):
        if obj.dtype.names and name in obj.dtype.names:
            return unwrap(obj[name])

    if isinstance(obj, dict):
        return unwrap(obj.get(name))

    return None


def to_python(value):
    """Convert NumPy values into JSON-friendly Python values."""
    if value is None:
        return None

    value = unwrap(value)

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S"}:
            return [
                item.decode("utf-8", errors="replace")
                if isinstance(item, bytes)
                else str(item)
                for item in value.ravel().tolist()
            ]

        if value.size <= 100:
            return value.tolist()

        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    if isinstance(value, np.generic):
        return value.item()

    return str(value)


def describe(value):
    """Summarize a field without dumping large EEG arrays."""
    if value is None:
        return None

    value = unwrap(value)

    if isinstance(value, np.ndarray):
        result = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

        if value.size <= 30:
            result["values"] = to_python(value)

        elif value.dtype.kind in "biuf":
            finite = value[np.isfinite(value)]

            if finite.size:
                result["min"] = float(finite.min())
                result["max"] = float(finite.max())

        elif value.dtype.kind in {"U", "S"}:
            result["values_preview"] = to_python(
                value.ravel()[:30]
            )

        return result

    return to_python(value)


def label_counts(y):
    """Count each label without assuming labels are consecutive integers."""
    if y is None:
        return {}

    arr = np.asarray(unwrap(y)).ravel()
    counts = {}

    for item in arr:
        if isinstance(item, bytes):
            key = item.decode("utf-8", errors="replace")
        else:
            key = str(item)

        counts[key] = counts.get(key, 0) + 1

    return counts


def safe_shape(value):
    if value is None:
        return None

    try:
        return list(np.asarray(value).shape)
    except Exception:
        return None


# ============================================================
# VALIDATE DATASET PATH
# ============================================================

if not ROOT.exists():
    raise FileNotFoundError(
        f"Dataset directory does not exist: {ROOT}"
    )

print("Dataset root:", ROOT)
print("Original data will only be read.")
print("Reports will be saved to:", OUT)


# ============================================================
# RECURSIVE EPOCH INSPECTION
# ============================================================

records = []

for group in ["Normals", "Patients"]:

    eeg_dir = ROOT / "epo_data" / group / "EEG"

    if not eeg_dir.exists():
        print(f"WARNING: Missing EEG directory: {eeg_dir}")
        continue

    # FIX: Search every subject/session subfolder recursively.
    mat_files = sorted(eeg_dir.rglob("*.mat"))

    print(f"\n{group}: found {len(mat_files)} MATLAB files")

    for path in mat_files:

        record = {
            "group": group,
            "subject": path.parent.name,
            "session_file": path.name,
            "file": str(path),
            "error": None,
        }

        try:
            data = loadmat(
                path,
                squeeze_me=True,
                struct_as_record=False,
            )

            if "epo" not in data:
                raise ValueError(
                    "MAT file does not contain an 'epo' variable"
                )

            epo = unwrap(data["epo"])

            X = get_field(epo, "X")
            y = get_field(epo, "y")
            fs = get_field(epo, "fs")
            channels = get_field(epo, "channelsName")
            classes = get_field(epo, "className")
            eo = get_field(epo, "EO")
            ec = get_field(epo, "EC")

            X_array = (
                np.asarray(X) if X is not None else None
            )

            y_array = (
                np.asarray(y).ravel() if y is not None else None
            )

            record.update({
                "X_shape": safe_shape(X_array),
                "X_dtype": (
                    str(X_array.dtype)
                    if X_array is not None else None
                ),
                "X_finite_fraction": None,
                "X_min": None,
                "X_max": None,
                "X_nan_count": None,
                "X_inf_count": None,
                "y_shape": safe_shape(y_array),
                "label_counts": label_counts(y),
                "fs": describe(fs),
                "channelsName": describe(channels),
                "className": describe(classes),
                "EO": describe(eo),
                "EC": describe(ec),
            })

            # Basic numeric integrity checks, without modifying X.
            if (
                X_array is not None
                and np.issubdtype(X_array.dtype, np.number)
            ):
                finite_mask = np.isfinite(X_array)

                record["X_finite_fraction"] = (
                    float(finite_mask.mean())
                    if X_array.size else None
                )

                record["X_nan_count"] = int(
                    np.isnan(X_array).sum()
                )

                record["X_inf_count"] = int(
                    np.isinf(X_array).sum()
                )

                finite_values = X_array[finite_mask]

                if finite_values.size:
                    record["X_min"] = float(
                        finite_values.min()
                    )
                    record["X_max"] = float(
                        finite_values.max()
                    )

        except Exception as exc:
            record["error"] = repr(exc)

        records.append(record)


# ============================================================
# SAVE REPORTS
# ============================================================

json_path = OUT / "epoch_inspection.json"

json_path.write_text(
    json.dumps(records, indent=2, default=str),
    encoding="utf-8",
)

csv_path = OUT / "epoch_inventory.csv"

columns = [
    "group",
    "subject",
    "session_file",
    "file",
    "X_shape",
    "X_dtype",
    "X_finite_fraction",
    "X_min",
    "X_max",
    "X_nan_count",
    "X_inf_count",
    "y_shape",
    "label_counts",
    "fs",
    "channelsName",
    "className",
    "EO",
    "EC",
    "error",
]

with csv_path.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=columns,
        extrasaction="ignore",
    )

    writer.writeheader()

    for record in records:
        row = {}

        for key in columns:
            value = record.get(key)

            if isinstance(value, (dict, list)):
                value = json.dumps(value, default=str)

            row[key] = value

        writer.writerow(row)


# ============================================================
# PRINT SUMMARY
# ============================================================

successful = [
    r for r in records if r.get("error") is None
]

failed = [
    r for r in records if r.get("error") is not None
]

print("\n" + "=" * 60)
print("DATASET 3 EPOCH INSPECTION SUMMARY")
print("=" * 60)

print("Total files inspected:", len(records))
print("Successful inspections:", len(successful))
print("Failed inspections:", len(failed))

for group in ["Normals", "Patients"]:
    group_records = [
        r for r in records if r["group"] == group
    ]

    group_success = [
        r for r in group_records if r.get("error") is None
    ]

    print(
        f"{group}: {len(group_success)} successful / "
        f"{len(group_records)} discovered"
    )

# Print two representative files from each group.
for group in ["Normals", "Patients"]:

    examples = [
        r for r in successful if r["group"] == group
    ]

    print(f"\n--- REPRESENTATIVE EEG EPOCHS: {group} ---")

    for record in examples[:2]:
        print(json.dumps(record, indent=2, default=str))

if failed:
    print("\n--- FIRST INSPECTION ERRORS ---")

    for record in failed[:5]:
        print(record["file"], "->", record["error"])

print("\nJSON report:", json_path)
print("CSV report:", csv_path)
print("\nInspection finished. Original dataset files were only read.")
