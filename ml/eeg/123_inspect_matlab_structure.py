
from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat

ROOT = Path(
    r"D:\neurosymphony-eeg-bci\major datasets\28955456"
)

OUT = Path(
    r"D:\neurosymphony-eeg-bci\experiments\dataset3_28955456_audit"
)
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = [
    ROOT / "epo_data" / "Normals" / "EEG" / "subject1" / "1_epo.mat",
    ROOT / "epo_data" / "Patients" / "EEG" / "subject1" / "1_epo.mat",
]


def describe(value, depth=0, max_depth=4):
    """Describe MATLAB/NumPy objects without coercing ragged structures."""

    if value is None:
        return None

    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "value": value.decode("utf-8", errors="replace")[:500],
        }

    if isinstance(value, str):
        return {"type": "str", "value": value[:500]}

    if isinstance(value, np.generic):
        return describe(value.item(), depth, max_depth)

    if isinstance(value, np.ndarray):
        info = {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

        if depth >= max_depth:
            return info

        if value.dtype.names:
            info["field_names"] = list(value.dtype.names)
            return info

        if value.size <= 12:
            info["items"] = [
                describe(item, depth + 1, max_depth)
                for item in value.flat
            ]
        elif value.dtype.kind in "biuf":
            finite = value[np.isfinite(value)]
            if finite.size:
                info["min"] = float(finite.min())
                info["max"] = float(finite.max())

        return info

    if isinstance(value, np.void):
        info = {"type": "numpy.void"}

        if value.dtype.names:
            info["fields"] = {
                name: describe(
                    value[name], depth + 1, max_depth
                )
                for name in value.dtype.names
            }

        return info

    if hasattr(value, "_fieldnames"):
        info = {
            "type": "mat_struct",
            "fields": {},
        }

        if depth < max_depth:
            for name in value._fieldnames:
                info["fields"][name] = describe(
                    getattr(value, name),
                    depth + 1,
                    max_depth,
                )

        return info

    if isinstance(value, (int, float, bool)):
        return {"type": type(value).__name__, "value": value}

    return {
        "type": type(value).__name__,
        "repr": repr(value)[:500],
    }


report = []

for path in TARGETS:
    print("\n" + "=" * 70)
    print("FILE:", path)

    if not path.exists():
        print("FILE NOT FOUND")
        continue

    data = loadmat(
        path,
        squeeze_me=False,
        struct_as_record=False,
    )

    print("TOP-LEVEL VARIABLES:", [
        key for key in data if not key.startswith("__")
    ])

    if "epo" not in data:
        print("No 'epo' variable found.")
        continue

    epo = data["epo"]

    # Preserve the original structure; do not squeeze it into
    # a potentially ambiguous or ragged NumPy array.
    if isinstance(epo, np.ndarray) and epo.size == 1:
        epo = epo.flat[0]

    if hasattr(epo, "_fieldnames"):
        fields = epo._fieldnames
    elif isinstance(epo, np.void) and epo.dtype.names:
        fields = list(epo.dtype.names)
    else:
        fields = []

    print("EPO FIELDS:", fields)

    field_report = {}

    for name in fields:
        if hasattr(epo, "_fieldnames"):
            value = getattr(epo, name)
        else:
            value = epo[name]

        field_report[name] = describe(value)

        print(f"\nFIELD: {name}")
        print(json.dumps(
            field_report[name],
            indent=2,
            ensure_ascii=False,
            default=str,
        )[:6000])

    report.append({
        "file": str(path),
        "fields": fields,
        "details": field_report,
    })

output = OUT / "matlab_structure_details.json"
output.write_text(
    json.dumps(report, indent=2, ensure_ascii=False, default=str),
    encoding="utf-8",
)

print("\nSaved detailed structure report:", output)
print("Original dataset files were only read.")
