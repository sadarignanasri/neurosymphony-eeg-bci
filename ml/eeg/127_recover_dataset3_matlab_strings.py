
from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat

ROOT = Path(r"D:\neurosymphony-eeg-bci")
DATASET = ROOT / "major datasets" / "28955456"
OUT = ROOT / "experiments" / "dataset3_28955456_audit"
OUT.mkdir(parents=True, exist_ok=True)


def describe(value, depth=0):
    result = {
        "type": type(value).__name__,
        "repr": repr(value)[:1200],
    }

    if isinstance(value, np.ndarray):
        result.update({
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        })

        if value.dtype.names:
            result["fields"] = list(value.dtype.names)
            if depth < 3:
                result["field_details"] = {
                    name: describe(value[name], depth + 1)
                    for name in value.dtype.names
                }
        elif value.dtype == object and depth < 3:
            result["items"] = [
                describe(item, depth + 1)
                for item in value.flat[:20]
            ]

    return result


def main():
    reports = []

    for group in ("Normals", "Patients"):
        path = (
            DATASET / "epo_data" / group / "EEG"
            / "subject1" / "1_epo.mat"
        )

        mat = loadmat(path, squeeze_me=False, struct_as_record=True)
        epo = mat["epo"][0, 0]

        reports.append({
            "group": group,
            "file": str(path),
            "channelsName": describe(epo["channelsName"]),
            "className": describe(epo["className"]),
            "labels": np.asarray(epo["y"]).reshape(-1).astype(int).tolist(),
        })

    source = DATASET / "code" / "EpoDataMaking" / "MakeDatasetFromRaw.m"
    source_lines = source.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()

    relevant_source = [
        {"line": i + 1, "text": line}
        for i, line in enumerate(source_lines)
        if "channels_name" in line.lower()
        or "className" in line
        or "eeg_data" in line.lower()
        or "left_mark" in line.lower()
        or "right_mark" in line.lower()
    ]

    report = {
        "matlab_metadata": reports,
        "original_source_references": relevant_source,
        "note": (
            "Structural inspection only. No original dataset files "
            "were modified."
        ),
    }

    output = OUT / "matlab_string_recovery.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nReport saved: {output}")


if __name__ == "__main__":
    main()
