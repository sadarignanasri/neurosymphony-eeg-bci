
from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat

ROOT = Path(r"D:\neurosymphony-eeg-bci")
DATASET = ROOT / "major datasets" / "28955456"
OUTPUT = ROOT / "experiments" / "dataset3_28955456_audit"
OUTPUT.mkdir(parents=True, exist_ok=True)


def decode_matlab_value(value, depth=0):
    """Convert common MATLAB metadata structures to readable Python values."""
    if depth > 8:
        return repr(value)

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, str):
        return value

    if isinstance(value, np.ndarray):
        if value.dtype.names:
            result = {}
            for name in value.dtype.names:
                result[name] = decode_matlab_value(
                    value[name], depth + 1
                )
            return result

        if value.dtype == object:
            return [
                decode_matlab_value(item, depth + 1)
                for item in value.flat
            ]

        if value.size == 1:
            return decode_matlab_value(value.item(), depth + 1)

        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, (list, tuple)):
        return [decode_matlab_value(item, depth + 1) for item in value]

    return repr(value)


def inspect_file(group):
    path = (
        DATASET / "epo_data" / group / "EEG"
        / "subject1" / "1_epo.mat"
    )

    if not path.exists():
        return {"group": group, "error": f"File not found: {path}"}

    mat = loadmat(path, squeeze_me=False, struct_as_record=True)
    if "epo" not in mat:
        return {"group": group, "error": "MATLAB variable 'epo' not found"}

    epo = mat["epo"]

    if epo.size != 1 or not epo.dtype.names:
        return {
            "group": group,
            "error": "Unexpected MATLAB structure",
            "dtype": str(epo.dtype),
            "shape": list(epo.shape),
        }

    fields = epo.dtype.names
    result = {
        "group": group,
        "file": str(path),
        "fields": list(fields),
    }

    X = epo["X"][0, 0]
    y = np.asarray(epo["y"][0, 0]).reshape(-1)
    fs = np.asarray(epo["fs"][0, 0]).reshape(-1)

    result["X_shape"] = list(X.shape)
    result["X_dtype"] = str(X.dtype)
    result["sampling_frequency"] = decode_matlab_value(fs)
    result["trial_count"] = int(y.size)
    result["label_counts"] = {
        str(int(label)): int(np.sum(y == label))
        for label in np.unique(y)
    }
    result["labels_valid_binary"] = bool(np.isin(y, [0, 1]).all())
    result["finite_fraction"] = float(np.isfinite(X).mean())

    for field in ("channelsName", "className"):
        if field in fields:
            result[field] = decode_matlab_value(epo[field][0, 0])

    result["EO_shape"] = (
        list(epo["EO"][0, 0].shape) if "EO" in fields else None
    )
    result["EC_shape"] = (
        list(epo["EC"][0, 0].shape) if "EC" in fields else None
    )

    return result


def main():
    reports = [
        inspect_file("Normals"),
        inspect_file("Patients"),
    ]

    report_path = OUTPUT / "decoded_metadata.json"
    report_path.write_text(
        json.dumps(reports, indent=2, default=str),
        encoding="utf-8",
    )

    print("\n=== DATASET 3 DECODED METADATA ===")
    print(json.dumps(reports, indent=2, default=str))
    print(f"\nReport saved: {report_path}")
    print("Original dataset files were read only.")


if __name__ == "__main__":
    main()
