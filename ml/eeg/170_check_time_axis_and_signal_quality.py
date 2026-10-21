from pathlib import Path
from scipy.io import loadmat
import numpy as np
import json

ROOTS = [
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults 3"),
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults-2 2"),
]

SUBJECTS = ["S01", "S03", "S05", "S07", "S08",
            "S09", "S10", "S13", "S14", "S16"]
CLASSES = ["FF", "FR", "RF", "RR"]

def find_file(relative_path):
    for root in ROOTS:
        path = root / relative_path
        if path.exists():
            return path
    return None

# Locate the shared time-axis file.
time_path = None
for root in ROOTS:
    matches = list(root.rglob("EEG_temporal_period_scale.mat"))
    if matches:
        time_path = matches[0]
        break

if time_path is None:
    raise FileNotFoundError("EEG_temporal_period_scale.mat not found.")

time_mat = loadmat(time_path)
time_keys = [k for k in time_mat if not k.startswith("__")]
print("Time-axis file:", time_path)
print("Time-axis variables:", time_keys)

time_report = {}
for key in time_keys:
    arr = np.asarray(time_mat[key])
    if np.issubdtype(arr.dtype, np.number):
        flat = arr.astype(float).ravel()
        time_report[key] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "first_values": flat[:10].tolist(),
            "last_values": flat[-10:].tolist(),
            "minimum": float(np.min(flat)),
            "maximum": float(np.max(flat)),
            "strictly_increasing": bool(np.all(np.diff(flat) > 0)),
        }
        print(f"\n{key}: {time_report[key]}")

report = {
    "time_axis_file": str(time_path),
    "time_axis": time_report,
    "subjects": [],
}

for subject in SUBJECTS:
    rel = Path(subject) / "EEG" / "preprocessed" / f"{subject}_preprocessed.mat"
    path = find_file(rel)
    if path is None:
        print(f"\n{subject}: preprocessed file not found")
        continue

    mat = loadmat(path)
    subject_info = {"subject": subject, "classes": {}}
    print(f"\n{subject}")

    for cls in CLASSES:
        data = np.asarray(mat[cls], dtype=np.float32)
        # Shape: channels x time x epochs.
        # Assess per-epoch absolute peak and robust amplitude quantiles.
        peaks = np.max(np.abs(data), axis=(0, 1))
        sample_abs = np.abs(data)
        q99 = float(np.quantile(sample_abs, 0.99))
        q999 = float(np.quantile(sample_abs, 0.999))
        details = {
            "shape": list(data.shape),
            "epoch_peak_median": float(np.median(peaks)),
            "epoch_peak_p95": float(np.quantile(peaks, 0.95)),
            "epoch_peak_max": float(np.max(peaks)),
            "absolute_sample_q99": q99,
            "absolute_sample_q999": q999,
        }
        subject_info["classes"][cls] = details
        print(
            f"  {cls}: peak median={details['epoch_peak_median']:.2f}, "
            f"peak p95={details['epoch_peak_p95']:.2f}, "
            f"peak max={details['epoch_peak_max']:.2f}, "
            f"|x| q99={q99:.2f}, |x| q99.9={q999:.2f}"
        )

    report["subjects"].append(subject_info)

out = Path("results/eeg/visual_motivation/step170_signal_quality.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\nReport saved:", out)
print("Read-only inspection; no source data or models modified.")
