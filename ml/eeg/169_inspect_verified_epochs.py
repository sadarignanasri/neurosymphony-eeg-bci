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

def locate_file(subject):
    for root in ROOTS:
        path = root / subject / "EEG" / "preprocessed" / f"{subject}_preprocessed.mat"
        if path.exists():
            return path
    return None

report = []

for subject in SUBJECTS:
    path = locate_file(subject)
    if path is None:
        print(f"{subject}: FILE NOT FOUND")
        continue

    mat = loadmat(path)
    subject_report = {
        "subject": subject,
        "file": str(path),
        "classes": {},
    }

    print(f"\n{subject} — {path.name}")

    for cls in CLASSES:
        if cls not in mat:
            raise KeyError(f"{subject}: class '{cls}' missing from MAT file")

        data = np.asarray(mat[cls])
        if data.ndim != 3 or data.shape[:2] != (31, 6500):
            raise ValueError(
                f"{subject}/{cls}: unexpected shape {data.shape}"
            )

        finite = np.isfinite(data)
        values = data[finite]

        details = {
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "finite_fraction": float(finite.mean()),
            "min": float(values.min()) if values.size else None,
            "max": float(values.max()) if values.size else None,
            "mean": float(values.mean()) if values.size else None,
            "std": float(values.std()) if values.size else None,
        }
        subject_report["classes"][cls] = details

        print(
            f"  {cls}: shape={data.shape}, dtype={data.dtype}, "
            f"finite={details['finite_fraction']:.6f}, "
            f"min={details['min']}, max={details['max']}, "
            f"std={details['std']}"
        )

        if not finite.all():
            raise ValueError(f"Non-finite values found in {subject}/{cls}")

    report.append(subject_report)

out = Path("results/eeg/visual_motivation/step169_signal_inspection.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")

print(f"\nInspected subjects: {len(report)}")
print("Report saved:", out)
print("No source data or existing models modified.")
