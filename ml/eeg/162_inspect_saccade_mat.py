from pathlib import Path
from scipy.io import loadmat
import numpy as np

ROOTS = [
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults 3"),
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults-2 2"),
]

SUBJECTS = ["S02", "S04", "S06", "S11", "S12", "S15"]

def inspect(value, prefix=""):
    if isinstance(value, np.ndarray):
        print(f"{prefix}shape={value.shape}, dtype={value.dtype}")

        if value.dtype == object or value.dtype.names:
            for index, item in enumerate(value.flat[:5]):
                inspect(item, f"{prefix}[{index}] ")
        elif value.size <= 12:
            print(f"{prefix}values={value.ravel().tolist()}")

    elif hasattr(value, "_fieldnames"):
        print(f"{prefix}MATLAB structure fields: {value._fieldnames}")
        for field in value._fieldnames:
            inspect(getattr(value, field), f"{prefix}.{field} ")

    else:
        print(f"{prefix}type={type(value).__name__}, value={str(value)[:160]}")

for subject in SUBJECTS:
    print(f"\n{'=' * 75}\n{subject}\n{'=' * 75}")

    seen = set()
    for root in ROOTS:
        folder = root / subject / "EEG" / "preprocessed"
        if not folder.exists():
            continue

        for path in sorted(folder.glob("*_saccade.mat")):
            import hashlib
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen:
                print(f"DUPLICATE CONTENT: {path}")
                continue
            seen.add(digest)

            print(f"\nFILE: {path}")
            try:
                mat = loadmat(path, struct_as_record=False, squeeze_me=False)
                for key, value in mat.items():
                    if not key.startswith("__"):
                        inspect(value, f"{key} ")
            except Exception as exc:
                print("ERROR:", exc)

print("\nInspection complete. No files modified.")
