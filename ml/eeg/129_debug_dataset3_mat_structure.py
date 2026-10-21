
from pathlib import Path
import numpy as np
from scipy.io import loadmat

ROOT = Path(r"D:\neurosymphony-eeg-bci\major datasets\28955456")

candidates = [
    ROOT / "epo_data" / "Normals" / "EEG" / "subject1" / "1_epo.mat",
    ROOT / "epo_data" / "Patients" / "EEG" / "subject1" / "1_epo.mat",
]

def describe(name, value):
    print(f"\n{name}")
    print("  type:", type(value))
    if isinstance(value, np.ndarray):
        print("  shape:", value.shape)
        print("  dtype:", value.dtype)
        if value.dtype.names:
            print("  fields:", value.dtype.names)

for path in candidates:
    if not path.exists():
        print(f"\nFile not found: {path}")
        continue

    print("\n" + "=" * 70)
    print("FILE:", path)

    mat = loadmat(path, squeeze_me=False, struct_as_record=True)
    describe("mat['epo']", mat["epo"])

    epo_array = mat["epo"]
    epo = epo_array[0, 0]

    for field in ("X", "y", "fs", "channelsName", "className"):
        value = epo[field]
        describe(f"epo[{field!r}]", value)

        if isinstance(value, np.ndarray) and value.size == 1:
            unwrapped = value.item()
            describe(f"epo[{field!r}].item()", unwrapped)

    # Compare direct field extraction with object unwrapping.
    for method in ("index", "item"):
        try:
            if method == "index":
                X = np.asarray(epo["X"][0, 0])
                y = np.asarray(epo["y"][0, 0])
            else:
                X = np.asarray(epo["X"].item())
                y = np.asarray(epo["y"].item())

            print(f"\nExtraction method: {method}")
            print("  X shape:", X.shape, "dtype:", X.dtype)
            print("  y shape:", y.shape, "dtype:", y.dtype)
            if X.size:
                print("  X finite fraction:", np.isfinite(X).mean())
            if y.size:
                print("  y first 10:", y.reshape(-1)[:10])
        except Exception as exc:
            print(f"\nExtraction method {method} failed: {exc}")
