import sys
import platform
import importlib.util
import json
from pathlib import Path

packages = [
    "numpy",
    "scipy",
    "pandas",
    "sklearn",
    "mne",
    "torch",
    "tensorflow",
    "keras",
    "matplotlib",
]

report = {
    "python_version": sys.version,
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "packages": {},
}

print("PYTHON ENVIRONMENT")
print("Python:", sys.version.split()[0])
print("Executable:", sys.executable)
print("Platform:", platform.platform())
print("Architecture:", platform.machine())

print("\nPACKAGE AVAILABILITY")
for package in packages:
    available = importlib.util.find_spec(package) is not None
    report["packages"][package] = available
    print(f"{package}: {'INSTALLED' if available else 'NOT INSTALLED'}")

if report["packages"].get("torch"):
    try:
        import torch
        report["pytorch"] = {
            "version": torch.__version__,
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "cuda_available": bool(torch.cuda.is_available()),
        }
        print("\nPYTORCH")
        print(json.dumps(report["pytorch"], indent=2))
    except Exception as exc:
        report["pytorch_error"] = str(exc)
        print("PyTorch inspection error:", exc)

if report["packages"].get("tensorflow"):
    try:
        import tensorflow as tf
        report["tensorflow"] = {
            "version": tf.__version__,
            "physical_devices": [
                {"name": d.name, "type": d.device_type}
                for d in tf.config.list_physical_devices()
            ],
        }
        print("\nTENSORFLOW")
        print(json.dumps(report["tensorflow"], indent=2))
    except Exception as exc:
        report["tensorflow_error"] = str(exc)
        print("TensorFlow inspection error:", exc)

out = Path("results/eeg/visual_motivation/step175_ml_environment.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\nReport saved:", out)
print("No packages installed or changed.")
print("No dataset or model files modified.")
