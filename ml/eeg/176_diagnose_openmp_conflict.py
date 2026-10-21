import subprocess
import sys
import json
from pathlib import Path

modules = {
    "numpy": "import numpy",
    "scipy": "import scipy",
    "pandas": "import pandas",
    "scikit-learn": "import sklearn",
    "mne": "import mne",
    "pytorch": "import torch",
    "matplotlib": "import matplotlib",
}

results = {}

print("OPENMP IMPORT DIAGNOSTICS")
print("Python executable:", sys.executable)

for name, statement in modules.items():
    code = f"print('IMPORT_START'); {statement}; print('IMPORT_OK')"
    process = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )

    result = {
        "return_code": process.returncode,
        "stdout": process.stdout.strip(),
        "stderr": process.stderr.strip()[-3000:],
        "passed": process.returncode == 0 and "IMPORT_OK" in process.stdout,
    }
    results[name] = result

    status = "PASS" if result["passed"] else "FAIL"
    print(f"\n{name}: {status} (exit code {process.returncode})")

    if result["stdout"]:
        print("  stdout:", result["stdout"][-500:])
    if result["stderr"]:
        print("  stderr:", result["stderr"][-1000:])

out = Path("results/eeg/visual_motivation/step176_openmp_diagnostics.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2), encoding="utf-8")

print("\nReport saved:", out)
print("No packages or existing models modified.")
