import sys
import os
import subprocess
import importlib.metadata as metadata
import json
from pathlib import Path

report = {
    "python_executable": sys.executable,
    "python_version": sys.version,
    "environment": {
        key: value
        for key, value in os.environ.items()
        if any(term in key.upper() for term in (
            "OMP", "MKL", "OPENBLAS", "CONDA", "DYLD"
        ))
    },
    "packages": {},
}

for package in ["torch", "numpy", "scipy", "scikit-learn"]:
    try:
        dist = metadata.distribution(package)
        report["packages"][package] = {
            "version": dist.version,
            "location": str(dist.locate_file("")),
        }
    except metadata.PackageNotFoundError:
        report["packages"][package] = "NOT FOUND"

print("PYTHON:", report["python_executable"])
print("\nPACKAGE METADATA")
print(json.dumps(report["packages"], indent=2))

print("\nOPENMP-RELATED ENVIRONMENT VARIABLES")
print(json.dumps(report["environment"], indent=2))

print("\nCONDA PACKAGE RECORDS")
result = subprocess.run(
    ["conda", "list", "--show-channel-urls"],
    capture_output=True,
    text=True,
    timeout=60,
)

if result.returncode == 0:
    relevant = [
        line for line in result.stdout.splitlines()
        if any(name in line.lower() for name in (
            "torch", "pytorch", "openmp", "libomp",
            "llvm", "mkl", "numpy", "scipy", "scikit-learn"
        ))
    ]
    report["conda_relevant_packages"] = relevant
    for line in relevant:
        print(line)
else:
    report["conda_list_error"] = result.stderr
    print("Could not read Conda package records:")
    print(result.stderr)

out = Path("results/eeg/visual_motivation/step177_torch_installation.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\nReport saved:", out)
print("No packages installed, removed, or changed.")
print("No source data or models modified.")
