import subprocess
import sys
import os
import json
from pathlib import Path

report = {}

torch_lib = Path(
    "/Applications/miniconda3/lib/python3.13/site-packages/torch/lib"
)
omp_path = torch_lib / "libomp.dylib"

print("PYTORCH LIBRARY DIRECTORY:", torch_lib)
print("PyTorch bundled libomp exists:", omp_path.exists())

if omp_path.exists():
    result = subprocess.run(
        ["otool", "-L", str(omp_path)],
        capture_output=True,
        text=True,
    )
    print("\nPYTORCH libomp DEPENDENCIES")
    print(result.stdout)
    report["torch_libomp_dependencies"] = result.stdout

# Inspect other common locations for OpenMP runtimes.
search_roots = [
    Path("/Applications/miniconda3/lib"),
    Path("/Applications/miniconda3/lib/python3.13/site-packages"),
]

found = set()
for root in search_roots:
    if root.exists():
        for pattern in ("libomp.dylib", "libiomp5.dylib"):
            for path in root.rglob(pattern):
                found.add(str(path))

report["openmp_libraries_found"] = sorted(found)

print("\nOPENMP LIBRARY FILES")
for path in sorted(found):
    print(path)

# Import PyTorch in a fresh process with thread counts constrained.
# This is diagnostic only; it does not suppress the OpenMP safety check.
code = (
    "import torch; "
    "print('TORCH_IMPORT_OK', torch.__version__); "
    "print('MPS_BUILT', torch.backends.mps.is_built()); "
    "print('MPS_AVAILABLE', torch.backends.mps.is_available())"
)
env = os.environ.copy()
env["OMP_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"
env["OPENBLAS_NUM_THREADS"] = "1"

result = subprocess.run(
    [sys.executable, "-c", code],
    capture_output=True,
    text=True,
    env=env,
    timeout=60,
)

report["torch_import_test"] = {
    "return_code": result.returncode,
    "stdout": result.stdout,
    "stderr": result.stderr[-4000:],
}

print("\nPYTORCH IMPORT TEST WITH THREAD LIMITS")
print("Exit code:", result.returncode)
if result.stdout:
    print(result.stdout)
if result.stderr:
    print(result.stderr[-2000:])

out = Path("results/eeg/visual_motivation/step179_openmp_resolution.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\nReport saved:", out)
print("No packages or models modified.")
