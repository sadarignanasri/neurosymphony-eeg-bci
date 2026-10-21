import subprocess
import sys
import json
from pathlib import Path
import importlib.util

report = {
    "python": sys.executable,
    "torch_spec": None,
    "library_checks": [],
}

spec = importlib.util.find_spec("torch")
if spec is None or not spec.submodule_search_locations:
    raise RuntimeError("PyTorch package directory not found.")

torch_dir = Path(list(spec.submodule_search_locations)[0])
report["torch_spec"] = str(torch_dir)

print("PYTORCH DIRECTORY:", torch_dir)

# Inspect a small selection of native libraries without importing torch.
libraries = []
for pattern in (
    "lib/libtorch_cpu.dylib",
    "lib/libtorch.dylib",
    "lib/libc10.dylib",
    "_C*.so",
):
    libraries.extend(torch_dir.glob(pattern))

for lib in sorted(set(libraries)):
    print(f"\nLIBRARY: {lib}")
    result = subprocess.run(
        ["otool", "-L", str(lib)],
        capture_output=True,
        text=True,
        timeout=30,
    )

    item = {
        "library": str(lib),
        "return_code": result.returncode,
        "dependencies": result.stdout.splitlines(),
        "error": result.stderr.strip(),
    }
    report["library_checks"].append(item)

    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if any(term in line.lower() for term in (
                "omp", "mkl", "openblas", "accelerate"
            )):
                print(" ", line.strip())
    else:
        print("otool error:", result.stderr.strip())

out = Path("results/eeg/visual_motivation/step178_torch_library_dependencies.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\nReport saved:", out)
print("No libraries imported, packages changed, or models modified.")
