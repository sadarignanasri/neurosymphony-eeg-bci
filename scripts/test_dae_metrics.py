"""
Validate Phase 2 DAE evaluation metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.evaluation.dae_metrics import (
    evaluate_reconstruction,
)


print("=" * 75)
print("PHASE 2 — DAE METRICS VALIDATION")
print("=" * 75)

rng = np.random.default_rng(42)

reference = rng.normal(
    0.0,
    1.0,
    size=(8, 512),
)

noise = rng.normal(
    0.0,
    0.05,
    size=reference.shape,
)

estimate = reference + noise

print()
print("[1] Evaluating reconstruction...")

metrics = evaluate_reconstruction(
    reference,
    estimate,
)

for key, value in metrics.items():
    print(
        f"{key}: {value}"
    )

assert np.isfinite(
    metrics["mse"]
)

assert np.isfinite(
    metrics["rmse"]
)

assert np.isfinite(
    metrics["snr_db"]
)

assert np.isfinite(
    metrics["correlation"]
)

print("PASS: All metrics are finite.")

print()
print("[2] Checking perfect reconstruction...")

perfect = evaluate_reconstruction(
    reference,
    reference,
)

assert perfect["mse"] == 0.0
assert perfect["rmse"] == 0.0
assert perfect["correlation"] > 0.999999

print("PASS: Perfect reconstruction behaves correctly.")

print()
print("=" * 75)
print("DAE METRICS VALIDATION PASSED")
print("=" * 75)
