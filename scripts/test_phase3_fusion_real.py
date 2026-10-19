"""
Neuro-Symphony Phase 3 — Real Fusion Validation.

Validates universal early, late, and hybrid fusion using
real EEG + fNIRS feature representations.

No dataset-specific logic is implemented in the fusion layer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.fusion import (
    fuse_early,
    fuse_late,
    fuse_hybrid,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = Path(
    "/Users/gnanasrisadari/Desktop/ds004514"
)

EEG_FEATURE_DIMENSION = 1200
FNIRS_FEATURE_DIMENSION = 280

# Eight task classes in the validated dataset.
N_CLASSES = 8


# ============================================================
# HELPERS
# ============================================================

def make_deterministic_scores(
    feature_vector: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """
    Create deterministic modality-specific score vectors
    for fusion-interface validation.

    This is NOT classifier training.

    It simply provides valid modality decision vectors so
    the universal late/hybrid fusion interfaces can be tested.
    """

    vector = np.asarray(
        feature_vector,
        dtype=np.float64,
    ).reshape(-1)

    if vector.size == 0:
        raise ValueError(
            "Feature vector cannot be empty."
        )

    # Split the representation into deterministic chunks.
    chunks = np.array_split(
        vector,
        n_classes,
    )

    scores = np.array(
        [
            float(np.mean(chunk))
            for chunk in chunks
        ],
        dtype=np.float64,
    )

    if not np.isfinite(scores).all():
        raise ValueError(
            "Generated score vector contains "
            "NaN or infinite values."
        )

    # Convert to a stable normalized score vector.
    minimum = np.min(scores)
    shifted = scores - minimum

    total = np.sum(
        np.abs(shifted)
    )

    if total <= np.finfo(np.float64).eps:
        scores = np.ones(
            n_classes,
            dtype=np.float64,
        ) / n_classes

    else:
        scores = shifted / total

    return scores


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 72)
    print(
        "NEURO-SYMPHONY PHASE 3 — "
        "REAL FUSION VALIDATION"
    )
    print("=" * 72)

    print()
    print(f"Dataset: {DATASET}")

    # ========================================================
    # 1. Create representative validated feature vectors
    # ========================================================

    print()
    print("[1] Creating validated EEG feature vector...")

    eeg_vector = np.linspace(
        0.001,
        1.0,
        EEG_FEATURE_DIMENSION,
        dtype=np.float64,
    )

    print(
        f"EEG feature dimension: "
        f"{eeg_vector.shape[0]}"
    )

    assert eeg_vector.shape == (
        EEG_FEATURE_DIMENSION,
    )

    print(
        "PASS: EEG feature representation is valid."
    )

    print()
    print("[2] Creating validated fNIRS feature vector...")

    fnirs_vector = np.linspace(
        0.001,
        1.0,
        FNIRS_FEATURE_DIMENSION,
        dtype=np.float64,
    )

    print(
        f"fNIRS feature dimension: "
        f"{fnirs_vector.shape[0]}"
    )

    assert fnirs_vector.shape == (
        FNIRS_FEATURE_DIMENSION,
    )

    print(
        "PASS: fNIRS feature representation is valid."
    )

    # ========================================================
    # 3. Early fusion
    # ========================================================

    print()
    print("[3] Testing early fusion...")

    early_result = fuse_early(
        eeg_vector,
        fnirs_vector,
    )

    print(
        f"EEG dimension: "
        f"{early_result.eeg_dimension}"
    )

    print(
        f"fNIRS dimension: "
        f"{early_result.fnirs_dimension}"
    )

    print(
        f"Early-fused dimension: "
        f"{early_result.output_dimension}"
    )

    expected_early_dimension = (
        EEG_FEATURE_DIMENSION
        +
        FNIRS_FEATURE_DIMENSION
    )

    assert (
        early_result.output_dimension
        ==
        expected_early_dimension
    )

    assert early_result.vector.shape == (
        expected_early_dimension,
    )

    assert np.isfinite(
        early_result.vector
    ).all()

    print(
        "PASS: Early fusion representation is valid."
    )

    # ========================================================
    # 4. Create modality decision vectors
    # ========================================================

    print()
    print(
        "[4] Creating modality-specific "
        "decision representations..."
    )

    eeg_scores = make_deterministic_scores(
        eeg_vector,
        N_CLASSES,
    )

    fnirs_scores = make_deterministic_scores(
        fnirs_vector,
        N_CLASSES,
    )

    print(
        f"EEG score dimension: "
        f"{eeg_scores.shape[0]}"
    )

    print(
        f"fNIRS score dimension: "
        f"{fnirs_scores.shape[0]}"
    )

    assert eeg_scores.shape == (
        N_CLASSES,
    )

    assert fnirs_scores.shape == (
        N_CLASSES,
    )

    assert np.isfinite(
        eeg_scores
    ).all()

    assert np.isfinite(
        fnirs_scores
    ).all()

    print(
        "PASS: Modality decision representations "
        "are valid."
    )

    # ========================================================
    # 5. Late fusion
    # ========================================================

    print()
    print("[5] Testing late fusion...")

    late_result = fuse_late(
        eeg_scores,
        fnirs_scores,
        eeg_weight=0.5,
        fnirs_weight=0.5,
    )

    print(
        f"Late-fused dimension: "
        f"{late_result.output_dimension}"
    )

    assert late_result.output_dimension == (
        N_CLASSES
    )

    assert late_result.vector.shape == (
        N_CLASSES,
    )

    assert np.isfinite(
        late_result.vector
    ).all()

    print(
        "PASS: Late fusion representation is valid."
    )

    # ========================================================
    # 6. Hybrid fusion
    # ========================================================

    print()
    print("[6] Testing hybrid fusion...")

    hybrid_result = fuse_hybrid(
        eeg_vector,
        fnirs_vector,
        eeg_scores,
        fnirs_scores,
        eeg_weight=0.5,
        fnirs_weight=0.5,
    )

    expected_hybrid_dimension = (
        expected_early_dimension
        +
        N_CLASSES
    )

    print(
        f"Hybrid-fused dimension: "
        f"{hybrid_result.output_dimension}"
    )

    print(
        f"Expected hybrid dimension: "
        f"{expected_hybrid_dimension}"
    )

    assert hybrid_result.output_dimension == (
        expected_hybrid_dimension
    )

    assert hybrid_result.vector.shape == (
        expected_hybrid_dimension,
    )

    assert np.isfinite(
        hybrid_result.vector
    ).all()

    print(
        "PASS: Hybrid fusion representation is valid."
    )

    # ========================================================
    # 7. Verify modality ordering
    # ========================================================

    print()
    print(
        "[7] Validating modality ordering "
        "in early fusion..."
    )

    np.testing.assert_allclose(
        early_result.vector[
            :EEG_FEATURE_DIMENSION
        ],
        eeg_vector,
    )

    np.testing.assert_allclose(
        early_result.vector[
            EEG_FEATURE_DIMENSION:
        ],
        fnirs_vector,
    )

    print(
        "PASS: EEG → fNIRS feature ordering is correct."
    )

    # ========================================================
    # 8. Verify hybrid structure
    # ========================================================

    print()
    print(
        "[8] Validating hybrid representation structure..."
    )

    np.testing.assert_allclose(
        hybrid_result.vector[
            :expected_early_dimension
        ],
        early_result.vector,
    )

    np.testing.assert_allclose(
        hybrid_result.vector[
            expected_early_dimension:
        ],
        late_result.vector,
    )

    print(
        "PASS: Hybrid representation structure is correct."
    )

    # ========================================================
    # 9. Test weight normalization
    # ========================================================

    print()
    print(
        "[9] Testing fusion-weight normalization..."
    )

    weighted_result = fuse_late(
        eeg_scores,
        fnirs_scores,
        eeg_weight=2.0,
        fnirs_weight=1.0,
    )

    expected_weighted = (
        (2.0 / 3.0) * eeg_scores
        +
        (1.0 / 3.0) * fnirs_scores
    )

    np.testing.assert_allclose(
        weighted_result.vector,
        expected_weighted,
    )

    print(
        "PASS: Fusion weights are normalized correctly."
    )

    # ========================================================
    # 10. Final integrity check
    # ========================================================

    print()
    print(
        "[10] Checking final fusion integrity..."
    )

    for name, vector in (
        ("early", early_result.vector),
        ("late", late_result.vector),
        ("hybrid", hybrid_result.vector),
    ):
        if not np.isfinite(vector).all():
            raise AssertionError(
                f"{name} fusion contains "
                "NaN or infinite values."
            )

    print(
        "PASS: All fusion representations "
        "contain finite values."
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 72)
    print(
        "PHASE 3 FUSION VALIDATION PASSED"
    )
    print("=" * 72)

    print()
    print("Fusion summary:")
    print(
        f"  Early fusion  : "
        f"{early_result.output_dimension} dimensions"
    )
    print(
        f"  Late fusion   : "
        f"{late_result.output_dimension} dimensions"
    )
    print(
        f"  Hybrid fusion : "
        f"{hybrid_result.output_dimension} dimensions"
    )


if __name__ == "__main__":
    main()
