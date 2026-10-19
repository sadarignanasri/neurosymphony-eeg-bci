"""
STEP 34 — CREATE LEAKAGE-SAFE TRAIN / VALIDATION / TEST SPLITS
ALL 12 SUBJECTS

Purpose
-------
Create deterministic, stratified train/validation/test splits
using trial indices only.

No feature values are used for splitting.

Per subject:
    720 total trials
    504 train       (70%)
    108 validation  (15%)
    108 test        (15%)

Expected class distribution:
    8 classes
    90 trials per class

The same split is later used for spectral features and models.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_ROOT = (
    PROJECT_ROOT /
    "data" /
    "features"
)

OUTPUT_ROOT = (
    PROJECT_ROOT /
    "data" /
    "splits"
)


# ============================================================
# CONFIGURATION
# ============================================================

SUBJECTS = [
    f"sub-{i:02d}"
    for i in range(1, 13)
]

EXPECTED_TRIALS = 720
EXPECTED_CLASSES = 8
TRIALS_PER_CLASS = 90

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# ============================================================
# LABELS
# ============================================================

LABEL_NAMES = [
    "animal_auditory_imagery_task",
    "animal_silent_naming_task",
    "animal_tactile_imagery_task",
    "animal_visual_imagery_task",
    "tool_auditory_imagery_task",
    "tool_silent_naming_task",
    "tool_tactile_imagery_task",
    "tool_visual_imagery_task",
]


# ============================================================
# VALIDATION HELPERS
# ============================================================

def print_separator():
    print("\n" + "=" * 90)


def validate_indices(
    train_indices,
    val_indices,
    test_indices
):
    """
    Verify that:
      - all indices are unique
      - no overlap exists
      - every trial is assigned exactly once
      - expected split sizes are correct
    """

    train_set = set(train_indices.tolist())
    val_set = set(val_indices.tolist())
    test_set = set(test_indices.tolist())

    # --------------------------------------------------------
    # Split sizes
    # --------------------------------------------------------

    if len(train_indices) != 504:
        raise ValueError(
            f"Expected 504 train trials, "
            f"found {len(train_indices)}"
        )

    if len(val_indices) != 108:
        raise ValueError(
            f"Expected 108 validation trials, "
            f"found {len(val_indices)}"
        )

    if len(test_indices) != 108:
        raise ValueError(
            f"Expected 108 test trials, "
            f"found {len(test_indices)}"
        )

    # --------------------------------------------------------
    # No overlap
    # --------------------------------------------------------

    if train_set & val_set:
        raise ValueError(
            "Train/validation overlap detected."
        )

    if train_set & test_set:
        raise ValueError(
            "Train/test overlap detected."
        )

    if val_set & test_set:
        raise ValueError(
            "Validation/test overlap detected."
        )

    # --------------------------------------------------------
    # Complete coverage
    # --------------------------------------------------------

    all_indices = (
        train_set |
        val_set |
        test_set
    )

    expected_indices = set(
        range(EXPECTED_TRIALS)
    )

    if all_indices != expected_indices:
        missing = expected_indices - all_indices
        extra = all_indices - expected_indices

        raise ValueError(
            f"Trial coverage error.\n"
            f"Missing: {sorted(missing)}\n"
            f"Extra: {sorted(extra)}"
        )

    # --------------------------------------------------------
    # Individual uniqueness
    # --------------------------------------------------------

    if len(train_set) != len(train_indices):
        raise ValueError(
            "Duplicate train indices detected."
        )

    if len(val_set) != len(val_indices):
        raise ValueError(
            "Duplicate validation indices detected."
        )

    if len(test_set) != len(test_indices):
        raise ValueError(
            "Duplicate test indices detected."
        )

    print("✓ Split sizes verified.")
    print("✓ No overlap between splits.")
    print("✓ All 720 trials assigned exactly once.")
    print("✓ No duplicate indices.")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def create_labels():
    """
    Create the expected 720 trial labels.

    Dataset structure:
        8 classes × 90 trials
    """

    labels = np.repeat(
        np.arange(EXPECTED_CLASSES),
        TRIALS_PER_CLASS
    )

    if len(labels) != EXPECTED_TRIALS:
        raise ValueError(
            "Generated label vector has incorrect length."
        )

    return labels


def print_class_distribution(
    name,
    indices,
    labels
):

    subset_labels = labels[indices]

    counts = np.bincount(
        subset_labels,
        minlength=EXPECTED_CLASSES
    )

    print(f"\n{name} class distribution:")

    for label_id, count in enumerate(counts):

        print(
            f"  {label_id}: "
            f"{LABEL_NAMES[label_id]:<35} "
            f"{count}"
        )

    return counts


# ============================================================
# PROCESS ONE SUBJECT
# ============================================================

def process_subject(subject):

    print_separator()
    print(f"{subject} — CREATE SPLIT")
    print_separator()

    # --------------------------------------------------------
    # Create labels
    # --------------------------------------------------------

    labels = create_labels()

    print(
        f"Total trials : "
        f"{len(labels)}"
    )

    print(
        f"Classes      : "
        f"{len(np.unique(labels))}"
    )

    # --------------------------------------------------------
    # Verify original class balance
    # --------------------------------------------------------

    original_counts = np.bincount(
        labels,
        minlength=EXPECTED_CLASSES
    )

    print("\nOriginal class distribution:")

    for label_id, count in enumerate(
        original_counts
    ):

        print(
            f"  {label_id}: "
            f"{LABEL_NAMES[label_id]:<35} "
            f"{count}"
        )

        if count != TRIALS_PER_CLASS:
            raise ValueError(
                f"Class {label_id} does not contain "
                f"{TRIALS_PER_CLASS} trials."
            )

    print(
        "\n✓ Original class balance verified."
    )

    # --------------------------------------------------------
    # Trial indices
    # --------------------------------------------------------

    indices = np.arange(
        EXPECTED_TRIALS
    )

    # --------------------------------------------------------
    # First split:
    #
    # 70% train
    # 30% temporary
    # --------------------------------------------------------

    train_indices, temp_indices = (
        train_test_split(
            indices,
            test_size=(
                VAL_RATIO + TEST_RATIO
            ),
            stratify=labels,
            random_state=RANDOM_SEED,
            shuffle=True,
        )
    )

    # --------------------------------------------------------
    # Second split:
    #
    # 15% validation
    # 15% test
    #
    # temp = 30%
    # half of temp = 15%
    # --------------------------------------------------------

    temp_labels = labels[temp_indices]

    val_indices, test_indices = (
        train_test_split(
            temp_indices,
            test_size=0.5,
            stratify=temp_labels,
            random_state=RANDOM_SEED,
            shuffle=True,
        )
    )

    # --------------------------------------------------------
    # Sort indices for reproducibility/readability
    # --------------------------------------------------------

    train_indices = np.sort(
        train_indices
    )

    val_indices = np.sort(
        val_indices
    )

    test_indices = np.sort(
        test_indices
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    print("\nValidating split...")

    validate_indices(
        train_indices,
        val_indices,
        test_indices
    )

    # --------------------------------------------------------
    # Class distributions
    # --------------------------------------------------------

    train_counts = print_class_distribution(
        "TRAIN",
        train_indices,
        labels
    )

    val_counts = print_class_distribution(
        "VALIDATION",
        val_indices,
        labels
    )

    test_counts = print_class_distribution(
        "TEST",
        test_indices,
        labels
    )

    # --------------------------------------------------------
    # Check stratification
    # --------------------------------------------------------

    expected_train_min = 63
    expected_train_max = 63

    if not np.all(
        train_counts == expected_train_min
    ):
        raise ValueError(
            "Unexpected training class distribution."
        )

    # Validation and test can differ by one trial
    # for individual classes because 90 cannot be
    # divided into exact 15% integer counts.
    #
    # Therefore each class should contain either
    # 13 or 14 trials in validation/test.

    if not np.all(
        np.isin(val_counts, [13, 14])
    ):
        raise ValueError(
            "Unexpected validation class distribution."
        )

    if not np.all(
        np.isin(test_counts, [13, 14])
    ):
        raise ValueError(
            "Unexpected test class distribution."
        )

    if val_counts.sum() != 108:
        raise ValueError(
            "Validation class counts do not sum to 108."
        )

    if test_counts.sum() != 108:
        raise ValueError(
            "Test class counts do not sum to 108."
        )

    print(
        "\n✓ Training class balance verified."
    )

    print(
        "✓ Validation class balance verified."
    )

    print(
        "✓ Test class balance verified."
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = (
        OUTPUT_ROOT /
        subject
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save index arrays
    # --------------------------------------------------------

    train_file = (
        output_dir /
        "train_indices.npy"
    )

    val_file = (
        output_dir /
        "val_indices.npy"
    )

    test_file = (
        output_dir /
        "test_indices.npy"
    )

    np.save(
        train_file,
        train_indices
    )

    np.save(
        val_file,
        val_indices
    )

    np.save(
        test_file,
        test_indices
    )

    # --------------------------------------------------------
    # Create metadata
    # --------------------------------------------------------

    metadata_rows = []

    for index in range(
        EXPECTED_TRIALS
    ):

        if index in set(train_indices):
            split = "train"

        elif index in set(val_indices):
            split = "validation"

        elif index in set(test_indices):
            split = "test"

        else:
            raise ValueError(
                f"Trial {index} was not assigned."
            )

        metadata_rows.append(
            {
                "subject": subject,
                "trial_index": index,
                "label_id": int(labels[index]),
                "label_name": LABEL_NAMES[
                    labels[index]
                ],
                "split": split,
                "random_seed": RANDOM_SEED,
            }
        )

    metadata = pd.DataFrame(
        metadata_rows
    )

    metadata_file = (
        output_dir /
        "split_metadata.csv"
    )

    metadata.to_csv(
        metadata_file,
        index=False
    )

    # --------------------------------------------------------
    # Final subject summary
    # --------------------------------------------------------

    print_separator()
    print(
        f"{subject} — SPLIT COMPLETE"
    )
    print_separator()

    print(
        f"Total trials : "
        f"{EXPECTED_TRIALS}"
    )

    print(
        f"Train        : "
        f"{len(train_indices)}"
    )

    print(
        f"Validation   : "
        f"{len(val_indices)}"
    )

    print(
        f"Test         : "
        f"{len(test_indices)}"
    )

    print("\nSaved:")

    print(
        f"  {train_file}"
    )

    print(
        f"  {val_file}"
    )

    print(
        f"  {test_file}"
    )

    print(
        f"  {metadata_file}"
    )

    print(
        "\n✓ Subject split verified."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_separator()
    print(
        "STEP 34 — LEAKAGE-SAFE DATASET SPLITS"
    )
    print(
        "ALL 12 SUBJECTS"
    )
    print_separator()

    print(
        f"Subjects           : "
        f"{len(SUBJECTS)}"
    )

    print(
        f"Total trials/subject: "
        f"{EXPECTED_TRIALS}"
    )

    print(
        f"Train ratio        : "
        f"{TRAIN_RATIO:.0%}"
    )

    print(
        f"Validation ratio   : "
        f"{VAL_RATIO:.0%}"
    )

    print(
        f"Test ratio         : "
        f"{TEST_RATIO:.0%}"
    )

    print(
        f"Random seed        : "
        f"{RANDOM_SEED}"
    )

    # --------------------------------------------------------
    # Verify ratios
    # --------------------------------------------------------

    if (
        abs(
            TRAIN_RATIO +
            VAL_RATIO +
            TEST_RATIO -
            1.0
        ) > 1e-9
    ):
        raise ValueError(
            "Train/validation/test ratios "
            "must sum to 1.0."
        )

    completed = []

    # --------------------------------------------------------
    # Process every subject
    # --------------------------------------------------------

    for subject in SUBJECTS:

        process_subject(subject)

        completed.append(
            subject
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print_separator()
    print(
        "STEP 34 — FINAL SUMMARY"
    )
    print_separator()

    print(
        f"Subjects requested : "
        f"{len(SUBJECTS)}"
    )

    print(
        f"Subjects completed : "
        f"{len(completed)}"
    )

    print(
        "Subjects failed    : 0"
    )

    print("\nCompleted:")

    for subject in completed:
        print(
            f"  ✓ {subject}"
        )

    print("\nExpected pooled totals:")

    print(
        f"  Train       : "
        f"{len(SUBJECTS) * 504}"
    )

    print(
        f"  Validation  : "
        f"{len(SUBJECTS) * 108}"
    )

    print(
        f"  Test        : "
        f"{len(SUBJECTS) * 108}"
    )

    print(
        f"  Total       : "
        f"{len(SUBJECTS) * 720}"
    )

    print("\n" + "=" * 90)
    print(
        "✓ ALL 12 SUBJECTS SPLIT SUCCESSFULLY"
    )
    print("=" * 90)

    print(
        "\nNext step: "
        "build the first all-subject baseline classifier."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()