"""
Neuro-Symphony
Step 34: Create a fixed stratified train/validation/test split — sub-01

Dataset:
    720 labelled trials
    8 classes
    90 trials per class

Split:
    70% Train
    15% Validation
    15% Test

Actual split:
    Train      = 504 trials
    Validation = 108 trials
    Test       = 108 trials

Because each class contains 90 trials:

    Train      = 63 trials per class
    Validation = 13 or 14 trials per class
    Test       = 14 or 13 trials per class

This is unavoidable because:

    90 × 15% = 13.5

Important:
    - Split is performed at trial level.
    - Stratification uses labels only.
    - EEG feature values are NOT used to create the split.
    - Test set remains untouched for final evaluation.
    - Fixed random seed makes the split reproducible.

No model training.
No feature scaling.
No feature selection.
"""


# ============================================================================
# IMPORTS
# ============================================================================

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================================
# PROJECT ROOT
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# INPUT
# ============================================================================

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "temporal_spectral_sub01"
    / "sub-01_spectral_metadata.csv"
)


# ============================================================================
# OUTPUT
# ============================================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "sub01"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


TRAIN_INDICES_FILE = (
    OUTPUT_DIR
    / "train_indices.npy"
)

VAL_INDICES_FILE = (
    OUTPUT_DIR
    / "val_indices.npy"
)

TEST_INDICES_FILE = (
    OUTPUT_DIR
    / "test_indices.npy"
)

SPLIT_METADATA_FILE = (
    OUTPUT_DIR
    / "split_metadata.csv"
)


# ============================================================================
# PARAMETERS
# ============================================================================

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

EXPECTED_TRIALS = 720
EXPECTED_CLASSES = 8
TRIALS_PER_CLASS = 90


# ============================================================================
# START
# ============================================================================

print("=" * 80)
print("NEURO-SYMPHONY — SUB-01 TRAIN / VALIDATION / TEST SPLIT")
print("=" * 80)

print("\nInput:")
print(METADATA_FILE)

print("\nOutput:")
print(OUTPUT_DIR)


# ============================================================================
# VALIDATE INPUT
# ============================================================================

if not METADATA_FILE.exists():

    raise FileNotFoundError(
        f"Metadata file not found:\n{METADATA_FILE}"
    )

print("\n✓ Metadata file found.")


# ============================================================================
# LOAD METADATA
# ============================================================================

print("\n" + "=" * 80)
print("LOADING METADATA")
print("=" * 80)

metadata = pd.read_csv(
    METADATA_FILE
)

print("\nMetadata shape:")
print(metadata.shape)


if len(metadata) != EXPECTED_TRIALS:

    raise RuntimeError(
        f"Expected {EXPECTED_TRIALS} trials, "
        f"found {len(metadata)}."
    )


# ============================================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================================

required_columns = [
    "label",
    "label_name",
]

for column in required_columns:

    if column not in metadata.columns:

        raise RuntimeError(
            f"Required column '{column}' "
            f"was not found in metadata."
        )


# ============================================================================
# LOAD LABELS
# ============================================================================

labels = metadata[
    "label"
].to_numpy()

label_names = metadata[
    "label_name"
].to_numpy()


unique_labels = np.unique(
    labels
)


print("\nUnique numeric labels:")
print(unique_labels)


if len(unique_labels) != EXPECTED_CLASSES:

    raise RuntimeError(
        f"Expected {EXPECTED_CLASSES} classes, "
        f"found {len(unique_labels)}."
    )


# ============================================================================
# ORIGINAL CLASS DISTRIBUTION
# ============================================================================

print("\nClass distribution:")


class_distribution = (
    metadata[
        [
            "label",
            "label_name"
        ]
    ]
    .drop_duplicates()
    .sort_values("label")
)


for _, row in class_distribution.iterrows():

    label = int(
        row["label"]
    )

    name = row["label_name"]

    count = int(
        np.sum(
            labels == label
        )
    )

    print(
        f"  {label}: "
        f"{name:<35} "
        f"{count} trials"
    )

    if count != TRIALS_PER_CLASS:

        raise RuntimeError(
            f"Class {label} has {count} trials. "
            f"Expected {TRIALS_PER_CLASS}."
        )


print(
    "\n✓ All 8 classes contain exactly 90 trials."
)


# ============================================================================
# ALL TRIAL INDICES
# ============================================================================

all_indices = np.arange(
    EXPECTED_TRIALS
)


# ============================================================================
# FIRST SPLIT — TRAIN + TEMPORARY
# ============================================================================

print("\n" + "=" * 80)
print("CREATING STRATIFIED SPLIT")
print("=" * 80)

print("\nRequested proportions:")

print(
    f"  Train      : "
    f"{TRAIN_SIZE * 100:.1f}%"
)

print(
    f"  Validation : "
    f"{VALIDATION_SIZE * 100:.1f}%"
)

print(
    f"  Test       : "
    f"{TEST_SIZE * 100:.1f}%"
)


# 70% → train
# 30% → temporary

train_indices, temporary_indices = train_test_split(
    all_indices,
    test_size=(
        VALIDATION_SIZE
        + TEST_SIZE
    ),
    stratify=labels,
    random_state=RANDOM_STATE
)


# ============================================================================
# SECOND SPLIT — VALIDATION + TEST
# ============================================================================

temporary_labels = labels[
    temporary_indices
]


validation_indices, test_indices = train_test_split(
    temporary_indices,
    test_size=0.50,
    stratify=temporary_labels,
    random_state=RANDOM_STATE
)


# ============================================================================
# SORT INDICES
# ============================================================================

train_indices = np.sort(
    train_indices
)

validation_indices = np.sort(
    validation_indices
)

test_indices = np.sort(
    test_indices
)


# ============================================================================
# VALIDATE SPLIT SIZES
# ============================================================================

print("\nActual split sizes:")

print(
    f"  Train      : "
    f"{len(train_indices)}"
)

print(
    f"  Validation : "
    f"{len(validation_indices)}"
)

print(
    f"  Test       : "
    f"{len(test_indices)}"
)


if len(train_indices) != 504:

    raise RuntimeError(
        f"Expected 504 training trials, "
        f"found {len(train_indices)}."
    )


if len(validation_indices) != 108:

    raise RuntimeError(
        f"Expected 108 validation trials, "
        f"found {len(validation_indices)}."
    )


if len(test_indices) != 108:

    raise RuntimeError(
        f"Expected 108 test trials, "
        f"found {len(test_indices)}."
    )


# ============================================================================
# CHECK FOR OVERLAP
# ============================================================================

print("\n" + "=" * 80)
print("CHECKING FOR DATA SPLIT OVERLAP")
print("=" * 80)


train_set = set(
    train_indices
)

validation_set = set(
    validation_indices
)

test_set = set(
    test_indices
)


train_val_overlap = (
    train_set
    &
    validation_set
)

train_test_overlap = (
    train_set
    &
    test_set
)

validation_test_overlap = (
    validation_set
    &
    test_set
)


print(
    f"\nTrain ∩ Validation: "
    f"{len(train_val_overlap)}"
)

print(
    f"Train ∩ Test: "
    f"{len(train_test_overlap)}"
)

print(
    f"Validation ∩ Test: "
    f"{len(validation_test_overlap)}"
)


if train_val_overlap:

    raise RuntimeError(
        "Train and validation sets overlap."
    )


if train_test_overlap:

    raise RuntimeError(
        "Train and test sets overlap."
    )


if validation_test_overlap:

    raise RuntimeError(
        "Validation and test sets overlap."
    )


print(
    "\n✓ No overlap between any splits."
)


# ============================================================================
# CHECK COMPLETE COVERAGE
# ============================================================================

combined_indices = np.concatenate(
    [
        train_indices,
        validation_indices,
        test_indices,
    ]
)


combined_indices = np.sort(
    combined_indices
)


if not np.array_equal(
    combined_indices,
    all_indices
):

    raise RuntimeError(
        "The train/validation/test sets "
        "do not cover exactly all 720 trials."
    )


print(
    "✓ All 720 trials are assigned exactly once."
)


# ============================================================================
# CLASS DISTRIBUTION FUNCTION
# ============================================================================

def print_split_distribution(
    split_name,
    indices,
):

    split_labels = labels[
        indices
    ]

    print(
        f"\n{split_name}: "
        f"{len(indices)} trials"
    )

    for label in sorted(
        unique_labels
    ):

        count = int(
            np.sum(
                split_labels == label
            )
        )

        name = metadata.loc[
            metadata["label"] == label,
            "label_name"
        ].iloc[0]

        print(
            f"  {label}: "
            f"{name:<35} "
            f"{count}"
        )


# ============================================================================
# PRINT CLASS DISTRIBUTIONS
# ============================================================================

print("\n" + "=" * 80)
print("STRATIFICATION CHECK")
print("=" * 80)


print_split_distribution(
    "TRAIN",
    train_indices
)

print_split_distribution(
    "VALIDATION",
    validation_indices
)

print_split_distribution(
    "TEST",
    test_indices
)


# ============================================================================
# VERIFY CLASS BALANCE
# ============================================================================

print("\n" + "=" * 80)
print("VERIFYING CLASS BALANCE")
print("=" * 80)


EXPECTED_TRAIN_PER_CLASS = 63


for label in sorted(
    unique_labels
):

    train_count = int(
        np.sum(
            labels[
                train_indices
            ] == label
        )
    )

    val_count = int(
        np.sum(
            labels[
                validation_indices
            ] == label
        )
    )

    test_count = int(
        np.sum(
            labels[
                test_indices
            ] == label
        )
    )


    print(
        f"Label {label}: "
        f"Train={train_count}, "
        f"Val={val_count}, "
        f"Test={test_count}"
    )


    # Every class must have exactly 90 trials.
    if (
        train_count
        + val_count
        + test_count
        != TRIALS_PER_CLASS
    ):

        raise RuntimeError(
            f"Class {label} does not contain exactly "
            f"{TRIALS_PER_CLASS} trials across "
            f"the three splits."
        )


    # 70% of 90 = 63 exactly.
    if train_count != EXPECTED_TRAIN_PER_CLASS:

        raise RuntimeError(
            f"Unexpected train count for "
            f"class {label}: "
            f"{train_count}. "
            f"Expected {EXPECTED_TRAIN_PER_CLASS}."
        )


    # Since:
    #
    # 90 × 15% = 13.5
    #
    # validation/test can legitimately be:
    #
    # 13 / 14
    #
    # or:
    #
    # 14 / 13
    #
    # but they must sum to 27.

    if val_count not in (13, 14):

        raise RuntimeError(
            f"Invalid validation count for "
            f"class {label}: "
            f"{val_count}."
        )


    if test_count not in (13, 14):

        raise RuntimeError(
            f"Invalid test count for "
            f"class {label}: "
            f"{test_count}."
        )


    if (
        val_count
        + test_count
        != 27
    ):

        raise RuntimeError(
            f"Validation + test count for "
            f"class {label} must equal 27, "
            f"found {val_count + test_count}."
        )


print(
    "\n✓ Every class contains exactly 90 trials."
)

print(
    "✓ Every class contains exactly 63 training trials."
)

print(
    "✓ Validation/test counts are 13/14 or 14/13."
)

print(
    "✓ Class stratification is valid."
)

print(
    "✓ Overall split is 504 / 108 / 108."
)


# ============================================================================
# CREATE SPLIT METADATA
# ============================================================================

print("\n" + "=" * 80)
print("CREATING SPLIT METADATA")
print("=" * 80)


split_assignment = np.empty(
    EXPECTED_TRIALS,
    dtype=object
)


split_assignment[
    train_indices
] = "train"


split_assignment[
    validation_indices
] = "validation"


split_assignment[
    test_indices
] = "test"


split_metadata = metadata.copy()


split_metadata[
    "split"
] = split_assignment


split_metadata[
    "split_random_state"
] = RANDOM_STATE


# ============================================================================
# SAVE SPLIT
# ============================================================================

print("\n" + "=" * 80)
print("SAVING SPLIT")
print("=" * 80)


np.save(
    TRAIN_INDICES_FILE,
    train_indices
)


np.save(
    VAL_INDICES_FILE,
    validation_indices
)


np.save(
    TEST_INDICES_FILE,
    test_indices
)


split_metadata.to_csv(
    SPLIT_METADATA_FILE,
    index=False
)


print("\n✓ Saved training indices:")
print(TRAIN_INDICES_FILE)


print("\n✓ Saved validation indices:")
print(VAL_INDICES_FILE)


print("\n✓ Saved test indices:")
print(TEST_INDICES_FILE)


print("\n✓ Saved complete split metadata:")
print(SPLIT_METADATA_FILE)


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("STEP 34 COMPLETE")
print("=" * 80)


print("\nDataset:")
print(
    "  720 total trials"
)


print("\nFixed split:")

print(
    "  Train      : 504 trials"
)

print(
    "  Validation : 108 trials"
)

print(
    "  Test       : 108 trials"
)


print("\nPer-class distribution:")

print(
    "  Train      : 63 trials per class"
)

print(
    "  Validation : 13 or 14 trials per class"
)

print(
    "  Test       : 13 or 14 trials per class"
)


print("\nRandom seed:")
print(
    f"  {RANDOM_STATE}"
)


print("\nLeakage checks:")

print(
    "  ✓ No train/validation overlap"
)

print(
    "  ✓ No train/test overlap"
)

print(
    "  ✓ No validation/test overlap"
)

print(
    "  ✓ All 720 trials used exactly once"
)


print("\nNo model was trained.")
print("No scaling was performed.")
print("No feature selection was performed.")
print("The test set remains untouched.")


print(
    "\nThe fixed leakage-safe split is ready "
    "for model training."
)


print("=" * 80)