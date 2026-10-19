import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = "data/features/ds004514_fnirs.csv"

OUTPUT_DIR = "data/features"

OUTPUT_PATH = (
    "data/features/ds004514_fnirs_engineered.csv"
)

RESULTS_DIR = "results/fnirs"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# HELPER
# ============================================================

def print_section(title):

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print_section(
    "NEURO-SYMPHONY — fNIRS FEATURE ENGINEERING"
)

print("\nInput:")
print(INPUT_PATH)

if not os.path.exists(INPUT_PATH):

    raise FileNotFoundError(
        f"Input dataset not found:\n{INPUT_PATH}"
    )

df = pd.read_csv(
    INPUT_PATH
)

print("\nOriginal shape:")
print(df.shape)


# ============================================================
# STANDARDIZE LABEL COLUMN
# ============================================================

if "Label" in df.columns and "label" not in df.columns:

    df = df.rename(
        columns={
            "Label": "label"
        }
    )

elif "label" not in df.columns:

    raise ValueError(
        "Neither 'label' nor 'Label' column exists."
    )


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "subject_id",
    "trial_id",
    "label_name",
    "label"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


# ============================================================
# IDENTIFY RAW fNIRS FEATURES
# ============================================================

metadata_columns = [
    "subject_id",
    "trial_id",
    "label_name",
    "label"
]

raw_features = [
    column
    for column in df.columns
    if column not in metadata_columns
]

print_section(
    "RAW fNIRS FEATURES"
)

print(
    "Number of raw fNIRS features:",
    len(raw_features)
)

for feature in raw_features:

    print(
        " ",
        feature
    )


# ============================================================
# CHECK NUMERIC FEATURES
# ============================================================

non_numeric = [
    column
    for column in raw_features
    if not pd.api.types.is_numeric_dtype(
        df[column]
    )
]

if non_numeric:

    raise ValueError(
        f"Non-numeric fNIRS features found: {non_numeric}"
    )


# ============================================================
# START ENGINEERED DATASET
# ============================================================

engineered = df[
    metadata_columns
].copy()


# ============================================================
# 1. RAW FEATURES
# ============================================================

print_section(
    "1. ADDING RAW fNIRS FEATURES"
)

for feature in raw_features:

    engineered[
        feature
    ] = df[
        feature
    ]

print(
    "Added:",
    len(raw_features),
    "raw features"
)


# ============================================================
# 2. GLOBAL TRIAL-LEVEL STATISTICS
# ============================================================

print_section(
    "2. GLOBAL TRIAL-LEVEL STATISTICS"
)

X = df[
    raw_features
]

engineered[
    "fnirs_global_mean"
] = X.mean(
    axis=1,
    skipna=True
)

engineered[
    "fnirs_global_std"
] = X.std(
    axis=1,
    skipna=True
)

engineered[
    "fnirs_global_median"
] = X.median(
    axis=1,
    skipna=True
)

engineered[
    "fnirs_global_min"
] = X.min(
    axis=1,
    skipna=True
)

engineered[
    "fnirs_global_max"
] = X.max(
    axis=1,
    skipna=True
)

engineered[
    "fnirs_global_range"
] = (
    engineered["fnirs_global_max"]
    -
    engineered["fnirs_global_min"]
)

engineered[
    "fnirs_global_rms"
] = np.sqrt(
    (
        X.pow(2)
        .mean(
            axis=1,
            skipna=True
        )
    )
)

print(
    "Added 7 global statistics"
)


# ============================================================
# 3. WAVELENGTH-SPECIFIC FEATURES
# ============================================================

print_section(
    "3. WAVELENGTH-SPECIFIC FEATURES"
)

features_785 = [
    feature
    for feature in raw_features
    if "785" in feature
]

features_830 = [
    feature
    for feature in raw_features
    if "830" in feature
]

print(
    "785 nm features:",
    len(features_785)
)

print(
    "830 nm features:",
    len(features_830)
)

if features_785:

    X785 = df[
        features_785
    ]

    engineered[
        "wavelength_785_mean"
    ] = X785.mean(
        axis=1,
        skipna=True
    )

    engineered[
        "wavelength_785_std"
    ] = X785.std(
        axis=1,
        skipna=True
    )

    engineered[
        "wavelength_785_median"
    ] = X785.median(
        axis=1,
        skipna=True
    )

    engineered[
        "wavelength_785_min"
    ] = X785.min(
        axis=1,
        skipna=True
    )

    engineered[
        "wavelength_785_max"
    ] = X785.max(
        axis=1,
        skipna=True
    )

    engineered[
        "wavelength_785_range"
    ] = (
        engineered["wavelength_785_max"]
        -
        engineered["wavelength_785_min"]
    )

if features_830:

    X830 = df[
        features_830
    ]

    engineered[
        "wavelength_830_mean"
    ] = X830.mean(
        axis=1,
        skipna=True
    )

    engineered[
        "wavelength_830_std"
    ] = X830.std(
        axis=1,
        skipna=True
    )

    engineered[
        "wavelength_830_median"
    ] = X830.median(
        axis=1,
        skipna=True
    )

    engineered[
        "wavelength_830_min"
    ] = X830.min(
        axis=1,
        skipna=True
    )

    engineered[
        "wavelength_830_max"
    ] = X830.max(
        axis=1,
        skipna=True
    )

    engineered[
        "wavelength_830_range"
    ] = (
        engineered["wavelength_830_max"]
        -
        engineered["wavelength_830_min"]
    )

print(
    "Added wavelength-level statistics"
)


# ============================================================
# 4. 785 vs 830 FEATURES
# ============================================================

print_section(
    "4. 785 nm vs 830 nm COMPARISON FEATURES"
)

# Match source-detector pairs.
#
# Example:
# S1_D3 785
# S1_D3 830

pair_785 = {}

pair_830 = {}

for feature in raw_features:

    if feature.endswith(" 785"):

        pair_name = feature[:-4].strip()

        pair_785[pair_name] = feature

    elif feature.endswith(" 830"):

        pair_name = feature[:-4].strip()

        pair_830[pair_name] = feature


common_pairs = sorted(
    set(pair_785)
    &
    set(pair_830)
)

print(
    "Common source-detector pairs:",
    len(common_pairs)
)

print(
    "\nPairs:"
)

for pair in common_pairs:

    print(
        " ",
        pair
    )


EPSILON = 1e-12

pair_feature_names = []


for pair in common_pairs:

    col_785 = pair_785[
        pair
    ]

    col_830 = pair_830[
        pair
    ]

    x785 = df[
        col_785
    ]

    x830 = df[
        col_830
    ]

    # --------------------------------------------------------
    # Difference
    # --------------------------------------------------------

    diff_name = (
        f"{pair}_785_830_difference"
    )

    engineered[
        diff_name
    ] = (
        x830
        -
        x785
    )

    pair_feature_names.append(
        diff_name
    )


    # --------------------------------------------------------
    # Absolute difference
    # --------------------------------------------------------

    abs_diff_name = (
        f"{pair}_785_830_abs_difference"
    )

    engineered[
        abs_diff_name
    ] = (
        x830
        -
        x785
    ).abs()

    pair_feature_names.append(
        abs_diff_name
    )


    # --------------------------------------------------------
    # Mean
    # --------------------------------------------------------

    mean_name = (
        f"{pair}_785_830_mean"
    )

    engineered[
        mean_name
    ] = pd.concat(
        [x785, x830],
        axis=1
    ).mean(
        axis=1,
        skipna=True
    )

    pair_feature_names.append(
        mean_name
    )


    # --------------------------------------------------------
    # Ratio
    #
    # We protect against division by zero.
    # --------------------------------------------------------

    ratio_name = (
        f"{pair}_830_to_785_ratio"
    )

    denominator = x785.abs()

    engineered[
        ratio_name
    ] = (
        x830
        /
        denominator.clip(
            lower=EPSILON
        )
    )

    pair_feature_names.append(
        ratio_name
    )


print(
    "\nAdded pairwise wavelength features:",
    len(pair_feature_names)
)


# ============================================================
# 5. SOURCE-LEVEL FEATURES
# ============================================================

print_section(
    "5. SOURCE-LEVEL FEATURES"
)

sources = sorted(
    {
        feature.split("_")[0]
        for feature in raw_features
    }
)

print(
    "Detected sources:",
    sources
)

source_feature_count = 0


for source in sources:

    source_features = [
        feature
        for feature in raw_features
        if feature.startswith(
            source + "_"
        )
    ]

    if not source_features:
        continue

    source_data = df[
        source_features
    ]

    engineered[
        f"{source}_mean"
    ] = source_data.mean(
        axis=1,
        skipna=True
    )

    engineered[
        f"{source}_std"
    ] = source_data.std(
        axis=1,
        skipna=True
    )

    engineered[
        f"{source}_median"
    ] = source_data.median(
        axis=1,
        skipna=True
    )

    engineered[
        f"{source}_min"
    ] = source_data.min(
        axis=1,
        skipna=True
    )

    engineered[
        f"{source}_max"
    ] = source_data.max(
        axis=1,
        skipna=True
    )

    source_feature_count += 5


print(
    "Added source-level features:",
    source_feature_count
)


# ============================================================
# 6. NUMBER OF AVAILABLE MEASUREMENTS
# ============================================================

print_section(
    "6. AVAILABLE MEASUREMENT COUNT"
)

engineered[
    "fnirs_available_measurements"
] = X.notna().sum(
    axis=1
)

print(
    "Added available-measurement count."
)


# ============================================================
# 7. MISSING FRACTION
# ============================================================

engineered[
    "fnirs_missing_fraction"
] = (
    X.isna()
    .mean(
        axis=1
    )
)

print(
    "Added missing-measurement fraction."
)


# ============================================================
# IMPORTANT:
# DO NOT IMPUTE HERE
# ============================================================

print_section(
    "MISSING VALUE HANDLING"
)

print(
    "NaN values are intentionally preserved."
)

print(
    "Imputation will be performed inside the ML"
)

print(
    "pipeline using statistics learned ONLY"
)

print(
    "from the training data."
)


# ============================================================
# FINAL FEATURE IDENTIFICATION
# ============================================================

final_feature_columns = [
    column
    for column in engineered.columns
    if column not in metadata_columns
]


# ============================================================
# CHECK DUPLICATES
# ============================================================

duplicate_feature_rows = engineered[
    final_feature_columns
].duplicated().sum()


# ============================================================
# CHECK MISSING / INFINITE
# ============================================================

missing_total = int(
    engineered[
        final_feature_columns
    ]
    .isna()
    .sum()
    .sum()
)

infinite_total = int(
    np.isinf(
        engineered[
            final_feature_columns
        ]
        .select_dtypes(
            include=np.number
        )
        .to_numpy()
    )
    .sum()
)


# ============================================================
# FEATURE STATISTICS
# ============================================================

feature_statistics = (
    engineered[
        final_feature_columns
    ]
    .describe()
    .T
)

feature_statistics[
    "missing_count"
] = (
    engineered[
        final_feature_columns
    ]
    .isna()
    .sum()
)

feature_statistics[
    "missing_percentage"
] = (
    feature_statistics[
        "missing_count"
    ]
    /
    len(engineered)
    *
    100
)

statistics_path = os.path.join(
    RESULTS_DIR,
    "engineered_feature_statistics.csv"
)

feature_statistics.to_csv(
    statistics_path
)


# ============================================================
# SAVE ENGINEERED DATASET
# ============================================================

engineered.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print_section(
    "FINAL FEATURE ENGINEERING SUMMARY"
)

print(
    "Original rows:",
    len(df)
)

print(
    "Original columns:",
    len(df.columns)
)

print(
    "Original fNIRS features:",
    len(raw_features)
)

print(
    "Engineered rows:",
    len(engineered)
)

print(
    "Engineered columns:",
    len(engineered.columns)
)

print(
    "Total ML features:",
    len(final_feature_columns)
)

print(
    "Missing feature values:",
    missing_total
)

print(
    "Infinite feature values:",
    infinite_total
)

print(
    "Duplicate feature rows:",
    duplicate_feature_rows
)


# ============================================================
# FEATURE GROUP SUMMARY
# ============================================================

print_section(
    "FEATURE GROUP SUMMARY"
)

print(
    "Raw fNIRS features:",
    len(raw_features)
)

print(
    "Global statistics:",
    7
)

print(
    "Wavelength statistics:",
    12
)

print(
    "Pairwise wavelength features:",
    len(pair_feature_names)
)

print(
    "Source-level features:",
    source_feature_count
)

print(
    "Availability features:",
    2
)


# ============================================================
# OUTPUT PATHS
# ============================================================

print_section(
    "OUTPUT FILES"
)

print(
    "Engineered dataset:"
)

print(
    OUTPUT_PATH
)

print(
    "\nFeature statistics:"
)

print(
    statistics_path
)


# ============================================================
# COMPLETED
# ============================================================

print("\n")
print("=" * 70)
print(
    "fNIRS FEATURE ENGINEERING COMPLETED SUCCESSFULLY"
)
print("=" * 70)