
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:\neurosymphony-eeg-bci")

NPZ_PATH = ROOT / "experiments" / "ds004022_verified_epochs" / "eeg_epochs.npz"
RESULTS_DIR = ROOT / "experiments" / "ds004022_experiment100_results"

with np.load(NPZ_PATH, allow_pickle=True) as archive:
    data = {key: archive[key] for key in archive.files}

y = np.asarray(data["y"]).reshape(-1)
subjects = np.asarray(data["subjects"]).astype(str).reshape(-1)
runs = np.asarray(data["runs"]).astype(str).reshape(-1)

print("\n========== SAVED CLASS NAMES ==========")
if "class_names" in data:
    print(data["class_names"])
else:
    print("No class_names array found.")

print("\n========== CUE MARKERS ==========")
if "cue_markers" in data:
    cues = np.asarray(data["cue_markers"]).reshape(-1)
    print("Unique cue markers:", np.unique(cues))
    print("\nLabel-to-cue combinations:")
    print(pd.crosstab(
        pd.Series(y, name="label_id"),
        pd.Series(cues.astype(str), name="cue_marker"),
    ))
else:
    cues = None
    print("No cue_markers array found.")

print("\n========== LABEL COUNTS BY SUBJECT ==========")
print(pd.crosstab(
    pd.Series(subjects, name="subject"),
    pd.Series(y, name="label_id"),
))

print("\n========== LABEL COUNTS BY RUN ==========")
print(pd.crosstab(
    [pd.Series(subjects, name="subject"),
     pd.Series(runs, name="run")],
    pd.Series(y, name="label_id"),
))

flag_path = RESULTS_DIR / "trials_to_review.csv"

print("\n========== FLAGGED TRIALS ==========")
if flag_path.exists():
    flagged = pd.read_csv(flag_path)

    columns = [
        "trial_index",
        "subject",
        "run",
        "label_id",
        "class_name",
        "mean_channel_std",
        "max_channel_std",
        "mean_channel_peak_to_peak",
        "max_channel_peak_to_peak",
        "mean_absolute_amplitude",
        "fraction_zero_std_channels",
        "has_extreme_amplitude",
        "std_outlier_flag",
    ]

    columns = [c for c in columns if c in flagged.columns]

    if flagged.empty:
        print("No flagged trials.")
    else:
        print(flagged[columns].to_string(index=False))
else:
    print("Flagged-trials CSV not found.")

print("\n========== COMPLETED ==========")
print("This script only reads existing files. It does not modify the dataset.")
