"""
Neuro-Symphony
Step 29: Inspect labelled task epochs for sub-01

This script does NOT extract or modify EEG data.
It verifies the timing and structure of the 720 labelled trials.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVENTS_TSV = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sub-01"
    / "sub-01_task-eeg_events.tsv"
)

SFREQ = 2048.0

TASK_LABELS = [
    "animal_tactile_imagery_task",
    "animal_visual_imagery_task",
    "animal_auditory_imagery_task",
    "animal_silent_naming_task",
    "tool_tactile_imagery_task",
    "tool_visual_imagery_task",
    "tool_auditory_imagery_task",
    "tool_silent_naming_task",
]


# ============================================================
# LOAD EVENTS
# ============================================================

events = pd.read_csv(EVENTS_TSV, sep="\t")

events = events.sort_values("sample").reset_index(drop=True)

print("=" * 80)
print("NEURO-SYMPHONY — SUB-01 TASK EPOCH INSPECTION")
print("=" * 80)


# ============================================================
# CALCULATE NEXT EVENT USING FULL EVENT STREAM
# ============================================================

events["next_event_sample"] = events["sample"].shift(-1)

events["next_event_trial_type"] = (
    events["trial_type"].shift(-1)
)

events["time_to_next_event"] = (
    events["next_event_sample"] - events["sample"]
) / SFREQ


# ============================================================
# SELECT ONLY THE 8 TASK EVENTS
# ============================================================

task_events = events[
    events["trial_type"].isin(TASK_LABELS)
].copy()

task_events = task_events.reset_index(drop=True)


# ============================================================
# BASIC COUNTS
# ============================================================

print("\n" + "=" * 80)
print("TASK EVENT COUNTS")
print("=" * 80)

print("\nTotal labelled task events:", len(task_events))

print("\nTrials per class:")
print(
    task_events["trial_type"]
    .value_counts()
    .sort_index()
)


# ============================================================
# VERIFY 720 TRIALS
# ============================================================

expected_trials = 8 * 90

print("\nExpected trials:", expected_trials)
print("Actual trials  :", len(task_events))

if len(task_events) == expected_trials:
    print("✓ Exactly 720 labelled trials found.")
else:
    print("✗ Trial count does not match 720.")


# ============================================================
# VERIFY EACH CLASS HAS 90 TRIALS
# ============================================================

counts = task_events["trial_type"].value_counts()

print("\n" + "=" * 80)
print("CLASS BALANCE")
print("=" * 80)

for label in TASK_LABELS:
    count = int(counts.get(label, 0))

    if count == 90:
        print(f"✓ {label}: {count}")
    else:
        print(f"✗ {label}: {count}")


# ============================================================
# DISPLAY FIRST 20 TASK EVENTS
# ============================================================

print("\n" + "=" * 80)
print("TASK EVENT TIMING")
print("=" * 80)

print("\nFirst 20 labelled trials:\n")

display_columns = [
    "trial_type",
    "sample",
    "next_event_sample",
    "next_event_trial_type",
    "time_to_next_event",
]

print(
    task_events[display_columns]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# TIMING STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("TIME TO NEXT EVENT STATISTICS")
print("=" * 80)

timing = task_events["time_to_next_event"].dropna()

print(f"\nMean   : {timing.mean():.6f} sec")
print(f"Median : {timing.median():.6f} sec")
print(f"Min    : {timing.min():.6f} sec")
print(f"Max    : {timing.max():.6f} sec")
print(f"Std    : {timing.std():.6f} sec")


# ============================================================
# CHECK 3-SECOND WINDOW
# ============================================================

WINDOW_SECONDS = 3.0
WINDOW_SAMPLES = int(WINDOW_SECONDS * SFREQ)

print("\n" + "=" * 80)
print("3-SECOND EPOCH CHECK")
print("=" * 80)

print(f"\nSampling rate : {SFREQ} Hz")
print(f"Window        : {WINDOW_SECONDS} seconds")
print(f"Samples       : {WINDOW_SAMPLES}")

enough_time = timing >= WINDOW_SECONDS

print(
    f"\nTrials with >= 3 seconds before next event: "
    f"{enough_time.sum()} / {len(timing)}"
)

print(
    f"Trials with < 3 seconds before next event: "
    f"{(~enough_time).sum()} / {len(timing)}"
)


# ============================================================
# CHECK TASK → NEXT EVENT TYPES
# ============================================================

print("\n" + "=" * 80)
print("TASK → NEXT EVENT TYPES")
print("=" * 80)

print(
    task_events["next_event_trial_type"]
    .value_counts(dropna=False)
)


# ============================================================
# CHECK SAMPLE ALIGNMENT
# ============================================================

print("\n" + "=" * 80)
print("SAMPLE ALIGNMENT")
print("=" * 80)

first_trial = task_events.iloc[0]

start_sample = int(first_trial["sample"])
end_sample = start_sample + WINDOW_SAMPLES

print("\nExample first trial:")
print(f"Label        : {first_trial['trial_type']}")
print(f"Start sample : {start_sample}")
print(f"End sample   : {end_sample}")
print(
    f"Duration     : {(end_sample - start_sample) / SFREQ:.3f} sec"
)

print(
    f"Next event   : {first_trial['next_event_trial_type']}"
)

print(
    f"Time to next : {first_trial['time_to_next_event']:.6f} sec"
)


# ============================================================
# CHECK WHETHER 3-SECOND WINDOW FITS
# ============================================================

print("\n" + "=" * 80)
print("FINAL 3-SECOND WINDOW VALIDATION")
print("=" * 80)

if enough_time.all():
    print(
        "\n✓ All 720 task events have at least 3 seconds "
        "before the next event."
    )
else:
    bad_count = int((~enough_time).sum())

    print(
        f"\n⚠ {bad_count} trials have less than 3 seconds "
        "before the next event."
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)

print("\nNo EEG data was extracted.")
print("No preprocessing was performed.")
print("No model was trained.")