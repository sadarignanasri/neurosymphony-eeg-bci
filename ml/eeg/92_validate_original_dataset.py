
from pathlib import Path
import json
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT / "major datasets" / "ds004514_real"
OUTPUT_DIR = PROJECT / "experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = [
    "animal_tactile_imagery_task",
    "animal_visual_imagery_task",
    "animal_auditory_imagery_task",
    "animal_silent_naming_task",
    "tool_tactile_imagery_task",
    "tool_visual_imagery_task",
    "tool_auditory_imagery_task",
    "tool_silent_naming_task",
]

rows = []
problems = []

for subject_dir in sorted(DATA_ROOT.glob("sub-*")):
    subject = subject_dir.name
    eeg_dir = subject_dir / "eeg"
    bdf_files = list(eeg_dir.glob("*_eeg.bdf"))
    event_files = list(eeg_dir.glob("*_events.tsv"))

    if len(bdf_files) != 1:
        problems.append(
            f"{subject}: expected 1 EEG BDF, found {len(bdf_files)}"
        )

    if len(event_files) != 1:
        problems.append(
            f"{subject}: expected 1 EEG events TSV, found {len(event_files)}"
        )
        continue

    events = pd.read_csv(event_files[0], sep="\t")
    required = {"trial_type", "onset", "sample"}

    if not required.issubset(events.columns):
        problems.append(
            f"{subject}: missing columns {required - set(events.columns)}"
        )
        continue

    task_events = events[events["trial_type"].isin(TASKS)].copy()

    counts = task_events["trial_type"].value_counts()

    for task in TASKS:
        rows.append({
            "subject": subject,
            "task": task,
            "label": TASKS.index(task),
            "trial_count": int(counts.get(task, 0)),
        })

    if len(task_events) != 720:
        problems.append(
            f"{subject}: expected 720 task events, found {len(task_events)}"
        )

    if task_events["sample"].isna().any():
        problems.append(f"{subject}: missing event sample indices")

    if task_events["sample"].duplicated().any():
        problems.append(f"{subject}: duplicate task event sample indices")

    if set(task_events["trial_type"]) != set(TASKS):
        problems.append(f"{subject}: one or more task classes are missing")

report = pd.DataFrame(rows)
report_path = OUTPUT_DIR / "original_eeg_label_audit.csv"
report.to_csv(report_path, index=False)

summary = {
    "dataset": str(DATA_ROOT),
    "subjects_audited": int(report["subject"].nunique()),
    "task_event_count": int(report["trial_count"].sum()),
    "expected_classes": TASKS,
    "label_mapping": {task: i for i, task in enumerate(TASKS)},
    "problems": problems,
    "passed": len(problems) == 0,
}

summary_path = OUTPUT_DIR / "original_eeg_label_audit.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
print(f"\nCSV report: {report_path}")
print(f"JSON report: {summary_path}")
