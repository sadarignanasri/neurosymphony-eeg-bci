from pathlib import Path
import hashlib
import mne
import json

ROOTS = [
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults 3"),
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults-2 2"),
]

SUBJECTS = [f"S{i:02d}" for i in range(1, 17)]
seen_hashes = set()
report = []

for subject in SUBJECTS:
    files = []
    for root in ROOTS:
        folder = root / subject / "EEG" / "raw"
        if folder.exists():
            files.extend(folder.glob("*.edf"))

    subject_rows = []
    for path in sorted(files):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()

        if digest in seen_hashes:
            subject_rows.append({
                "file": str(path),
                "duplicate_content": True,
            })
            continue

        seen_hashes.add(digest)

        try:
            raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
            row = {
                "file": str(path),
                "duplicate_content": False,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                "sampling_rate": float(raw.info["sfreq"]),
                "channels": len(raw.ch_names),
                "duration_seconds": round(
                    raw.n_times / raw.info["sfreq"], 2
                ),
                "annotation_count": len(raw.annotations),
                "trigger_counts_by_description": {},
            }

            for desc in raw.annotations.description:
                row["trigger_counts_by_description"][desc] = (
                    row["trigger_counts_by_description"].get(desc, 0) + 1
                )

            subject_rows.append(row)

        except Exception as exc:
            subject_rows.append({
                "file": str(path),
                "duplicate_content": False,
                "error": str(exc),
            })

    report.append({"subject": subject, "files": subject_rows})

    print(f"\n{subject}")
    if not subject_rows:
        print("  NO RAW EDF FILE FOUND")
    for row in subject_rows:
        if row.get("duplicate_content"):
            print(f"  DUPLICATE: {row['file']}")
        elif "error" in row:
            print(f"  ERROR: {row['file']} -> {row['error']}")
        else:
            print(
                f"  {Path(row['file']).name}: "
                f"{row['duration_seconds']}s, "
                f"{row['channels']} channels, "
                f"{row['sampling_rate']} Hz, "
                f"{row['annotation_count']} annotations"
            )

out = Path("results/eeg/visual_motivation/step167_raw_edf_inventory.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"\nReport saved: {out}")
print("No original data was modified.")
