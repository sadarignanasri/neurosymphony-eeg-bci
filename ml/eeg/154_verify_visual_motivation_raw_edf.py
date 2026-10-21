from pathlib import Path
import json

try:
    import mne
except ImportError:
    raise SystemExit(
        "MNE is not installed in this Python environment. "
        "Activate your project environment before continuing."
    )

ROOTS = [
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults 3"),
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults-2 2"),
]

OUT_DIR = Path("results/eeg/visual_motivation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

report = {
    "purpose": "Raw EDF integrity audit; no preprocessing performed",
    "datasets": [],
    "exact_duplicate_edf_files": [],
    "warnings": [],
}

seen_hashes = {}

import hashlib

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

for root in ROOTS:
    dataset = {
        "folder": str(root),
        "edf_files": [],
        "subject_count_with_edf": 0,
    }

    if not root.exists():
        report["warnings"].append(f"Missing folder: {root}")
        continue

    edf_files = sorted(root.glob("S*/*.edf"))
    # Also support EDFs nested deeper in a subject's folder.
    edf_files = sorted(set(edf_files) | set(root.glob("S*/**/*.edf")))

    subjects_found = set()

    for path in edf_files:
        subject = next(
            (part for part in path.parts if part.startswith("S") and part[1:].isdigit()),
            "UNKNOWN",
        )
        subjects_found.add(subject)
        digest = sha256(path)

        entry = {
            "subject": subject,
            "file": str(path),
            "sha256": digest,
            "duplicate_of": None,
        }

        if digest in seen_hashes:
            entry["duplicate_of"] = seen_hashes[digest]
            report["exact_duplicate_edf_files"].append({
                "file": str(path),
                "same_as": seen_hashes[digest],
            })
        else:
            seen_hashes[digest] = str(path)

        try:
            raw = mne.io.read_raw_edf(
                str(path),
                preload=False,
                verbose="ERROR",
            )

            sfreq = float(raw.info["sfreq"])
            duration = float(raw.n_times / sfreq)
            channel_names = list(raw.ch_names)
            annotations = []

            if raw.annotations is not None:
                for onset, dur, desc in zip(
                    raw.annotations.onset,
                    raw.annotations.duration,
                    raw.annotations.description,
                ):
                    annotations.append({
                        "onset_seconds": float(onset),
                        "duration_seconds": float(dur),
                        "description": str(desc),
                    })

            entry.update({
                "status": "readable",
                "sampling_rate_hz": sfreq,
                "channel_count": len(channel_names),
                "channel_names": channel_names,
                "sample_count": int(raw.n_times),
                "duration_seconds": duration,
                "annotation_count": len(annotations),
                "annotations": annotations,
            })

            print(
                f"{subject} | {path.name} | "
                f"{sfreq:g} Hz | {len(channel_names)} channels | "
                f"{duration:.2f} s | {len(annotations)} annotations"
            )

            raw.close()

        except Exception as exc:
            entry.update({
                "status": "error",
                "error": str(exc),
            })
            print(f"ERROR | {path} | {exc}")

        dataset["edf_files"].append(entry)

    dataset["subject_count_with_edf"] = len(subjects_found)
    report["datasets"].append(dataset)

report_path = OUT_DIR / "step154_raw_edf_audit.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\n========== STEP 154 SUMMARY ==========")
for dataset in report["datasets"]:
    print("Dataset:", Path(dataset["folder"]).name)
    print("Readable EDF files:", sum(
        x["status"] == "readable" for x in dataset["edf_files"]
    ))
    print("Subjects with EDF files:", dataset["subject_count_with_edf"])

print("Exact duplicate EDF files:", len(report["exact_duplicate_edf_files"]))
print("Warnings:", report["warnings"] or "None")
print("Report saved to:", report_path)
print("No raw data or existing models were modified.")
