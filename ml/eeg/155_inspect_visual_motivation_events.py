from pathlib import Path
import json
from collections import Counter
import mne

ROOTS = [
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults 3"),
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults-2 2"),
]

report = {"datasets": []}

for root in ROOTS:
    if not root.exists():
        continue

    dataset = {"folder": str(root), "files": []}
    edf_files = sorted(set(root.glob("S*/**/*.edf")))

    for path in edf_files:
        raw = mne.io.read_raw_edf(
            str(path), preload=False, verbose="ERROR"
        )

        annotations = raw.annotations
        descriptions = [str(x).strip() for x in annotations.description]
        counts = Counter(descriptions)

        entry = {
            "file": str(path),
            "subject": path.parts[-3] if path.parent.name == "EEG" else next(
                (part for part in path.parts
                 if part.startswith("S") and part[1:].isdigit()),
                "UNKNOWN",
            ),
            "sampling_rate_hz": float(raw.info["sfreq"]),
            "channel_names": list(raw.ch_names),
            "annotation_counts": dict(counts),
            "first_30_annotations": [
                {
                    "onset_seconds": float(annotations.onset[i]),
                    "duration_seconds": float(annotations.duration[i]),
                    "description": descriptions[i],
                }
                for i in range(min(30, len(annotations)))
            ],
        }

        dataset["files"].append(entry)
        print(f"\nFILE: {path}")
        print("CHANNELS:", raw.ch_names)
        print("ANNOTATION COUNTS:", dict(counts))
        print("FIRST 15 ANNOTATIONS:")
        for item in entry["first_30_annotations"][:15]:
            print(
                f"  {item['onset_seconds']:.3f}s | "
                f"{item['description']} | "
                f"duration={item['duration_seconds']:.3f}s"
            )

        raw.close()

    report["datasets"].append(dataset)

out = Path("results/eeg/visual_motivation/step155_event_inspection.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("\nSaved:", out)
