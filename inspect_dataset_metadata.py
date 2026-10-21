
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent / "major datasets"

TARGETS = [
    "ds004514_real",
    "ds004022",
    "A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults",
    "Simultaneous EEG-fNIRS Data on Learning Capability",
]

KEYWORDS = (
    "samplingfrequency",
    "channelcount",
    "taskname",
    "taskdescription",
    "event",
    "epoch",
    "trial",
    "class",
    "label",
    "sampling rate",
    "channel",
)

def inspect(name):
    folder = ROOT / name
    print("\n" + "=" * 70)
    print("DATASET:", name)
    print("=" * 70)

    if not folder.is_dir():
        print("MISSING:", folder)
        return

    files = [
        p for p in folder.rglob("*")
        if p.is_file()
        and ".git" not in p.parts
        and ".datalad" not in p.parts
    ]

    metadata = [
        p for p in files
        if p.suffix.lower() == ".json"
        and p.stat().st_size < 500_000
    ]

    for path in metadata[:30]:
        try:
            data = json.loads(
                path.read_text(encoding="utf-8")
            )
        except (ValueError, OSError):
            continue

        if not isinstance(data, dict):
            continue

        relevant = {
            key: value
            for key, value in data.items()
            if any(word in key.lower() for word in KEYWORDS)
        }

        if relevant:
            print("\nFILE:", path.relative_to(folder))
            print(json.dumps(relevant, indent=2))

    event_files = [
        p for p in files
        if "event" in p.name.lower()
        and p.suffix.lower() in {".tsv", ".txt", ".csv"}
    ]

    for path in event_files[:5]:
        print("\nEVENT FILE:", path.relative_to(folder))
        try:
            print(
                path.read_text(
                    encoding="utf-8", errors="replace"
                )[:1200]
            )
        except OSError as error:
            print("Read error:", error)

if __name__ == "__main__":
    for name in TARGETS:
        inspect(name)

    print("\nMetadata inspection completed. No files modified.")
