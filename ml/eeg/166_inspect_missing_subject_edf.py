from pathlib import Path
import hashlib
import mne

ROOTS = [
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults 3"),
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults-2 2"),
]

SUBJECTS = ["S02", "S04", "S06", "S11", "S12", "S15"]
seen = set()

for subject in SUBJECTS:
    print(f"\n{'=' * 80}\n{subject}\n{'=' * 80}")

    files = []
    for root in ROOTS:
        raw_dir = root / subject / "EEG" / "raw"
        if raw_dir.exists():
            files.extend(raw_dir.glob("*.edf"))

    for path in sorted(files):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            print(f"DUPLICATE CONTENT — skipped: {path}")
            continue
        seen.add(digest)

        print(f"\nFile: {path}")
        try:
            raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
            print(f"  Sampling rate: {raw.info['sfreq']} Hz")
            print(f"  Channels: {len(raw.ch_names)}")
            print(f"  Duration: {raw.n_times / raw.info['sfreq']:.2f} seconds")
            print(f"  Channel names: {raw.ch_names}")

            annotations = raw.annotations
            print(f"  Annotation count: {len(annotations)}")
            print("  First 12 annotations:")
            for onset, duration, description in zip(
                annotations.onset[:12],
                annotations.duration[:12],
                annotations.description[:12],
            ):
                print(
                    f"    onset={onset:.3f}s, "
                    f"duration={duration:.3f}s, event={description}"
                )
        except Exception as exc:
            print(f"  ERROR: {exc}")

print("\nInspection complete. No EEG files were modified.")
