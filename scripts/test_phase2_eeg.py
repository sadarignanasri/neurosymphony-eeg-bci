
import os
import sys

import matplotlib.pyplot as plt
import mne

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.data.data_loader import DataLoader
from src.preprocessing.eeg import purify_eeg


def main():

    # Phase 1: locate and load the EEG
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

    dataset_root = os.path.join(
        project_root,
        "..",
        "ds004514"
    )

    loader = DataLoader(dataset_root)

    print("Loading real EEG...")
    raw = loader.load_raw_bdf("sub-01")

    print("\nOriginal EEG:")
    print(raw)

    # Phase 2: purification
    print("\nRunning Phase 2 EEG purification...")

    purified = purify_eeg(raw)

    print("\nPurified EEG:")
    print(purified)

    # Compare first channel
    raw_copy = raw.copy().crop(tmin=0, tmax=10)
    purified_copy = purified.copy().crop(tmin=0, tmax=10)

    raw_data = raw_copy.get_data(picks="eeg")[0]
    purified_data = purified_copy.get_data(picks="eeg")[0]

    times = raw_copy.times

    # Plot
    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(12, 6))

    plt.plot(
        times,
        raw_data,
        label="Original EEG",
        alpha=0.7
    )

    plt.plot(
        times,
        purified_data,
        label="Phase 2 purified EEG",
        alpha=0.8
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Phase 2 EEG Purification - Channel 0")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = "outputs/phase2_eeg_comparison.png"

    plt.savefig(output_path)
    plt.close()

    print("\nSaved comparison to:")
    print(output_path)


if __name__ == "__main__":
    main()