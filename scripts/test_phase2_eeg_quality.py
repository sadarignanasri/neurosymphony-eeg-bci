
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.data.data_loader import DataLoader
from src.preprocessing.eeg import purify_eeg


def main():

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

    dataset_root = os.path.join(
        project_root,
        "..",
        "ds004514"
    )

    print("Loading real EEG...")

    loader = DataLoader(dataset_root)

    raw = loader.load_raw_bdf("sub-01")

    print("\nOriginal EEG:")
    print(raw)

    # --------------------------------------------------
    # Use only the first 30 seconds for development
    # --------------------------------------------------

    raw_segment = raw.copy().crop(
        tmin=0,
        tmax=30
    )

    print("\nTesting on first 30 seconds.")

    # --------------------------------------------------
    # Phase 2 purification
    # --------------------------------------------------

    purified = purify_eeg(raw_segment)

    print("\nPurified EEG:")
    print(purified)

    # --------------------------------------------------
    # Extract Channel 0
    # --------------------------------------------------

    original_data = raw_segment.get_data(
        picks="eeg"
    )[0]

    purified_data = purified.get_data(
        picks="eeg"
    )[0]

    # --------------------------------------------------
    # Basic statistics
    # --------------------------------------------------

    print("\n==============================")
    print("EEG QUALITY COMPARISON")
    print("==============================")

    print("\nOriginal EEG")
    print("Mean:", np.mean(original_data))
    print("Std :", np.std(original_data))
    print("Min :", np.min(original_data))
    print("Max :", np.max(original_data))

    print("\nPurified EEG")
    print("Mean:", np.mean(purified_data))
    print("Std :", np.std(purified_data))
    print("Min :", np.min(purified_data))
    print("Max :", np.max(purified_data))

    # --------------------------------------------------
    # NaN check
    # --------------------------------------------------

    print("\nNaN check")

    print(
        "Original NaNs:",
        np.isnan(original_data).sum()
    )

    print(
        "Purified NaNs:",
        np.isnan(purified_data).sum()
    )

    # --------------------------------------------------
    # Correlation
    # --------------------------------------------------

    correlation = np.corrcoef(
        original_data,
        purified_data
    )[0, 1]

    print("\nCorrelation:")
    print(correlation)

    # --------------------------------------------------
    # Plot comparison
    # --------------------------------------------------

    os.makedirs("outputs", exist_ok=True)

    times = raw_segment.times

    plt.figure(figsize=(14, 6))

    plt.plot(
        times,
        original_data,
        label="Original EEG",
        alpha=0.7
    )

    plt.plot(
        times,
        purified_data,
        label="Phase 2 purified EEG",
        alpha=0.7
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.title(
        "Phase 2 EEG Quality Check - Channel 0"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = (
        "outputs/phase2_eeg_quality.png"
    )

    plt.savefig(output_path)

    plt.close()

    print("\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()