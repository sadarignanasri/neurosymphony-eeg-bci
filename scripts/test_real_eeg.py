import os
import mne
import matplotlib.pyplot as plt

dataset_name = "ds004514"
subject_id = "sub-01"

output_dir = rf"F:\experiments\{dataset_name}\{subject_id}\eeg"

os.makedirs(output_dir, exist_ok=True)

file_path = r"F:\ds004514_real\sub-01\eeg\sub-01_task-eeg_eeg.bdf"

print("Opening metadata...")

raw = mne.io.read_raw_bdf(file_path, preload=False)

print(raw)

print("Selecting first 10 seconds...")

raw.crop(tmin=0, tmax=10)

print("Loading cropped EEG...")

raw.load_data()

print("Downsampling to 256 Hz...")

raw.resample(256)

print("Extracting first EEG channel...")

data, times = raw[:1]

plt.figure(figsize=(12, 4))

plt.plot(times, data[0])

plt.title("Real EEG Signal")

plt.xlabel("Time (s)")

plt.ylabel("Amplitude")

plt.grid(True)

save_path = os.path.join(output_dir, "real_eeg_plot.png")

plt.savefig(save_path, dpi=300)

print(f"EEG plot saved at: {save_path}")

plt.show()

print("EEG loaded successfully!")

print(raw.info)
