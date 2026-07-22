import os
import mne
import matplotlib.pyplot as plt

dataset_name = "ds004514"
subject_id = "sub-01"

output_dir = rf"F:\experiments\{dataset_name}\{subject_id}\fnirs"

os.makedirs(output_dir, exist_ok=True)

file_path = r"F:\ds004514_real\sub-01\nirs\sub-01_task-nirs_nirs.snirf"

print("Loading real fNIRS...")

raw = mne.io.read_raw_snirf(file_path, preload=True)

print(raw)

print("Cropping first 20 seconds...")

raw.crop(tmin=0, tmax=20)

print("Selecting first fNIRS channel...")

data, times = raw[:1]

plt.figure(figsize=(12, 4))

plt.plot(times, data[0])

plt.title("Real fNIRS Signal")

plt.xlabel("Time (s)")

plt.ylabel("Amplitude")

plt.grid(True)

save_path = os.path.join(output_dir, "real_fnirs_plot.png")

plt.savefig(save_path, dpi=300)

print(f"fNIRS plot saved at: {save_path}")

plt.show()

print("Real fNIRS loaded successfully.")