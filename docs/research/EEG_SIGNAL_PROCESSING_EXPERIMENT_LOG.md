# NeuroSymphony — EEG Signal-Processing Research Log

## 1. Research objective

Investigate EEG signal-processing and feature-extraction techniques
for their effect on classification performance across 12 subjects
and 8 task classes.

The objective is to conduct controlled, reproducible experiments.
No predetermined accuracy is guaranteed or required. Improvements,
declines, and inconclusive findings will all be documented.

## 2. Dataset configuration

- Dataset: DS004514
- Subjects: 12
- Trials per subject: 720
- Task classes: 8
- EEG channels: 64
- Sampling rate: 2048 Hz
- Samples per trial: 6142
- Trial duration: approximately 3.00098 seconds

## 3. Existing preprocessing baseline

Script:
ml/eeg/31_preprocess_all_subjects_temporal.py

Implementation:
src/preprocessing/eeg.py

Current processing sequence:
1. Identify and type EEG and auxiliary channels.
2. Rename BioSemi channels.
3. Apply the BioSemi-64 montage.
4. Select 64 EEG channels.
5. Apply a 1–40 Hz FIR band-pass filter.
6. Apply a 50 Hz notch filter.
7. Mark Iz, TP8, and P6 as bad channels.
8. Apply spherical-spline interpolation.
9. Apply average reference.
10. Check for NaN and infinite values.

Note: The detailed implementation and parameter settings must be
consulted before changing any preprocessing operation.

## 4. Existing spectral-feature baseline

Step 32:
ml/eeg/32_temporal_fft_all_subjects.py

- Applies a real FFT along the temporal axis.
- Calculates magnitude as abs(FFT).
- Calculates raw power as magnitude squared.
- Saves the frequency vector, magnitude, and power.
- Uses disk-backed arrays and processes trials in chunks.

Step 33:
ml/eeg/33_extract_temporal_spectral_features_all_subjects.py

- Retains frequencies from 1 to 40 Hz.
- Extracts five band-power features per EEG channel.
- Delta: 1–4 Hz.
- Theta: 4–8 Hz.
- Alpha: 8–13 Hz.
- Beta: 13–30 Hz.
- Gamma: 30–40 Hz.
- Calculates mean spectral power within each band.
- Output shape: (720, 64, 5).
- Flattened feature count: 320 per trial.

## 5. Initial classifier findings

The existing baseline experiments indicate substantial overfitting.
Random Forest achieved approximately 20.06% mean test accuracy across
the 12 subjects in the previous classifier comparison.

This is an exploratory baseline, not a guaranteed performance target.
The existing test results have already been examined and must not be
used repeatedly to select new processing techniques.

## 6. Known limitations and questions

- Verify the preprocessing helper implementation before changing filters.
- Verify FFT scaling and windowing assumptions.
- Investigate possible spectral leakage.
- Compare alternative spectral feature representations.
- Investigate artifact and outlier sensitivity.
- Verify trial-label and channel-order alignment before interpreting
  physiological patterns.
- Record software versions and exact experiment configurations.
- Avoid fitting preprocessing or normalization parameters using
  validation or test data.
- Keep experimental outputs separate from the baseline.

## 7. Experiment register

| Experiment ID | Technique | Status |
|---|---|---|
| EEG-SP-01 | Hann window before FFT | Planned |
| EEG-SP-02 | Alternative band-power representation | Planned |
| EEG-SP-03 | Artifact and outlier handling | Planned |

## 8. Experiment reporting requirements

For every experiment, record:
- Hypothesis and scientific motivation.
- Dataset, subjects, and trial splits.
- Exact processing parameters.
- Classifier and fixed configuration.
- Validation accuracy, macro-F1, and balanced accuracy.
- Per-subject results and variability.
- Final test results only after the experiment is locked.
- Computational cost, limitations, and interpretation.

No experiment will be described as an improvement unless the
comparison supports that conclusion.
