import numpy as np
import mne



# ============================================================
# DS004514 — BIOSEMI-64 EEG CHANNELS
# ============================================================

EEG_CHANNELS = [
    f"A{i}" for i in range(1, 33)
] + [
    f"B{i}" for i in range(1, 33)
]


# ============================================================
# AUXILIARY CHANNELS
# ============================================================

AUXILIARY_CHANNELS = [
    "EXG1",
    "EXG2",
    "EXG3",
    "EXG4",
    "EXG5",
    "EXG6",
    "EXG7",
    "EXG8",
    "GSR1",
    "GSR2",
    "Erg1",
    "Erg2",
    "Resp",
    "Plet",
    "Temp",
]


# ============================================================
# DS004514 BIOSEMI → STANDARD ELECTRODE NAMES
# ============================================================

CHANNEL_MAPPING = {
    "A1": "Fp1",
    "A2": "AF7",
    "A3": "AF3",
    "A4": "F1",
    "A5": "F3",
    "A6": "F5",
    "A7": "F7",
    "A8": "FT7",
    "A9": "FC5",
    "A10": "FC3",
    "A11": "FC1",
    "A12": "C1",
    "A13": "C3",
    "A14": "C5",
    "A15": "T7",
    "A16": "TP7",
    "A17": "CP5",
    "A18": "CP3",
    "A19": "CP1",
    "A20": "P1",
    "A21": "P3",
    "A22": "P5",
    "A23": "P7",
    "A24": "P9",
    "A25": "PO7",
    "A26": "PO3",
    "A27": "O1",
    "A28": "Iz",
    "A29": "Oz",
    "A30": "POz",
    "A31": "Pz",
    "A32": "CPz",

    "B1": "Fpz",
    "B2": "Fp2",
    "B3": "AF8",
    "B4": "AF4",
    "B5": "AFz",
    "B6": "Fz",
    "B7": "F2",
    "B8": "F4",
    "B9": "F6",
    "B10": "F8",
    "B11": "FT8",
    "B12": "FC6",
    "B13": "FC4",
    "B14": "FC2",
    "B15": "FCz",
    "B16": "Cz",
    "B17": "C2",
    "B18": "C4",
    "B19": "C6",
    "B20": "T8",
    "B21": "TP8",
    "B22": "CP6",
    "B23": "CP4",
    "B24": "CP2",
    "B25": "P2",
    "B26": "P4",
    "B27": "P6",
    "B28": "P8",
    "B29": "P10",
    "B30": "PO8",
    "B31": "PO4",
    "B32": "O2",
}


# ============================================================
# FINAL PHASE 2 BAD CHANNELS
# ============================================================

FINAL_BAD_CHANNELS = [
    "Iz",
    "TP8",
    "P6",
]


# ============================================================
# PREPARE EEG CHANNELS
# ============================================================

def prepare_eeg_channels(raw):
    """
    Identify and type the 64 actual EEG channels
    in the DS004514 BioSemi recording.
    """

    if raw is None:
        raise ValueError("EEG input cannot be None.")

    prepared = raw.copy()

    # --------------------------------------------------------
    # 1. Find expected 64 BioSemi EEG channels
    # --------------------------------------------------------

    available_eeg = [
        ch
        for ch in EEG_CHANNELS
        if ch in prepared.ch_names
    ]

    if len(available_eeg) != 64:
        raise ValueError(
            f"Expected 64 EEG channels, "
            f"but found {len(available_eeg)}."
        )

    # --------------------------------------------------------
    # 2. Mark A1-A32 and B1-B32 as EEG
    # --------------------------------------------------------

    prepared.set_channel_types(
        {
            ch: "eeg"
            for ch in available_eeg
        }
    )

    # --------------------------------------------------------
    # 3. Mark auxiliary channels as miscellaneous
    # --------------------------------------------------------

    for ch in AUXILIARY_CHANNELS:
        if ch in prepared.ch_names:
            prepared.set_channel_types(
                {ch: "misc"}
            )

    # --------------------------------------------------------
    # 4. Mark Status as stimulus channel
    # --------------------------------------------------------

    if "Status" in prepared.ch_names:
        prepared.set_channel_types(
            {"Status": "stim"}
        )

    return prepared


# ============================================================
# APPLY DS004514 BIOSEMI MONTAGE
# ============================================================

def apply_eeg_montage(raw):
    """
    Rename DS004514 BioSemi channels to standard
    electrode names and apply the BioSemi-64 montage.
    """

    mounted = raw.copy()

    # --------------------------------------------------------
    # 1. Rename BioSemi channels
    # --------------------------------------------------------

    mounted.rename_channels(CHANNEL_MAPPING)

    # --------------------------------------------------------
    # 2. Apply BioSemi-64 spatial montage
    # --------------------------------------------------------

    montage = mne.channels.make_standard_montage(
        "biosemi64"
    )

    mounted.set_montage(
        montage,
        on_missing="raise"
    )

    return mounted


# ============================================================
# PRODUCTION EEG PURIFICATION
# ============================================================

def purify_eeg(raw):
    """
    Apply the finalized deterministic Phase 2 EEG
    preprocessing pipeline.

    Pipeline:

        80-channel raw recording
            ↓
        Identify 64 EEG channels
            ↓
        Type auxiliary channels
            ↓
        BioSemi → standard electrode names
            ↓
        BioSemi-64 montage
            ↓
        1–40 Hz band-pass
            ↓
        50 Hz notch
            ↓
        Mark Iz, TP8, P6 as bad
            ↓
        Spherical-spline interpolation
            ↓
        Average reference
            ↓
        Final 64-channel EEG
    """

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    if raw is None:
        raise ValueError(
            "EEG input cannot be None."
        )

    if not isinstance(
        raw,
        mne.io.BaseRaw
    ):
        raise TypeError(
            "EEG input must be an MNE Raw object."
        )

    # Never modify original Raw object
    purified = raw.copy()

    # ========================================================
    # 1. IDENTIFY AND TYPE EEG CHANNELS
    # ========================================================

    purified = prepare_eeg_channels(
        purified
    )

    # ========================================================
    # 2. RENAME + MONTAGE
    # ========================================================

    purified = apply_eeg_montage(
        purified
    )

    # ========================================================
    # 3. SELECT EEG CHANNELS
    # ========================================================

    eeg_picks = mne.pick_types(
        purified.info,
        eeg=True,
        exclude=[]
    )

    if len(eeg_picks) != 64:
        raise ValueError(
            f"Expected 64 EEG channels after preparation, "
            f"but found {len(eeg_picks)}."
        )

    # ========================================================
    # 4. BAND-PASS FILTER: 1–40 Hz
    # ========================================================

    purified.filter(
        l_freq=1.0,
        h_freq=40.0,
        picks=eeg_picks,
        fir_design="firwin"
    )

    # ========================================================
    # 5. 50 Hz POWER-LINE NOTCH
    # ========================================================

    purified.notch_filter(
        freqs=[50.0],
        picks=eeg_picks
    )

    # ========================================================
    # 6. FINAL BAD CHANNEL CONFIGURATION
    # ========================================================

    purified.info["bads"] = FINAL_BAD_CHANNELS.copy()

    # ========================================================
    # 7. SPHERICAL-SPLINE INTERPOLATION
    # ========================================================

    purified.interpolate_bads(
        reset_bads=False,
        method=dict(
            eeg="spline"
        )
    )

    # ========================================================
    # 8. AVERAGE REFERENCE
    # ========================================================

    purified.set_eeg_reference(
        "average",
        projection=False
    )

    # ========================================================
    # 9. FINAL SAFETY CHECK
    # ========================================================

    final_data = purified.get_data(
        picks=eeg_picks
    )

    if np.isnan(final_data).any():
        raise ValueError(
            "Final EEG contains NaN values."
        )

    if np.isinf(final_data).any():
        raise ValueError(
            "Final EEG contains infinite values."
        )

    return purified