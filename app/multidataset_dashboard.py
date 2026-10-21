from __future__ import annotations

import time
import warnings
import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
DATASETS = ROOT / "major datasets"

st.set_page_config(
    page_title="NeuroSymphony | Multi-Dataset EEG",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 NeuroSymphony — Multi-Dataset EEG Dashboard")
st.caption("Research demonstration | Prerecorded EEG | No live hardware acquisition")

DATASET_OPTIONS = {
    "ds004022 — Motor Actions": "ds004022",
    "28955456 — Motor Imagery": "28955456",
    "ds004514_real — Sensory Imagery": "ds004514_real",
}

TASK_LABELS_514 = {
    "animal_visual_imagery_task": "Visual imagery",
    "animal_auditory_imagery_task": "Auditory imagery",
    "animal_tactile_imagery_task": "Tactile imagery",
    "animal_silent_naming_task": "Silent naming",
}


# ============================================================
# DATASET 1: ds004022
# ============================================================

@st.cache_data(show_spinner=False)
def load_ds004022():
    path = EXPERIMENTS / "ds004022_verified_epochs" / "eeg_epochs.npz"

    if not path.is_file():
        raise FileNotFoundError(f"Missing EEG data: {path}")

    with np.load(path, allow_pickle=True) as data:
        return {
            "X": np.asarray(data["X"]),
            "y": np.asarray(data["y"]),
            "subjects": np.asarray(data["subjects"]).astype(str),
            "class_names": np.asarray(data["class_names"]).astype(str),
            "channel_names": np.asarray(data["channel_names"]).astype(str),
            "fs": float(np.asarray(data["sampling_rate"]).item()),
        }


@st.cache_resource(show_spinner=False)
def load_ds004022_model():
    model_path = (
        EXPERIMENTS
        / "ds004022_model_results"
        / "best_model_trained_on_all_subjects.joblib"
    )
    train_script = ROOT / "ml" / "eeg" / "98_train_ds004022_models.py"

    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model: {model_path}")

    if not train_script.is_file():
        raise FileNotFoundError(f"Missing feature extractor: {train_script}")

    import joblib
    from sklearn.exceptions import InconsistentVersionWarning

    namespace = runpy.run_path(
        str(train_script),
        run_name="neurosymphony_feature_loader",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        artifact = joblib.load(model_path)

    return (
        artifact,
        namespace["extract_features"],
        [str(item.message) for item in caught],
    )


# ============================================================
# DATASET 2: 28955456
# ============================================================

@st.cache_data(show_spinner=False)
def load_ds28955456_predictions():
    path = (
        EXPERIMENTS
        / "dataset3_28955456_training"
        / "experiment134_corrected_features"
        / "out_of_fold_predictions.csv"
    )

    if not path.is_file():
        raise FileNotFoundError(f"Missing saved predictions: {path}")

    df = pd.read_csv(path)

    required = {
        "feature_family",
        "model",
        "participant",
        "group",
        "source_file",
        "trial_number",
        "true_label",
        "predicted_label",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    selected = df[
        (df["feature_family"].astype(str).str.lower() == "combined")
        & (df["model"].astype(str).str.upper() == "RBF_SVM")
    ].copy()

    if selected.empty:
        pairs = (
            df.groupby(["feature_family", "model"])
            .size()
            .sort_values(ascending=False)
        )

        if pairs.empty:
            raise ValueError("No prediction experiment rows were found.")

        feature_family, model_name = pairs.index[0]

        selected = df[
            (df["feature_family"] == feature_family)
            & (df["model"] == model_name)
        ].copy()

    selected["trial_number"] = pd.to_numeric(
        selected["trial_number"],
        errors="coerce",
    )
    selected = selected.dropna(subset=["trial_number"]).copy()
    selected["trial_number"] = selected["trial_number"].astype(int)

    return selected.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_mat_trial(mat_path_string, trial_number):
    from scipy.io import loadmat

    path = Path(mat_path_string)

    if not path.is_file():
        raise FileNotFoundError(f"Original MAT file not found: {path}")

    mat = loadmat(
        path,
        struct_as_record=False,
        squeeze_me=False,
    )

    if "epo" not in mat:
        raise ValueError(f"MAT file has no 'epo' structure: {path}")

    epo = mat["epo"][0, 0]
    X = np.asarray(epo.X, dtype=np.float32)
    y = np.asarray(epo.y).reshape(-1).astype(int)
    fs = float(np.asarray(epo.fs).reshape(-1)[0])

    # Original format: samples x channels x trials.
    if X.ndim != 3 or X.shape[2] != len(y):
        raise ValueError(f"Unexpected EEG shape: {X.shape}")

    if not 0 <= trial_number < X.shape[2]:
        raise IndexError(f"Trial index {trial_number} is outside the MAT file.")

    signal = X[:, :, trial_number].T
    return signal, fs, int(y[trial_number])


# ============================================================
# DATASET 3: ds004514_real
# ============================================================

@st.cache_data(show_spinner=False)
def list_514_subjects():
    root = DATASETS / "ds004514_real"

    return sorted(
        path.name
        for path in root.glob("sub-*")
        if path.is_dir()
        and (
            path / "eeg" / f"{path.name}_task-eeg_eeg.bdf"
        ).is_file()
    )


@st.cache_data(show_spinner=False)
def load_514_events(subject):
    path = (
        DATASETS
        / "ds004514_real"
        / subject
        / "eeg"
        / f"{subject}_task-eeg_events.tsv"
    )

    if not path.is_file():
        raise FileNotFoundError(f"Missing events file: {path}")

    events = pd.read_csv(path, sep="\t")

    if not {"onset", "trial_type"}.issubset(events.columns):
        raise ValueError("Event TSV requires onset and trial_type.")

    events = events[
        events["trial_type"].isin(TASK_LABELS_514)
    ].copy()

    events["label"] = events["trial_type"].map(TASK_LABELS_514)
    return events.sort_values("onset").reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def open_514_raw(subject):
    import mne

    path = (
        DATASETS
        / "ds004514_real"
        / subject
        / "eeg"
        / f"{subject}_task-eeg_eeg.bdf"
    )

    if not path.is_file():
        raise FileNotFoundError(f"Missing BDF recording: {path}")

    # IMPORTANT: do not load the complete recording into RAM.
    return mne.io.read_raw_bdf(
        path,
        preload=False,
        verbose="ERROR",
    )


# ============================================================
# COMMON VISUALIZATION
# ============================================================

def plot_signal(signal, fs, channel_name, title):
    signal = np.asarray(signal).squeeze()

    if signal.ndim != 1 or len(signal) == 0:
        st.warning("No valid signal is available to display.")
        return

    # Downsample for plotting only.
    stride = max(1, int(np.ceil(len(signal) / 5000)))
    plotted_signal = signal[::stride]
    seconds = np.arange(len(signal))[::stride] / fs

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(seconds, plotted_signal, linewidth=0.8)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Recorded amplitude")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "multi_ds022_index": 0,
    "multi_ds289_index": 0,
    "multi_ds514_index": 0,
    "replay_running": False,
    "multi_replay_speed": 1,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def get_index_key(dataset_name):
    return {
        "ds004022": "multi_ds022_index",
        "28955456": "multi_ds289_index",
        "ds004514_real": "multi_ds514_index",
    }[dataset_name]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Dataset controls")

    selected_label = st.selectbox(
        "Select EEG dataset",
        list(DATASET_OPTIONS.keys()),
        key="multi_selected_dataset",
    )

    dataset = DATASET_OPTIONS[selected_label]
    index_key = get_index_key(dataset)

    if st.button("Reset selected dataset", use_container_width=True):
        st.session_state[index_key] = 0
        st.session_state.replay_running = False
        st.rerun()

    st.divider()
    st.header("🎛️ Replay controls")

    st.selectbox(
        "Playback speed",
        options=[1, 2, 3, 5],
        format_func=lambda value: (
            f"{value} trial/event(s) per second"
            if value > 1
            else "1 trial/event per second"
        ),
        key="multi_replay_speed",
    )

    start_col, pause_col = st.columns(2)

    with start_col:
        if st.button(
            "▶ Start",
            key="multi_replay_start",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.replay_running = True
            st.rerun()

    with pause_col:
        if st.button(
            "⏸ Pause",
            key="multi_replay_pause",
            use_container_width=True,
        ):
            st.session_state.replay_running = False
            st.rerun()

    if st.button(
        "↻ Reset",
        key="multi_replay_reset",
        use_container_width=True,
    ):
        st.session_state.replay_running = False
        st.session_state[index_key] = 0
        st.rerun()

    st.caption(
        "Playback uses prerecorded EEG. It does not acquire signals "
        "from physical hardware."
    )

    if st.session_state.replay_running:
        st.success("▶ Playing")
    else:
        st.info("⏸ Paused")


st.divider()


# ============================================================
# REPLAY DISPLAY
# ============================================================

replay_interval = 1.0 / max(
    1,
    int(st.session_state.multi_replay_speed),
)


@st.fragment(
    run_every=replay_interval
    if st.session_state.replay_running
    else None
)
def render_replay():
    if st.session_state.replay_running:
        active_index_key = get_index_key(dataset)
        st.session_state[active_index_key] += 1

    # --------------------------------------------------------
    # DATASET 1: ds004022
    # --------------------------------------------------------

    if dataset == "ds004022":
        try:
            data = load_ds004022()
        except Exception as exc:
            st.error(f"Could not load ds004022: {exc}")
            return

        X = data["X"]
        y = data["y"]
        subjects = data["subjects"]
        class_names = data["class_names"]
        channels = data["channel_names"]
        fs = data["fs"]

        st.subheader("Motor action EEG replay")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("EEG trials", f"{len(X):,}")
        m2.metric("EEG channels", str(X.shape[1]))
        m3.metric("Sampling rate", f"{fs:.0f} Hz")
        m4.metric("Action classes", str(len(class_names)))

        st.info(
            f"Dataset: ds004022 | "
            f"Subjects: {len(np.unique(subjects))} | "
            f"Trial duration: {X.shape[2] / fs:.1f} seconds"
        )

        subject_options = ["All subjects"] + sorted(
            np.unique(subjects).tolist()
        )

        chosen_subject = st.selectbox(
            "Filter by subject",
            subject_options,
            key="multi_022_subject",
        )

        if chosen_subject == "All subjects":
            valid_indices = np.arange(len(X))
        else:
            valid_indices = np.flatnonzero(subjects == chosen_subject)

        if len(valid_indices) == 0:
            st.warning("No trials are available for this subject.")
            return

        position = st.session_state.multi_ds022_index % len(valid_indices)
        trial_idx = int(valid_indices[position])

        channel_name = st.selectbox(
            "EEG channel",
            list(channels),
            key="multi_022_channel",
        )
        channel_idx = list(channels).index(channel_name)

        left, right = st.columns([1.7, 1])

        with left:
            plot_signal(
                X[trial_idx, channel_idx],
                fs,
                channel_name,
                f"Recorded EEG | Trial {trial_idx + 1} | {channel_name}",
            )

        with right:
            st.subheader("Trial details")
            st.write(f"**Subject:** {subjects[trial_idx]}")
            st.write(
                f"**Reference label:** {class_names[int(y[trial_idx])]}"
            )

            try:
                artifact, extract_features, version_warnings = (
                    load_ds004022_model()
                )

                if version_warnings:
                    st.warning(
                        "Saved model scikit-learn version differs from "
                        "the installed version. Compatibility is unverified."
                    )

                started = time.perf_counter()
                features, _ = extract_features(
                    X[trial_idx:trial_idx + 1],
                    fs,
                )

                prediction = artifact["model"].predict(features)[0]
                latency_ms = (time.perf_counter() - started) * 1000
                prediction_idx = int(prediction)

                if not 0 <= prediction_idx < len(class_names):
                    raise ValueError(
                        f"Unexpected model prediction: {prediction}"
                    )

                st.metric(
                    "Predicted action",
                    str(class_names[prediction_idx]),
                )
                st.metric("Inference latency", f"{latency_ms:.2f} ms")

                model = artifact["model"]

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(features)[0]
                    st.write("Model output scores")

                    for name, probability in zip(
                        class_names,
                        probabilities,
                    ):
                        st.progress(
                            float(np.clip(probability, 0, 1)),
                            text=f"{name}: {probability:.1%}",
                        )

            except Exception as exc:
                st.warning(f"Model inference unavailable: {exc}")

        st.caption(
            "The saved model was trained on this dataset. "
            "This display is not an independent test-accuracy measurement."
        )

        if st.button("Next EEG trial", key="next_ds022"):
            st.session_state.multi_ds022_index += 1
            st.rerun()

    # --------------------------------------------------------
    # DATASET 2: 28955456
    # --------------------------------------------------------

    elif dataset == "28955456":
        try:
            predictions = load_ds28955456_predictions()
        except Exception as exc:
            st.error(f"Could not load historical predictions: {exc}")
            return

        if predictions.empty:
            st.warning("No saved prediction rows are available.")
            return

        row_idx = st.session_state.multi_ds289_index % len(predictions)
        row = predictions.iloc[row_idx]

        st.subheader("Left/right motor imagery")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Saved prediction rows", f"{len(predictions):,}")
        m2.metric("EEG channels", "32")
        m3.metric("Sampling rate", "256 Hz")
        m4.metric("Imagery classes", "2")

        st.info(
            "Historical out-of-fold predictions | "
            f"Experiment: {row['feature_family']} / {row['model']}"
        )

        st.warning(
            "This section replays saved evaluation predictions. "
            "It does not run a newly trained model. "
            "Normals/Patients identifies the participant group, "
            "not the target class."
        )

        try:
            signal, fs, mat_label = load_mat_trial(
                str(row["source_file"]),
                int(row["trial_number"]),
            )
        except Exception as exc:
            st.error(f"Could not load the original EEG trial: {exc}")
            st.dataframe(
                predictions.head(50),
                use_container_width=True,
                hide_index=True,
            )
            return

        channel_names = [
            "FC1", "AF3", "AF4", "CP1", "CP2", "CP6", "Cz", "C3",
            "C4", "T7", "T8", "FC2", "FC5", "FC6", "Pz", "CP5",
            "PO3", "PO4", "Oz", "FP2", "FP1", "FZ", "F3", "F4",
            "F7", "F8", "P3", "P4", "P7", "P8", "O1", "O2",
        ][:signal.shape[0]]

        channel_name = st.selectbox(
            "EEG channel",
            channel_names,
            key="multi_289_channel",
        )
        channel_idx = channel_names.index(channel_name)

        left, right = st.columns([1.7, 1])

        with left:
            plot_signal(
                signal[channel_idx],
                fs,
                channel_name,
                (
                    f"Original EEG | {row['participant']} | "
                    f"Trial {int(row['trial_number']) + 1}"
                ),
            )

        with right:
            label_names = {
                0: "Left motor imagery",
                1: "Right motor imagery",
            }

            true_label = int(row["true_label"])
            predicted_label = int(row["predicted_label"])

            st.subheader("Historical prediction")
            st.write(f"**Participant:** {row['participant']}")
            st.write(f"**Cohort:** {row['group']}")

            st.metric(
                "Saved predicted label",
                label_names.get(predicted_label, str(predicted_label)),
            )
            st.metric(
                "Reference label",
                label_names.get(true_label, str(true_label)),
            )
            st.metric(
                "Saved prediction match",
                "Correct" if true_label == predicted_label else "Incorrect",
            )
            st.caption("Probability scores were not stored in this CSV.")

        st.subheader("Saved prediction history")

        st.dataframe(
            predictions[
                [
                    "participant",
                    "group",
                    "trial_number",
                    "true_label",
                    "predicted_label",
                    "feature_family",
                    "model",
                ]
            ].head(100),
            use_container_width=True,
            hide_index=True,
        )

        if st.button("Next saved trial", key="next_ds289"):
            st.session_state.multi_ds289_index += 1
            st.rerun()

    # --------------------------------------------------------
    # DATASET 3: ds004514_real
    # --------------------------------------------------------

    else:
        subjects = list_514_subjects()

        if not subjects:
            st.error(
                "No ds004514_real BDF recordings were found. "
                "Check the major datasets folder."
            )
            return

        st.subheader("Sensory imagery and silent naming")

        chosen_subject = st.selectbox(
            "Select EEG subject",
            subjects,
            key="multi_514_subject",
        )

        try:
            events = load_514_events(chosen_subject)
            raw = open_514_raw(chosen_subject)
        except Exception as exc:
            st.error(f"Could not load recording or events: {exc}")
            return

        if events.empty:
            st.warning(
                "No recognized imagery/naming task events were found "
                "in this subject's event file."
            )
            return

        position = st.session_state.multi_ds514_index % len(events)
        event = events.iloc[position]

        fs = float(raw.info["sfreq"])
        start_seconds = max(0.0, float(event["onset"]))
        segment_seconds = 3.0

        start_sample = int(round(start_seconds * fs))
        stop_sample = min(
            raw.n_times,
            start_sample + int(round(segment_seconds * fs)),
        )

        if start_sample >= raw.n_times or stop_sample <= start_sample:
            st.error("The selected event is outside the recording.")
            return

        # Select EEG channels without preloading the full recording.
        eeg_raw = raw.copy().pick("eeg")
        channel_names = eeg_raw.ch_names

        # Only the selected 3-second segment is read from disk.
        segment = eeg_raw.get_data(
            start=start_sample,
            stop=stop_sample,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("EEG subjects", str(len(subjects)))
        m2.metric("EEG channels", str(segment.shape[0]))
        m3.metric("Sampling rate", f"{fs:.0f} Hz")
        m4.metric("Recognized task events", f"{len(events):,}")

        task_label = TASK_LABELS_514[event["trial_type"]]

        st.info(
            f"Original BDF | {chosen_subject} | {task_label} | "
            f"Event onset: {start_seconds:.3f} seconds"
        )

        st.warning(
            "A compatible trained classifier has not been verified for "
            "this dataset. This panel displays event-aligned EEG, "
            "not a model prediction."
        )

        channel_name = st.selectbox(
            "EEG channel",
            channel_names,
            key="multi_514_channel",
        )
        channel_idx = channel_names.index(channel_name)

        left, right = st.columns([1.7, 1])

        with left:
            plot_signal(
                segment[channel_idx],
                fs,
                channel_name,
                f"{task_label} | {chosen_subject} | {channel_name}",
            )

        with right:
            st.subheader("Task event details")
            st.metric("Task event", task_label)

            st.write(
                f"**Event code:** {event.get('value', 'not provided')}"
            )
            st.write(
                f"**Event onset:** {start_seconds:.3f} seconds"
            )
            st.write(
                f"**Displayed EEG segment:** "
                f"{len(segment[0]) / fs:.2f} seconds"
            )
            st.write("**Model prediction:** Not available")

            st.caption(
                "The label comes from trial_type in the original event file. "
                "This segment begins at event onset and has not been validated "
                "as a task-specific model epoch."
            )

        st.subheader("Task-event sequence")

        st.dataframe(
            events[["onset", "trial_type", "label"]].head(100),
            use_container_width=True,
            hide_index=True,
        )

        if st.button("Next task event", key="next_ds514"):
            st.session_state.multi_ds514_index += 1
            st.rerun()

    st.divider()
    st.caption(
        "NeuroSymphony research dashboard. Saved-model inference, historical "
        "evaluation predictions, and event-aligned EEG replay are distinct modes. "
        "Not intended for clinical diagnosis."
    )


render_replay()