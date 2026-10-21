
import time
import warnings
import runpy
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning

# ------------------------------------------------------------
# 1. PATHS AND PAGE CONFIGURATION
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "experiments" / "ds004022_verified_epochs" / "eeg_epochs.npz"
MODEL_PATH = ROOT / "experiments" / "ds004022_model_results" / "best_model_trained_on_all_subjects.joblib"
TRAIN_SCRIPT = ROOT / "ml" / "eeg" / "98_train_ds004022_models.py"

st.set_page_config(
    page_title="NeuroSymphony | EEG Replay",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 NeuroSymphony — EEG Replay Dashboard")
st.caption(
    "Research demonstration | Prerecorded EEG replay | "
    "Not live hardware acquisition"
)

# ------------------------------------------------------------
# 2. VERIFY REQUIRED FILES
# ------------------------------------------------------------
missing_files = [
    str(path)
    for path in (DATA_PATH, MODEL_PATH, TRAIN_SCRIPT)
    if not path.is_file()
]

if missing_files:
    st.error("Required project files are missing:")
    for path in missing_files:
        st.write(f"- `{path}`")
    st.stop()


# ------------------------------------------------------------
# 3. LOAD THE EXISTING DATA, MODEL, AND FEATURE EXTRACTOR
# ------------------------------------------------------------
@st.cache_resource
def load_resources():
    # runpy provides __file__ and other normal script globals.
    # run_name is deliberately not "__main__", so the training
    # script's main() function is not executed.
    namespace = runpy.run_path(
        str(TRAIN_SCRIPT),
        run_name="neurosymphony_feature_loader",
    )

    if "extract_features" not in namespace:
        raise RuntimeError(
            "Could not find extract_features() in the training script."
        )

    # Keep compatibility warnings visible rather than suppressing them.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        artifact = joblib.load(MODEL_PATH)

    data = np.load(DATA_PATH, allow_pickle=True)

    try:
        X = np.asarray(data["X"])
        y = np.asarray(data["y"])
        subjects = np.asarray(data["subjects"]).astype(str)
        class_names = np.asarray(data["class_names"]).astype(str)
        channel_names = np.asarray(data["channel_names"]).astype(str)
        sampling_rate = float(data["sampling_rate"].item())
    finally:
        data.close()

    required_keys = ("model", "feature_names")
    for key in required_keys:
        if key not in artifact:
            raise ValueError(f"Model artifact is missing key: {key}")

    if X.ndim != 3:
        raise ValueError(f"Expected 3D EEG data; received {X.shape}.")

    if not (len(X) == len(y) == len(subjects)):
        raise ValueError("EEG trials, labels, and subjects do not match.")

    if X.shape[1] != len(channel_names):
        raise ValueError("EEG channel count does not match channel names.")

    if len(class_names) != len(np.unique(y)):
        raise ValueError("Class names do not match the stored label classes.")

    if not np.isfinite(X).all():
        raise ValueError("EEG data contains NaN or infinite values.")

    if sampling_rate != float(artifact.get("sampling_rate", 500.0)):
        raise ValueError("Dataset and model sampling rates do not match.")

    if X.shape[1] != int(artifact.get("n_channels", X.shape[1])):
        raise ValueError("Dataset and model channel counts do not match.")

    if len(artifact["feature_names"]) != X.shape[1] * 16:
        raise ValueError("Expected feature count does not match model metadata.")

    return (
        X,
        y,
        subjects,
        class_names,
        channel_names,
        sampling_rate,
        artifact,
        namespace["extract_features"],
        [str(item.message) for item in caught],
    )


try:
    (
        X,
        y,
        subjects,
        class_names,
        channel_names,
        fs,
        artifact,
        extract_features,
        version_warnings,
    ) = load_resources()
except Exception as exc:
    st.error(f"Could not load or validate the replay pipeline: {exc}")
    st.exception(exc)
    st.stop()

model = artifact["model"]

if version_warnings:
    st.warning(
        "The saved model was serialized with a different scikit-learn "
        "version. It loaded, but compatibility has not been fully verified. "
        "Do not treat its predictions as validated until this is resolved."
    )
    with st.expander("Compatibility warning details"):
        for warning_text in version_warnings:
            st.code(warning_text)

# ------------------------------------------------------------
# 4. DATASET OVERVIEW
# ------------------------------------------------------------
metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("EEG trials", f"{len(X):,}")
metric2.metric("EEG channels", str(X.shape[1]))
metric3.metric("Sampling rate", f"{fs:.0f} Hz")
metric4.metric("Action classes", str(len(class_names)))

st.info(
    f"Dataset: ds004022 | Subjects: {len(np.unique(subjects))} | "
    f"Window: {X.shape[2] / fs:.1f} seconds | "
    f"Model: {artifact.get('model_name', 'Saved classifier')}"
)

# ------------------------------------------------------------
# 5. SESSION STATE
# ------------------------------------------------------------
defaults = {
    "replay_position": 0,
    "replay_history": [],
    "replay_playing": False,
    "replay_last_result": None,
    "replay_subject": "All subjects",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ------------------------------------------------------------
# 6. SIDEBAR CONTROLS
# ------------------------------------------------------------
with st.sidebar:
    st.header("🎛️ Replay controls")

    subject_options = ["All subjects"] + sorted(np.unique(subjects).tolist())

    previous_subject = st.session_state["replay_subject"]
    if previous_subject not in subject_options:
        previous_subject = "All subjects"

    selected_subject = st.selectbox(
        "Select EEG subject",
        subject_options,
        index=subject_options.index(previous_subject),
    )

    if selected_subject != st.session_state["replay_subject"]:
        st.session_state["replay_subject"] = selected_subject
        st.session_state["replay_position"] = 0
        st.session_state["replay_history"] = []
        st.session_state["replay_last_result"] = None
        st.session_state["replay_playing"] = False

    if selected_subject == "All subjects":
        trial_indices = np.arange(len(X))
    else:
        trial_indices = np.flatnonzero(subjects == selected_subject)

    start_col, pause_col = st.columns(2)

    with start_col:
        if st.button(
            "▶ Start",
            type="primary",
            use_container_width=True,
        ):
            st.session_state["replay_playing"] = True

    with pause_col:
        if st.button("⏸ Pause", use_container_width=True):
            st.session_state["replay_playing"] = False

    if st.button("↻ Reset", use_container_width=True):
        st.session_state["replay_position"] = 0
        st.session_state["replay_history"] = []
        st.session_state["replay_last_result"] = None
        st.session_state["replay_playing"] = False

    st.caption(
        "Playback processes successive prerecorded five-second trials. "
        "It does not acquire EEG from a physical device."
    )

if len(trial_indices) == 0:
    st.warning("No trials are available for this subject.")
    st.stop()

# ------------------------------------------------------------
# 7. REPLAY AND PREDICTION DASHBOARD
# ------------------------------------------------------------
@st.fragment(run_every="1s")
def replay_panel():
    indices = trial_indices
    if len(indices) == 0:
        st.warning("No trials available.")
        return

    position = st.session_state["replay_position"] % len(indices)
    trial_index = int(indices[position])

    # One inference is performed per fragment update while playing.
    if st.session_state["replay_playing"]:
        try:
            started_at = time.perf_counter()

            features, _ = extract_features(
                X[trial_index:trial_index + 1],
                fs,
            )

            if features.shape != (1, len(artifact["feature_names"])):
                raise ValueError(
                    f"Feature shape {features.shape} does not match "
                    f"the model's expected feature count "
                    f"({len(artifact['feature_names'])})."
                )

            if not np.isfinite(features).all():
                raise ValueError("Extracted features contain invalid values.")

            prediction = model.predict(features)[0]

            if not hasattr(model, "predict_proba"):
                raise ValueError(
                    "This saved classifier does not support predict_proba()."
                )

            probabilities = model.predict_proba(features)[0]
            latency_ms = (time.perf_counter() - started_at) * 1000

            # Model labels are 0, 1, 2, 3 and class_names follows that order.
            predicted_index = int(prediction)
            if not 0 <= predicted_index < len(class_names):
                raise ValueError(
                    f"Unexpected predicted class: {prediction}"
                )

            result = {
                "trial": trial_index,
                "subject": str(subjects[trial_index]),
                "prediction": str(class_names[predicted_index]),
                "probabilities": probabilities.tolist(),
                "actual": str(class_names[int(y[trial_index])]),
                "latency_ms": float(latency_ms),
            }

            st.session_state["replay_last_result"] = result
            st.session_state["replay_history"].append(result)
            st.session_state["replay_history"] = (
                st.session_state["replay_history"][-30:]
            )
            st.session_state["replay_position"] = (
                st.session_state["replay_position"] + 1
            )

        except Exception as exc:
            st.session_state["replay_playing"] = False
            st.error(f"Inference failed: {exc}")
            st.exception(exc)

    result = st.session_state["replay_last_result"]

    signal_trial_index = trial_index
    if result is not None and not st.session_state["replay_playing"]:
        signal_trial_index = result["trial"]

    # --------------------------------------------------------
    # SIGNAL VIEW
    # --------------------------------------------------------
    left, right = st.columns([1.7, 1])

    with left:
        st.subheader("📈 Recorded EEG signal")

        channel_index = st.selectbox(
            "EEG channel",
            options=list(range(X.shape[1])),
            format_func=lambda i: str(channel_names[i]),
            key="replay_channel",
        )

        signal = X[signal_trial_index, channel_index]
        seconds = np.arange(len(signal)) / fs

        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(seconds, signal, linewidth=0.8)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Recorded amplitude")
        ax.set_title(
            f"Trial {signal_trial_index + 1} — "
            f"Channel {channel_names[channel_index]}"
        )
        ax.grid(alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            f"Subject: {subjects[signal_trial_index]} | "
            f"Trial duration: {len(signal) / fs:.1f} seconds"
        )

    # --------------------------------------------------------
    # PREDICTION VIEW
    # --------------------------------------------------------
    with right:
        st.subheader("🧠 Latest prediction")

        if result is None:
            st.write("Click Start to begin replay and generate predictions.")
        else:
            st.metric("Predicted action", result["prediction"])
            st.metric(
                "Inference latency",
                f"{result['latency_ms']:.2f} ms",
            )

            st.caption(
                f"Trial {result['trial'] + 1} | "
                f"Subject {result['subject']}"
            )

            st.write("Model output scores")

            for name, probability in zip(
                class_names,
                result["probabilities"],
            ):
                st.progress(
                    float(np.clip(probability, 0.0, 1.0)),
                    text=f"{name}: {probability:.1%}",
                )

            st.caption(
                f"Stored reference label: {result['actual']}. "
                "The all-subjects model was trained on this dataset; "
                "this comparison is not independent test accuracy."
            )

    # --------------------------------------------------------
    # RECENT PREDICTIONS
    # --------------------------------------------------------
    st.subheader("📋 Recent prediction history")

    history = st.session_state["replay_history"]

    if history:
        rows = [
            {
                "Trial": item["trial"] + 1,
                "Subject": item["subject"],
                "Predicted action": item["prediction"],
                "Stored reference": item["actual"],
                "Latency (ms)": round(item["latency_ms"], 2),
            }
            for item in reversed(history)
        ]

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("Predictions will appear here after replay starts.")

    if st.session_state["replay_playing"]:
        st.caption("▶ Replay running — advancing through saved EEG trials.")
    else:
        st.caption("⏸ Replay paused.")


replay_panel()
