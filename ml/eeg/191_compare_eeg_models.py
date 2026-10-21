from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("default")

DATA_DIR = Path("data/processed/visual_motivation_125hz_normalized")
MANIFEST_PATH = DATA_DIR / "dataset_manifest.csv"
OUTPUT_DIR = Path("results/eeg/visual_motivation/step191_model_comparison")
MODEL_DIR = OUTPUT_DIR / "models"

CLASSES = ["FF", "FR", "RF", "RR"]
RANDOM_STATE = 42
PCA_COMPONENTS = 100

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(MANIFEST_PATH)
if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
    raise FileExistsError(
        f"Output directory already contains files: {OUTPUT_DIR}. "
        "Nothing was overwritten."
    )

manifest = pd.read_csv(MANIFEST_PATH)

def load_split(split):
    rows = manifest[manifest["split"] == split]
    X_parts, y_parts = [], []

    for row in rows.itertuples(index=False):
        arr = np.load(row.output_file, allow_pickle=False)
        if arr.ndim != 3 or arr.shape[1:] != (31, 1625):
            raise ValueError(f"Unexpected shape: {row.output_file}")
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite values: {row.output_file}")

        # Flatten each epoch to 50,375 features.
        X_parts.append(arr.reshape(arr.shape[0], -1))
        y_parts.extend([row.label] * arr.shape[0])

    if not X_parts:
        raise ValueError(f"No data found for split: {split}")

    return np.concatenate(X_parts).astype(np.float32), np.asarray(y_parts)

print("Loading subject-wise splits...")
X_train, y_train = load_split("train")
X_val, y_val = load_split("validation")

print("Train:", X_train.shape, pd.Series(y_train).value_counts().to_dict())
print("Validation:", X_val.shape, pd.Series(y_val).value_counts().to_dict())
print("Test data will not be loaded or evaluated in this step.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# PCA is fitted only on training subjects.
# StandardScaler(with_mean=False) avoids a second large dense-centering copy.
print(f"\nFitting PCA with {PCA_COMPONENTS} components on training data...")
pca_start = time.perf_counter()
pca = PCA(
    n_components=PCA_COMPONENTS,
    svd_solver="randomized",
    random_state=RANDOM_STATE,
)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
pca_seconds = time.perf_counter() - pca_start

if not np.isfinite(X_train_pca).all() or not np.isfinite(X_val_pca).all():
    raise ValueError("PCA output contains non-finite values.")

joblib.dump(pca, MODEL_DIR / "training_fitted_pca.joblib")

print("PCA train shape:", X_train_pca.shape)
print("PCA validation shape:", X_val_pca.shape)
print("PCA fit + transform seconds:", round(pca_seconds, 2))
print(
    "Explained variance ratio (100 components):",
    round(float(pca.explained_variance_ratio_.sum()), 6),
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "Linear SVM": SVC(
        kernel="linear",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "RBF SVM": SVC(
        kernel="rbf",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        n_jobs=-1,
    ),
    "LDA": LinearDiscriminantAnalysis(
        solver="lsqr",
        shrinkage="auto",
    ),
}

results = []
reports = {}
confusions = {}

for name, model in models.items():
    print(f"\n{'=' * 65}\nTraining: {name}\n{'=' * 65}")

    start = time.perf_counter()
    model.fit(X_train_pca, y_train)
    training_seconds = time.perf_counter() - start

    start = time.perf_counter()
    predictions = model.predict(X_val_pca)
    prediction_seconds = time.perf_counter() - start

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val, predictions, labels=CLASSES,
        average="macro", zero_division=0
    )
    balanced = balanced_accuracy_score(y_val, predictions)

    results.append({
        "model": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced,
        "training_seconds": training_seconds,
        "validation_prediction_seconds": prediction_seconds,
        "pca_components": PCA_COMPONENTS,
    })

    reports[name] = classification_report(
        y_val, predictions, labels=CLASSES,
        output_dict=True, zero_division=0
    )
    confusions[name] = confusion_matrix(
        y_val, predictions, labels=CLASSES
    ).tolist()

    model_path = MODEL_DIR / (
        name.lower().replace(" ", "_") + ".joblib"
    )
    joblib.dump(model, model_path)

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")
    print(f"Balanced accuracy: {balanced:.4f}")
    print(f"Training seconds:  {training_seconds:.2f}")
    print(f"Prediction secs:   {prediction_seconds:.2f}")
    print("Per-class F1:")
    for label in CLASSES:
        print(
            f"  {label}: "
            f"{reports[name][label]['f1-score']:.4f}"
        )

leaderboard = pd.DataFrame(results).sort_values(
    ["macro_f1", "balanced_accuracy"],
    ascending=False,
)
leaderboard.to_csv(OUTPUT_DIR / "validation_leaderboard.csv", index=False)

(OUTPUT_DIR / "classification_reports.json").write_text(
    json.dumps(reports, indent=2), encoding="utf-8"
)
(OUTPUT_DIR / "confusion_matrices.json").write_text(
    json.dumps(
        {
            "class_order": CLASSES,
            "validation_confusion_matrices": confusions,
        },
        indent=2,
    ),
    encoding="utf-8",
)

summary = {
    "step": 191,
    "models_trained": list(models.keys()),
    "training_epochs": int(len(y_train)),
    "validation_epochs": int(len(y_val)),
    "test_data_used": False,
    "pca_components": PCA_COMPONENTS,
    "pca_explained_variance_ratio_sum": float(
        pca.explained_variance_ratio_.sum()
    ),
    "pca_fit_seconds": pca_seconds,
    "classes": CLASSES,
    "best_validation_model_by_macro_f1": str(leaderboard.iloc[0]["model"]),
    "best_validation_macro_f1": float(leaderboard.iloc[0]["macro_f1"]),
    "output_directory": str(OUTPUT_DIR),
}

(OUTPUT_DIR / "summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

print("\n" + "=" * 75)
print("STEP 191 COMPLETE — VALIDATION LEADERBOARD")
print("=" * 75)
print(leaderboard.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nBest by validation macro F1:", summary["best_validation_model_by_macro_f1"])
print("Results:", OUTPUT_DIR)
print("Models:", MODEL_DIR)
print("Test subjects were not used.")
