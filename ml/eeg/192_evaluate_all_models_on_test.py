from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

DATA_DIR = Path("data/processed/visual_motivation_125hz_normalized")
STEP191_DIR = Path(
    "results/eeg/visual_motivation/step191_model_comparison"
)
MODEL_DIR = STEP191_DIR / "models"
OUTPUT_DIR = Path(
    "results/eeg/visual_motivation/step192_test_evaluation"
)

CLASSES = ["FF", "FR", "RF", "RR"]

if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
    raise FileExistsError(
        f"Output directory is not empty: {OUTPUT_DIR}. "
        "Nothing was overwritten."
    )

manifest_path = DATA_DIR / "dataset_manifest.csv"
if not manifest_path.exists():
    raise FileNotFoundError(manifest_path)

manifest = pd.read_csv(manifest_path)
test_rows = manifest[manifest["split"] == "test"]

if set(test_rows["subject"]) != {"S03", "S14"}:
    raise ValueError(
        f"Unexpected test subjects: {sorted(test_rows['subject'].unique())}"
    )

def load_test_data():
    X_parts, y_parts = [], []

    for row in test_rows.itertuples(index=False):
        arr = np.load(row.output_file, allow_pickle=False)

        if arr.shape != (int(row.epoch_count), 31, 1625):
            raise ValueError(f"Unexpected shape: {row.output_file}")
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite values: {row.output_file}")

        X_parts.append(arr.reshape(arr.shape[0], -1))
        y_parts.extend([row.label] * arr.shape[0])

    return np.concatenate(X_parts).astype(np.float32), np.asarray(y_parts)

print("Loading held-out test subjects only...")
X_test, y_test = load_test_data()

if len(y_test) != 413:
    raise ValueError(f"Expected 413 test epochs, found {len(y_test)}")

print("Test shape:", X_test.shape)
print("Test subjects:", sorted(test_rows["subject"].unique()))
print("True class counts:", pd.Series(y_test).value_counts().to_dict())

pca_path = MODEL_DIR / "training_fitted_pca.joblib"
if not pca_path.exists():
    raise FileNotFoundError(pca_path)

# Reuse the PCA fitted on training subjects in Step 191.
pca = joblib.load(pca_path)
X_test_pca = pca.transform(X_test)

if not np.isfinite(X_test_pca).all():
    raise ValueError("PCA test output contains non-finite values")

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Random Forest": "random_forest.joblib",
    "Extra Trees": "extra_trees.joblib",
    "Linear SVM": "linear_svm.joblib",
    "RBF SVM": "rbf_svm.joblib",
    "KNN": "knn.joblib",
    "LDA": "lda.joblib",
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
results = []
reports = {}
confusions = {}

for name, filename in model_files.items():
    model_path = MODEL_DIR / filename
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = joblib.load(model_path)

    start = time.perf_counter()
    predictions = model.predict(X_test_pca)
    prediction_seconds = time.perf_counter() - start

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, labels=CLASSES,
        average="macro", zero_division=0
    )
    balanced = balanced_accuracy_score(y_test, predictions)

    results.append({
        "model": name,
        "test_accuracy": accuracy,
        "test_accuracy_percent": accuracy * 100,
        "test_macro_f1": macro_f1,
        "test_balanced_accuracy": balanced,
        "test_prediction_seconds": prediction_seconds,
        "test_epochs": len(y_test),
        "pca_components": 100,
    })

    reports[name] = classification_report(
        y_test, predictions, labels=CLASSES,
        output_dict=True, zero_division=0
    )
    confusions[name] = confusion_matrix(
        y_test, predictions, labels=CLASSES
    ).tolist()

    print(f"\n{name}")
    print(f"  Test accuracy:       {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"  Test macro F1:       {macro_f1:.4f}")
    print(f"  Test balanced acc.:  {balanced:.4f}")
    print("  Per-class F1:")
    for label in CLASSES:
        print(f"    {label}: {reports[name][label]['f1-score']:.4f}")

leaderboard = pd.DataFrame(results).sort_values(
    ["test_macro_f1", "test_balanced_accuracy"],
    ascending=False,
)
leaderboard.to_csv(OUTPUT_DIR / "test_leaderboard.csv", index=False)

(OUTPUT_DIR / "test_classification_reports.json").write_text(
    json.dumps(reports, indent=2), encoding="utf-8"
)
(OUTPUT_DIR / "test_confusion_matrices.json").write_text(
    json.dumps(
        {"class_order": CLASSES, "confusion_matrices": confusions},
        indent=2,
    ),
    encoding="utf-8",
)

summary = {
    "step": 192,
    "test_subjects": sorted(test_rows["subject"].unique()),
    "test_epochs": len(y_test),
    "models_evaluated": list(model_files.keys()),
    "pca_fitted_on_training_data_only": True,
    "models_retrained": False,
    "test_results_used_for_prior_model_selection": False,
    "best_test_model_by_macro_f1": str(leaderboard.iloc[0]["model"]),
    "best_test_macro_f1": float(leaderboard.iloc[0]["test_macro_f1"]),
    "best_test_accuracy_percent": float(
        leaderboard.iloc[0]["test_accuracy_percent"]
    ),
    "output_directory": str(OUTPUT_DIR),
}

(OUTPUT_DIR / "summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

print("\n" + "=" * 80)
print("STEP 192 COMPLETE — HELD-OUT TEST RESULTS")
print("=" * 80)
print(leaderboard.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nResults saved to:", OUTPUT_DIR)
print("No models were retrained or overwritten.")
