import sys
import numpy as np
import pandas as pd
import sklearn
from pathlib import Path

root = Path("data/processed/visual_motivation_125hz_normalized")
manifest_path = root / "dataset_manifest.csv"

print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)

if not manifest_path.exists():
    raise FileNotFoundError(manifest_path)

df = pd.read_csv(manifest_path)
print("Manifest rows:", len(df))
print("Subjects:", df["subject"].nunique())
print("Splits:", df.groupby("split")["subject"].nunique().to_dict())
print("Epochs:", int(df["epoch_count"].sum()))
print("Features per epoch:", 31 * 1625)

train = df[df["split"] == "train"]
print("Training epochs:", int(train["epoch_count"].sum()))
print("Validation epochs:", int(df.loc[df["split"] == "validation", "epoch_count"].sum()))
print("Test epochs:", int(df.loc[df["split"] == "test", "epoch_count"].sum()))

assert len(df) == 40
assert int(df["epoch_count"].sum()) == 1874
assert int(train["epoch_count"].sum()) == 1118

print("\nBASELINE ENVIRONMENT CHECK: PASS")
print("No model trained and no test predictions made.")
