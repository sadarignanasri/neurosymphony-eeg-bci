from pathlib import Path
from scipy.io import loadmat
import pandas as pd
import json

ROOTS = [
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults 3"),
    Path("data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults-2 2"),
]
CLASSES = ["FF", "FR", "RF", "RR"]

summary_path = next(
    (r / "Total_epoch.xlsx" for r in ROOTS if (r / "Total_epoch.xlsx").exists()),
    None,
)
if summary_path is None:
    raise FileNotFoundError("Official Total_epoch.xlsx not found.")

summary = pd.read_excel(summary_path)
summary.columns = [str(c).strip() for c in summary.columns]
summary = summary.rename(columns={summary.columns[0]: "Subject"})

results = []
for _, row in summary.iterrows():
    subject = str(row["Subject"]).strip()
    if subject.lower() == "total":
        continue

    expected = {c: int(row[c]) for c in CLASSES}
    candidates = []

    for root in ROOTS:
        candidates.extend(
            root.glob(f"{subject}/EEG/preprocessed/{subject}_preprocessed.mat")
        )

    # Prefer a valid file and skip byte-identical duplicates.
    observed = None
    selected = None
    seen_hashes = set()

    import hashlib
    for path in candidates:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)

        try:
            mat = loadmat(path)
            counts = {
                c: int(mat[c].shape[2])
                if c in mat and getattr(mat[c], "ndim", 0) == 3
                and mat[c].shape[:2] == (31, 6500)
                else None
                for c in CLASSES
            }
            if all(v is not None for v in counts.values()):
                observed, selected = counts, str(path)
                break
        except Exception:
            continue

    if observed is None:
        status = "NO_PREPROCESSED_EPOCH_ARRAYS"
        total = None
    else:
        status = "PASS" if observed == expected else "COUNT_MISMATCH"
        total = sum(observed.values())

    results.append({
        "subject": subject,
        "status": status,
        "expected_counts": expected,
        "observed_counts": observed,
        "total_retained_epochs": total,
        "preprocessed_file": selected,
    })

    print(
        f"{subject}: {status} | "
        f"expected={expected} | observed={observed} | total={total}"
    )

print("\nSUMMARY")
print("Subjects:", len(results))
print("Matching preprocessed files:", sum(r["status"] == "PASS" for r in results))
print("Missing preprocessed arrays:", sum(
    r["status"] == "NO_PREPROCESSED_EPOCH_ARRAYS" for r in results
))
print("Count mismatches:", sum(r["status"] == "COUNT_MISMATCH" for r in results))
print("Verified retained epochs:", sum(
    r["total_retained_epochs"] or 0 for r in results
))

out = Path("results/eeg/visual_motivation/step168_final_preprocessed_validation.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2), encoding="utf-8")
print("Report saved:", out)
print("No source data or models modified.")
