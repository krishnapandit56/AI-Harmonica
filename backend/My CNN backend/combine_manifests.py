
from pathlib import Path
import pandas as pd

DATASET_DIR = Path("KSHMR_Dataset")

FEATURE_FILE = DATASET_DIR / "feature_manifest.csv"
REVIEW_FILE = DATASET_DIR / "label_review_labeled.csv"
OUTPUT_FILE = DATASET_DIR / "final_training_manifest.csv"

# Load both CSV files
features = pd.read_csv(FEATURE_FILE)
review = pd.read_csv(REVIEW_FILE)

# Normalize audio paths so both CSVs can be matched
def normalize_path(value):
    return str(value).replace("\\", "/").strip().lower()

features["match_path"] = features["original_filepath"].map(
    normalize_path
)
review["match_path"] = review["filepath"].map(normalize_path)

# Create a lookup table for reviewed labels
review_lookup = review.drop_duplicates(
    subset="match_path"
).set_index("match_path")

# Start with successfully generated spectrograms
result = features.copy()

# Replace labels when a reviewed label exists for the same audio file
result["final_label"] = result.apply(
    lambda row: (
        review_lookup.loc[row["match_path"], "assigned_label"]
        if row["match_path"] in review_lookup.index
        else row["label"]
    ),
    axis=1
)

# Remove categories we don't want to train on
excluded_labels = {
    "exclude_main_drums",
    "needs_review",
    "unknown",
    "nan",
    ""
}

result = result[
    ~result["final_label"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(excluded_labels)
].copy()

# Save only useful columns
columns_to_keep = [
    "feature_path",
    "original_filepath",
    "final_label"
]

result = result[columns_to_keep]

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("Manifest combination complete.")
print("Spectrogram rows originally:", len(features))
print("Eligible rows after filtering:", len(result))
print("\nSamples per final label:")
print(result["final_label"].value_counts().to_string())
print("\nSaved to:", OUTPUT_FILE)
