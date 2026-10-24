
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

BASE_DIR = Path("KSHMR_Dataset")

features_file = BASE_DIR / "feature_manifest.csv"
dataset_file = BASE_DIR / "dataset_manifest.csv"
output_file = BASE_DIR / "final_training_manifest.csv"

features = pd.read_csv(features_file)
dataset = pd.read_csv(dataset_file)

print("Feature manifest columns:", features.columns.tolist())
print("Dataset manifest columns:", dataset.columns.tolist())

# Match paths consistently.
def normalize_path(path):
    return str(path).replace("\\", "/").strip().lower()

features["path_key"] = features["original_filepath"].map(normalize_path)
dataset["path_key"] = dataset["filepath"].map(normalize_path)

# Recover split_group from the original dataset manifest.
if "split_group" not in dataset.columns:
    raise ValueError(
        "split_group is missing from dataset_manifest.csv. "
        "We need to inspect the manifest before splitting."
    )

group_lookup = dataset[["path_key", "split_group"]].drop_duplicates(
    subset="path_key"
)

df = features.merge(
    group_lookup,
    on="path_key",
    how="left",
    validate="many_to_one"
)

missing_groups = df["split_group"].isna().sum()
print("Rows missing split_group:", missing_groups)

if missing_groups:
    print(df.loc[df["split_group"].isna(), "original_filepath"].head(10))
    raise ValueError("Some samples could not be matched to their split groups.")

# Use 15% for test, then 15% of the remaining data for validation.
# This gives approximately 70% train, 15% validation, 15% test.
first_split = GroupShuffleSplit(
    n_splits=1, test_size=0.15, random_state=42
)
train_val_idx, test_idx = next(
    first_split.split(df, groups=df["split_group"])
)

train_val = df.iloc[train_val_idx].copy()
test = df.iloc[test_idx].copy()

second_split = GroupShuffleSplit(
    n_splits=1, test_size=0.15 / 0.85, random_state=42
)
train_idx, val_idx = next(
    second_split.split(train_val, groups=train_val["split_group"])
)

train = train_val.iloc[train_idx].copy()
val = train_val.iloc[val_idx].copy()

df["split"] = ""
df.loc[train.index, "split"] = "train"
df.loc[val.index, "split"] = "validation"
df.loc[test.index, "split"] = "test"

# Remove the helper column; keep split_group for auditing.
df = df.drop(columns=["path_key"])

df.to_csv(output_file, index=False)

print("\nSplit counts:")
print(df["split"].value_counts())
print("\nClass counts by split:")
print(pd.crosstab(df["split"], df["label"]))
print("\nSaved to:", output_file)
