
from pathlib import Path
import pandas as pd

path = Path("KSHMR_Dataset/final_training_manifest.csv")
df = pd.read_csv(path)

print("Total rows:", len(df))
print("Missing feature paths:", df["feature_path"].isna().sum())
print("Missing audio paths:", df["original_filepath"].isna().sum())
print("Missing labels:", df["final_label"].isna().sum())

print("\nLabels that should not be present:")
excluded = {
    "exclude_main_drums",
    "needs_review",
    "unknown",
    "nan",
    ""
}

bad = df[
    df["final_label"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(excluded)
]

print(bad["final_label"].value_counts().to_string())
print("\nFinal category counts:")
print(df["final_label"].value_counts().to_string())

print("\nMissing spectrogram files:")
missing = [
    p for p in df["feature_path"].dropna()
    if not (Path("KSHMR_Dataset") / p).is_file()
]
print(len(missing))
