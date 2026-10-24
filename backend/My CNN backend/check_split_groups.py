
import pandas as pd

path = "KSHMR_Dataset/feature_manifest.csv"
df = pd.read_csv(path)

print("Columns:")
print(df.columns.tolist())

print("\nSample counts by label:")
print(df["label"].value_counts().to_string())

if "original_index" in df.columns:
    print("\nRows with duplicate original indices:")
    print(df["original_index"].duplicated().sum())
