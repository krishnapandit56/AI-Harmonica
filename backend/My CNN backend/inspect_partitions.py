
import pandas as pd
from pathlib import Path

partition_dir = Path("openmic-2018/partitions")

for file in sorted(partition_dir.glob("*.csv")):
    df = pd.read_csv(file, header=None)

    print(f"\nFile: {file.name}")
    print("Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print("\nColumns:", df.columns.tolist())
