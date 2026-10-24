
from pathlib import Path
import numpy as np
import pandas as pd

root = Path("openmic-2018/processed")

for split in ["train", "test"]:
    manifest = pd.read_csv(root / f"{split}_manifest.csv")

    labels = np.load(root / f"{split}_labels.npz")

    print(f"\n===== {split.upper()} =====")
    print("Manifest rows:", len(manifest))
    print("Sample IDs:", labels["sample_keys"].shape)
    print("Targets:", labels["targets"].shape)
    print("Masks:", labels["masks"].shape)
    print("Relevance:", labels["relevance"].shape)
    print("Instruments:", len(labels["instruments"]))

    feature_path = Path(manifest.iloc[0]["feature_path"])
    feature = np.load(feature_path)

    print("Example spectrogram shape:", feature.shape)
    print("Feature dtype:", feature.dtype)
    print("Feature range:", float(feature.min()), float(feature.max()))

    assert len(manifest) == len(labels["sample_keys"])
    assert labels["targets"].shape == labels["masks"].shape
    assert labels["targets"].shape[1] == 20
    assert feature.ndim == 2
    assert feature.shape[0] == 128

print("\nOutput verification complete.")
