
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("openmic-2018/processed")

manifest = pd.read_csv(ROOT / "train_manifest.csv")
labels = np.load(ROOT / "train_labels.npz", allow_pickle=True)

targets = labels["targets"]
masks = labels["masks"]

# Instrument order saved by the preprocessing script.
if "instruments" in labels.files:
    instruments = labels["instruments"].tolist()
else:
    # Read the order from the saved model checkpoint.
    import torch

    checkpoint = torch.load(
        Path("models/openmic_instrument_cnn.pth"),
        map_location="cpu",
        weights_only=True,
    )
    instruments = checkpoint["instruments"]

targets = targets.astype(bool)
known = masks.astype(bool)

rows = []

for i, instrument in enumerate(instruments):
    valid = known[:, i]
    positive = np.sum(targets[valid, i])
    negative = np.sum(~targets[valid, i])
    uncertain = np.sum(~valid)

    rows.append({
        "instrument": instrument,
        "positive": int(positive),
        "negative": int(negative),
        "uncertain": int(uncertain),
        "known_labels": int(valid.sum()),
        "positive_percent_of_known": (
            round(100 * positive / valid.sum(), 2)
            if valid.sum() else 0.0
        ),
    })

result = pd.DataFrame(rows).sort_values(
    "positive",
    ascending=False,
)

print("Training clips:", len(manifest))
print("\nClass distribution:")
print(result.to_string(index=False))

output = Path("evaluation_results")
output.mkdir(exist_ok=True)
result.to_csv(output / "training_class_balance.csv", index=False)

print("\nSaved: evaluation_results/training_class_balance.csv")
