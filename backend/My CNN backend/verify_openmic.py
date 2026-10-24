
from pathlib import Path
import pandas as pd

# Dataset paths
DATASET_DIR = Path("openmic-2018")
AUDIO_DIR = DATASET_DIR / "audio"
LABELS_FILE = DATASET_DIR / "openmic-2018-aggregated-labels.csv"
PARTITIONS_DIR = DATASET_DIR / "partitions"

# Load instrument labels
labels = pd.read_csv(LABELS_FILE)

# Collect audio sample keys
audio_keys = {
    file.stem
    for file in AUDIO_DIR.rglob("*.ogg")
}

label_keys = set(labels["sample_key"].astype(str))

print("===== OpenMIC Dataset Verification =====")
print("Unique audio clips:", len(audio_keys))
print("Unique label keys:", len(label_keys))

missing_audio = label_keys - audio_keys
missing_labels = audio_keys - label_keys

print("Label keys without audio:", len(missing_audio))
print("Audio clips without labels:", len(missing_labels))

print("\nInstrument categories:")
print(sorted(labels["instrument"].unique()))

print("\nPartition files:")
if PARTITIONS_DIR.exists():
    for file in PARTITIONS_DIR.rglob("*"):
        if file.is_file():
            print(file.relative_to(DATASET_DIR))
else:
    print("Partitions folder not found.")

print("\nVerification complete.")
