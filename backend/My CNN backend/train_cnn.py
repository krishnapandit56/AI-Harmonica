
from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# 1. Configuration
# -----------------------------
SEED = 42
BATCH_SIZE = 16
MAX_EPOCHS = 20
PATIENCE = 5
LEARNING_RATE = 0.001

BASE_DIR = Path("KSHMR_Dataset")
MANIFEST_PATH = BASE_DIR / "final_training_manifest.csv"
FEATURE_DIR = BASE_DIR / "features" / "mel_spectrograms"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")
print("Training device:", DEVICE)


# -----------------------------
# 2. Load manifest and labels
# -----------------------------
df = pd.read_csv(MANIFEST_PATH)

required_columns = {"feature_path", "label", "split"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing manifest columns: {missing_columns}")

df["label"] = df["label"].astype(str).str.strip()

classes = sorted(df["label"].unique())
class_to_idx = {name: i for i, name in enumerate(classes)}

print("\nInstrument classes:")
for name, idx in class_to_idx.items():
    print(f"{idx}: {name}")

# Resolve paths relative to KSHMR_Dataset.
def resolve_feature_path(value):
    path = Path(str(value).replace("\\", "/"))

    if path.is_absolute() and path.exists():
        return path

    candidates = [
        BASE_DIR / path,
        BASE_DIR / "features" / "mel_spectrograms" / path.name,
        path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find spectrogram: {value}"
    )


df["resolved_path"] = df["feature_path"].apply(resolve_feature_path)

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

if min(len(train_df), len(val_df), len(test_df)) == 0:
    raise ValueError("Train, validation, or test split is empty.")

print("\nSamples:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# -----------------------------
# 3. Dataset loader
# -----------------------------
class SpectrogramDataset(Dataset):
    def __init__(self, frame):
        self.paths = frame["resolved_path"].tolist()
        self.labels = [
            class_to_idx[label] for label in frame["label"]
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        spec = np.load(self.paths[index]).astype(np.float32)

        if spec.shape != (128, 216):
            raise ValueError(
                f"Unexpected shape {spec.shape} in {self.paths[index]}"
            )

        # Add the channel dimension: (128, 216) -> (1, 128, 216)
        spec = torch.from_numpy(spec).unsqueeze(0)
        label = torch.tensor(self.labels[index], dtype=torch.long)

        return spec, label


train_loader = DataLoader(
    SpectrogramDataset(train_df),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    SpectrogramDataset(val_df),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    SpectrogramDataset(test_df),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# -----------------------------
# 4. Define our CNN
# -----------------------------
class InstrumentCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((4, 6)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 6, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = InstrumentCNN(num_classes=len(classes)).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

print("\nModel created.")
print("Number of parameters:", sum(p.numel() for p in model.parameters()))


# -----------------------------
# 5. Training and evaluation
# -----------------------------
def run_epoch(loader, training):
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, labels in loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct / total_samples,
    )


best_val_loss = float("inf")
best_state = None
epochs_without_improvement = 0

for epoch in range(1, MAX_EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, training=True)
    val_loss, val_acc = run_epoch(val_loader, training=False)

    print(
        f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train acc: {train_acc:.1%} | "
        f"Val loss: {val_loss:.4f} | "
        f"Val acc: {val_acc:.1%}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping: validation loss stopped improving.")
        break


# -----------------------------
# 6. Save the best model
# -----------------------------
if best_state is None:
    raise RuntimeError("Training did not produce a model checkpoint.")

model.load_state_dict(best_state)

model_path = MODEL_DIR / "instrument_cnn.pth"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "input_shape": [1, 128, 216],
    },
    model_path,
)

print("\nBest model saved to:", model_path)


# -----------------------------
# 7. Final test evaluation
# -----------------------------
test_loss, test_acc = run_epoch(test_loader, training=False)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.1%}")
print("Training complete.")
