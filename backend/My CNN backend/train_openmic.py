
from pathlib import Path
import random
import copy

import numpy as np
import pandas as pd
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

ROOT = Path("openmic-2018/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001
VALIDATION_RATIO = 0.10

# Use CPU for compatibility with your current PyTorch installation.
DEVICE = torch.device("cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Device:", DEVICE)


# ============================================================
# 2. LOAD PREPROCESSED DATA
# ============================================================

train_manifest = pd.read_csv(ROOT / "train_manifest.csv")
train_labels = np.load(ROOT / "train_labels.npz")

test_manifest = pd.read_csv(ROOT / "test_manifest.csv")
test_labels = np.load(ROOT / "test_labels.npz")

instruments = train_labels["instruments"].tolist()

assert len(instruments) == 20
assert len(train_manifest) == len(train_labels["sample_keys"])
assert len(test_manifest) == len(test_labels["sample_keys"])

assert np.array_equal(
    train_manifest["sample_key"].astype(str).to_numpy(),
    train_labels["sample_keys"].astype(str)
)

assert np.array_equal(
    test_manifest["sample_key"].astype(str).to_numpy(),
    test_labels["sample_keys"].astype(str)
)

print("Instruments:", instruments)
print("Training clips:", len(train_manifest))
print("Test clips:", len(test_manifest))


# ============================================================
# 3. CREATE TRAINING / VALIDATION SPLIT
# ============================================================

all_indices = np.random.permutation(len(train_manifest))

validation_size = int(len(all_indices) * VALIDATION_RATIO)

val_indices = all_indices[:validation_size]
train_indices = all_indices[validation_size:]

print("Training subset:", len(train_indices))
print("Validation subset:", len(val_indices))


# ============================================================
# 4. DATASET CLASS
# ============================================================

class OpenMICDataset(Dataset):

    def __init__(self, manifest, labels, indices):
        self.manifest = manifest.reset_index(drop=True)
        self.targets = labels["targets"].astype(np.float32)
        self.masks = labels["masks"].astype(np.float32)
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])

        feature_path = Path(
            self.manifest.iloc[index]["feature_path"]
        )

        if not feature_path.is_absolute():
            feature_path = feature_path.resolve()

        spectrogram = np.load(feature_path).astype(np.float32)

        # CNN input shape: (channels, frequency, time)
        spectrogram = torch.from_numpy(
            spectrogram[np.newaxis, :, :]
        )

        target = torch.from_numpy(self.targets[index])
        mask = torch.from_numpy(self.masks[index])

        return spectrogram, target, mask


train_dataset = OpenMICDataset(
    train_manifest, train_labels, train_indices
)

val_dataset = OpenMICDataset(
    train_manifest, train_labels, val_indices
)

# Windows-friendly: num_workers=0.
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 5. DEFINE THE CNN
# ============================================================

class InstrumentCNN(nn.Module):

    def __init__(self, number_of_instruments=20):
        super().__init__()

        self.features = nn.Sequential(
            # Input: (B, 1, 128, 216)
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Avoid a large fully connected layer.
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(128, number_of_instruments)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = InstrumentCNN(len(instruments)).to(DEVICE)

print("\nCNN created successfully.")
print("Trainable parameters:",
      sum(p.numel() for p in model.parameters()
          if p.requires_grad))


# ============================================================
# 6. MASKED MULTI-LABEL LOSS
# ============================================================

def masked_bce_loss(logits, targets, masks):
    """
    Ignore uncertain labels using masks.
    logits must remain raw; BCEWithLogitsLoss applies
    the numerically stable sigmoid + BCE calculation.
    """
    element_losses = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none"
    )

    masked_losses = element_losses * masks

    # Normalize by the number of known labels.
    return masked_losses.sum() / masks.sum().clamp(min=1.0)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 7. EVALUATION METRICS
# ============================================================

@torch.no_grad()
def evaluate(loader):
    model.eval()

    total_loss = 0.0
    total_known_labels = 0.0

    tp = np.zeros(len(instruments), dtype=np.float64)
    fp = np.zeros(len(instruments), dtype=np.float64)
    fn = np.zeros(len(instruments), dtype=np.float64)

    for spectrograms, targets, masks in loader:
        spectrograms = spectrograms.to(DEVICE)
        targets = targets.to(DEVICE)
        masks = masks.to(DEVICE)

        logits = model(spectrograms)

        losses = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        total_loss += (losses * masks).sum().item()
        total_known_labels += masks.sum().item()

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()

        # Ignore unknown entries when counting metrics.
        valid = masks.bool()

        tp += ((predictions == 1) & (targets == 1) & valid).sum(
            dim=0
        ).cpu().numpy()

        fp += ((predictions == 1) & (targets == 0) & valid).sum(
            dim=0
        ).cpu().numpy()

        fn += ((predictions == 0) & (targets == 1) & valid).sum(
            dim=0
        ).cpu().numpy()

    average_loss = total_loss / max(total_known_labels, 1.0)

    precision_per_class = tp / np.maximum(tp + fp, 1)
    recall_per_class = tp / np.maximum(tp + fn, 1)

    f1_per_class = (
        2 * precision_per_class * recall_per_class
        / np.maximum(precision_per_class + recall_per_class, 1e-12)
    )

    # Exclude classes with no positive ground-truth examples
    # in this evaluation subset from macro F1.
    supported_classes = (tp + fn) > 0

    macro_f1 = (
        float(f1_per_class[supported_classes].mean())
        if supported_classes.any()
        else 0.0
    )

    micro_precision = tp.sum() / max(tp.sum() + fp.sum(), 1)
    micro_recall = tp.sum() / max(tp.sum() + fn.sum(), 1)

    micro_f1 = (
        2 * micro_precision * micro_recall
        / max(micro_precision + micro_recall, 1e-12)
    )

    return average_loss, macro_f1, float(micro_f1)


# ============================================================
# 8. TRAIN THE MODEL
# ============================================================

best_val_loss = float("inf")
best_state = None
best_epoch = 0

checkpoint_path = MODEL_DIR / "openmic_instrument_cnn.pth"

for epoch in range(1, EPOCHS + 1):

    model.train()

    total_loss = 0.0
    total_known_labels = 0.0

    for batch_number, (spectrograms, targets, masks) in enumerate(
        train_loader, start=1
    ):
        spectrograms = spectrograms.to(DEVICE)
        targets = targets.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        logits = model(spectrograms)
        loss = masked_bce_loss(logits, targets, masks)

        loss.backward()
        optimizer.step()

        known = masks.sum().item()
        total_loss += loss.item() * known
        total_known_labels += known

        if batch_number % 100 == 0:
            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Batch {batch_number}/{len(train_loader)}"
            )

    train_loss = total_loss / max(total_known_labels, 1.0)

    val_loss, val_macro_f1, val_micro_f1 = evaluate(val_loader)

    print(
        f"\nEpoch {epoch}/{EPOCHS}\n"
        f"Train loss: {train_loss:.4f}\n"
        f"Validation loss: {val_loss:.4f}\n"
        f"Validation macro F1: {val_macro_f1:.4f}\n"
        f"Validation micro F1: {val_micro_f1:.4f}\n"
    )

    # Save a copy of the best weights in memory.
    if np.isfinite(val_loss) and val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())

        torch.save(
            {
                "model_state_dict": best_state,
                "instruments": instruments,
                "input_shape": [1, 128, 216],
                "best_epoch": best_epoch,
                "validation_loss": best_val_loss,
                "threshold": 0.5
            },
            checkpoint_path
        )

        print("Saved best model:", checkpoint_path)

    print("-" * 60)


# ============================================================
# 9. FINAL VALIDATION SUMMARY
# ============================================================

if best_state is None:
    raise RuntimeError(
        "Training did not produce a valid checkpoint. "
        "Check the dataset and loss values."
    )

model.load_state_dict(best_state)

print("\nTraining complete!")
print("Best epoch:", best_epoch)
print(f"Best validation loss: {best_val_loss:.4f}")
print("Best model saved to:", checkpoint_path)
print("\nThe official test set has NOT been used for training.")
print("We will evaluate it separately in the next step.")
