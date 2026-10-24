
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path("openmic-2018/processed")
MODEL_PATH = Path("models/openmic_instrument_cnn.pth")
OUTPUT_DIR = Path("evaluation_results")

BATCH_SIZE = 16
THRESHOLD = 0.50
DEVICE = torch.device("cpu")


# ==================================================
# CNN ARCHITECTURE
# Must match train_openmic.py
# ==================================================

class InstrumentCNN(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ==================================================
# TEST DATASET
# ==================================================

class OpenMICTestDataset(Dataset):
    def __init__(self, manifest, targets, masks):
        self.manifest = manifest.reset_index(drop=True)
        self.targets = targets.astype(np.float32)
        self.masks = masks.astype(np.float32)

        path_columns = [
            "feature_path",
            "spectrogram_path",
            "feature_file",
            "path",
        ]

        self.path_column = next(
            (c for c in path_columns if c in self.manifest.columns),
            None,
        )

        if self.path_column is None:
            raise ValueError(
                "Feature-path column not found. Manifest columns: "
                + str(list(self.manifest.columns))
            )

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        feature_path = Path(
            str(self.manifest.iloc[index][self.path_column])
        )

        if not feature_path.is_absolute():
            if not feature_path.exists():
                feature_path = Path.cwd() / feature_path

        if not feature_path.exists():
            raise FileNotFoundError(
                f"Spectrogram not found: {feature_path}"
            )

        spectrogram = np.load(feature_path).astype(np.float32)

        # Expected feature shape: (128, 216)
        spectrogram = torch.from_numpy(spectrogram).unsqueeze(0)

        target = torch.from_numpy(self.targets[index])
        mask = torch.from_numpy(self.masks[index])

        return spectrogram, target, mask


# ==================================================
# METRICS
# ==================================================

def calculate_metrics(y_true, y_pred, masks, instruments):
    """
    Only known labels are evaluated.
    Uncertain OpenMIC labels are excluded by masks.
    """

    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    valid = masks.astype(bool)

    correct = y_true == y_pred

    # Overall label accuracy
    label_accuracy = (
        100.0 * correct[valid].mean()
        if valid.any()
        else 0.0
    )

    # Exact-match accuracy over known labels in each clip
    clip_has_labels = valid.any(axis=1)
    exact_matches = np.all(
        (~valid) | correct,
        axis=1,
    )

    exact_match_accuracy = (
        100.0 * exact_matches[clip_has_labels].mean()
        if clip_has_labels.any()
        else 0.0
    )

    # Micro precision, recall and F1
    actual = y_true[valid]
    predicted = y_pred[valid]

    tp = np.sum(actual & predicted)
    fp = np.sum(~actual & predicted)
    fn = np.sum(actual & ~predicted)

    micro_precision = (
        tp / (tp + fp) if tp + fp else 0.0
    )
    micro_recall = (
        tp / (tp + fn) if tp + fn else 0.0
    )
    micro_f1 = (
        2 * micro_precision * micro_recall
        / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )

    # Per-instrument metrics
    rows = []

    for i, name in enumerate(instruments):
        known = valid[:, i]

        true_i = y_true[known, i]
        pred_i = y_pred[known, i]

        tp_i = int(np.sum(true_i & pred_i))
        fp_i = int(np.sum(~true_i & pred_i))
        fn_i = int(np.sum(true_i & ~pred_i))
        tn_i = int(np.sum(~true_i & ~pred_i))

        precision = (
            tp_i / (tp_i + fp_i)
            if tp_i + fp_i else 0.0
        )
        recall = (
            tp_i / (tp_i + fn_i)
            if tp_i + fn_i else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        accuracy = (
            100.0 * (tp_i + tn_i) / len(true_i)
            if len(true_i) else 0.0
        )

        rows.append({
            "instrument": name,
            "known_labels": len(true_i),
            "true_positives": tp_i,
            "false_positives": fp_i,
            "true_negatives": tn_i,
            "false_negatives": fn_i,
            "accuracy_percent": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    per_instrument = pd.DataFrame(rows)

    # Macro F1: average of instrument-level F1 scores
    macro_f1 = (
        per_instrument["f1"].mean()
        if len(per_instrument)
        else 0.0
    )

    metrics = {
        "label_accuracy_percent": label_accuracy,
        "exact_match_accuracy_percent": exact_match_accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
    }

    return metrics, per_instrument


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":
    print("Device:", DEVICE)

    manifest_path = ROOT / "test_manifest.csv"
    labels_path = ROOT / "test_labels.npz"

    for path in [MODEL_PATH, manifest_path, labels_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # Load checkpoint
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    instruments = checkpoint["instruments"]

    model = InstrumentCNN(num_classes=len(instruments))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print("Best training epoch:", checkpoint.get("best_epoch", "Unknown"))
    print("Number of instruments:", len(instruments))

    # Load official test data
    manifest = pd.read_csv(manifest_path)
    label_data = np.load(labels_path, allow_pickle=True)

    targets = label_data["targets"]
    masks = label_data["masks"]

    if not (len(manifest) == len(targets) == len(masks)):
        raise ValueError(
            "Test manifest, targets, and masks have different lengths."
        )

    dataset = OpenMICTestDataset(manifest, targets, masks)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    all_targets = []
    all_masks = []
    all_predictions = []

    total_loss = 0.0
    total_known_labels = 0

    loss_function = nn.BCEWithLogitsLoss(reduction="none")

    print("\nEvaluating official test set...")
    print("Test clips:", len(dataset))

    with torch.no_grad():
        for batch_index, (features, batch_targets, batch_masks) in enumerate(loader):
            features = features.to(DEVICE)
            batch_targets = batch_targets.to(DEVICE)
            batch_masks = batch_masks.to(DEVICE)

            logits = model(features)

            element_loss = loss_function(logits, batch_targets)
            masked_loss = element_loss * batch_masks

            known_count = int(batch_masks.sum().item())

            total_loss += masked_loss.sum().item()
            total_known_labels += known_count

            probabilities = torch.sigmoid(logits)
            predictions = probabilities >= THRESHOLD

            all_targets.append(batch_targets.cpu().numpy())
            all_masks.append(batch_masks.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

            completed = min(
                (batch_index + 1) * BATCH_SIZE,
                len(dataset),
            )

            if (batch_index + 1) % 50 == 0 or completed == len(dataset):
                print(f"Evaluated {completed}/{len(dataset)} clips")

    y_true = np.concatenate(all_targets, axis=0)
    y_masks = np.concatenate(all_masks, axis=0)
    y_pred = np.concatenate(all_predictions, axis=0)

    metrics, per_instrument = calculate_metrics(
        y_true,
        y_pred,
        y_masks,
        instruments,
    )

    test_loss = (
        total_loss / total_known_labels
        if total_known_labels
        else float("nan")
    )

    # Save evaluation reports
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "test_loss": test_loss,
        **metrics,
        "threshold": THRESHOLD,
        "test_clips": len(dataset),
        "known_labels": total_known_labels,
    }

    summary_path = OUTPUT_DIR / "test_summary.csv"
    per_instrument_path = OUTPUT_DIR / "per_instrument_metrics.csv"

    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    per_instrument.to_csv(per_instrument_path, index=False)

    # Print overall results
    print("\n" + "=" * 60)
    print("OFFICIAL TEST SET RESULTS")
    print("=" * 60)

    print(f"Test loss:               {test_loss:.4f}")
    print(
        f"Label accuracy:          "
        f"{metrics['label_accuracy_percent']:.2f}%"
    )
    print(
        f"Exact-match accuracy:    "
        f"{metrics['exact_match_accuracy_percent']:.2f}%"
    )
    print(f"Macro F1:                {metrics['macro_f1']:.4f}")
    print(f"Micro F1:                {metrics['micro_f1']:.4f}")
    print(f"Micro precision:         {metrics['micro_precision']:.4f}")
    print(f"Micro recall:            {metrics['micro_recall']:.4f}")
    print(f"Threshold:               {THRESHOLD:.2f}")

    print("\nPER-INSTRUMENT RESULTS")
    print("-" * 60)

    print(
        per_instrument[
            [
                "instrument",
                "accuracy_percent",
                "precision",
                "recall",
                "f1",
            ]
        ].to_string(
            index=False,
            formatters={
                "accuracy_percent": "{:.2f}%".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1": "{:.4f}".format,
            },
        )
    )

    print("\nReports saved:")
    print("Summary:", summary_path)
    print("Per-instrument metrics:", per_instrument_path)
    print("\nEvaluation complete. Model weights were not changed.")
