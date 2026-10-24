
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import soundfile as sf

# Dataset paths
DATASET_DIR = Path("KSHMR_Dataset")
AUDIO_DIR = DATASET_DIR / "audio"
MANIFEST_PATH = DATASET_DIR / "dataset_manifest.csv"
OUTPUT_DIR = DATASET_DIR / "features" / "mel_spectrograms"

# Audio processing settings
SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
MAX_DURATION = 5.0  # seconds

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST_PATH)

required_columns = {"filepath", "candidate_instrument_label"}
missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing manifest columns: {missing}")

# Save metadata for successfully processed features
feature_rows = []
failed_files = []

for index, row in df.iterrows():
    relative_path = Path(str(row["filepath"]))

    # Manifest paths are relative to KSHMR_Dataset
    audio_path = DATASET_DIR / relative_path

    if not audio_path.is_file():
        failed_files.append({
            "filepath": str(audio_path),
            "reason": "File not found"
        })
        continue

    label = str(row["candidate_instrument_label"]).strip()

    if not label or label.lower() in {"nan", "unknown", "needs_review"}:
        failed_files.append({
            "filepath": str(audio_path),
            "reason": "Missing or uncertain label"
        })
        continue

    try:
        # Load audio as mono at the target sample rate
        y, sr = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True,
            duration=MAX_DURATION
        )

        if len(y) == 0:
            raise ValueError("Audio contains no samples")

        # Convert waveform into a Mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=2.0
        )

        # Convert power to decibels
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # Normalize dimensions to a fixed time length
        target_frames = int(
            MAX_DURATION * SAMPLE_RATE / HOP_LENGTH
        ) + 1

        if log_mel.shape[1] < target_frames:
            pad_width = target_frames - log_mel.shape[1]
            log_mel = np.pad(
                log_mel,
                ((0, 0), (0, pad_width)),
                mode="constant",
                constant_values=log_mel.min()
            )
        else:
            log_mel = log_mel[:, :target_frames]

        # Store one spectrogram per file
        feature_name = f"sample_{index:07d}.npy"
        feature_path = OUTPUT_DIR / feature_name
        np.save(feature_path, log_mel.astype(np.float32))

        feature_rows.append({
            "feature_path": str(feature_path.relative_to(DATASET_DIR)),
            "original_filepath": str(relative_path),
            "label": label,
            "original_index": index,
            "n_mels": log_mel.shape[0],
            "n_frames": log_mel.shape[1]
        })

    except Exception as exc:
        failed_files.append({
            "filepath": str(audio_path),
            "reason": str(exc)
        })

    if (index + 1) % 100 == 0:
        print(
            f"Processed {index + 1}/{len(df)} manifest rows; "
            f"features created: {len(feature_rows)}"
        )

# Save feature metadata
feature_manifest = pd.DataFrame(feature_rows)
feature_manifest.to_csv(
    DATASET_DIR / "feature_manifest.csv",
    index=False
)

# Save errors for inspection
pd.DataFrame(failed_files).to_csv(
    DATASET_DIR / "preprocessing_errors.csv",
    index=False
)

print("\nPreprocessing complete.")
print(f"Manifest rows: {len(df)}")
print(f"Features created: {len(feature_rows)}")
print(f"Files skipped or failed: {len(failed_files)}")
print(f"Spectrogram folder: {OUTPUT_DIR}")
print(f"Feature manifest: {DATASET_DIR / 'feature_manifest.csv'}")
