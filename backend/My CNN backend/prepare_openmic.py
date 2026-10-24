
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATASET_DIR = Path("openmic-2018")
AUDIO_DIR = DATASET_DIR / "audio"
LABELS_FILE = DATASET_DIR / "openmic-2018-aggregated-labels.csv"
PARTITIONS_DIR = DATASET_DIR / "partitions"

OUTPUT_DIR = DATASET_DIR / "processed"
FEATURE_DIR = OUTPUT_DIR / "mel_spectrograms"

SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 1024
MAX_DB_RANGE = 80.0

# Label thresholds. These are initial settings to evaluate.
POSITIVE_THRESHOLD = 0.70
NEGATIVE_THRESHOLD = 0.30

# Set to an integer such as 100 to test a small batch first.
# Set to None to process all clips.
MAX_SAMPLES_PER_SPLIT = None


# ==========================================
# 2. CHECK INPUT FILES
# ==========================================

for path in [
    AUDIO_DIR,
    LABELS_FILE,
    PARTITIONS_DIR / "split01_train.csv",
    PARTITIONS_DIR / "split01_test.csv",
]:
    if not path.exists():
        raise FileNotFoundError(f"Required path not found: {path}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 3. LOAD INSTRUMENT LABELS
# ==========================================

labels_df = pd.read_csv(LABELS_FILE)

instruments = sorted(labels_df["instrument"].unique())
instrument_to_index = {
    instrument: index
    for index, instrument in enumerate(instruments)
}

print("Number of instruments:", len(instruments))
print("Instruments:", instruments)


# ==========================================
# 4. LOAD OFFICIAL PARTITIONS
# ==========================================

train_keys = (
    pd.read_csv(PARTITIONS_DIR / "split01_train.csv", header=None)[0]
    .astype(str)
    .str.strip()
    .tolist()
)

test_keys = (
    pd.read_csv(PARTITIONS_DIR / "split01_test.csv", header=None)[0]
    .astype(str)
    .str.strip()
    .tolist()
)

train_set = set(train_keys)
test_set = set(test_keys)

if train_set & test_set:
    raise ValueError("Training and test partitions overlap!")

all_keys = train_set | test_set
label_keys = set(labels_df["sample_key"].astype(str))

if all_keys != label_keys:
    raise ValueError(
        "Partition IDs and label IDs do not match. "
        f"Missing labels: {len(all_keys - label_keys)}, "
        f"extra labels: {len(label_keys - all_keys)}"
    )

print("Training clips:", len(train_keys))
print("Test clips:", len(test_keys))


# ==========================================
# 5. INDEX AUDIO FILES
# ==========================================

print("\nIndexing audio files...")

audio_paths = {
    file.stem: file
    for file in AUDIO_DIR.rglob("*.ogg")
}

if len(audio_paths) != len(all_keys):
    print("Warning: audio files found:", len(audio_paths))
    print("Expected clips:", len(all_keys))

missing_audio = all_keys - set(audio_paths)

if missing_audio:
    raise FileNotFoundError(
        f"{len(missing_audio)} partition clips have no audio file. "
        f"Examples: {list(missing_audio)[:5]}"
    )

print("All partition audio files found.")


# ==========================================
# 6. BUILD MULTI-LABEL TARGETS AND MASKS
# ==========================================

# Targets:
#   1 = likely present
#   0 = likely absent
#
# Masks:
#   1 = use this assessment for training
#   0 = ignore this uncertain assessment

def build_label_arrays(sample_keys):
    key_to_row = {
        key: index
        for index, key in enumerate(sample_keys)
    }

    n_samples = len(sample_keys)
    n_instruments = len(instruments)

    targets = np.zeros(
        (n_samples, n_instruments),
        dtype=np.float32
    )

    masks = np.zeros(
        (n_samples, n_instruments),
        dtype=np.float32
    )

    relevance_values = np.full(
        (n_samples, n_instruments),
        np.nan,
        dtype=np.float32
    )

    subset = labels_df[
        labels_df["sample_key"].astype(str).isin(key_to_row)
    ]

    for row in subset.itertuples(index=False):
        sample_key = str(row.sample_key)
        instrument = row.instrument
        relevance = float(row.relevance)

        sample_index = key_to_row[sample_key]
        instrument_index = instrument_to_index[instrument]

        relevance_values[sample_index, instrument_index] = relevance

        if relevance >= POSITIVE_THRESHOLD:
            targets[sample_index, instrument_index] = 1.0
            masks[sample_index, instrument_index] = 1.0

        elif relevance <= NEGATIVE_THRESHOLD:
            targets[sample_index, instrument_index] = 0.0
            masks[sample_index, instrument_index] = 1.0

        # Intermediate scores remain masked out.

    return targets, masks, relevance_values


# ==========================================
# 7. EXTRACT LOG-MEL SPECTROGRAMS
# ==========================================

def extract_log_mel(audio_path):
    audio, sr = sf.read(
        str(audio_path),
        dtype="float32",
        always_2d=True
    )

    # Convert stereo to mono.
    audio = audio.mean(axis=1)

    # Resample to 22,050 Hz when necessary.
    if sr != SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=SAMPLE_RATE
        )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max,
        top_db=MAX_DB_RANGE
    )

    # Convert the approximately -80 to 0 dB range
    # into a compact 0 to 1 representation.
    log_mel = (log_mel + MAX_DB_RANGE) / MAX_DB_RANGE
    log_mel = np.clip(log_mel, 0.0, 1.0)

    # Save as float16 to reduce disk usage.
    return log_mel.astype(np.float16)


# ==========================================
# 8. PROCESS TRAIN AND TEST SEPARATELY
# ==========================================

def process_split(split_name, sample_keys):
    if MAX_SAMPLES_PER_SPLIT is not None:
        sample_keys = sample_keys[:MAX_SAMPLES_PER_SPLIT]

    print(f"\nProcessing {split_name}: {len(sample_keys)} clips")

    targets, masks, relevance = build_label_arrays(sample_keys)

    manifest_rows = []
    failed = []

    for index, sample_key in enumerate(sample_keys, start=1):
        audio_path = audio_paths[sample_key]

        feature_path = FEATURE_DIR / f"{sample_key}.npy"

        try:
            if not feature_path.exists():
                feature = extract_log_mel(audio_path)
                np.save(feature_path, feature)

            manifest_rows.append({
                "sample_key": sample_key,
                "audio_path": str(audio_path),
                "feature_path": str(feature_path),
                "split": split_name,
            })

        except Exception as exc:
            failed.append((sample_key, str(exc)))
            print(f"\nFailed: {sample_key}: {exc}")

        if index % 100 == 0 or index == len(sample_keys):
            print(
                f"\rProcessed {index}/{len(sample_keys)}",
                end="",
                flush=True
            )

    print()

    # Keep the manifest aligned with the label arrays.
    successful_indices = [
        index
        for index, key in enumerate(sample_keys)
        if key not in {failed_key for failed_key, _ in failed}
    ]

    successful_keys = [sample_keys[i] for i in successful_indices]

    manifest = pd.DataFrame(manifest_rows)

    manifest.to_csv(
        OUTPUT_DIR / f"{split_name}_manifest.csv",
        index=False
    )

    np.savez_compressed(
        OUTPUT_DIR / f"{split_name}_labels.npz",
        sample_keys=np.asarray(successful_keys),
        instruments=np.asarray(instruments),
        targets=targets[successful_indices],
        masks=masks[successful_indices],
        relevance=relevance[successful_indices],
    )

    print(f"Successful clips: {len(manifest_rows)}")
    print(f"Failed clips: {len(failed)}")

    if failed:
        pd.DataFrame(
            failed,
            columns=["sample_key", "error"]
        ).to_csv(
            OUTPUT_DIR / f"{split_name}_failed.csv",
            index=False
        )

    return len(manifest_rows), len(failed)


# ==========================================
# 9. RUN PREPROCESSING
# ==========================================

print("\nStarting OpenMIC preprocessing...")
print("Output folder:", OUTPUT_DIR.resolve())

train_success, train_failed = process_split(
    "train",
    train_keys
)

test_success, test_failed = process_split(
    "test",
    test_keys
)

print("\n========== PREPROCESSING SUMMARY ==========")
print("Training clips processed:", train_success)
print("Training failures:", train_failed)
print("Test clips processed:", test_success)
print("Test failures:", test_failed)
print("Number of instruments:", len(instruments))
print("Feature directory:", FEATURE_DIR.resolve())
print("===========================================")
