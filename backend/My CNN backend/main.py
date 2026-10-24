
from pathlib import Path
import io

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# New OpenMIC model; old KSHMR model remains untouched.
MODEL_PATH = BASE_DIR / "models" / "openmic_instrument_cnn.pth"

SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 1024
SEGMENT_SECONDS = 10
MAX_UPLOAD_SIZE = 20 * 1024 * 1024

DEVICE = torch.device("cpu")


# ============================================================
# 2. CNN ARCHITECTURE
# Must match train_openmic.py exactly.
# ============================================================

class InstrumentCNN(nn.Module):

    def __init__(self, number_of_instruments=20):
        super().__init__()

        self.features = nn.Sequential(
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

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(128, number_of_instruments),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ============================================================
# 3. LOAD OPENMIC CHECKPOINT
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"OpenMIC model not found: {MODEL_PATH}\n"
        "Make sure train_openmic.py has saved the checkpoint."
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

CLASSES = checkpoint["instruments"]

model = InstrumentCNN(number_of_instruments=len(CLASSES))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

print("Loaded model:", MODEL_PATH)
print("Number of instrument categories:", len(CLASSES))


# ============================================================
# 4. FASTAPI AND CORS
# ============================================================

app = FastAPI(title="AI Instrument Detector - OpenMIC")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "OpenMIC instrument detector backend is running",
        "model": "OpenMIC multi-label CNN",
        "number_of_instruments": len(CLASSES),
    }


# ============================================================
# 5. AUDIO PREPROCESSING
# Must match prepare_openmic.py.
# ============================================================

def audio_to_spectrogram(audio_segment):
    mel = librosa.feature.melspectrogram(
        y=audio_segment,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
        top_db=80,
    )

    # Match OpenMIC preprocessing normalization.
    mel_db = (mel_db + 80.0) / 80.0
    mel_db = np.clip(mel_db, 0.0, 1.0)

    # Expected shape: (1, 1, 128, 216)
    return torch.from_numpy(
        mel_db.astype(np.float32)
    ).unsqueeze(0).unsqueeze(0)


# ============================================================
# 6. PREDICT MULTIPLE INSTRUMENTS
# ============================================================

@torch.inference_mode()
def analyze_audio_bytes(audio_bytes):
    try:
        audio, sr = sf.read(
            io.BytesIO(audio_bytes),
            dtype="float32",
        )
    except (sf.SoundFileError, ValueError) as exc:
        raise ValueError(
            "Unable to decode audio. Please upload a valid WAV, "
            "FLAC, OGG, AIFF, or AIF file."
        ) from exc

    if audio.size == 0:
        raise ValueError("The uploaded audio file is empty.")

    # Convert stereo to mono.
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if audio.ndim != 1:
        raise ValueError("Unsupported audio channel format.")

    # Resample to the training sample rate.
    if sr != SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=SAMPLE_RATE,
        )

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains invalid numerical values.")

    if len(audio) == 0:
        raise ValueError("No usable audio samples were found.")

    duration = len(audio) / SAMPLE_RATE
    segment_samples = SAMPLE_RATE * SEGMENT_SECONDS

    segment_scores = []

    # Analyze the entire recording in 10-second segments.
    for start in range(0, len(audio), segment_samples):
        segment = audio[start:start + segment_samples]

        # Pad the last short segment to 10 seconds.
        if len(segment) < segment_samples:
            segment = np.pad(
                segment,
                (0, segment_samples - len(segment)),
            )

        spectrogram = audio_to_spectrogram(segment)

        logits = model(spectrogram.to(DEVICE))

        # Independent scores for every instrument.
        probabilities = torch.sigmoid(logits)[0].cpu().numpy()

        segment_scores.append(probabilities)

    segment_scores = np.asarray(segment_scores)

    # Average scores across segments for a song-level ranking.
    song_scores = segment_scores.mean(axis=0)

    ranking = np.argsort(song_scores)[::-1]

    # Initial threshold only; tune after proper evaluation.
    threshold = 0.5

    detected = [
        {
            "instrument": CLASSES[int(i)],
            "score": round(float(song_scores[i]), 4),
            "score_percent": round(float(song_scores[i] * 100), 2),
        }
        for i in ranking
        if song_scores[i] >= threshold
    ]

    all_scores = [
        {
            "instrument": CLASSES[int(i)],
            "score": round(float(song_scores[i]), 4),
            "score_percent": round(float(song_scores[i] * 100), 2),
        }
        for i in ranking
    ]

    return {
        "duration_seconds": round(duration, 2),
        "segments_analyzed": len(segment_scores),
        "predicted_instruments": detected,
        "ranked_instruments": all_scores,
        "threshold": threshold,
        "message": "Audio analyzed successfully",
        "model_status": (
            "Experimental: trained for one epoch; "
            "predictions may be inaccurate."
        ),
    }


# ============================================================
# 7. FRONTEND API ENDPOINT
# ============================================================

@app.post("/analyze")
async def analyze_endpoint(audio: UploadFile = File(...)):
    if not audio.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    allowed_extensions = (".wav", ".flac", ".ogg", ".aiff", ".aif")

    if not audio.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail="Please upload WAV, FLAC, OGG, or AIFF audio.",
        )

    try:
        contents = await audio.read(MAX_UPLOAD_SIZE + 1)

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="The uploaded audio file is empty.",
            )

        if len(contents) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Audio file exceeds the 20 MB limit.",
            )

        result = analyze_audio_bytes(contents)
        result["filename"] = audio.filename

        return result

    except HTTPException:
        raise
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process audio: {str(exc)}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc
    finally:
        await audio.close()
