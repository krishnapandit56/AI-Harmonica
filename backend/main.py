"""
FastAPI Backend — SonusAI Musical Instrument Classifier
Powered by PANNs CNN14 + OpenMIC CNN Ensemble
"""

import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from .ensemble_classifier import EnsembleInstrumentClassifier
from .instruments_data import INSTRUMENT_CATALOG, INSTRUMENT_FAMILIES

app = FastAPI(
    title="SonusAI — Musical Instrument Classifier API",
    description=(
        "Identifies musical instruments in audio songs using an ensemble of "
        "PANNs CNN14 (AudioSet/Google) and a custom-trained OpenMIC CNN."
    ),
    version="2.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Demo audio directory ──────────────────────────────────────────────────────
DEMO_DIR = Path(__file__).parent / "demo_audio"
DEMO_DIR.mkdir(exist_ok=True)


def generate_sample_demo_tracks():
    """Generate synthetic demo audio files for instant testing."""
    sr = 32000
    duration = 8.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Demo 1: Acoustic Piano chords
    piano_path = DEMO_DIR / "piano_keys_demo.wav"
    if not piano_path.exists():
        chord_c  = [261.63, 329.63, 392.00, 523.25]
        chord_am = [220.00, 261.63, 329.63, 440.00]
        sig = np.zeros_like(t)
        t1 = t[t < 4.0]
        e1 = np.exp(-1.5 * t1)
        for f in chord_c:
            sig[:len(t1)] += 0.25 * e1 * (np.sin(2*np.pi*f*t1) + 0.5*np.sin(4*np.pi*f*t1) + 0.2*np.sin(6*np.pi*f*t1))
        t2 = t[t >= 4.0] - 4.0
        e2 = np.exp(-1.5 * t2)
        off = len(t1)
        for f in chord_am:
            sig[off:off+len(t2)] += 0.25 * e2 * (np.sin(2*np.pi*f*t2) + 0.5*np.sin(4*np.pi*f*t2) + 0.2*np.sin(6*np.pi*f*t2))
        sig = sig / (np.max(np.abs(sig)) + 1e-6) * 0.8
        sf.write(str(piano_path), sig.astype(np.float32), sr)

    # Demo 2: Plucked guitar + bass
    guitar_path = DEMO_DIR / "guitar_bass_demo.wav"
    if not guitar_path.exists():
        sig = np.zeros_like(t)
        bass_f = 55.0
        env_b = np.exp(-0.8 * t)
        sig += 0.45 * env_b * (np.sin(2*np.pi*bass_f*t) + 0.7*np.sin(4*np.pi*bass_f*t))
        notes = [110.0, 164.81, 220.0, 277.18, 329.63, 440.0, 554.37, 659.25]
        for idx, start_sec in enumerate(np.arange(0, duration, 0.5)):
            freq = notes[idx % len(notes)]
            sub_t = t[t >= start_sec] - start_sec
            sub_t = sub_t[sub_t < 1.2]
            s_idx = int(start_sec * sr)
            e_idx = s_idx + len(sub_t)
            env_p = np.exp(-4.5 * sub_t)
            sig[s_idx:e_idx] += 0.22 * env_p * (np.sin(2*np.pi*freq*sub_t) + 0.6*np.sin(4*np.pi*freq*sub_t) + 0.3*np.sin(6*np.pi*freq*sub_t))
        sig = sig / (np.max(np.abs(sig)) + 1e-6) * 0.85
        sf.write(str(guitar_path), sig.astype(np.float32), sr)


try:
    generate_sample_demo_tracks()
except Exception as e:
    print(f"[Demo] Generation notice: {e}")


# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    try:
        clf = EnsembleInstrumentClassifier.get_instance()
        print(f"[Startup] Ensemble classifier ready on {clf.device.upper()}.")
    except Exception as e:
        print(f"[Startup] Ensemble init error: {e}")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    clf = EnsembleInstrumentClassifier.get_instance()
    return {
        "status": "healthy",
        "version": "2.0.0 (Ensemble)",
        "models": {
            "cnn14": {
                "name":    "PANNs CNN14",
                "device":  clf.device,
                "classes": len(clf.panns_instrument_indices),
            },
            "openmic": {
                "name":       "OpenMIC Instrument CNN",
                "classes":    len(clf.openmic_instruments),
                "instruments": clf.openmic_instruments,
            },
        },
        "ensemble": {
            "strategy": "Weighted Score Fusion",
            "default_weights": {"cnn14": 0.60, "openmic": 0.40},
        },
        "sample_rate_cnn14":   32000,
        "sample_rate_openmic": 22050,
    }


# ── Instrument Catalog ─────────────────────────────────────────────────────────
@app.get("/api/catalog")
async def get_instrument_catalog():
    clf = EnsembleInstrumentClassifier.get_instance()
    catalog_by_family = {}
    for fam_key, fam_data in INSTRUMENT_FAMILIES.items():
        catalog_by_family[fam_key] = {**fam_data, "instruments": []}

    for key, inst_meta in INSTRUMENT_CATALOG.items():
        fam = inst_meta["family"]
        if fam in catalog_by_family:
            catalog_by_family[fam]["instruments"].append({
                "label":             key,
                "name":              inst_meta["name"],
                "role":              inst_meta["role"],
                "freq_range":        inst_meta["freq_range"],
                "description":       inst_meta["description"],
                "characteristics":   inst_meta["characteristics"],
                "fun_fact":          inst_meta["fun_fact"],
                "icon":              inst_meta["icon"],
                "audioset_supported": key in clf.panns_label_to_idx,
                "openmic_supported":  any(
                    key in [p for p in INSTRUMENT_CATALOG.keys() if p in clf.openmic_inst_to_idx]
                ),
            })

    return {
        "families":           list(INSTRUMENT_FAMILIES.values()),
        "catalog_by_family":  catalog_by_family,
        "total_supported":    len(INSTRUMENT_CATALOG),
        "openmic_instruments": clf.openmic_instruments,
    }


# ── Demo listing & streaming ───────────────────────────────────────────────────
@app.get("/api/demos")
async def list_demos():
    return [
        {
            "id":              "piano_keys_demo",
            "name":            "Acoustic Piano & Harmonic Chords",
            "description":     "Rich piano harmony progression with natural acoustic body decay.",
            "duration":        "8s",
            "expected_family": "Keyboards & Synths",
        },
        {
            "id":              "guitar_bass_demo",
            "name":            "Plucked Acoustic Guitar & Bass",
            "description":     "Dynamic plucked string arpeggios over a low-frequency bass foundation.",
            "duration":        "8s",
            "expected_family": "Strings & Plucked",
        },
    ]


@app.get("/api/demos/{demo_id}")
async def get_demo_audio(demo_id: str):
    fp = DEMO_DIR / f"{demo_id}.wav"
    if not fp.exists():
        raise HTTPException(status_code=404, detail="Demo track not found.")
    return FileResponse(fp, media_type="audio/wav", filename=f"{demo_id}.wav")


@app.post("/api/analyze-demo/{demo_id}")
async def analyze_demo(
    demo_id: str,
    cnn14_weight:   float = Query(default=0.60, ge=0.0, le=1.0, description="Weight for PANNs CNN14 (0–1)"),
    openmic_weight: float = Query(default=0.40, ge=0.0, le=1.0, description="Weight for OpenMIC CNN (0–1)"),
):
    fp = DEMO_DIR / f"{demo_id}.wav"
    if not fp.exists():
        raise HTTPException(status_code=404, detail="Demo track not found.")
    clf = EnsembleInstrumentClassifier.get_instance()
    try:
        result = clf.analyze_audio_file(str(fp), cnn14_weight=cnn14_weight, openmic_weight=openmic_weight)
        result["file_name"] = f"{demo_id}.wav"
        return result
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis error: {e}")


# ── Main upload & analyze endpoint ─────────────────────────────────────────────
@app.post("/api/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    cnn14_weight:   float = Query(default=0.60, ge=0.0, le=1.0, description="Weight for PANNs CNN14 (0–1)"),
    openmic_weight: float = Query(default=0.40, ge=0.0, le=1.0, description="Weight for OpenMIC CNN (0–1)"),
):
    """
    Upload an audio song (MP3/WAV/FLAC/OGG/M4A/AAC) and classify every
    musical instrument using the PANNs CNN14 + OpenMIC CNN ensemble.

    Query params:
      - cnn14_weight   (default 0.60): how much to trust PANNs CNN14
      - openmic_weight (default 0.40): how much to trust the OpenMIC CNN
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    valid_ext = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".webm", ".wma"}
    file_ext  = Path(file.filename).suffix.lower()
    if file_ext not in valid_ext and not (file.content_type or "").startswith("audio/"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{file_ext}'. Upload MP3, WAV, FLAC, OGG, M4A or AAC.",
        )

    tmp_path = os.path.join(tempfile.gettempdir(), f"upload_{os.getpid()}_{file.filename}")
    try:
        with open(tmp_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        clf = EnsembleInstrumentClassifier.get_instance()
        result = clf.analyze_audio_file(tmp_path, cnn14_weight=cnn14_weight, openmic_weight=openmic_weight)
        result["file_name"]       = file.filename
        result["file_size_bytes"] = os.path.getsize(tmp_path)
        return JSONResponse(content=result)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ensemble inference error: {e}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
