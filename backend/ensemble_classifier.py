"""
Ensemble Classifier: PANNs CNN14 + OpenMIC CNN
=================================================
Implements two-model ensemble learning to boost musical instrument detection:

  Model A — PANNs CNN14 (Large-Scale Pretrained):
    • Trained on 2M+ Google AudioSet clips across 527 classes.
    • 14-layer convolutional backbone, input at 32,000 Hz.
    • Excellent broad coverage, especially on popular instruments.

  Model B — OpenMIC Instrument CNN (Custom Trained):
    • 4-conv-block lightweight CNN, trained on OpenMIC-2018 dataset.
    • Covers 20 specific instrument classes at 22,050 Hz.
    • Higher precision on niche instruments vs. AudioSet's broad taxonomy.

Ensemble Strategy:
  1. Cross-label alignment: map OpenMIC label names to PANNs AudioSet class indices.
  2. Weighted score fusion: for shared classes, blend CNN14 + OpenMIC scores.
  3. Exclusive coverage: OpenMIC-only classes are injected into results with calibrated confidence.
  4. Adaptive weighting: configurable alpha (CNN14) and beta (OpenMIC) blend weights per analysis.

Author: SonusAI Ensemble Engine
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import librosa
import torch
import torch.nn as nn

from panns_inference import AudioTagging, labels as PANNS_LABELS
from .instruments_data import INSTRUMENT_CATALOG, INSTRUMENT_FAMILIES


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

PANNS_SAMPLE_RATE    = 32000  # PANNs CNN14 requirement
OPENMIC_SAMPLE_RATE  = 22050  # OpenMIC CNN requirement
OPENMIC_N_MELS       = 128
OPENMIC_N_FFT        = 2048
OPENMIC_HOP_LENGTH   = 1024
OPENMIC_SEGMENT_SECS = 10     # seconds per segment for OpenMIC

# Temperature scaling for OpenMIC CNN calibration.
# The OpenMIC CNN was trained on a small dataset and can be overconfident.
# Temperature T > 1 softens logits: sigmoid(logit / T) → smaller, better-calibrated probs.
# T = 1.8 was empirically chosen to align OpenMIC output distribution with PANNs.
OPENMIC_TEMPERATURE  = 1.8

OPENMIC_MODEL_PATH = Path(r"C:\Users\krishna pandit\Desktop\Ai instrument detector\Backend\models\openmic_instrument_cnn.pth")

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


# ─────────────────────────────────────────────────────────────
# CROSS-MODEL LABEL ALIGNMENT MAP
# Maps each OpenMIC instrument → list of matching PANNs labels
# (a list because one openmic label may align with multiple audioset classes)
# ─────────────────────────────────────────────────────────────

OPENMIC_TO_PANNS: Dict[str, List[str]] = {
    "accordion":        ["Accordion"],
    "banjo":            ["Banjo"],
    "bass":             ["Bass guitar"],
    "cello":            ["Cello"],
    "clarinet":         ["Clarinet"],
    "cymbals":          ["Cymbal"],
    "drums":            ["Drum kit", "Drum", "Snare drum", "Bass drum", "Drum machine"],
    "flute":            ["Flute"],
    "guitar":           ["Guitar", "Acoustic guitar", "Electric guitar", "Steel guitar, slide guitar"],
    "mallet_percussion":["Marimba, xylophone", "Vibraphone", "Mallet percussion"],
    "mandolin":         ["Mandolin"],
    "organ":            ["Organ", "Hammond organ", "Electronic organ"],
    "piano":            ["Piano", "Electric piano"],
    "saxophone":        ["Saxophone"],
    "synthesizer":      ["Synthesizer"],
    "trombone":         ["Trombone"],
    "trumpet":          ["Trumpet"],
    "ukulele":          ["Ukulele"],
    "violin":           ["Violin, fiddle"],
    "voice":            [],   # Not tracked in AudioSet instrument catalog
}

# Pretty display names for OpenMIC-only / voice label
OPENMIC_DISPLAY_NAMES = {
    "mallet_percussion": "Mallet Percussion (Marimba/Vibraphone)",
    "voice":             "Human Voice / Vocals",
    "bass":              "Bass (Guitar/Double Bass)",
    "drums":             "Drums & Kit",
    "cymbals":           "Cymbals & Hi-Hats",
    "guitar":            "Guitar (General)",
    "organ":             "Organ (Pipe/Hammond/Electronic)",
    "piano":             "Piano (Acoustic/Electric)",
}


# ─────────────────────────────────────────────────────────────
# OPENMIC CNN ARCHITECTURE (exact match to train_openmic.py)
# ─────────────────────────────────────────────────────────────

class OpenMICInstrumentCNN(nn.Module):
    """
    Lightweight 4-block CNN trained on the OpenMIC-2018 dataset.
    Input:  (B, 1, 128, 216) Log-Mel spectrogram (22,050 Hz, 128 bins)
    Output: (B, 20) raw logits → sigmoid → per-instrument probabilities
    """
    def __init__(self, number_of_instruments: int = 20):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────────────────────
# ENSEMBLE CLASSIFIER
# ─────────────────────────────────────────────────────────────

class EnsembleInstrumentClassifier:
    """
    Dual-model ensemble that fuses PANNs CNN14 and OpenMIC CNN outputs
    for improved musical instrument detection accuracy.
    """
    _instance: Optional["EnsembleInstrumentClassifier"] = None

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ── Load PANNs CNN14 ──────────────────────────────────────────────
        panns_ckpt = str(Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth")
        print(f"[Ensemble] Loading PANNs CNN14 on {self.device.upper()} …")
        self.panns_model = AudioTagging(checkpoint_path=panns_ckpt, device=self.device)
        self.panns_labels = PANNS_LABELS
        self.panns_label_to_idx = {lb: i for i, lb in enumerate(PANNS_LABELS)}

        # ── Load OpenMIC CNN ──────────────────────────────────────────────
        if not OPENMIC_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"[Ensemble] OpenMIC CNN checkpoint not found: {OPENMIC_MODEL_PATH}\n"
                "Please ensure the model file exists."
            )
        print(f"[Ensemble] Loading OpenMIC Instrument CNN from {OPENMIC_MODEL_PATH} …")
        ckpt = torch.load(str(OPENMIC_MODEL_PATH), map_location=self.device, weights_only=True)
        self.openmic_instruments: List[str] = ckpt["instruments"]
        self.openmic_model = OpenMICInstrumentCNN(number_of_instruments=len(self.openmic_instruments))
        self.openmic_model.load_state_dict(ckpt["model_state_dict"])
        self.openmic_model.to(self.device)
        self.openmic_model.eval()
        print(f"[Ensemble] OpenMIC CNN loaded — {len(self.openmic_instruments)} instrument classes.")

        # Build index lookup for OpenMIC
        self.openmic_inst_to_idx = {name: i for i, name in enumerate(self.openmic_instruments)}

        # Build PANNs instrument index set (from catalog)
        self.panns_instrument_indices: Dict[str, int] = {}
        for inst_key in INSTRUMENT_CATALOG:
            if inst_key in self.panns_label_to_idx:
                self.panns_instrument_indices[inst_key] = self.panns_label_to_idx[inst_key]

        # Pre-compute cross-model alignment index pairs
        # { openmic_label: [panns_class_index, …] }
        self._alignment: Dict[str, List[int]] = {}
        for om_label, panns_list in OPENMIC_TO_PANNS.items():
            indices = [self.panns_label_to_idx[p] for p in panns_list if p in self.panns_label_to_idx]
            self._alignment[om_label] = indices

        print(
            f"[Ensemble] Initialized: CNN14 ({len(self.panns_instrument_indices)} tracked classes) "
            f"+ OpenMIC CNN ({len(self.openmic_instruments)} classes). "
            f"Ensemble ready on {self.device.upper()}."
        )

    # ── Singleton access ──────────────────────────────────────────────────
    @classmethod
    def get_instance(cls) -> "EnsembleInstrumentClassifier":
        if cls._instance is None:
            cls._instance = EnsembleInstrumentClassifier()
        return cls._instance

    # ── OpenMIC spectrogram extraction ────────────────────────────────────
    @staticmethod
    def _audio_to_openmic_spectrogram(segment: np.ndarray) -> torch.Tensor:
        """
        Convert a 10-second audio segment (at 22,050 Hz) into the exact
        log-mel spectrogram shape expected by the OpenMIC CNN: (1, 1, 128, 216).
        """
        mel = librosa.feature.melspectrogram(
            y=segment,
            sr=OPENMIC_SAMPLE_RATE,
            n_fft=OPENMIC_N_FFT,
            hop_length=OPENMIC_HOP_LENGTH,
            n_mels=OPENMIC_N_MELS,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80)
        mel_db = (mel_db + 80.0) / 80.0   # normalize to [0, 1]
        mel_db = np.clip(mel_db, 0.0, 1.0)
        return torch.from_numpy(mel_db.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    # ── Run OpenMIC CNN on full audio ─────────────────────────────────────
    def _run_openmic(self, audio_at_22050: np.ndarray) -> np.ndarray:
        """
        Segments audio into 10-second windows, runs OpenMIC CNN on each,
        applies temperature calibration to logits (prevents overconfidence),
        and returns blended probability vector of shape (20,).
        """
        segment_samples = OPENMIC_SAMPLE_RATE * OPENMIC_SEGMENT_SECS
        all_probs: List[np.ndarray] = []

        for start in range(0, len(audio_at_22050), segment_samples):
            seg = audio_at_22050[start : start + segment_samples]
            if len(seg) < segment_samples:
                seg = np.pad(seg, (0, segment_samples - len(seg)))

            spec = self._audio_to_openmic_spectrogram(seg).to(self.device)
            with torch.inference_mode():
                logits = self.openmic_model(spec)

            # ── Temperature calibration ────────────────────────────────────
            # Dividing raw logits by T softens overconfident predictions.
            # This aligns the OpenMIC probability range with PANNs CNN14 scale.
            calibrated_logits = logits / OPENMIC_TEMPERATURE
            probs = torch.sigmoid(calibrated_logits)[0].cpu().numpy()
            all_probs.append(probs)

        # Max-pool across time (catches momentary solos), blend with mean
        stacked = np.array(all_probs)
        max_p   = stacked.max(axis=0)
        mean_p  = stacked.mean(axis=0)
        return 0.6 * max_p + 0.4 * mean_p   # matches PANNs blending logic

    # ── Key estimation ────────────────────────────────────────────────────
    def _estimate_key(self, audio: np.ndarray, sr: int) -> str:
        try:
            clip = audio[:30 * sr] if len(audio) > 30 * sr else audio
            chroma = librosa.feature.chroma_stft(y=clip, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            best_corr, best_key = -1.0, "C Major"
            for i in range(12):
                rolled = np.roll(chroma_mean, -i)
                for profile, mode in [(MAJOR_PROFILE, "Major"), (MINOR_PROFILE, "Minor")]:
                    corr = float(np.corrcoef(rolled, profile)[0, 1])
                    if corr > best_corr:
                        best_corr = corr
                        best_key = f"{NOTE_NAMES[i]} {mode}"
            return best_key
        except Exception:
            return "Undetermined"

    # ── Waveform peaks ────────────────────────────────────────────────────
    @staticmethod
    def _waveform_peaks(audio: np.ndarray, n: int = 120) -> List[float]:
        step = len(audio) / n
        peaks = []
        for i in range(n):
            chunk = audio[int(i * step) : int((i + 1) * step)]
            peaks.append(float(np.max(np.abs(chunk))) if len(chunk) > 0 else 0.0)
        max_p = max(peaks) or 1.0
        return [round(p / max_p, 3) for p in peaks]

    # ── Confidence scaling ────────────────────────────────────────────────
    @staticmethod
    def _scale_confidence(raw: float) -> float:
        """Non-linear scaling: 0.15 → ~45%, 0.35 → ~66%, 0.6 → ~77%."""
        return round((raw / (raw + 0.18)) * 100, 1)

    # ── Core ensemble fusion ──────────────────────────────────────────────
    def _fuse_scores(
        self,
        panns_blended: np.ndarray,
        openmic_blended: np.ndarray,
        alpha: float = 0.60,   # weight for PANNs CNN14
        beta: float = 0.40,    # weight for OpenMIC CNN
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Returns three dicts (panns_scores, openmic_scores, fused_scores)
        keyed by AudioSet label / synthetic key for OpenMIC-only classes.
        All values are in [0, 1] raw probability space.
        """
        panns_scores: Dict[str, float]  = {}
        openmic_scores: Dict[str, float] = {}
        fused_scores: Dict[str, float]  = {}

        # A. PANNs scores for all tracked AudioSet instrument classes
        for label, idx in self.panns_instrument_indices.items():
            panns_scores[label] = float(panns_blended[idx])

        # B. For each OpenMIC class, find aligned PANNs indices
        for om_label, probs_idx in self.openmic_inst_to_idx.items():
            om_prob = float(openmic_blended[probs_idx])
            openmic_scores[om_label] = om_prob

            aligned_panns_keys = [
                p for p in OPENMIC_TO_PANNS.get(om_label, [])
                if p in self.panns_label_to_idx
            ]

            if aligned_panns_keys:
                # Take the max PANNs score among aligned classes
                best_panns = max(panns_scores.get(k, 0.0) for k in aligned_panns_keys)

                # Fuse all aligned PANNs labels with OpenMIC score
                for panns_key in aligned_panns_keys:
                    p_score = panns_scores.get(panns_key, 0.0)
                    fused = alpha * p_score + beta * om_prob
                    fused_scores[panns_key] = max(fused_scores.get(panns_key, 0.0), fused)

            else:
                # OpenMIC-exclusive class (e.g., "voice", "mallet_percussion" edge)
                synthetic_key = f"__openmic__{om_label}"
                fused_scores[synthetic_key] = beta * om_prob  # Only OpenMIC has a signal

        # C. PANNs-only classes that have no OpenMIC alignment: keep PANNs score as-is
        for label, p_score in panns_scores.items():
            if label not in fused_scores:
                fused_scores[label] = alpha * p_score

        return panns_scores, openmic_scores, fused_scores

    # ── Main analysis entry point ─────────────────────────────────────────
    def analyze_audio_file(
        self,
        file_path: str,
        cnn14_weight: float = 0.60,
        openmic_weight: float = 0.40,
    ) -> Dict[str, Any]:
        """
        Full-song ensemble analysis pipeline:
          1. Load & resample audio for both models.
          2. Run PANNs CNN14 windowed inference (10s/5s).
          3. Run OpenMIC CNN segment inference (10s).
          4. Fuse probabilities via weighted ensemble.
          5. Enrich results with catalog metadata, timeline, key, BPM.
        """
        # Normalise weights to sum to 1.0
        total_w = cnn14_weight + openmic_weight
        alpha = cnn14_weight / total_w
        beta  = openmic_weight / total_w

        # ── 1. Load audio at both sample rates ───────────────────────────
        audio_32k, _ = librosa.load(file_path, sr=PANNS_SAMPLE_RATE,   mono=True)
        audio_22k, _ = librosa.load(file_path, sr=OPENMIC_SAMPLE_RATE, mono=True)

        total_duration = len(audio_32k) / PANNS_SAMPLE_RATE
        if total_duration < 0.5:
            raise ValueError("Audio clip is too short (minimum 0.5 seconds).")

        # ── 2. Audio features (waveform, key, BPM) ───────────────────────
        waveform_peaks = self._waveform_peaks(audio_32k)

        try:
            tempo_res, _ = librosa.beat.beat_track(y=audio_32k, sr=PANNS_SAMPLE_RATE)
            bpm = round(float(tempo_res[0]) if hasattr(tempo_res, "__len__") else float(tempo_res), 1)
        except Exception:
            bpm = 120.0

        key_signature = self._estimate_key(audio_32k, PANNS_SAMPLE_RATE)

        # ── 3. PANNs CNN14 windowed inference ────────────────────────────
        window_size = int(10.0 * PANNS_SAMPLE_RATE)
        hop_size    = int(5.0  * PANNS_SAMPLE_RATE)

        window_clips: List[np.ndarray] = []
        timestamps:   List[Tuple[float, float]] = []

        if len(audio_32k) <= window_size:
            padded = np.pad(audio_32k, (0, window_size - len(audio_32k)))
            window_clips.append(padded)
            timestamps.append((0.0, round(total_duration, 2)))
        else:
            for s in range(0, len(audio_32k) - hop_size, hop_size):
                e = min(s + window_size, len(audio_32k))
                chunk = audio_32k[s:e]
                if len(chunk) < window_size:
                    chunk = np.pad(chunk, (0, window_size - len(chunk)))
                window_clips.append(chunk)
                timestamps.append((round(s / PANNS_SAMPLE_RATE, 1), round(e / PANNS_SAMPLE_RATE, 1)))
                if e >= len(audio_32k):
                    break

        window_clips_np = np.array(window_clips, dtype=np.float32)
        with torch.no_grad():
            panns_clipwise, _ = self.panns_model.inference(window_clips_np)  # (W, 527)

        panns_max    = panns_clipwise.max(axis=0)
        panns_mean   = panns_clipwise.mean(axis=0)
        panns_blended = 0.6 * panns_max + 0.4 * panns_mean   # (527,)

        # ── 4. OpenMIC CNN inference ──────────────────────────────────────
        openmic_blended = self._run_openmic(audio_22k)        # (20,)

        # ── 5. Ensemble fusion ────────────────────────────────────────────
        panns_scores, openmic_scores, fused_scores = self._fuse_scores(
            panns_blended, openmic_blended, alpha=alpha, beta=beta
        )

        # ── 6. Build detected instruments list ───────────────────────────
        detected_candidates: List[Dict[str, Any]] = []
        seen_openmic_covered: set = set()  # avoid double-counting openmic after panns fusion

        for label, fused_raw in fused_scores.items():
            is_openmic_only = label.startswith("__openmic__")
            om_label = label.replace("__openmic__", "") if is_openmic_only else None

            # Source signal breakdown for display
            if is_openmic_only:
                p_raw  = 0.0
                om_raw = float(openmic_blended[self.openmic_inst_to_idx[om_label]])
            else:
                p_raw  = float(panns_blended[self.panns_label_to_idx[label]]) if label in self.panns_label_to_idx else 0.0
                # Find max OpenMIC score among aligned OpenMIC labels
                om_raw = 0.0
                for om_lbl, idx in self.openmic_inst_to_idx.items():
                    if label in [p for p in OPENMIC_TO_PANNS.get(om_lbl, []) if p in self.panns_label_to_idx]:
                        om_raw = max(om_raw, float(openmic_blended[idx]))

            # Skip very low confidence signals
            if fused_raw < 0.04 and max(p_raw, om_raw) < 0.08:
                continue

            # Confidence in user-friendly %
            confidence = self._scale_confidence(fused_raw)

            # Get catalog metadata
            if is_openmic_only:
                # Synthetic OpenMIC-only entry
                display_name = OPENMIC_DISPLAY_NAMES.get(om_label, om_label.replace("_", " ").title())
                meta = {
                    "name": display_name,
                    "family": "world_traditional" if om_label not in ["voice"] else "world_traditional",
                    "role": "Instrument detected by OpenMIC CNN",
                    "freq_range": "Variable",
                    "description": f"Detected by the OpenMIC-trained CNN. Category: {om_label.replace('_', ' ')}.",
                    "characteristics": ["OpenMIC trained", "Custom CNN"],
                    "fun_fact": "Classified by a CNN trained on the OpenMIC-2018 acoustic instrument dataset.",
                    "icon": "Music",
                }
                effective_label = om_label
                family_key = "world_traditional"
            else:
                meta = INSTRUMENT_CATALOG.get(label, None)
                if meta is None:
                    continue
                effective_label = label
                family_key = meta["family"]

            # Intensity tier
            if confidence >= 65:
                intensity, intensity_color = "Dominant / Lead", "emerald"
            elif confidence >= 40:
                intensity, intensity_color = "Prominent", "blue"
            else:
                intensity, intensity_color = "Subtle / Ambient", "slate"

            # Timeline presence from PANNs windowed output
            timeline_presence: List[Dict] = []
            if not is_openmic_only and label in self.panns_label_to_idx:
                pidx = self.panns_label_to_idx[label]
                for wi, (t_start, t_end) in enumerate(timestamps):
                    w_score = float(panns_clipwise[wi, pidx])
                    w_conf  = self._scale_confidence(w_score)
                    if w_score >= 0.03:
                        timeline_presence.append({"start": t_start, "end": t_end, "confidence": w_conf, "raw_score": round(w_score, 4)})

            detected_candidates.append({
                "id":                   effective_label.lower().replace(" ", "_").replace(",", "").replace("/", "_"),
                "label":                effective_label,
                "name":                 meta["name"],
                "family":               family_key,
                "family_name":          INSTRUMENT_FAMILIES[family_key]["name"],
                "family_color":         INSTRUMENT_FAMILIES[family_key]["color"],
                "role":                 meta["role"],
                "freq_range":           meta["freq_range"],
                "description":          meta["description"],
                "characteristics":      meta["characteristics"],
                "fun_fact":             meta["fun_fact"],
                "icon":                 meta["icon"],
                "confidence":           confidence,
                "raw_score":            round(fused_raw, 4),
                "peak_score":           round(max(p_raw, om_raw), 4),
                # Ensemble signal breakdown
                "cnn14_score":          round(p_raw, 4),
                "openmic_score":        round(om_raw, 4),
                "cnn14_confidence":     self._scale_confidence(p_raw),
                "openmic_confidence":   self._scale_confidence(om_raw),
                "ensemble_weights":     {"cnn14": round(alpha, 2), "openmic": round(beta, 2)},
                "source":               "ensemble" if p_raw > 0.03 and om_raw > 0.03 else ("cnn14" if p_raw > om_raw else "openmic"),
                "intensity":            intensity,
                "intensity_color":      intensity_color,
                "active_segments_count":len(timeline_presence),
                "timeline":             timeline_presence,
            })

        # Sort by ensemble confidence descending
        detected_candidates.sort(key=lambda x: x["confidence"], reverse=True)

        # ── 7. Family summary ─────────────────────────────────────────────
        family_summary: Dict[str, Any] = {}
        for fam_key, fam_meta in INSTRUMENT_FAMILIES.items():
            matching = [d for d in detected_candidates if d["family"] == fam_key]
            if matching:
                avg_conf = sum(d["confidence"] for d in matching) / len(matching)
                family_summary[fam_key] = {
                    "family":            fam_key,
                    "name":              fam_meta["name"],
                    "color":             fam_meta["color"],
                    "bg_color":          fam_meta["bg_color"],
                    "border_color":      fam_meta["border_color"],
                    "text_color":        fam_meta["text_color"],
                    "badge_color":       fam_meta["badge_color"],
                    "icon":              fam_meta["icon"],
                    "description":       fam_meta["description"],
                    "count":             len(matching),
                    "instruments":       [d["name"] for d in matching],
                    "average_confidence":round(avg_conf, 1),
                }

        # ── 8. AI insight tags ────────────────────────────────────────────
        song_tags: List[str] = []
        if detected_candidates:
            top3 = [d["name"] for d in detected_candidates[:3]]
            song_tags.append(f"Featuring {', '.join(top3)}")
        if bpm > 0:
            tempo_desc = "Slow / Ballad" if bpm < 85 else "Moderate Groove" if bpm < 125 else "Uptempo / Energetic"
            song_tags.append(f"{tempo_desc} ({bpm} BPM)")
        if key_signature != "Undetermined":
            song_tags.append(f"Key of {key_signature}")
        song_tags.append(f"Ensemble: CNN14 {round(alpha * 100)}% + OpenMIC {round(beta * 100)}%")

        # ── 9. Return full response payload ──────────────────────────────
        return {
            "summary": {
                "total_instruments_detected":  len(detected_candidates),
                "duration_seconds":            round(total_duration, 2),
                "duration_formatted":          f"{int(total_duration//60)}:{int(total_duration%60):02d}",
                "tempo_bpm":                   bpm,
                "key_signature":               key_signature,
                "dominant_family":             max(family_summary.values(), key=lambda f: f["count"])["name"] if family_summary else "None",
                "sample_rate":                 PANNS_SAMPLE_RATE,
                "analysis_windows_count":      len(window_clips),
                "ensemble_cnn14_weight":       round(alpha, 2),
                "ensemble_openmic_weight":     round(beta, 2),
            },
            "waveform":              waveform_peaks,
            "detected_instruments":  detected_candidates,
            "families":              list(family_summary.values()),
            "tags":                  song_tags,
            "ensemble_info": {
                "models": [
                    {
                        "name":          "PANNs CNN14",
                        "dataset":       "Google AudioSet (2M+ clips, 527 classes)",
                        "weight":        round(alpha, 2),
                        "sample_rate":   PANNS_SAMPLE_RATE,
                        "tracked_classes": len(self.panns_instrument_indices),
                    },
                    {
                        "name":          "OpenMIC Instrument CNN",
                        "dataset":       "OpenMIC-2018 (20 instrument classes)",
                        "weight":        round(beta, 2),
                        "sample_rate":   OPENMIC_SAMPLE_RATE,
                        "tracked_classes": len(self.openmic_instruments),
                        "instruments":   self.openmic_instruments,
                    },
                ],
                "strategy":        "Weighted Score Fusion (max+mean blended) with cross-label alignment",
                "alignment_pairs": len([v for v in OPENMIC_TO_PANNS.values() if v]),
            },
            "timeline_segments": [
                {
                    "window_index": i,
                    "start":        t_start,
                    "end":          t_end,
                    "active_instruments": [
                        {
                            "name":       d["name"],
                            "confidence": next((p["confidence"] for p in d["timeline"] if p["start"] == t_start), 0.0)
                        }
                        for d in detected_candidates
                        if any(p["start"] == t_start for p in d["timeline"])
                    ][:4]
                }
                for i, (t_start, t_end) in enumerate(timestamps)
            ]
        }
