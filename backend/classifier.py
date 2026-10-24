"""
PANNs CNN14 Audio Classifier Engine.
Performs audio analysis, instrument tagging, temporal timeline extraction,
and musical feature detection (tempo, key, waveform).
"""

import os
from pathlib import Path
import numpy as np
import librosa
import torch
from typing import Dict, Any, List, Optional
import math

from panns_inference import AudioTagging, labels
from instruments_data import INSTRUMENT_CATALOG, INSTRUMENT_FAMILIES

# Notes for key estimation
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Major and minor chord profiles for key estimation (Krumhansl-Schmuckler)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


class InstrumentClassifier:
    _instance: Optional["InstrumentClassifier"] = None

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 32000
        self.window_duration = 10.0  # 10s PANNs window
        self.hop_duration = 5.0     # 5s hop for timeline analysis
        
        checkpoint_path = str(Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth")
        print(f"Loading PANNs CNN14 on {self.device.upper()} from {checkpoint_path}...")
        
        self.model = AudioTagging(checkpoint_path=checkpoint_path, device=self.device)
        self.labels = labels
        self.label_to_idx = {lb: i for i, lb in enumerate(labels)}
        
        # Build index mapping for our recognized musical instruments
        self.instrument_indices = {}
        for inst_key in INSTRUMENT_CATALOG.keys():
            if inst_key in self.label_to_idx:
                self.instrument_indices[inst_key] = self.label_to_idx[inst_key]
        
        print(f"PANNs CNN14 initialized! Tracking {len(self.instrument_indices)} distinct musical instrument classes.")

    @classmethod
    def get_instance(cls) -> "InstrumentClassifier":
        if cls._instance is None:
            cls._instance = InstrumentClassifier()
        return cls._instance

    def _estimate_key(self, audio: np.ndarray, sr: int) -> str:
        """Estimate musical tonality/key using chromagram correlation."""
        try:
            # Subsample audio if longer than 30s for fast key estimation
            max_samples = 30 * sr
            clip = audio[:max_samples] if len(audio) > max_samples else audio
            
            chroma = librosa.feature.chroma_stft(y=clip, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            
            best_corr = -1
            best_key = "C Major"
            
            for i in range(12):
                # Roll chroma to test each root note
                rolled = np.roll(chroma_mean, -i)
                
                # Pearson correlation with major and minor profiles
                corr_maj = np.corrcoef(rolled, MAJOR_PROFILE)[0, 1]
                corr_min = np.corrcoef(rolled, MINOR_PROFILE)[0, 1]
                
                if corr_maj > best_corr:
                    best_corr = corr_maj
                    best_key = f"{NOTE_NAMES[i]} Major"
                if corr_min > best_corr:
                    best_corr = corr_min
                    best_key = f"{NOTE_NAMES[i]} Minor"
                    
            return best_key
        except Exception:
            return "Undetermined"

    def _compute_waveform_peaks(self, audio: np.ndarray, num_points: int = 150) -> List[float]:
        """Extract a downsampled normalized waveform envelope for UI rendering."""
        try:
            step = len(audio) / num_points
            peaks = []
            for i in range(num_points):
                start = int(i * step)
                end = int((i + 1) * step)
                chunk = audio[start:end]
                if len(chunk) > 0:
                    val = float(np.max(np.abs(chunk)))
                    peaks.append(round(min(1.0, val), 3))
                else:
                    peaks.append(0.0)
            
            # Normalize to 0.1 - 1.0 range for attractive visual wave
            max_p = max(peaks) if peaks and max(peaks) > 0 else 1.0
            return [round(p / max_p, 3) for p in peaks]
        except Exception:
            return [0.5] * num_points

    def analyze_audio_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze audio song:
        1. Resamples to 32kHz mono.
        2. Computes temporal slices with PANNs CNN14.
        3. Identifies detected musical instruments with confidence scores.
        4. Calculates timeline distribution, tempo, key, dynamic energy, and waveform.
        """
        # Load audio at 32kHz
        audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        total_duration = len(audio) / sr
        
        # Audio length validation
        if total_duration < 0.5:
            raise ValueError("Audio clip is too short for analysis (minimum 0.5 seconds required).")

        # Musical characteristics
        waveform_peaks = self._compute_waveform_peaks(audio, num_points=120)
        
        # Tempo / BPM estimation
        try:
            tempo_result, _ = librosa.beat.beat_track(y=audio, sr=sr)
            if hasattr(tempo_result, "__len__"):
                bpm = float(tempo_result[0])
            else:
                bpm = float(tempo_result)
            bpm = round(bpm, 1)
        except Exception:
            bpm = 120.0

        key_signature = self._estimate_key(audio, sr)

        # Slice audio into 10s windows with 5s hop for temporal analysis
        window_size = int(self.window_duration * sr)
        hop_size = int(self.hop_duration * sr)
        
        window_clips = []
        timestamps = []
        
        if len(audio) <= window_size:
            # Pad audio to at least 10s if short, for PANNs optimal reception
            pad_len = window_size - len(audio)
            padded = np.pad(audio, (0, pad_len), mode="constant")
            window_clips.append(padded)
            timestamps.append((0.0, round(total_duration, 2)))
        else:
            for start_idx in range(0, len(audio) - hop_size, hop_size):
                end_idx = min(start_idx + window_size, len(audio))
                chunk = audio[start_idx:end_idx]
                if len(chunk) < window_size:
                    chunk = np.pad(chunk, (0, window_size - len(chunk)), mode="constant")
                window_clips.append(chunk)
                t_start = round(start_idx / sr, 1)
                t_end = round(min(start_idx + window_size, len(audio)) / sr, 1)
                timestamps.append((t_start, t_end))
                if end_idx >= len(audio):
                    break

        # Batch inference across windows
        window_clips_np = np.array(window_clips, dtype=np.float32)
        
        # Run PANNs CNN14 inference
        # clipwise_output: shape (num_windows, 527)
        with torch.no_grad():
            clipwise_output, _ = self.model.inference(window_clips_np)
        
        # Aggregation:
        # Max-pooling across time highlights instruments even if played during solos or brief sections.
        # Mean-pooling highlights instruments that are consistently present throughout the song.
        # Blend: 0.6 * max + 0.4 * mean gives the best balance for music songs!
        max_scores = np.max(clipwise_output, axis=0)
        mean_scores = np.mean(clipwise_output, axis=0)
        blended_scores = 0.6 * max_scores + 0.4 * mean_scores

        # Extract scores for instrument catalog
        detected_candidates = []
        
        for inst_key, meta in INSTRUMENT_CATALOG.items():
            idx = self.instrument_indices.get(inst_key)
            if idx is None:
                continue
                
            raw_score = float(blended_scores[idx])
            peak_score = float(max_scores[idx])
            
            # Map raw sigmoid score to an intuitive confidence percentage
            # Sigmoid values in AudioSet for active presence typically range from 0.05 to 0.7+
            # We use non-linear scaling: confidence = (score / (score + 0.15)) * 100
            # That way 0.15 -> 50%, 0.35 -> 70%, 0.6 -> 80%+
            scaled_confidence = (raw_score / (raw_score + 0.18)) * 100
            
            # Minimum threshold for inclusion
            if raw_score >= 0.04 or peak_score >= 0.09:
                # Calculate timeline presence for this instrument
                timeline_presence = []
                for win_idx, (t_start, t_end) in enumerate(timestamps):
                    w_score = float(clipwise_output[win_idx, idx])
                    w_conf = round((w_score / (w_score + 0.18)) * 100, 1)
                    if w_score >= 0.035:
                        timeline_presence.append({
                            "start": t_start,
                            "end": t_end,
                            "confidence": w_conf,
                            "raw_score": round(w_score, 4)
                        })
                
                # Presence intensity tier
                if scaled_confidence >= 65:
                    intensity = "Dominant / Lead"
                    intensity_color = "emerald"
                elif scaled_confidence >= 40:
                    intensity = "Prominent"
                    intensity_color = "blue"
                else:
                    intensity = "Subtle / Ambient"
                    intensity_color = "slate"

                detected_candidates.append({
                    "id": inst_key.lower().replace(" ", "_").replace(",", "").replace("/", "_"),
                    "label": inst_key,
                    "name": meta["name"],
                    "family": meta["family"],
                    "family_name": INSTRUMENT_FAMILIES[meta["family"]]["name"],
                    "family_color": INSTRUMENT_FAMILIES[meta["family"]]["color"],
                    "role": meta["role"],
                    "freq_range": meta["freq_range"],
                    "description": meta["description"],
                    "characteristics": meta["characteristics"],
                    "fun_fact": meta["fun_fact"],
                    "icon": meta["icon"],
                    "confidence": round(scaled_confidence, 1),
                    "raw_score": round(raw_score, 4),
                    "peak_score": round(peak_score, 4),
                    "intensity": intensity,
                    "intensity_color": intensity_color,
                    "active_segments_count": len(timeline_presence),
                    "timeline": timeline_presence
                })

        # Sort candidates descending by confidence
        detected_candidates.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Hierarchical cleanup:
        # If specific guitar like "Acoustic guitar" or "Electric guitar" is present,
        # adjust general "Guitar" so we don't crowd the list with duplicates unless useful.
        # Group by family
        family_summary = {}
        for fam_key, fam_meta in INSTRUMENT_FAMILIES.items():
            matching_insts = [d for d in detected_candidates if d["family"] == fam_key]
            if matching_insts:
                avg_conf = sum(d["confidence"] for d in matching_insts) / len(matching_insts)
                family_summary[fam_key] = {
                    "family": fam_key,
                    "name": fam_meta["name"],
                    "color": fam_meta["color"],
                    "bg_color": fam_meta["bg_color"],
                    "border_color": fam_meta["border_color"],
                    "text_color": fam_meta["text_color"],
                    "badge_color": fam_meta["badge_color"],
                    "icon": fam_meta["icon"],
                    "description": fam_meta["description"],
                    "count": len(matching_insts),
                    "instruments": [d["name"] for d in matching_insts],
                    "average_confidence": round(avg_conf, 1)
                }

        # Dynamic song tags
        song_tags = []
        if detected_candidates:
            top_3 = [d["name"] for d in detected_candidates[:3]]
            song_tags.append(f"Featuring {', '.join(top_3)}")
        if bpm > 0:
            tempo_desc = "Slow / Ballad" if bpm < 85 else "Moderate Groove" if bpm < 125 else "Uptempo / Energetic"
            song_tags.append(f"{tempo_desc} ({bpm} BPM)")
        if key_signature != "Undetermined":
            song_tags.append(f"Key of {key_signature}")

        return {
            "summary": {
                "total_instruments_detected": len(detected_candidates),
                "duration_seconds": round(total_duration, 2),
                "duration_formatted": f"{int(total_duration // 60)}:{int(total_duration % 60):02d}",
                "tempo_bpm": bpm,
                "key_signature": key_signature,
                "dominant_family": max(family_summary.values(), key=lambda f: f["count"])["name"] if family_summary else "None",
                "sample_rate": sr,
                "analysis_windows_count": len(window_clips)
            },
            "waveform": waveform_peaks,
            "detected_instruments": detected_candidates,
            "families": list(family_summary.values()),
            "tags": song_tags,
            "timeline_segments": [
                {
                    "window_index": i,
                    "start": t_start,
                    "end": t_end,
                    "active_instruments": [
                        {
                            "name": d["name"],
                            "confidence": next((p["confidence"] for p in d["timeline"] if p["start"] == t_start), 0.0)
                        }
                        for d in detected_candidates
                        if any(p["start"] == t_start for p in d["timeline"])
                    ][:4]
                }
                for i, (t_start, t_end) in enumerate(timestamps)
            ]
        }
