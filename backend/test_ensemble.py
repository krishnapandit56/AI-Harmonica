"""Quick smoke test for the ensemble backend."""
import sys
from pathlib import Path

backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from main import generate_sample_demo_tracks, DEMO_DIR
from ensemble_classifier import EnsembleInstrumentClassifier

print("=" * 60)
print("ENSEMBLE BACKEND SMOKE TEST")
print("=" * 60)

print("\n1. Generating demo audio tracks …")
generate_sample_demo_tracks()
print("   Done.")

print("\n2. Initialising EnsembleInstrumentClassifier …")
clf = EnsembleInstrumentClassifier.get_instance()
print(f"   Device: {clf.device.upper()}")
print(f"   CNN14 tracked instrument classes: {len(clf.panns_instrument_indices)}")
print(f"   OpenMIC CNN instrument classes: {len(clf.openmic_instruments)}")
print(f"   OpenMIC classes: {clf.openmic_instruments}")

print("\n3. Analysing piano_keys_demo.wav …")
res = clf.analyze_audio_file(str(DEMO_DIR / "piano_keys_demo.wav"), cnn14_weight=0.60, openmic_weight=0.40)
s = res["summary"]
print(f"   Duration: {s['duration_formatted']} | Key: {s['key_signature']} | BPM: {s['tempo_bpm']}")
print(f"   Dominant family: {s['dominant_family']}")
print(f"   Ensemble weights  CNN14={s['ensemble_cnn14_weight']}  OpenMIC={s['ensemble_openmic_weight']}")
print(f"   Total detected: {s['total_instruments_detected']}")
print("   Top 5 instruments:")
for d in res["detected_instruments"][:5]:
    src = d.get("source", "?")
    c14 = d.get("cnn14_confidence", 0)
    om  = d.get("openmic_confidence", 0)
    print(f"     [{src:8s}] {d['name']:45s} fused={d['confidence']}%  (CNN14={c14}%, OpenMIC={om}%)")

print("\n4. Analysing guitar_bass_demo.wav …")
res2 = clf.analyze_audio_file(str(DEMO_DIR / "guitar_bass_demo.wav"), cnn14_weight=0.60, openmic_weight=0.40)
s2 = res2["summary"]
print(f"   Dominant family: {s2['dominant_family']} | Detected: {s2['total_instruments_detected']}")
print("   Top 5 instruments:")
for d in res2["detected_instruments"][:5]:
    src = d.get("source", "?")
    c14 = d.get("cnn14_confidence", 0)
    om  = d.get("openmic_confidence", 0)
    print(f"     [{src:8s}] {d['name']:45s} fused={d['confidence']}%  (CNN14={c14}%, OpenMIC={om}%)")

print("\n5. Ensemble info:")
ei = res["ensemble_info"]
print(f"   Strategy: {ei['strategy']}")
for m in ei["models"]:
    print(f"   - {m['name']}  weight={m['weight']}  classes={m['tracked_classes']}")

print("\n" + "=" * 60)
print("ALL ENSEMBLE TESTS PASSED!")
print("=" * 60)
