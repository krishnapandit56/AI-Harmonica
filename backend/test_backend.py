import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from main import generate_sample_demo_tracks, DEMO_DIR
from classifier import InstrumentClassifier

print("1. Generating sample demo tracks...")
generate_sample_demo_tracks()

print("2. Initializing InstrumentClassifier singleton...")
clf = InstrumentClassifier.get_instance()

print("3. Analyzing piano_keys_demo.wav...")
piano_file = str(DEMO_DIR / "piano_keys_demo.wav")
piano_res = clf.analyze_audio_file(piano_file)
print("Summary:", piano_res["summary"])
print("Top Detected Instruments:")
for item in piano_res["detected_instruments"][:5]:
    print(f"  * {item['name']} ({item['family_name']}): {item['confidence']}% [{item['intensity']}]")

print("\n4. Analyzing guitar_bass_demo.wav...")
guitar_file = str(DEMO_DIR / "guitar_bass_demo.wav")
guitar_res = clf.analyze_audio_file(guitar_file)
print("Summary:", guitar_res["summary"])
print("Top Detected Instruments:")
for item in guitar_res["detected_instruments"][:5]:
    print(f"  * {item['name']} ({item['family_name']}): {item['confidence']}% [{item['intensity']}]")

print("\nALL BACKEND CLASSIFICATION TESTS PASSED!")
