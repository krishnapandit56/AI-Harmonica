
from pathlib import Path
import pandas as pd
import re

DATASET_DIR = Path("KSHMR_Dataset")
INPUT_CSV = DATASET_DIR / "label_review.csv"
OUTPUT_CSV = DATASET_DIR / "label_review_labeled.csv"

df = pd.read_csv(INPUT_CSV)

def assign_label(row):
    name = str(row["filename"]).lower()
    folder = str(row["original_relative_path"]).lower()
    text = f"{folder} {name}"

    # Exclude the complex orchestral Main Drums category
    if "main_drums" in text or "main drums" in text:
        return "exclude_main_drums"

    # Specific user-requested percussion categories
    if re.search(r"percussion[_ -]?(roll|flam)", name):
        return "Percussion"

    # Transitions are their own category
    if "transition" in text or "transitions" in text:
        return "Transitions"

    # Specific instrument and sound names
    rules = [
        (r"closed[_ -]?hat|open[_ -]?hat|hi[_ -]?hat|hihat", "Hi-hat"),
        (r"\bkick\b|_kick_|^kick_", "Kick"),
        (r"\bsnare\b|_snare_|^snare_", "Snare"),
        (r"\bclap\b|_clap_|^clap_", "Clap"),
        (r"crash|cymbal", "Cymbal"),
        (r"\btimpani\b", "Timpani"),
        (r"tibetan[_ -]?gong|chinese[_ -]?gong|\bgong\b", "Gong"),
        (r"cowbell|sleigh[_ -]?bells|tambourine|shaker|woodblock",
         "Percussion"),
        (r"drum[_ -]?ending|drum[_ -]?endings", "Drum Ending"),
        (r"deep[_ -]?impact|impact", "Impact"),
        (r"enhancer", "Drum Enhancer"),
        (r"beatbox[_ -]?pop", "Beatbox"),
        (r"roll|flam", "Percussion"),
        (r"rim[_ -]?shot|cross[_ -]?stick|rim", "Snare"),
    ]

    for pattern, label in rules:
        if re.search(pattern, name):
            return label

    # Existing reliable labels can be retained
    old_label = str(row.get("candidate_instrument_label", "")).strip()
    if old_label.lower() not in {"", "nan", "unknown", "needs_review"}:
        return old_label

    # Don't guess when filename evidence is insufficient
    return "needs_review"


df["assigned_label"] = df.apply(assign_label, axis=1)

df["label_status"] = df["assigned_label"].apply(
    lambda x: "excluded" if x == "exclude_main_drums"
    else "needs_review" if x == "needs_review"
    else "assigned"
)

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("Labeling complete.")
print("Rows processed:", len(df))
print("\nAssigned labels:")
print(df["assigned_label"].value_counts().to_string())
print("\nSaved to:", OUTPUT_CSV)
