"""
prepare_data.py
----------------
Run this FIRST, from your project root (same folder as Reviews.csv).
It regenerates everything the dashboard needs inside data/:
    data/quality_summary.json
    data/reviews_sample.csv
    data/reviews_clean.csv

Usage:
    python prepare_data.py
"""

import json
import os
import re

import pandas as pd

RAW_FILE = "Reviews.csv"   # change this if your raw file has a different name
DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(RAW_FILE):
    raise FileNotFoundError(
        f"Could not find '{RAW_FILE}' in the current folder. "
        f"Place your raw dataset here or update RAW_FILE at the top of this script."
    )

print(f"Loading {RAW_FILE} ...")
df_raw = pd.read_csv(RAW_FILE)

# ---------------------------------------------------------------------------
# 1. Data quality summary (on the FULL raw data)
# ---------------------------------------------------------------------------
original_shape = df_raw.shape
missing = df_raw.isnull().sum().to_dict()
numerical_cols = df_raw.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df_raw.select_dtypes(include=["object"]).columns.tolist()
dup_text = int(df_raw.duplicated(subset=["Text"]).sum())

df = df_raw.dropna(subset=["ProfileName", "Summary"])
df = df.drop_duplicates(subset=["Text"])
df = df[df["Text"].str.len() > 5]
cleaned_full_shape = df.shape

quality_summary = {
    "original_shape": list(original_shape),
    "missing_values": {k: int(v) for k, v in missing.items()},
    "numerical_columns": numerical_cols,
    "categorical_columns": categorical_cols,
    "duplicate_text_count": dup_text,
    "cleaned_full_shape": list(cleaned_full_shape),
}

with open(os.path.join(DATA_DIR, "quality_summary.json"), "w") as f:
    json.dump(quality_summary, f, indent=2)
print("Saved data/quality_summary.json")
print(quality_summary)

# ---------------------------------------------------------------------------
# 2. Sample for training (15,000 rows)
# ---------------------------------------------------------------------------
df_final = df[["Score", "Time", "Summary", "Text"]]
sample_size = min(15000, len(df_final))
sample = df_final.sample(n=sample_size, random_state=42)
sample.to_csv(os.path.join(DATA_DIR, "reviews_sample.csv"), index=False)
print(f"Saved data/reviews_sample.csv ({sample.shape})")

# ---------------------------------------------------------------------------
# 3. Clean + engineer features -> reviews_clean.csv
# ---------------------------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


sample = sample.dropna(subset=["Text", "Score"])
sample = sample.drop_duplicates(subset=["Text"])
sample["clean_text"] = sample["Text"].apply(clean_text)
sample = sample[sample["clean_text"].str.len() > 5]
sample["review_length"] = sample["clean_text"].apply(lambda t: len(t.split()))


def score_to_sentiment(score):
    if score >= 4:
        return "Positive"
    elif score == 3:
        return "Neutral"
    else:
        return "Negative"


sample["sentiment"] = sample["Score"].apply(score_to_sentiment)
sample["date"] = pd.to_datetime(sample["Time"], unit="s")
sample["year_month"] = sample["date"].dt.to_period("M").astype(str)

sample.to_csv(os.path.join(DATA_DIR, "reviews_clean.csv"), index=False)
print(f"Saved data/reviews_clean.csv ({sample.shape})")

print("\nDone. Now run: python train_model.py")