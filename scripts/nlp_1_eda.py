"""
nlp_1_eda.py — Phase 1: Data Loading & Exploratory Data Analysis

Owner: Hemakshi (XKKW4) | Branch: hemakshi/data-prep

Input:
    data/raw/Womens Clothing E-Commerce Reviews.csv

Output:
    data/processed/reviews_clean.csv    — full cleaned dataset
    data/processed/reviews_genz.csv     — Gen Z reviews only (ages 18-26)
    data/processed/reviews_older.csv    — older reviews only (ages 27+)
    outputs/eda_review_length_dist.png
    outputs/eda_rating_dist.png
    outputs/eda_age_group_counts.png
    outputs/eda_wordcloud_genz.png
    outputs/eda_wordcloud_older.png
"""

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from wordcloud import WordCloud
from tqdm import tqdm

warnings.filterwarnings("ignore")
np.random.seed(42)
sns.set_style("whitegrid")

for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent
RAW_CSV    = ROOT / "data" / "raw" / "Womens Clothing E-Commerce Reviews.csv"
PROC_DIR   = ROOT / "data" / "processed"
OUT_DIR    = ROOT / "outputs"

PROC_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Guard: raw dataset must exist
# ---------------------------------------------------------------------------
if not RAW_CSV.exists():
    print(f"[ERROR] Raw dataset not found at: {RAW_CSV}")
    print("Download it from https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews")
    print("and place it at data/raw/Womens Clothing E-Commerce Reviews.csv")
    sys.exit(1)

# =============================================================================
# 1. LOAD
# =============================================================================
print("Loading dataset …")
df = pd.read_csv(RAW_CSV, index_col=0)
df.columns = (
    df.columns.str.strip().str.lower()
    .str.replace(" ", "_").str.replace(r"[^a-z0-9_]", "", regex=True)
)
TEXT_COL = "review_text"
print(f"  Shape: {df.shape}  |  Columns: {list(df.columns)}")

# =============================================================================
# 2. AGE SEGMENTATION
# =============================================================================
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df = df.dropna(subset=["age"])
df["age"] = df["age"].astype(int)
df = df[(df["age"] >= 18) & (df["age"] <= 100)].reset_index(drop=True)

df["age_group"] = df["age"].apply(lambda a: "gen_z" if a <= 26 else "older")

# =============================================================================
# 3. TEXT CLEANING
# =============================================================================
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)   # keep apostrophes
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Cleaning text …")
df = df.dropna(subset=[TEXT_COL])
df[TEXT_COL] = [clean_text(t) for t in tqdm(df[TEXT_COL], desc="  clean")]
df = df.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)
df["word_count"] = df[TEXT_COL].str.split().str.len()
df = df[df["word_count"] >= 10].reset_index(drop=True)

# Keep only required columns
KEEP_COLS = [TEXT_COL, "rating", "age", "age_group",
             "clothing_id", "division_name", "department_name",
             "class_name", "recommended_ind"]
KEEP_COLS = [c for c in KEEP_COLS if c in df.columns]
df = df[KEEP_COLS]

gz_df  = df[df["age_group"] == "gen_z"].reset_index(drop=True)
old_df = df[df["age_group"] == "older"].reset_index(drop=True)

# =============================================================================
# 4. SUMMARY
# =============================================================================
gz_wc  = gz_df[TEXT_COL].str.split().str.len()
old_wc = old_df[TEXT_COL].str.split().str.len()

print(f"\nTotal reviews after cleaning : {len(df):,}")
print(f"Gen Z reviews                : {len(gz_df):,}")
print(f"Older reviews                : {len(old_df):,}")
print(f"Avg review length (Gen Z)    : {gz_wc.mean():.1f} words")
print(f"Avg review length (Older)    : {old_wc.mean():.1f} words")

# =============================================================================
# 5. SAVE CSVs
# =============================================================================
df.to_csv(PROC_DIR / "reviews_clean.csv",  index=False)
gz_df.to_csv(PROC_DIR / "reviews_genz.csv",  index=False)
old_df.to_csv(PROC_DIR / "reviews_older.csv", index=False)
print(f"\nSaved CSVs to {PROC_DIR}/")

# =============================================================================
# 6. FIGURES
# =============================================================================
FIG_KW = dict(figsize=(10, 6), dpi=150)
PALETTE = {"gen_z": "#6C63FF", "older": "#FF6584"}

# — Figure 1: review length distribution
fig, ax = plt.subplots(**FIG_KW)
for label, subset in [("gen_z", gz_df), ("older", old_df)]:
    ax.hist(subset[TEXT_COL].str.split().str.len().clip(upper=300),
            bins=40, alpha=0.6, label=label, color=PALETTE[label], edgecolor="none")
ax.set_title("Review Length Distribution by Age Group")
ax.set_xlabel("Word Count")
ax.set_ylabel("Number of Reviews")
ax.legend()
fig.savefig(OUT_DIR / "eda_review_length_dist.png", bbox_inches="tight")
plt.close(fig)

# — Figure 2: rating distribution
fig, ax = plt.subplots(**FIG_KW)
x = np.arange(1, 6)
w = 0.35
gz_r  = gz_df["rating"].value_counts().reindex(range(1, 6), fill_value=0)
old_r = old_df["rating"].value_counts().reindex(range(1, 6), fill_value=0)
ax.bar(x - w/2, gz_r,  w, label="gen_z",  color=PALETTE["gen_z"],  alpha=0.85)
ax.bar(x + w/2, old_r, w, label="older",  color=PALETTE["older"], alpha=0.85)
ax.set_title("Star Rating Distribution by Age Group")
ax.set_xlabel("Rating")
ax.set_ylabel("Number of Reviews")
ax.set_xticks(x)
ax.legend()
fig.savefig(OUT_DIR / "eda_rating_dist.png", bbox_inches="tight")
plt.close(fig)

# — Figure 3: age group counts
fig, ax = plt.subplots(**FIG_KW)
counts = df["age_group"].value_counts().reindex(["gen_z", "older"])
bars = ax.bar(counts.index, counts.values,
              color=[PALETTE[k] for k in counts.index], width=0.4)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30, f"{val:,}",
            ha="center", va="bottom")
ax.set_title("Review Count by Age Group")
ax.set_ylabel("Number of Reviews")
fig.savefig(OUT_DIR / "eda_age_group_counts.png", bbox_inches="tight")
plt.close(fig)

# — Figure 4 & 5: word clouds
for label, subset, fname in [
    ("gen_z",  gz_df,  "eda_wordcloud_genz.png"),
    ("older",  old_df, "eda_wordcloud_older.png"),
]:
    text = " ".join(subset[TEXT_COL])
    wc = WordCloud(width=800, height=400, background_color="white",
                   colormap="cool", max_words=100,
                   random_state=42).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Word Cloud — {label}")
    fig.savefig(OUT_DIR / fname, bbox_inches="tight")
    plt.close(fig)

print(f"Saved figures to {OUT_DIR}/")

# =============================================================================
# DONE
# =============================================================================
created = [
    "data/processed/reviews_clean.csv",
    "data/processed/reviews_genz.csv",
    "data/processed/reviews_older.csv",
    "outputs/eda_review_length_dist.png",
    "outputs/eda_rating_dist.png",
    "outputs/eda_age_group_counts.png",
    "outputs/eda_wordcloud_genz.png",
    "outputs/eda_wordcloud_older.png",
]
print("\n[DONE] Files created:")
for f in created:
    print(f"  {f}")
