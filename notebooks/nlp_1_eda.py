# =============================================================================
# NLP Group Project — Women's E-Commerce Clothing Reviews
# Section: Data Loading & EDA
# Author: [Your Name]
#
# To install dependencies, run:
#   pip install pandas matplotlib seaborn nltk
#
# Place "Womens Clothing E-Commerce Reviews.csv" in the same folder as this
# script, or update DATA_PATH below.
#
# Outputs produced:
#   genz_reviews.csv       — cleaned Gen Z reviews (for teammates)
#   older_reviews.csv      — cleaned Older reviews (for teammates)
#   eda_overview.png       — age group counts, rating distribution, review length
#   eda_age_histogram.png  — full age distribution histogram
# =============================================================================

import re
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore")

for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

# ---------------------------------------------------------------------------
# Consistent colour palette — reused by topic modelling script too
# ---------------------------------------------------------------------------
PALETTE   = {"Gen Z (18–26)": "#6C63FF", "Older (27+)": "#FF6584"}
GZ_COLOR  = "#6C63FF"
OLD_COLOR = "#FF6584"

plt.rcParams.update({"figure.dpi": 130, "font.size": 11})


# =============================================================================
# SECTION 1 — DATA LOADING & PREPARATION
# =============================================================================
print("=" * 60)
print("SECTION 1: DATA LOADING & PREPARATION")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1.1  Load the raw CSV
# ---------------------------------------------------------------------------
DATA_PATH = "Womens Clothing E-Commerce Reviews.csv"

df = pd.read_csv(DATA_PATH, index_col=0)   # first column is an unnamed index
print(f"\nRaw dataset shape      : {df.shape}")
print(f"Columns                : {list(df.columns)}\n")

# ---------------------------------------------------------------------------
# 1.2  Standardise column names (lowercase + underscores)
# ---------------------------------------------------------------------------
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace(r"[^a-z0-9_]", "", regex=True)
)
# After standardisation the review column becomes 'review_text'
TEXT_COL = "review_text"

# ---------------------------------------------------------------------------
# 1.3  Drop duplicates and rows missing the two essential fields
# ---------------------------------------------------------------------------
n_raw = len(df)
df = df.drop_duplicates()
df = df.dropna(subset=[TEXT_COL, "age"])
print(f"After dropping duplicates & missing age/review : {len(df):,} rows "
      f"(removed {n_raw - len(df):,})")

# ---------------------------------------------------------------------------
# 1.4  Remove reviews with fewer than 10 words
# ---------------------------------------------------------------------------
df["word_count"] = df[TEXT_COL].astype(str).apply(lambda x: len(x.split()))
n_before = len(df)
df = df[df["word_count"] >= 10].reset_index(drop=True)
print(f"After removing short reviews (<10 words)       : {len(df):,} rows "
      f"(removed {n_before - len(df):,})")

# ---------------------------------------------------------------------------
# 1.5  Text normalisation — lowercase + strip special characters
# ---------------------------------------------------------------------------
def normalise_text(text: str) -> str:
    """
    Converts text to lowercase and removes all characters that are not
    letters, digits, or whitespace. Multiple spaces are collapsed to one.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df[TEXT_COL].apply(normalise_text)
print("Text normalisation applied (lowercase + special character removal).")

# ---------------------------------------------------------------------------
# 1.6  Age validation and generational segmentation
#       Gen Z  : 18–26   |   Older : 27+
# ---------------------------------------------------------------------------
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df = df.dropna(subset=["age"])
df["age"] = df["age"].astype(int)

# Exclude implausible ages
df = df[(df["age"] >= 18) & (df["age"] <= 99)].reset_index(drop=True)

df["age_group"] = df["age"].apply(
    lambda a: "Gen Z (18–26)" if a <= 26 else "Older (27+)"
)

gz_df  = df[df["age_group"] == "Gen Z (18–26)"].reset_index(drop=True)
old_df = df[df["age_group"] == "Older (27+)"].reset_index(drop=True)

print(f"\nGen Z  (18–26) reviews : {len(gz_df):,}")
print(f"Older  (27+)   reviews : {len(old_df):,}")
print(f"Total after cleaning   : {len(df):,}")

# ---------------------------------------------------------------------------
# 1.7  Export cleaned dataframes so teammates can load them directly
#      The topic modelling script loads these CSVs — run this file first.
# ---------------------------------------------------------------------------
gz_df.to_csv("genz_reviews.csv",   index=False)
old_df.to_csv("older_reviews.csv", index=False)
print("\nExported: genz_reviews.csv  |  older_reviews.csv")


# =============================================================================
# SECTION 2 — EXPLORATORY DATA ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 2: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ---------------------------------------------------------------------------
# 2.1  Summary statistics table
# ---------------------------------------------------------------------------
summary = df.groupby("age_group").agg(
    n_reviews         = ("clean_text",       "count"),
    mean_word_count   = ("word_count",        "mean"),
    median_word_count = ("word_count",        "median"),
    mean_rating       = ("rating",            "mean"),
    pct_recommended   = ("recommended_ind",   "mean"),
).round(2)
summary["pct_recommended"] = (summary["pct_recommended"] * 100).round(1)

print("\nSummary Statistics by Age Group:")
print(summary.to_string())

# ---------------------------------------------------------------------------
# 2.2  Three-panel overview figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Exploratory Data Analysis — Overview", fontsize=14, fontweight="bold")

# Panel A — Age group review counts
ax = axes[0]
counts = df["age_group"].value_counts().reindex(["Gen Z (18–26)", "Older (27+)"])
bars = ax.bar(
    counts.index, counts.values,
    color=[PALETTE[k] for k in counts.index],
    edgecolor="white", width=0.5,
)
for bar, val in zip(bars, counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 50,
        f"{val:,}", ha="center", va="bottom", fontsize=10,
    )
ax.set_title("Review Counts by Age Group")
ax.set_ylabel("Number of Reviews")
ax.set_ylim(0, counts.max() * 1.15)
ax.spines[["top", "right"]].set_visible(False)

# Panel B — Rating distribution by age group
ax = axes[1]
rating_gz  = gz_df["rating"].value_counts().sort_index()
rating_old = old_df["rating"].value_counts().sort_index()
x = np.arange(1, 6)
w = 0.35
ax.bar(x - w/2, rating_gz.reindex(x,  fill_value=0), w,
       label="Gen Z (18–26)", color=GZ_COLOR,  alpha=0.85, edgecolor="white")
ax.bar(x + w/2, rating_old.reindex(x, fill_value=0), w,
       label="Older (27+)",   color=OLD_COLOR, alpha=0.85, edgecolor="white")
ax.set_title("Rating Distribution by Age Group")
ax.set_xlabel("Star Rating")
ax.set_ylabel("Number of Reviews")
ax.set_xticks(x)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

# Panel C — Review length distribution (KDE with median lines)
ax = axes[2]
for label, subset, color in [
    ("Gen Z (18–26)", gz_df,  GZ_COLOR),
    ("Older (27+)",   old_df, OLD_COLOR),
]:
    lengths = subset["word_count"].clip(upper=300)
    lengths.plot.kde(ax=ax, label=label, color=color, linewidth=2)
    ax.axvline(lengths.median(), color=color, linestyle="--",
               linewidth=1.2, alpha=0.7)
ax.set_title("Review Length Distribution (word count)")
ax.set_xlabel("Word Count")
ax.set_ylabel("Density")
ax.set_xlim(0, 300)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("eda_overview.png", bbox_inches="tight")
plt.show()
print("\nSaved: eda_overview.png")

# ---------------------------------------------------------------------------
# 2.3  Full age distribution histogram
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(gz_df["age"],  bins=range(18, 100, 2), color=GZ_COLOR,
        alpha=0.75, label="Gen Z (18–26)", edgecolor="white")
ax.hist(old_df["age"], bins=range(18, 100, 2), color=OLD_COLOR,
        alpha=0.75, label="Older (27+)",   edgecolor="white")
ax.axvline(26.5, color="black", linestyle="--", linewidth=1.2,
           label="Gen Z cutoff (age 26)")
ax.set_title("Age Distribution of Reviewers", fontweight="bold")
ax.set_xlabel("Age")
ax.set_ylabel("Number of Reviews")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("eda_age_histogram.png", bbox_inches="tight")
plt.show()
print("Saved: eda_age_histogram.png")

print("\n" + "=" * 60)
print("EDA complete. Files produced:")
print("  genz_reviews.csv       — cleaned Gen Z reviews")
print("  older_reviews.csv      — cleaned Older reviews")
print("  eda_overview.png       — three-panel EDA figure")
print("  eda_age_histogram.png  — age distribution")
print("\nRun nlp_2_lda.py next.")
print("=" * 60)
