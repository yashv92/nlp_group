"""
nlp_2_sentiment.py — Phase 2: Sentiment Analysis

Owner: Prishita (WDYF5) | Branch: prishita/sentiment

Input:
    data/processed/reviews_clean.csv

Output:
    data/processed/reviews_sentiment.csv             — full dataset with all sentiment columns
    outputs/sentiment_ttest_results.csv              — t-tests + Cohen's d (Gen Z vs older)
    outputs/sentiment_model_agreement.csv            — VADER vs DistilBERT agreement summary
    outputs/sentiment_by_department.csv              — mean sentiment per department × age group
    outputs/sentiment_vader_boxplot.png
    outputs/sentiment_distilbert_boxplot.png
    outputs/sentiment_compound_hist.png
    outputs/sentiment_rating_correlation.png
    outputs/sentiment_intensity_boxplot.png          — emotional intensity (|compound|) by group
    outputs/sentiment_model_agreement_scatter.png    — VADER vs DistilBERT compound scatter
    outputs/sentiment_by_department.png              — mean VADER compound per department × group
"""

import subprocess
import sys
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Auto-install required packages
# ---------------------------------------------------------------------------
_REQUIRED = [
    ("numpy",          "numpy"),
    ("pandas",         "pandas"),
    ("matplotlib",     "matplotlib"),
    ("seaborn",        "seaborn"),
    ("tqdm",           "tqdm"),
    ("scipy",          "scipy"),
    ("vaderSentiment", "vaderSentiment"),
    ("transformers",   "transformers"),
    ("torch",          "torch"),
]

def _install(pip_name: str) -> None:
    print(f"  Installing {pip_name} …")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", pip_name]
    )

for import_name, pip_name in _REQUIRED:
    try:
        __import__(import_name)
    except ImportError:
        _install(pip_name)

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")
np.random.seed(42)
sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR  = ROOT / "outputs"
RAW_CSV  = ROOT / "data" / "raw" / "Womens Clothing E-Commerce Reviews.csv"

PROC_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_CSV     = PROC_DIR / "reviews_clean.csv"
SENTIMENT_CSV = PROC_DIR / "reviews_sentiment.csv"

# ---------------------------------------------------------------------------
# If Phase 1 output missing, build it inline from raw CSV
# ---------------------------------------------------------------------------
if not CLEAN_CSV.exists():
    print("[INFO] reviews_clean.csv not found — running inline preprocessing from raw CSV …")
    if not RAW_CSV.exists():
        print(f"[ERROR] Raw dataset not found at: {RAW_CSV}")
        sys.exit(1)

    raw = pd.read_csv(RAW_CSV, index_col=0)
    raw.columns = (
        raw.columns.str.strip().str.lower()
        .str.replace(" ", "_").str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    TEXT_COL_RAW = "review_text"

    raw["age"] = pd.to_numeric(raw["age"], errors="coerce")
    raw = raw.dropna(subset=["age"])
    raw["age"] = raw["age"].astype(int)
    raw = raw[(raw["age"] >= 18) & (raw["age"] <= 100)].reset_index(drop=True)
    raw["age_group"] = raw["age"].apply(lambda a: "gen_z" if a <= 26 else "older")

    def _clean(text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\s']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    raw = raw.dropna(subset=[TEXT_COL_RAW])
    raw[TEXT_COL_RAW] = [_clean(t) for t in tqdm(raw[TEXT_COL_RAW], desc="  clean")]
    raw = raw.drop_duplicates(subset=[TEXT_COL_RAW]).reset_index(drop=True)
    raw["word_count"] = raw[TEXT_COL_RAW].str.split().str.len()
    raw = raw[raw["word_count"] >= 10].reset_index(drop=True)

    keep = [TEXT_COL_RAW, "rating", "age", "age_group",
            "clothing_id", "division_name", "department_name",
            "class_name", "recommended_ind"]
    keep = [c for c in keep if c in raw.columns]
    raw = raw[keep]
    raw.to_csv(CLEAN_CSV, index=False)
    print(f"  Saved inline → {CLEAN_CSV}")

# ---------------------------------------------------------------------------
# Shared plot config
# ---------------------------------------------------------------------------
FIG_KW  = dict(figsize=(10, 6), dpi=150)
PALETTE = {"gen_z": "#6C63FF", "older": "#FF6584"}

# ---------------------------------------------------------------------------
# Helper: Cohen's d effect size
# ---------------------------------------------------------------------------
def cohens_d(a: pd.Series, b: pd.Series) -> float:
    """Compute Cohen's d for two independent samples."""
    n_a, n_b   = len(a), len(b)
    pooled_std = np.sqrt(
        ((n_a - 1) * a.std() ** 2 + (n_b - 1) * b.std() ** 2)
        / (n_a + n_b - 2)
    )
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0

# =============================================================================
# 1. LOAD
# =============================================================================
print("Loading cleaned dataset …")
df = pd.read_csv(CLEAN_CSV)
TEXT_COL = "review_text"
print(f"  Shape: {df.shape}")

# =============================================================================
# 2. VADER BASELINE
# =============================================================================
print("\nRunning VADER …")
analyzer = SentimentIntensityAnalyzer()

vader_rows = []
for text in tqdm(df[TEXT_COL], desc="  VADER"):
    scores = analyzer.polarity_scores(str(text))
    vader_rows.append(scores)

vader_df = pd.DataFrame(vader_rows)
df["vader_pos"]       = vader_df["pos"].values
df["vader_neg"]       = vader_df["neg"].values
df["vader_neu"]       = vader_df["neu"].values
df["vader_compound"]  = vader_df["compound"].values
# Emotional intensity: how strongly positive OR negative (regardless of direction)
df["vader_intensity"] = df["vader_compound"].abs()

print(f"  VADER compound  — mean: {df['vader_compound'].mean():.3f} "
      f"| std: {df['vader_compound'].std():.3f}")
print(f"  VADER intensity — mean: {df['vader_intensity'].mean():.3f}")

# =============================================================================
# 3. DISTILBERT SENTIMENT
# =============================================================================
print("\nRunning DistilBERT …")
try:
    import os, gc
    os.environ["TOKENIZERS_PARALLELISM"] = "false"   # suppress semaphore warnings

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch.nn.functional as F

    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
    BATCH_SIZE = 8
    MAX_LEN    = 64     # reviews are short; 64 tokens captures the sentiment well
    # Sample for DistilBERT — 2 000 rows is statistically robust and memory-safe on CPU
    DB_SAMPLE  = 2000

    print(f"  Sampling {DB_SAMPLE} rows for DistilBERT (CPU-safe; statistically robust)")
    db_df = df.sample(n=min(DB_SAMPLE, len(df)), random_state=42).copy()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    id2label = model.config.id2label   # {0: "NEGATIVE", 1: "POSITIVE"}

    labels_out = []
    scores_out = []
    texts_db   = db_df[TEXT_COL].tolist()

    with torch.no_grad():
        for i in tqdm(range(0, len(texts_db), BATCH_SIZE), desc="  DistilBERT"):
            batch = texts_db[i : i + BATCH_SIZE]
            enc   = tokenizer(
                batch,
                truncation=True,
                max_length=MAX_LEN,
                padding=True,
                return_tensors="pt",
            )
            logits = model(**enc).logits
            probs  = F.softmax(logits, dim=-1)
            pred_ids = probs.argmax(dim=-1).tolist()
            pred_scores = probs.max(dim=-1).values.tolist()
            labels_out.extend([id2label[p] for p in pred_ids])
            scores_out.extend(pred_scores)
            del enc, logits, probs
            gc.collect()

    db_df["distilbert_label"]    = labels_out
    db_df["distilbert_score"]    = scores_out
    db_df["distilbert_compound"] = [
        s if l == "POSITIVE" else -s for l, s in zip(labels_out, scores_out)
    ]
    db_df["distilbert_intensity"] = db_df["distilbert_compound"].abs()

    # Merge sampled DistilBERT columns back onto full df (NaN for unsampled rows)
    for col in ["distilbert_label", "distilbert_score",
                "distilbert_compound", "distilbert_intensity"]:
        df[col] = np.nan
    df.loc[db_df.index, "distilbert_label"]    = db_df["distilbert_label"].values
    df.loc[db_df.index, "distilbert_score"]    = db_df["distilbert_score"].values
    df.loc[db_df.index, "distilbert_compound"] = db_df["distilbert_compound"].values
    df.loc[db_df.index, "distilbert_intensity"]= db_df["distilbert_intensity"].values

    print(f"  DistilBERT compound  — mean: {df['distilbert_compound'].mean():.3f}")
    print(f"  DistilBERT intensity — mean: {df['distilbert_intensity'].mean():.3f}")

    DISTILBERT_AVAILABLE = True

except ImportError:
    print("  [WARNING] transformers not installed — skipping DistilBERT.")
    df["distilbert_label"]     = np.nan
    df["distilbert_score"]     = np.nan
    df["distilbert_compound"]  = np.nan
    df["distilbert_intensity"] = np.nan
    DISTILBERT_AVAILABLE = False

# =============================================================================
# 4. STATISTICAL TESTING (Gen Z vs older) — compound + intensity, with Cohen's d
# =============================================================================
print("\nRunning t-tests …")
gz  = df[df["age_group"] == "gen_z"]
old = df[df["age_group"] == "older"]

metrics_to_test = ["vader_compound", "vader_intensity"]
if DISTILBERT_AVAILABLE:
    metrics_to_test += ["distilbert_compound", "distilbert_intensity"]

ttest_rows = []
for metric in metrics_to_test:
    gz_vals  = gz[metric].dropna()
    old_vals = old[metric].dropna()
    if len(gz_vals) < 2 or len(old_vals) < 2:
        print(f"  Skipping {metric} — insufficient data")
        continue
    t_stat, p_value = stats.ttest_ind(gz_vals, old_vals, equal_var=False)
    d               = cohens_d(gz_vals, old_vals)
    magnitude       = (
        "negligible" if abs(d) < 0.2 else
        "small"      if abs(d) < 0.5 else
        "medium"     if abs(d) < 0.8 else
        "large"
    )
    ttest_rows.append({
        "metric":      metric,
        "t_stat":      round(t_stat, 4),
        "p_value":     round(p_value, 6),
        "significant": p_value < 0.05,
        "cohens_d":    round(d, 4),
        "effect_size": magnitude,
        "gen_z_mean":  round(gz_vals.mean(), 4),
        "older_mean":  round(old_vals.mean(), 4),
    })
    sig = "YES" if p_value < 0.05 else "NO"
    print(f"  {metric}: t={t_stat:.3f}, p={p_value:.4f}, d={d:.3f} ({magnitude}) "
          f"— significant: {sig}")

ttest_df = pd.DataFrame(ttest_rows)
ttest_df.to_csv(OUT_DIR / "sentiment_ttest_results.csv", index=False)
print(f"  Saved t-test results → outputs/sentiment_ttest_results.csv")

# =============================================================================
# 5. MODEL AGREEMENT ANALYSIS (VADER vs DistilBERT)
# =============================================================================
if DISTILBERT_AVAILABLE:
    print("\nComputing model agreement …")
    # Assign binary sentiment label from VADER (compound threshold 0.05, standard)
    df["vader_label"] = df["vader_compound"].apply(
        lambda c: "POSITIVE" if c >= 0.05 else ("NEGATIVE" if c <= -0.05 else "NEUTRAL")
    )
    # DistilBERT only outputs POSITIVE/NEGATIVE; map to match
    df["distilbert_label_binary"] = df["distilbert_label"]  # already POSITIVE/NEGATIVE

    # Restrict agreement/correlation to sampled rows (others have NaN distilbert cols)
    sampled = df.dropna(subset=["distilbert_compound"])

    # Agreement: both agree on POSITIVE or NEGATIVE (exclude NEUTRAL from VADER)
    non_neutral = sampled[sampled["vader_label"] != "NEUTRAL"]
    agree = (non_neutral["vader_label"] == non_neutral["distilbert_label_binary"])
    agreement_rate = agree.mean()

    agree_by_group = (
        non_neutral.groupby("age_group")
        .apply(lambda g: (g["vader_label"] == g["distilbert_label_binary"]).mean())
        .reset_index(name="agreement_rate")
    )
    agree_by_group["total_reviews"]    = non_neutral.groupby("age_group").size().values
    agree_by_group["agreement_pct"]    = (agree_by_group["agreement_rate"] * 100).round(1)

    print(f"  Overall model agreement (excl. VADER neutral): {agreement_rate:.1%}")
    print(agree_by_group.to_string(index=False))

    # Pearson correlation between continuous compound scores (sampled rows only)
    corr, corr_p = stats.pearsonr(
        sampled["vader_compound"].values,
        sampled["distilbert_compound"].values,
    )
    print(f"  Pearson r (compound scores): {corr:.3f}, p={corr_p:.2e}")

    agree_summary = pd.DataFrame([{
        "overall_agreement_rate":  round(agreement_rate, 4),
        "pearson_r":               round(corr, 4),
        "pearson_p":               round(corr_p, 6),
    }])
    agree_summary.to_csv(OUT_DIR / "sentiment_model_agreement.csv", index=False)
    agree_by_group.to_csv(OUT_DIR / "sentiment_model_agreement_by_group.csv", index=False)
    print("  Saved agreement stats → outputs/sentiment_model_agreement*.csv")

# =============================================================================
# 6. SENTIMENT BY DEPARTMENT
# =============================================================================
DEPT_COL = "department_name"
if DEPT_COL in df.columns:
    print(f"\nComputing sentiment by {DEPT_COL} …")
    dept_sent = (
        df.dropna(subset=[DEPT_COL])
        .groupby([DEPT_COL, "age_group"])
        .agg(
            mean_vader_compound=("vader_compound", "mean"),
            mean_vader_intensity=("vader_intensity", "mean"),
            review_count=("vader_compound", "count"),
        )
        .round(4)
        .reset_index()
    )
    dept_sent.to_csv(OUT_DIR / "sentiment_by_department.csv", index=False)
    print(f"  Saved → outputs/sentiment_by_department.csv")

# =============================================================================
# 7. FIGURES
# =============================================================================

# — Figure 1: VADER compound box plot by age group
fig, ax = plt.subplots(**FIG_KW)
sns.boxplot(
    data=df, x="age_group", y="vader_compound",
    palette=PALETTE, order=["gen_z", "older"], ax=ax,
    width=0.4, flierprops=dict(marker="o", alpha=0.3, markersize=2),
)
ax.set_title("VADER Compound Score by Age Group")
ax.set_xlabel("Age Group")
ax.set_ylabel("VADER Compound Score (−1 to +1)")
ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
# Annotate means
for i, group in enumerate(["gen_z", "older"]):
    m = df[df["age_group"] == group]["vader_compound"].mean()
    ax.text(i, m + 0.02, f"mean={m:.2f}", ha="center", fontsize=9, color="black")
fig.savefig(OUT_DIR / "sentiment_vader_boxplot.png", bbox_inches="tight")
plt.close(fig)

# — Figure 2: DistilBERT compound box plot by age group
if DISTILBERT_AVAILABLE:
    fig, ax = plt.subplots(**FIG_KW)
    sns.boxplot(
        data=df, x="age_group", y="distilbert_compound",
        palette=PALETTE, order=["gen_z", "older"], ax=ax,
        width=0.4, flierprops=dict(marker="o", alpha=0.3, markersize=2),
    )
    ax.set_title("DistilBERT Compound Score by Age Group")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("DistilBERT Compound Score (−1 to +1)")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    for i, group in enumerate(["gen_z", "older"]):
        m = df[df["age_group"] == group]["distilbert_compound"].mean()
        ax.text(i, m + 0.02, f"mean={m:.2f}", ha="center", fontsize=9, color="black")
    fig.savefig(OUT_DIR / "sentiment_distilbert_boxplot.png", bbox_inches="tight")
    plt.close(fig)
else:
    fig, ax = plt.subplots(**FIG_KW)
    ax.text(0.5, 0.5, "DistilBERT not available",
            ha="center", va="center", transform=ax.transAxes, fontsize=14)
    ax.set_title("DistilBERT Compound Score by Age Group")
    fig.savefig(OUT_DIR / "sentiment_distilbert_boxplot.png", bbox_inches="tight")
    plt.close(fig)

# — Figure 3: Overlaid histogram — 2-panel subplot (VADER | DistilBERT)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
for group, color in PALETTE.items():
    subset = df[df["age_group"] == group]
    axes[0].hist(subset["vader_compound"], bins=50, alpha=0.6,
                 label=group, color=color, edgecolor="none")
    if DISTILBERT_AVAILABLE:
        axes[1].hist(subset["distilbert_compound"], bins=50, alpha=0.6,
                     label=group, color=color, edgecolor="none")

axes[0].set_title("VADER Compound Score Distribution")
axes[0].set_xlabel("Compound Score")
axes[0].set_ylabel("Number of Reviews")
axes[0].legend()
axes[1].set_title("DistilBERT Compound Score Distribution")
axes[1].set_xlabel("Compound Score")
axes[1].set_ylabel("Number of Reviews")
axes[1].legend()
fig.suptitle("Sentiment Score Distributions by Age Group", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "sentiment_compound_hist.png", bbox_inches="tight")
plt.close(fig)

# — Figure 4: Scatter — VADER compound vs Rating, coloured by age group
fig, ax = plt.subplots(**FIG_KW)
for group, color in PALETTE.items():
    subset = df[df["age_group"] == group]
    jitter = np.random.uniform(-0.15, 0.15, size=len(subset))
    ax.scatter(
        subset["rating"] + jitter,
        subset["vader_compound"],
        c=color, alpha=0.15, s=6, label=group,
    )
# Mean per rating (all groups combined)
for rating in range(1, 6):
    mean_val = df[df["rating"] == rating]["vader_compound"].mean()
    ax.plot(rating, mean_val, "k^", markersize=8, zorder=5)
ax.set_title("VADER Compound Score vs Star Rating (by Age Group)")
ax.set_xlabel("Star Rating")
ax.set_ylabel("VADER Compound Score")
ax.set_xticks(range(1, 6))
ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
ax.legend(title="Age Group", markerscale=3)
fig.savefig(OUT_DIR / "sentiment_rating_correlation.png", bbox_inches="tight")
plt.close(fig)

# — Figure 5: Emotional intensity box plot (|compound|) by age group
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
sns.boxplot(
    data=df, x="age_group", y="vader_intensity",
    palette=PALETTE, order=["gen_z", "older"], ax=axes[0],
    width=0.4, flierprops=dict(marker="o", alpha=0.3, markersize=2),
)
axes[0].set_title("VADER Emotional Intensity by Age Group")
axes[0].set_xlabel("Age Group")
axes[0].set_ylabel("|VADER Compound| (0 to 1)")

if DISTILBERT_AVAILABLE:
    sns.boxplot(
        data=df, x="age_group", y="distilbert_intensity",
        palette=PALETTE, order=["gen_z", "older"], ax=axes[1],
        width=0.4, flierprops=dict(marker="o", alpha=0.3, markersize=2),
    )
    axes[1].set_title("DistilBERT Emotional Intensity by Age Group")
    axes[1].set_xlabel("Age Group")
    axes[1].set_ylabel("|DistilBERT Compound| (0 to 1)")
else:
    axes[1].text(0.5, 0.5, "DistilBERT not available",
                 ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set_title("DistilBERT Emotional Intensity by Age Group")

fig.suptitle("Emotional Intensity: Gen Z vs Older Consumers", fontsize=13)
fig.tight_layout()
fig.savefig(OUT_DIR / "sentiment_intensity_boxplot.png", bbox_inches="tight")
plt.close(fig)

# — Figure 6: VADER vs DistilBERT model agreement scatter
if DISTILBERT_AVAILABLE:
    fig, ax = plt.subplots(**FIG_KW)
    sampled_plot = df.dropna(subset=["distilbert_compound"])
    for group, color in PALETTE.items():
        subset = sampled_plot[sampled_plot["age_group"] == group]
        ax.scatter(
            subset["vader_compound"],
            subset["distilbert_compound"],
            c=color, alpha=0.3, s=6, label=group,
        )
    # Diagonal reference line (perfect agreement)
    ax.plot([-1, 1], [-1, 1], "k--", linewidth=1, alpha=0.5, label="perfect agreement")
    ax.set_title(f"VADER vs DistilBERT Compound Scores (Pearson r={corr:.2f})")
    ax.set_xlabel("VADER Compound Score")
    ax.set_ylabel("DistilBERT Compound Score")
    ax.legend(title="Age Group", markerscale=4)
    fig.savefig(OUT_DIR / "sentiment_model_agreement_scatter.png", bbox_inches="tight")
    plt.close(fig)

# — Figure 7: Mean VADER compound by department × age group
if DEPT_COL in df.columns:
    pivot = dept_sent.pivot(
        index=DEPT_COL, columns="age_group", values="mean_vader_compound"
    ).dropna()
    pivot = pivot.sort_values("gen_z", ascending=False)

    fig, ax = plt.subplots(**FIG_KW)
    x      = np.arange(len(pivot))
    w      = 0.35
    ax.bar(x - w/2, pivot.get("gen_z",  0), w, label="gen_z",  color=PALETTE["gen_z"],  alpha=0.85)
    ax.bar(x + w/2, pivot.get("older", 0), w, label="older", color=PALETTE["older"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=25, ha="right")
    ax.set_title("Mean VADER Sentiment by Department × Age Group")
    ax.set_ylabel("Mean VADER Compound Score")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.legend()
    fig.savefig(OUT_DIR / "sentiment_by_department.png", bbox_inches="tight")
    plt.close(fig)

print(f"\nSaved all figures to {OUT_DIR}/")

# =============================================================================
# 8. SAVE OUTPUT CSV
# =============================================================================
df.to_csv(SENTIMENT_CSV, index=False)
print(f"Saved sentiment dataset → {SENTIMENT_CSV}")

# =============================================================================
# DONE
# =============================================================================
created = [
    "data/processed/reviews_sentiment.csv",
    "outputs/sentiment_ttest_results.csv",
    "outputs/sentiment_model_agreement.csv",
    "outputs/sentiment_model_agreement_by_group.csv",
    "outputs/sentiment_by_department.csv",
    "outputs/sentiment_vader_boxplot.png",
    "outputs/sentiment_distilbert_boxplot.png",
    "outputs/sentiment_compound_hist.png",
    "outputs/sentiment_rating_correlation.png",
    "outputs/sentiment_intensity_boxplot.png",
    "outputs/sentiment_model_agreement_scatter.png",
    "outputs/sentiment_by_department.png",
]
print("\n[DONE] Files created:")
for f in created:
    print(f"  {f}")
