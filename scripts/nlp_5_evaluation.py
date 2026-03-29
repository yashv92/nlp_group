"""
nlp_5_evaluation.py -- Phase 5: Evaluation, Error Analysis & Results Summary

Owner: Diana (TLWQ1) | Branch: diana/evaluation-report

Input:
    data/processed/reviews_topics.csv      -- output of Phase 3
    outputs/clf_comparison.csv             -- output of Phase 4
    outputs/sentiment_ttest_results.csv    -- output of Phase 2
    outputs/aspect_ttest_results.csv       -- output of Phase 3
    outputs/linguistic_ttest_results.csv   -- output of Phase 4

Output:
    outputs/error_analysis.csv             -- misclassified reviews with metadata
    outputs/results_summary.md             -- structured markdown for the write-up
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT     = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR  = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Load reviews_topics.csv (required)
# ---------------------------------------------------------------------------
topics_csv = PROC_DIR / "reviews_topics.csv"
if not topics_csv.exists():
    print("ERROR: data/processed/reviews_topics.csv not found. Run nlp_3_topic_modelling.py first.")
    sys.exit(1)

print("Loading reviews_topics.csv ...")
df = pd.read_csv(topics_csv)

CLEAN_COL = "review_text_clean" if "review_text_clean" in df.columns else "review_text"
TEXT_COL  = "review_text"       if "review_text"       in df.columns else CLEAN_COL
GROUP_COL = "age_group"

df = df.dropna(subset=[CLEAN_COL, GROUP_COL]).reset_index(drop=True)
df[GROUP_COL] = df[GROUP_COL].str.strip().str.lower()
df = df[df[GROUP_COL].isin(["gen_z", "older"])].reset_index(drop=True)

print(f"  Rows: {len(df):,}  | Gen Z: {(df[GROUP_COL]=='gen_z').sum():,}  "
      f"| Older: {(df[GROUP_COL]=='older').sum():,}")

# ---------------------------------------------------------------------------
# Load optional results CSVs (degrade gracefully if missing)
# ---------------------------------------------------------------------------
def load_csv(path: Path, label: str) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    print(f"  [WARNING] {label} not found at {path} -- skipping.")
    return None

clf_df      = load_csv(OUT_DIR / "clf_comparison.csv",           "clf_comparison.csv")
sent_ttest  = load_csv(OUT_DIR / "sentiment_ttest_results.csv",  "sentiment_ttest_results.csv")
asp_ttest   = load_csv(OUT_DIR / "aspect_ttest_results.csv",     "aspect_ttest_results.csv")
ling_ttest  = load_csv(OUT_DIR / "linguistic_ttest_results.csv", "linguistic_ttest_results.csv")

# ============================================================================
# PART 1 -- ERROR ANALYSIS
# Rerun the best classifier from Phase 4 (Logistic Regression with TF-IDF)
# to obtain per-review predictions and identify misclassifications.
# ============================================================================
print("\n--- Part 1: Error Analysis ---")

vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X   = vec.fit_transform(df[CLEAN_COL])
y   = (df[GROUP_COL] == "gen_z").astype(int)

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

lr = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                        random_state=RANDOM_STATE)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

test_df = df.loc[idx_test].copy().reset_index(drop=True)
test_df["true_label"]      = y_test.values
test_df["predicted_label"] = y_pred
test_df["true_group"]      = test_df["true_label"].map({1: "gen_z", 0: "older"})
test_df["predicted_group"] = test_df["predicted_label"].map({1: "gen_z", 0: "older"})

misclassified = test_df[test_df["true_label"] != test_df["predicted_label"]].copy()

print(f"  Test set size    : {len(test_df):,}")
print(f"  Misclassified    : {len(misclassified):,} "
      f"({100 * len(misclassified) / len(test_df):.1f}%)")

gz_as_older = ((misclassified["true_group"] == "gen_z") &
               (misclassified["predicted_group"] == "older")).sum()
older_as_gz = ((misclassified["true_group"] == "older") &
               (misclassified["predicted_group"] == "gen_z")).sum()
print(f"  Gen Z predicted as Older : {gz_as_older:,}")
print(f"  Older predicted as Gen Z : {older_as_gz:,}")

# Top topics among misclassified reviews
if "topic_label" in misclassified.columns:
    top_topics = misclassified["topic_label"].value_counts().head(3)
    print(f"\n  Top 3 topics among misclassified reviews:")
    for topic, count in top_topics.items():
        print(f"    {topic}: {count}")

# Save error analysis CSV
error_cols = [c for c in
              [TEXT_COL, "true_group", "predicted_group",
               "vader_compound", "topic_label", "word_count",
               "age_group", "rating"]
              if c in misclassified.columns]
misclassified[error_cols].to_csv(OUT_DIR / "error_analysis.csv", index=False)
print(f"\n  Saved: error_analysis.csv  ({len(misclassified):,} rows)")

# ============================================================================
# PART 2 -- PRINT FULL EVALUATION SUMMARY TO CONSOLE
# ============================================================================
print("\n--- Part 2: Full Evaluation Summary ---")

if clf_df is not None:
    print("\nClassifier Comparison:")
    print(clf_df.to_string(index=False))

if sent_ttest is not None:
    print("\nSentiment T-Tests (Gen Z vs Older):")
    print(sent_ttest.to_string(index=False))

if asp_ttest is not None:
    sig_topics = asp_ttest[asp_ttest["significant"] == True] if "significant" in asp_ttest.columns else pd.DataFrame()
    print(f"\nAspect T-Tests: {len(asp_ttest)} topics tested, "
          f"{len(sig_topics)} with significant Gen Z vs Older difference (p < 0.05)")
    if len(sig_topics):
        print(sig_topics[["topic_label", "gen_z_mean", "older_mean", "p_value"]].to_string(index=False))

# ============================================================================
# PART 3 -- WRITE results_summary.md
# ============================================================================
print("\n--- Part 3: Writing results_summary.md ---")

# Dataset stats
n_total  = len(df)
n_gz     = (df[GROUP_COL] == "gen_z").sum()
n_older  = (df[GROUP_COL] == "older").sum()
pct_gz   = 100 * n_gz / n_total
pct_old  = 100 * n_older / n_total

# Sentiment stats
vader_gz_mean  = df[df[GROUP_COL] == "gen_z"]["vader_compound"].mean()  \
                 if "vader_compound" in df.columns else None
vader_old_mean = df[df[GROUP_COL] == "older"]["vader_compound"].mean()  \
                 if "vader_compound" in df.columns else None

# Classifier stats
lr_acc = clf_df.loc[clf_df["model"] == "LogisticRegression", "accuracy"].values[0] \
         if clf_df is not None and "LogisticRegression" in clf_df["model"].values else "N/A"
rf_acc = clf_df.loc[clf_df["model"] == "RandomForest", "accuracy"].values[0] \
         if clf_df is not None and "RandomForest" in clf_df["model"].values else "N/A"
lr_f1_gz = clf_df.loc[clf_df["model"] == "LogisticRegression", "f1_genz"].values[0] \
           if clf_df is not None and "LogisticRegression" in clf_df["model"].values else "N/A"
rf_f1_gz = clf_df.loc[clf_df["model"] == "RandomForest", "f1_genz"].values[0] \
           if clf_df is not None and "RandomForest" in clf_df["model"].values else "N/A"

# Significant sentiment metrics
sig_sent_metrics = []
if sent_ttest is not None and "significant" in sent_ttest.columns:
    sig_sent_metrics = sent_ttest[sent_ttest["significant"] == True]["metric"].tolist()

# Significant aspects
sig_aspect_list = []
if asp_ttest is not None and "significant" in asp_ttest.columns:
    sig_aspect_list = asp_ttest[asp_ttest["significant"] == True]["topic_label"].tolist()

# Significant linguistic features
sig_ling_list = []
if ling_ttest is not None and "significant" in ling_ttest.columns:
    sig_ling_list = ling_ttest[ling_ttest["significant"] == True]["feature"].tolist()

md = f"""# NLP Group Project — Results Summary

**Module:** MSIN0221 — Natural Language Processing
**Dataset:** Women's E-Commerce Clothing Reviews (~23k reviews)

---

## 1. Dataset Overview

- **Total reviews after cleaning:** {n_total:,}
- **Gen Z (ages 18–26):** {n_gz:,} reviews ({pct_gz:.1f}%)
- **Older (ages 27+):** {n_older:,} reviews ({pct_old:.1f}%)
- **Class imbalance note:** The dataset is heavily skewed toward older reviewers (~{pct_old:.0f}% older vs ~{pct_gz:.0f}% Gen Z). All classifiers use `class_weight='balanced'` to account for this.

---

## 2. Sentiment Analysis (Phase 2)

### VADER Compound Scores
- Gen Z mean: {f"{vader_gz_mean:.4f}" if vader_gz_mean is not None else "N/A"}
- Older mean: {f"{vader_old_mean:.4f}" if vader_old_mean is not None else "N/A"}

### Statistical Significance (Welch's t-test, alpha = 0.05)
{("- Significant metrics: " + ", ".join(sig_sent_metrics)) if sig_sent_metrics else "- No significant differences found between age groups on sentiment metrics."}

---

## 3. Topic Modelling (Phase 3)

- LDA coherence sweep tested k = 6, 8, 10, 12, 15
- Best k selected by highest c_v coherence score
{("### Aspects with Significant Gen Z vs Older Difference" + chr(10) + chr(10).join(f"- {t}" for t in sig_aspect_list)) if sig_aspect_list else "- No topics showed a statistically significant sentiment difference between age groups."}

---

## 4. Linguistic Analysis (Phase 4)

### Significant Linguistic Features (Welch's t-test, p < 0.05)
{chr(10).join(f"- {f}" for f in sig_ling_list) if sig_ling_list else "- No linguistic features showed significant differences between age groups."}

---

## 5. Classification Results (Phase 4)

| Model | Accuracy | F1 Gen Z | F1 Older |
|---|---|---|---|
| Logistic Regression | {lr_acc} | {lr_f1_gz} | {clf_df.loc[clf_df["model"]=="LogisticRegression","f1_older"].values[0] if clf_df is not None and "LogisticRegression" in clf_df["model"].values else "N/A"} |
| Random Forest | {rf_acc} | {rf_f1_gz} | {clf_df.loc[clf_df["model"]=="RandomForest","f1_older"].values[0] if clf_df is not None and "RandomForest" in clf_df["model"].values else "N/A"} |

**Note:** High overall accuracy is misleading given class imbalance. F1 for Gen Z is the more informative metric.

---

## 6. Error Analysis (Phase 5)

- Test set size: {len(test_df):,} reviews
- Total misclassified: {len(misclassified):,} ({100 * len(misclassified) / len(test_df):.1f}%)
- Gen Z predicted as Older: {gz_as_older:,}
- Older predicted as Gen Z: {older_as_gz:,}
{("- Top misclassified topics: " + ", ".join(top_topics.index.tolist())) if "topic_label" in misclassified.columns else ""}

---

*Generated by nlp_5_evaluation.py*
"""

summary_path = OUT_DIR / "results_summary.md"
summary_path.write_text(md, encoding="utf-8")
print(f"  Saved: results_summary.md")

# ============================================================================
# DONE
# ============================================================================
created = [
    "outputs/error_analysis.csv",
    "outputs/results_summary.md",
]
print("\n[DONE] Files created:")
for f in created:
    print(f"  {f}")
