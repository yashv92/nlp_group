"""
nlp_4_linguistic.py — Phase 4: Linguistic Analysis & Age Group Classifier

Owner: Yash (SSSP0) | Branch: yash/linguistic-classifier

Input:
    data/processed/reviews_topics.csv   (preferred — output of Phase 3)
    data/processed/reviews_clean.csv    (fallback — output of Phase 1)

    Both files must contain:
        review_text       — original text (punctuation + capitalisation preserved)
        review_text_clean — lowercased, punctuation-removed version
        age_group         — gen_z or older

    review_text is used for linguistic feature extraction (punctuation, caps, sentences).
    review_text_clean is used for TF-IDF and classification.

Output:
    outputs/linguistic_features_summary.csv  — mean/std of all features by age group
    outputs/linguistic_ttest_results.csv     — t-test results per feature
    outputs/tfidf_top_terms.csv              — top 30 TF-IDF terms per age group
    outputs/clf_logistic_report.csv          — Logistic Regression classification report
    outputs/clf_rf_report.csv                — Random Forest classification report
    outputs/clf_comparison.csv               — side-by-side model comparison
    outputs/ling_feature_comparison.png      — grouped bar chart (z-score normalised)
    outputs/ling_exclamation_dist.png        — histogram of exclamation count by group
    outputs/ling_ttr_boxplot.png             — type-token ratio box plot
    outputs/ling_wordcount_boxplot.png       — word count box plot
    outputs/clf_confusion_matrix_lr.png      — confusion matrix, Logistic Regression
    outputs/clf_confusion_matrix_rf.png      — confusion matrix, Random Forest
    outputs/clf_feature_importance_lr.png    — top 20 LR coefficients per class
    outputs/clf_roc_curve.png                — ROC curves for both models
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from scipy.stats import ttest_ind
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc
)
from tqdm import tqdm

warnings.filterwarnings("ignore")
# Fix random seed for reproducibility — ensures train/test splits and RF are identical across runs
np.random.seed(42)
sns.set_style("whitegrid")

for pkg in ["punkt", "punkt_tab", "stopwords"]:
    nltk.download(pkg, quiet=True)

STOP_WORDS   = set(stopwords.words("english"))
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR  = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_KW  = dict(figsize=(10, 6), dpi=150)
PALETTE = {"gen_z": "#6C63FF", "older": "#FF6584"}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
topics_csv = PROC_DIR / "reviews_topics.csv"
clean_csv  = PROC_DIR / "reviews_clean.csv"

if topics_csv.exists():
    print(f"Loading {topics_csv.name} …")
    df = pd.read_csv(topics_csv)
elif clean_csv.exists():
    print(f"reviews_topics.csv not found — falling back to {clean_csv.name} …")
    df = pd.read_csv(clean_csv)
else:
    print("ERROR: Run nlp_1_eda.py first")
    sys.exit(1)

# Map normalised column names back to actual column names in the file
col_lower = {c.lower().replace(" ", "_"): c for c in df.columns}
TEXT_COL  = col_lower.get("review_text")        # original — used for linguistic features
CLEAN_COL = col_lower.get("review_text_clean")  # cleaned  — used for TF-IDF & classification
GROUP_COL = col_lower.get("age_group")

if TEXT_COL is None or GROUP_COL is None:
    print(f"ERROR: Expected 'review_text' and 'age_group' columns. Found: {list(df.columns)}")
    sys.exit(1)

# If review_text_clean is absent (old CSV), fall back to review_text for NLP tasks
if CLEAN_COL is None:
    print("  Warning: 'review_text_clean' column not found — re-run nlp_1_eda.py to regenerate CSVs.")
    print("  Falling back to 'review_text' for TF-IDF and classification.")
    CLEAN_COL = TEXT_COL

df = df.dropna(subset=[TEXT_COL, GROUP_COL]).reset_index(drop=True)
df[GROUP_COL] = df[GROUP_COL].str.strip().str.lower()
df = df[df[GROUP_COL].isin(["gen_z", "older"])].reset_index(drop=True)

print(f"  Rows: {len(df):,}  |  Gen Z: {(df[GROUP_COL]=='gen_z').sum():,}  |  Older: {(df[GROUP_COL]=='older').sum():,}")

# ============================================================================
# PART 1 — LINGUISTIC FEATURE EXTRACTION
# ============================================================================
print("\n--- Part 1: Linguistic Feature Extraction ---")

def extract_features(text: str) -> dict:
    """
    Extract surface-level linguistic features from a single review string.

    These features capture stylistic and structural properties of writing that
    may differ between age groups — e.g. Gen Z reviewers may use more emphatic
    punctuation or shorter sentences, while older reviewers may write more
    formally with greater lexical variety.

    Parameters
    ----------
    text : str
        The original (uncleaned) review text, with punctuation and capitalisation
        preserved. Do NOT pass the lowercased/cleaned version — several features
        (exclamation_count, caps_word_count, question_count) depend on the original.

    Returns
    -------
    dict with keys:
        word_count        — total tokens; proxy for review verbosity
        char_count        — total characters excluding spaces
        sentence_count    — number of sentences; shorter sentences may suggest
                            informal writing style
        avg_word_length   — mean characters per token; correlates with vocabulary
                            sophistication (longer words tend to be less common)
        exclamation_count — raw count of '!' characters; marker of emotional intensity
        caps_word_count   — count of ALL-CAPS content words (>=2 chars, non-stopword);
                            caps are used for emphasis in informal writing
        type_token_ratio  — unique_words / total_words; a standard measure of lexical
                            diversity (TTR). Higher TTR = less repetition. Note: TTR
                            decreases naturally as text length increases, so interpret
                            alongside word_count.
        question_count    — raw count of '?' characters; questions may indicate
                            uncertainty or engagement with the reader
    """
    words     = str(text).split()
    total_w   = max(len(words), 1)   # guard against empty reviews causing division by zero
    unique_w  = len(set(w.lower() for w in words))
    chars_ns  = sum(len(w) for w in words)
    sentences = sent_tokenize(str(text))

    # Require len >= 2 to exclude single-letter tokens ("I", "A") which are
    # technically all-caps but carry no emphatic meaning
    caps_w = [w for w in words
              if len(w) >= 2 and w.isupper() and w.lower() not in STOP_WORDS]

    return {
        "word_count"        : len(words),
        "char_count"        : chars_ns,
        "sentence_count"    : len(sentences),
        "avg_word_length"   : round(chars_ns / total_w, 4),
        "exclamation_count" : text.count("!"),
        "caps_word_count"   : len(caps_w),
        "type_token_ratio"  : round(unique_w / total_w, 4),
        "question_count"    : text.count("?"),
    }

records = [extract_features(t) for t in tqdm(df[TEXT_COL], desc="  features")]
feat_df = pd.DataFrame(records)
for col in feat_df.columns:
    df[col] = feat_df[col].values

FEATURES = list(feat_df.columns)

# Group summary
summary_rows = []
for grp in ["gen_z", "older"]:
    sub = df[df[GROUP_COL] == grp][FEATURES]
    row = {"age_group": grp}
    for f in FEATURES:
        row[f"{f}_mean"] = round(sub[f].mean(), 4)
        row[f"{f}_std"]  = round(sub[f].std(),  4)
    summary_rows.append(row)

pd.DataFrame(summary_rows).to_csv(OUT_DIR / "linguistic_features_summary.csv", index=False)
print("  Saved: linguistic_features_summary.csv")

# Independent-samples t-tests: do Gen Z and older reviewers differ on each feature?
# H0: the population means are equal (no difference between age groups).
# We use Welch's t-test (equal_var=False) rather than Student's t-test because
# our two groups are very different sizes (~1,600 Gen Z vs ~20,800 older), which
# makes the equal-variance assumption of Student's test unreliable.
gz_sub  = df[df[GROUP_COL] == "gen_z"]
old_sub = df[df[GROUP_COL] == "older"]

ttest_rows = []
for f in FEATURES:
    t, p = ttest_ind(gz_sub[f], old_sub[f], equal_var=False)
    ttest_rows.append({
        "feature"     : f,
        "gen_z_mean"  : round(gz_sub[f].mean(), 4),
        "older_mean"  : round(old_sub[f].mean(), 4),
        "t_stat"      : round(t, 4),
        "p_value"     : round(p, 4),
        # Alpha = 0.05 is the conventional threshold for statistical significance;
        # a p-value below this means we reject H0 and treat the group difference
        # as unlikely to be due to chance alone
        "significant" : p < 0.05,
    })

ttest_df = pd.DataFrame(ttest_rows)
ttest_df.to_csv(OUT_DIR / "linguistic_ttest_results.csv", index=False)
print("  Saved: linguistic_ttest_results.csv")
print(ttest_df[["feature", "gen_z_mean", "older_mean", "p_value", "significant"]].to_string(index=False))

# ============================================================================
# PART 2 — TF-IDF VOCABULARY ANALYSIS
# ============================================================================
print("\n--- Part 2: TF-IDF Vocabulary Analysis ---")

# TF-IDF (Term Frequency–Inverse Document Frequency) weights a term by how often
# it appears in a document (TF) relative to how many documents it appears in (IDF).
# Terms that are common within one group but rare overall get high scores — these
# are the vocabulary items most characteristic of that group's writing.
#
# We fit a separate vectorizer on each age group rather than on the full corpus.
# Fitting on the full corpus would give IDF weights that reflect corpus-wide
# frequency, which would suppress terms that happen to be common in both groups
# even if one group uses them far more. Group-specific fitting isolates each
# group's vocabulary profile.
tfidf_rows = []
for grp in ["gen_z", "older"]:
    texts = df[df[GROUP_COL] == grp][CLEAN_COL].tolist()
    vec   = TfidfVectorizer(max_features=5000, stop_words="english")
    mat   = vec.fit_transform(texts)
    # Mean TF-IDF score across all documents in the group: terms with high mean
    # scores are both distinctive (high IDF) and frequently used (high TF) within
    # this group — a reliable proxy for group-characteristic vocabulary
    means = np.asarray(mat.mean(axis=0)).flatten()
    top30 = means.argsort()[::-1][:30]
    terms = np.array(vec.get_feature_names_out())[top30]
    for term, score in zip(terms, means[top30]):
        tfidf_rows.append({"term": term, "tfidf_score": round(float(score), 6), "age_group": grp})

pd.DataFrame(tfidf_rows).to_csv(OUT_DIR / "tfidf_top_terms.csv", index=False)
print("  Saved: tfidf_top_terms.csv")

# ============================================================================
# PART 3 — CLASSIFICATION
# ============================================================================
print("\n--- Part 3: Classification ---")

# Represent each review as a sparse TF-IDF vector.
# ngram_range=(1, 2) includes both unigrams and bigrams. Bigrams capture short
# phrases and help with negation (e.g. "not great" is one feature, not two
# separate words that cancel each other out). max_features=5000 keeps the
# feature space manageable given the ~22k document corpus.
vec_clf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vec_clf.fit_transform(df[CLEAN_COL])

# Binary target: 1 = Gen Z (ages 18–26), 0 = older (ages 27+)
y = (df[GROUP_COL] == "gen_z").astype(int)

# Stratified split preserves the original class ratio (~7% Gen Z) in both
# the training and test sets. Without stratification, random sampling could
# produce a test set with very few Gen Z examples, making evaluation unreliable.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

def save_report(y_true, y_pred, path: Path) -> dict:
    """
    Compute a per-class classification report and save it to CSV.

    Returns the report as a dict so callers can extract per-class metrics
    (precision, recall, F1) for the comparison table.
    """
    report = classification_report(
        y_true, y_pred,
        target_names=["older", "gen_z"],
        output_dict=True,
    )
    pd.DataFrame(report).transpose().to_csv(path)
    return report

# --- Model 1: Logistic Regression ---
# LR is included because its coefficients are directly interpretable: a positive
# coefficient on a term means that term's presence pushes the model toward
# predicting Gen Z; a negative coefficient pushes toward older. This gives us
# an explainable vocabulary-level signal to discuss in the write-up.
#
# class_weight='balanced' reweights each training sample inversely proportional
# to its class frequency. Without this, a model trained on ~93% older reviews
# could achieve 93% accuracy by always predicting "older" — the weighting forces
# the model to treat Gen Z and older errors equally during training.
print("  Training Logistic Regression …")
lr = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                        random_state=RANDOM_STATE)
lr.fit(X_train, y_train)
y_pred_lr  = lr.predict(X_test)
y_proba_lr = lr.predict_proba(X_test)[:, 1]
report_lr  = save_report(y_test, y_pred_lr, OUT_DIR / "clf_logistic_report.csv")
print("  Saved: clf_logistic_report.csv")

# --- Model 2: Random Forest ---
# RF is a non-linear ensemble that can capture interactions between features
# that LR cannot (e.g. a term that is distinctive only in combination with
# another term). It serves as a performance upper bound to test whether the
# linear assumption of LR is a bottleneck. class_weight='balanced' applied for
# the same imbalance reason as LR above.
print("  Training Random Forest …")
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE,
                             class_weight="balanced")
rf.fit(X_train, y_train)
y_pred_rf  = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]
report_rf  = save_report(y_test, y_pred_rf, OUT_DIR / "clf_rf_report.csv")
print("  Saved: clf_rf_report.csv")

# Comparison CSV
def row_from_report(report_dict: dict, model_name: str) -> dict:
    """Extract per-class precision, recall, and F1 into a flat dict for the comparison CSV."""
    return {
        "model"           : model_name,
        "accuracy"        : round(report_dict["accuracy"], 4),
        "precision_genz"  : round(report_dict["gen_z"]["precision"], 4),
        "recall_genz"     : round(report_dict["gen_z"]["recall"],    4),
        "f1_genz"         : round(report_dict["gen_z"]["f1-score"],  4),
        "precision_older" : round(report_dict["older"]["precision"], 4),
        "recall_older"    : round(report_dict["older"]["recall"],    4),
        "f1_older"        : round(report_dict["older"]["f1-score"],  4),
    }

comparison_df = pd.DataFrame([
    row_from_report(report_lr, "LogisticRegression"),
    row_from_report(report_rf, "RandomForest"),
])
comparison_df.to_csv(OUT_DIR / "clf_comparison.csv", index=False)
print("  Saved: clf_comparison.csv")
print(comparison_df.to_string(index=False))

# ============================================================================
# PART 4 — FIGURES
# ============================================================================
print("\n--- Part 4: Figures ---")

# Figure 1: z-score normalised feature comparison
# The eight features are on very different scales (word_count ~60, exclamation_count ~0.7).
# Plotting raw means side-by-side would make the chart unreadable — small-scale
# features would be invisible next to large-scale ones. We normalise each group
# mean by the overall feature mean and std, so the y-axis represents standard
# deviations from the corpus average, making all features directly comparable.
z_gz, z_old = [], []
for f in FEATURES:
    mu    = df[f].mean()
    sigma = df[f].std()
    if sigma > 0:
        z_gz.append((gz_sub[f].mean()  - mu) / sigma)
        z_old.append((old_sub[f].mean() - mu) / sigma)
    else:
        z_gz.append(0.0)
        z_old.append(0.0)

fig, ax = plt.subplots(**FIG_KW)
x, w = np.arange(len(FEATURES)), 0.35
ax.bar(x - w/2, z_gz,  w, label="gen_z",  color=PALETTE["gen_z"],  alpha=0.85)
ax.bar(x + w/2, z_old, w, label="older", color=PALETTE["older"], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(FEATURES, rotation=35, ha="right")
ax.set_ylabel("Z-Score")
ax.set_title("Linguistic Feature Means by Age Group (Z-Score Normalised)")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.legend()
fig.savefig(OUT_DIR / "ling_feature_comparison.png", bbox_inches="tight")
plt.close(fig)
print("  Saved: ling_feature_comparison.png")

# Figure 2: exclamation count histogram
fig, ax = plt.subplots(**FIG_KW)
clip_val = int(df["exclamation_count"].quantile(0.99)) + 1
for grp in ["gen_z", "older"]:
    vals = df[df[GROUP_COL] == grp]["exclamation_count"].clip(upper=clip_val)
    ax.hist(vals, bins=range(0, clip_val + 2), alpha=0.6, label=grp,
            color=PALETTE[grp], edgecolor="none")
ax.set_title("Exclamation Count Distribution by Age Group")
ax.set_xlabel("Exclamation Count")
ax.set_ylabel("Number of Reviews")
ax.legend()
fig.savefig(OUT_DIR / "ling_exclamation_dist.png", bbox_inches="tight")
plt.close(fig)
print("  Saved: ling_exclamation_dist.png")

# Figure 3: type-token ratio box plot
fig, ax = plt.subplots(**FIG_KW)
plot_data = [df[df[GROUP_COL] == grp]["type_token_ratio"].values for grp in ["gen_z", "older"]]
bp = ax.boxplot(plot_data, labels=["gen_z", "older"], patch_artist=True,
                medianprops=dict(color="black", linewidth=2))
for patch, grp in zip(bp["boxes"], ["gen_z", "older"]):
    patch.set_facecolor(PALETTE[grp])
    patch.set_alpha(0.75)
ax.set_title("Type-Token Ratio by Age Group")
ax.set_ylabel("Type-Token Ratio")
fig.savefig(OUT_DIR / "ling_ttr_boxplot.png", bbox_inches="tight")
plt.close(fig)
print("  Saved: ling_ttr_boxplot.png")

# Figure 4: word count box plot
fig, ax = plt.subplots(**FIG_KW)
plot_data = [df[df[GROUP_COL] == grp]["word_count"].clip(upper=300).values for grp in ["gen_z", "older"]]
bp = ax.boxplot(plot_data, labels=["gen_z", "older"], patch_artist=True,
                medianprops=dict(color="black", linewidth=2))
for patch, grp in zip(bp["boxes"], ["gen_z", "older"]):
    patch.set_facecolor(PALETTE[grp])
    patch.set_alpha(0.75)
ax.set_title("Word Count by Age Group (clipped at 300)")
ax.set_ylabel("Word Count")
fig.savefig(OUT_DIR / "ling_wordcount_boxplot.png", bbox_inches="tight")
plt.close(fig)
print("  Saved: ling_wordcount_boxplot.png")

# Figures 5 & 6: confusion matrices
for model_name, y_pred, fname in [
    ("Logistic Regression", y_pred_lr, "clf_confusion_matrix_lr.png"),
    ("Random Forest",       y_pred_rf, "clf_confusion_matrix_rf.png"),
]:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(**FIG_KW)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["older", "gen_z"],
                yticklabels=["older", "gen_z"],
                ax=ax, linewidths=0.5)
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.savefig(OUT_DIR / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")

# Figure 7: LR top 20 coefficients per class
# In a binary logistic regression, the single coefficient vector (lr.coef_[0])
# represents log-odds of the positive class (Gen Z = 1).
# A large positive coefficient means that term strongly predicts Gen Z.
# A large negative coefficient means that term strongly predicts older.
# This gives us a direct, interpretable window into which vocabulary is most
# diagnostic of each age group — something Random Forest cannot provide.
feature_names = np.array(vec_clf.get_feature_names_out())
coefs         = lr.coef_[0]
top_pos = coefs.argsort()[-20:][::-1]   # highest positive log-odds → gen_z predictors
top_neg = coefs.argsort()[:20]          # most negative log-odds → older predictors
top_idx = np.concatenate([top_pos, top_neg])
top_coefs = coefs[top_idx]
top_terms = feature_names[top_idx]
colors    = [PALETTE["gen_z"] if c > 0 else PALETTE["older"] for c in top_coefs]

fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
yp = np.arange(len(top_terms))
ax.barh(yp, top_coefs, color=colors, alpha=0.85, edgecolor="none")
ax.set_yticks(yp)
ax.set_yticklabels(top_terms, fontsize=9)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Coefficient")
ax.set_title("Top 20 Logistic Regression Coefficients\n(purple = gen_z, pink = older)")
fig.savefig(OUT_DIR / "clf_feature_importance_lr.png", bbox_inches="tight")
plt.close(fig)
print("  Saved: clf_feature_importance_lr.png")

# Figure 8: ROC curves
# The ROC curve plots True Positive Rate (recall for Gen Z) against False Positive
# Rate at every classification threshold. AUC (area under the curve) summarises
# the model's ability to rank Gen Z reviews above older reviews regardless of
# threshold — AUC = 0.5 is random, AUC = 1.0 is perfect.
# AUC is a better summary metric than accuracy here because accuracy is dominated
# by the majority class (older) and can be misleadingly high.
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
auc_lr = auc(fpr_lr, tpr_lr)
auc_rf = auc(fpr_rf, tpr_rf)

fig, ax = plt.subplots(**FIG_KW)
ax.plot(fpr_lr, tpr_lr, color=PALETTE["gen_z"],  lw=2,
        label=f"Logistic Regression (AUC = {auc_lr:.3f})")
ax.plot(fpr_rf, tpr_rf, color=PALETTE["older"], lw=2,
        label=f"Random Forest       (AUC = {auc_rf:.3f})")
ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1, label="Random baseline")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Age Group Classifier")
ax.legend()
fig.savefig(OUT_DIR / "clf_roc_curve.png", bbox_inches="tight")
plt.close(fig)
print("  Saved: clf_roc_curve.png")

# ============================================================================
# DONE
# ============================================================================
created = [
    "outputs/linguistic_features_summary.csv",
    "outputs/linguistic_ttest_results.csv",
    "outputs/tfidf_top_terms.csv",
    "outputs/clf_logistic_report.csv",
    "outputs/clf_rf_report.csv",
    "outputs/clf_comparison.csv",
    "outputs/ling_feature_comparison.png",
    "outputs/ling_exclamation_dist.png",
    "outputs/ling_ttr_boxplot.png",
    "outputs/ling_wordcount_boxplot.png",
    "outputs/clf_confusion_matrix_lr.png",
    "outputs/clf_confusion_matrix_rf.png",
    "outputs/clf_feature_importance_lr.png",
    "outputs/clf_roc_curve.png",
]
print("\n[DONE] Files created:")
for f in created:
    print(f"  {f}")
