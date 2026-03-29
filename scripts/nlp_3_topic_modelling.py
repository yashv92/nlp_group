"""
nlp_3_topic_modelling.py — Phase 3: LDA Topic Modelling + Aspect-Level Sentiment

Owner: Charles (SQBR7) | Branch: charles/topic-modelling

Input:
    data/processed/reviews_sentiment.csv   — output of Phase 2 (Prishita)
    Must contain: review_text_clean, age_group, vader_compound, distilbert_compound

Output:
    data/processed/reviews_topics.csv      — full dataset with topic_id and topic_label
    outputs/lda_coherence_scores.csv       — coherence score for each k tested
    outputs/aspect_sentiment_by_group.csv  — mean sentiment per topic x age group
    outputs/aspect_ttest_results.csv       — per-topic t-tests (gen_z vs older)
    outputs/lda_coherence_plot.png
    outputs/topic_sentiment_heatmap.png
    outputs/topic_distribution_bar.png
    outputs/tfidf_top_terms_genz.png
    outputs/tfidf_top_terms_older.png
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
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import ttest_ind
from tqdm import tqdm

warnings.filterwarnings("ignore")
np.random.seed(42)
sns.set_style("whitegrid")

for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR  = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_KW       = dict(figsize=(10, 6), dpi=150)
PALETTE      = {"gen_z": "#6C63FF", "older": "#FF6584"}
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
sentiment_csv = PROC_DIR / "reviews_sentiment.csv"
clean_csv     = PROC_DIR / "reviews_clean.csv"

if sentiment_csv.exists():
    print(f"Loading {sentiment_csv.name} ...")
    df = pd.read_csv(sentiment_csv)
elif clean_csv.exists():
    print("reviews_sentiment.csv not found -- falling back to reviews_clean.csv ...")
    df = pd.read_csv(clean_csv)
else:
    print("ERROR: Run nlp_1_eda.py first")
    sys.exit(1)

CLEAN_COL = "review_text_clean" if "review_text_clean" in df.columns else "review_text"
GROUP_COL = "age_group"

df = df.dropna(subset=[CLEAN_COL, GROUP_COL]).reset_index(drop=True)
df[GROUP_COL] = df[GROUP_COL].str.strip().str.lower()
df = df[df[GROUP_COL].isin(["gen_z", "older"])].reset_index(drop=True)

HAS_VADER      = "vader_compound"      in df.columns
HAS_DISTILBERT = "distilbert_compound" in df.columns

print(f"  Rows: {len(df):,}  |  Gen Z: {(df[GROUP_COL]=='gen_z').sum():,}  "
      f"|  Older: {(df[GROUP_COL]=='older').sum():,}")
print(f"  VADER available: {HAS_VADER}  |  DistilBERT available: {HAS_DISTILBERT}")

# ============================================================================
# PART 1 -- TOKENISATION
# ============================================================================
print("\n--- Part 1: Tokenisation ---")

stop_words = set(stopwords.words("english"))

# Fashion-domain stopwords: highly frequent in clothing reviews but carry no
# topical signal -- removing them stops topics from being dominated by generic
# review vocabulary and helps LDA find substantive aspect-level themes
DOMAIN_STOPWORDS = {
    "dress", "top", "shirt", "wear", "wearing", "wore", "item", "product",
    "order", "ordered", "size", "sized", "buy", "bought", "get", "got",
    "would", "could", "really", "also", "one", "like", "just", "make",
    "look", "looking", "come", "came", "back", "even", "though", "much",
    "little", "great", "good", "love", "nice", "well", "want",
}
stop_words.update(DOMAIN_STOPWORDS)
lemmatiser = WordNetLemmatizer()

def tokenise(text: str) -> list:
    """
    Tokenise a single review for LDA input.

    Steps: split on whitespace -> remove stopwords and domain noise words ->
    lemmatise -> keep alphabetic tokens of 3+ characters.

    The 3-character minimum filters out noise tokens and most abbreviations
    that don't contribute to interpretable topics.
    """
    tokens = str(text).split()
    return [
        lemmatiser.lemmatize(t) for t in tokens
        if t not in stop_words and len(t) >= 3 and t.isalpha()
    ]

print("Tokenising reviews ...")
df["tokens"] = [tokenise(t) for t in tqdm(df[CLEAN_COL], desc="  tokenise")]

# ============================================================================
# PART 2 -- GENSIM DICTIONARY & CORPUS
# ============================================================================
print("\n--- Part 2: Gensim Dictionary & Corpus ---")

dictionary = corpora.Dictionary(df["tokens"])
# no_below=5: exclude tokens seen in fewer than 5 documents (too rare / noisy)
# no_above=0.8: exclude tokens in more than 80% of documents (too generic)
dictionary.filter_extremes(no_below=5, no_above=0.8)
corpus = [dictionary.doc2bow(tokens) for tokens in df["tokens"]]

print(f"  Vocabulary: {len(dictionary):,} tokens  |  Corpus: {len(corpus):,} documents")

# ============================================================================
# PART 3 -- COHERENCE SWEEP k = [6, 8, 10, 12, 15]
# ============================================================================
print("\n--- Part 3: LDA Coherence Sweep ---")
print("Testing k = 6, 8, 10, 12, 15 ... (this will take several minutes on CPU)\n")

# c_v coherence is based on normalised pointwise mutual information and
# sliding window co-occurrence; it correlates best with human topic judgement
K_VALUES         = [6, 8, 10, 12, 15]
coherence_scores = []
lda_models       = []

for k in K_VALUES:
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=k,
        random_state=RANDOM_STATE,
        passes=10,
        alpha="auto",
        eta="auto",
        per_word_topics=True,
    )
    cm = CoherenceModel(
        model=lda,
        texts=df["tokens"].tolist(),
        dictionary=dictionary,
        coherence="c_v",
        processes=1,
    )
    score = cm.get_coherence()
    coherence_scores.append(score)
    lda_models.append(lda)
    print(f"  k = {k:2d}  |  c_v = {score:.4f}")

best_idx = int(np.argmax(coherence_scores))
best_k   = K_VALUES[best_idx]
best_lda = lda_models[best_idx]
print(f"\n  Best k = {best_k}  (c_v = {coherence_scores[best_idx]:.4f})")

pd.DataFrame({"n_topics": K_VALUES, "coherence_cv": coherence_scores}).to_csv(
    OUT_DIR / "lda_coherence_scores.csv", index=False)
print("  Saved: lda_coherence_scores.csv")

# ============================================================================
# PART 4 -- TOPIC LABELS
# ============================================================================
print(f"\n--- Part 4: Top Words per Topic (k={best_k}) ---")
for tid in range(best_k):
    words = [w for w, _ in best_lda.show_topic(tid, topn=10)]
    print(f"  Topic {tid:2d}: {', '.join(words)}")

# Edit these labels after inspecting the top words printed above.
# Entries beyond best_k are ignored; add more if best_k > 8.
TOPIC_LABELS = {
    0:  "fit_and_sizing",
    1:  "fabric_and_quality",
    2:  "style_and_appearance",
    3:  "price_and_value",
    4:  "delivery_and_service",
    5:  "comfort_and_feel",
    6:  "colour_and_design",
    7:  "occasion_and_versatility",
    8:  "returns_and_exchange",
    9:  "gifting_and_occasion",
    10: "brand_and_reputation",
    11: "online_shopping_experience",
    12: "body_fit_and_proportion",
    13: "washing_and_care",
    14: "gift_and_recommendation",
}

def get_label(topic_id: int) -> str:
    return TOPIC_LABELS.get(topic_id, f"topic_{topic_id}")

# ============================================================================
# PART 5 -- ASSIGN DOMINANT TOPIC
# ============================================================================
print("\n--- Part 5: Assigning Dominant Topics ---")

def dominant_topic(bow) -> int:
    """Return the topic id with the highest posterior probability for this document."""
    topics = best_lda.get_document_topics(bow, minimum_probability=0)
    return max(topics, key=lambda x: x[1])[0]

df["topic_id"]    = [dominant_topic(bow) for bow in tqdm(corpus, desc="  assign")]
df["topic_label"] = df["topic_id"].apply(get_label)

print("\n  Review counts per topic:")
print(df["topic_label"].value_counts().to_string())

# ============================================================================
# PART 6 -- ASPECT-LEVEL SENTIMENT
# ============================================================================
print("\n--- Part 6: Aspect-Level Sentiment ---")

compound_cols = []
if HAS_VADER:
    compound_cols.append("vader_compound")
if HAS_DISTILBERT:
    compound_cols.append("distilbert_compound")

if compound_cols:
    aspect_rows = []
    for topic in df["topic_label"].unique():
        for grp in ["gen_z", "older"]:
            sub = df[(df["topic_label"] == topic) & (df[GROUP_COL] == grp)]
            row = {"topic_label": topic, "age_group": grp, "n": len(sub)}
            for col in compound_cols:
                row[f"mean_{col}"] = round(sub[col].mean(), 4) if len(sub) > 0 else np.nan
            aspect_rows.append(row)

    aspect_df = pd.DataFrame(aspect_rows)
    aspect_df.to_csv(OUT_DIR / "aspect_sentiment_by_group.csv", index=False)
    print("  Saved: aspect_sentiment_by_group.csv")

    # Welch's t-test per topic: do gen_z and older differ on VADER compound?
    ttest_rows = []
    for topic in df["topic_label"].unique():
        gz_vals  = df[(df["topic_label"] == topic) &
                      (df[GROUP_COL] == "gen_z")]["vader_compound"].dropna()
        old_vals = df[(df["topic_label"] == topic) &
                      (df[GROUP_COL] == "older")]["vader_compound"].dropna()
        if len(gz_vals) > 1 and len(old_vals) > 1:
            t, p = ttest_ind(gz_vals, old_vals, equal_var=False)
            ttest_rows.append({
                "topic_label": topic,
                "gen_z_mean" : round(gz_vals.mean(),  4),
                "older_mean" : round(old_vals.mean(), 4),
                "t_stat"     : round(t, 4),
                "p_value"    : round(p, 4),
                "significant": p < 0.05,
            })

    pd.DataFrame(ttest_rows).to_csv(OUT_DIR / "aspect_ttest_results.csv", index=False)
    print("  Saved: aspect_ttest_results.csv")

# ============================================================================
# PART 7 -- SAVE reviews_topics.csv
# ============================================================================
df.drop(columns=["tokens"], errors="ignore").to_csv(
    PROC_DIR / "reviews_topics.csv", index=False)
print(f"\n  Saved: data/processed/reviews_topics.csv  ({len(df):,} rows)")

# ============================================================================
# PART 8 -- FIGURES
# ============================================================================
print("\n--- Part 8: Figures ---")

# Figure 1: coherence plot
fig, ax = plt.subplots(**FIG_KW)
ax.plot(K_VALUES, coherence_scores, marker="o", color=PALETTE["gen_z"],
        linewidth=2, markersize=7)
ax.axvline(best_k, color=PALETTE["older"], linestyle="--", linewidth=1.5,
           label=f"Best k = {best_k}  (c_v = {coherence_scores[best_idx]:.4f})")
ax.set_title("LDA Coherence Score vs Number of Topics")
ax.set_xlabel("Number of Topics (k)")
ax.set_ylabel("Coherence Score (c_v)")
ax.set_xticks(K_VALUES)
ax.legend()
fig.savefig(OUT_DIR / "lda_coherence_plot.png", bbox_inches="tight")
plt.close(fig)
print("  Saved: lda_coherence_plot.png")

# Figure 2: topic sentiment heatmap
if HAS_VADER and compound_cols:
    pivot = aspect_df.pivot(index="topic_label", columns="age_group",
                            values="mean_vader_compound").dropna()
    fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.5 + 2)), dpi=150)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                center=0, vmin=-0.5, vmax=0.5,
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Mean VADER Compound"},
                ax=ax)
    ax.set_title("Mean VADER Sentiment per Topic x Age Group")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.savefig(OUT_DIR / "topic_sentiment_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: topic_sentiment_heatmap.png")

# Figure 3: topic distribution bar chart
topic_order = [get_label(i) for i in range(best_k)]
cnt = (df.groupby([GROUP_COL, "topic_label"]).size().reset_index(name="count"))
tot = df.groupby(GROUP_COL).size().reset_index(name="total")
cnt = cnt.merge(tot, on=GROUP_COL)
cnt["proportion"] = cnt["count"] / cnt["total"]

fig, ax = plt.subplots(**FIG_KW)
x, w = np.arange(len(topic_order)), 0.35
for offset, grp in [(-w/2, "gen_z"), (w/2, "older")]:
    vals = []
    for t in topic_order:
        row = cnt[(cnt[GROUP_COL] == grp) & (cnt["topic_label"] == t)]
        vals.append(float(row["proportion"].values[0]) if len(row) else 0.0)
    ax.bar(x + offset, vals, w, label=grp, color=PALETTE[grp], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(topic_order, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Proportion of Reviews")
ax.set_title("Topic Distribution by Age Group")
ax.legend()
fig.savefig(OUT_DIR / "topic_distribution_bar.png", bbox_inches="tight")
plt.close(fig)
print("  Saved: topic_distribution_bar.png")

# Figures 4 & 5: TF-IDF top 20 terms per age group
for grp, fname in [("gen_z", "tfidf_top_terms_genz.png"),
                   ("older", "tfidf_top_terms_older.png")]:
    texts  = df[df[GROUP_COL] == grp][CLEAN_COL].tolist()
    vec    = TfidfVectorizer(max_features=5000, stop_words="english")
    mat    = vec.fit_transform(texts)
    means  = np.asarray(mat.mean(axis=0)).flatten()
    top20  = means.argsort()[::-1][:20]
    terms  = np.array(vec.get_feature_names_out())[top20]
    scores = means[top20]

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    yp = np.arange(len(terms))
    ax.barh(yp, scores[::-1], color=PALETTE[grp], alpha=0.85, edgecolor="none")
    ax.set_yticks(yp)
    ax.set_yticklabels(terms[::-1], fontsize=9)
    ax.set_xlabel("Mean TF-IDF Score")
    ax.set_title(f"Top 20 TF-IDF Terms -- {grp}")
    fig.savefig(OUT_DIR / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")

# ============================================================================
# DONE
# ============================================================================
created = [
    "data/processed/reviews_topics.csv",
    "outputs/lda_coherence_scores.csv",
    "outputs/aspect_sentiment_by_group.csv",
    "outputs/aspect_ttest_results.csv",
    "outputs/lda_coherence_plot.png",
    "outputs/topic_sentiment_heatmap.png",
    "outputs/topic_distribution_bar.png",
    "outputs/tfidf_top_terms_genz.png",
    "outputs/tfidf_top_terms_older.png",
]
print("\n[DONE] Files created:")
for f in created:
    print(f"  {f}")
