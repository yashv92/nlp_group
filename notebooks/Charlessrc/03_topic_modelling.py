# =============================================================================
# NLP Group Project — Women's E-Commerce Clothing Reviews
# Section: Topic Modelling (LDA) + Sentiment per Topic
# Author: [Your Name]
#
# *** Run nlp_1_eda.py first — this script loads the cleaned CSVs it exports.
#
# To install dependencies, run:
#   pip install pandas matplotlib seaborn nltk gensim wordcloud vaderSentiment scipy
#
# Outputs produced:
#   lda_coherence.png              — coherence score sweep (k = 8–15)
#   lda_wordclouds.png             — word cloud per topic
#   lda_topic_distribution.png    — topic proportions by age group
#   sentiment_heatmap.png          — mean VADER score per topic × group
#   sentiment_by_topic_group.png   — final comparison bar chart with significance
#   topic_sentiment_results.csv    — t-test results table for the report
# =============================================================================

pip install pandas matplotlib seaborn nltk gensim wordcloud vaderSentiment scipy

import re
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.stats import ttest_ind

warnings.filterwarnings("ignore")

for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

# ---------------------------------------------------------------------------
# Colour palette — matches Task 1 for visual consistency
# ---------------------------------------------------------------------------
GZ_COLOR  = "#6C63FF"
OLD_COLOR = "#FF6584"

plt.rcParams.update({"figure.dpi": 130, "font.size": 11})


# =============================================================================
# SECTION 1 — LOAD CLEANED DATA
# =============================================================================
print("=" * 60)
print("SECTION 1: LOADING CLEANED DATA")
print("=" * 60)

# Load the CSVs produced by nlp_1_eda.py
gz_df  = pd.read_csv("genz_reviews.csv")
old_df = pd.read_csv("older_reviews.csv")
df     = pd.concat([gz_df, old_df], ignore_index=True)

TEXT_COL = "review_text"

print(f"\nGen Z  reviews loaded : {len(gz_df):,}")
print(f"Older  reviews loaded : {len(old_df):,}")
print(f"Total                 : {len(df):,}")


# =============================================================================
# SECTION 2 — TOKENISATION
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 2: TOKENISATION")
print("=" * 60)

stop_words = set(stopwords.words("english"))

# Fashion-domain stop words — frequent but carry no topical signal
DOMAIN_STOPWORDS = {
    "dress", "top", "shirt", "wear", "wearing", "wore", "item", "product",
    "order", "ordered", "size", "sized", "buy", "bought", "get", "got",
    "would", "could", "really", "also", "one", "like", "just", "make",
    "look", "looking", "come", "came", "back", "even", "though", "much",
    "little", "great", "good", "love", "nice", "well", "want",
}
stop_words.update(DOMAIN_STOPWORDS)

lemmatiser = WordNetLemmatizer()

def tokenise(text: str) -> list[str]:
    """
    Tokenises a cleaned review string:
      1. Splits on whitespace
      2. Removes stop words and domain-specific noise words
      3. Lemmatises each token
      4. Filters tokens shorter than 3 characters or non-alphabetic
    """
    tokens = str(text).split()
    tokens = [
        lemmatiser.lemmatize(t) for t in tokens
        if t not in stop_words and len(t) >= 3 and t.isalpha()
    ]
    return tokens

print("Tokenising reviews …")
df["tokens"] = df["clean_text"].apply(tokenise)
print("Done.")


# =============================================================================
# SECTION 3 — BUILD GENSIM CORPUS
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 3: BUILDING GENSIM DICTIONARY & CORPUS")
print("=" * 60)

dictionary = corpora.Dictionary(df["tokens"])

# Remove tokens that appear in fewer than 10 documents (too rare to be
# meaningful topics) or in more than 60% of documents (too generic).
dictionary.filter_extremes(no_below=10, no_above=0.6)

corpus = [dictionary.doc2bow(tokens) for tokens in df["tokens"]]

print(f"Dictionary size after filtering : {len(dictionary):,} unique tokens")
print(f"Corpus size                      : {len(corpus):,} documents")


# =============================================================================
# SECTION 4 — COHERENCE SWEEP (k = 8 … 15)
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 4: LDA COHERENCE SWEEP")
print("=" * 60)
print("Testing k = 8 … 15 topics (this may take a few minutes) …\n")

K_RANGE          = range(8, 16)
coherence_scores = []
lda_models       = []

for k in K_RANGE:
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=k,
        random_state=42,
        passes=10,       # number of full passes through the corpus
        alpha="auto",    # asymmetric prior — learned from data
        eta="auto",
        per_word_topics=True,
    )
    cm = CoherenceModel(
        model=lda,
        texts=df["tokens"].tolist(),
        dictionary=dictionary,
        coherence="c_v",   # c_v is the most widely used coherence metric
    )
    score = cm.get_coherence()
    coherence_scores.append(score)
    lda_models.append(lda)
    print(f"  k = {k:2d}  |  coherence (c_v) = {score:.4f}")

# Select k with the highest coherence score
best_idx = int(np.argmax(coherence_scores))
best_k   = list(K_RANGE)[best_idx]
best_lda = lda_models[best_idx]

# ---------------------------------------------------------------------------
# Plot coherence scores
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(list(K_RANGE), coherence_scores,
        marker="o", color=GZ_COLOR, linewidth=2, markersize=7)
ax.axvline(best_k, color=OLD_COLOR, linestyle="--", linewidth=1.5,
           label=f"Optimal k = {best_k}  (c_v = {coherence_scores[best_idx]:.4f})")
ax.set_title("LDA Coherence Score vs. Number of Topics", fontweight="bold")
ax.set_xlabel("Number of Topics (k)")
ax.set_ylabel("Coherence Score (c_v)")
ax.set_xticks(list(K_RANGE))
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("lda_coherence.png", bbox_inches="tight")
plt.show()
print(f"\nOptimal number of topics selected : k = {best_k}")
print("Saved: lda_coherence.png")


# =============================================================================
# SECTION 5 — INSPECT TOPICS & ASSIGN MANUAL LABELS
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 5: TOP WORDS PER TOPIC")
print("=" * 60)
print(f"\nTop 10 words for each of the {best_k} topics:\n")

for topic_id in range(best_k):
    words = [w for w, _ in best_lda.show_topic(topic_id, topn=10)]
    print(f"  Topic {topic_id:2d}: {', '.join(words)}")

# ---------------------------------------------------------------------------
# MANUAL TOPIC LABELS
# After inspecting the top-word output above, update this dictionary with
# your own descriptive labels before proceeding. The example labels below
# are illustrative placeholders.
# If best_k > 8, add entries for the extra topic ids (8, 9, …).
# ---------------------------------------------------------------------------
TOPIC_LABELS = {
    0: "Fit & Sizing",
    1: "Fabric & Quality",
    2: "Style & Aesthetics",
    3: "Price & Value",
    4: "Comfort & Feel",
    5: "Colour & Design",
    6: "Delivery & Returns",
    7: "Occasion & Versatility",
    # 8: "Label for topic 8",
    # 9: "Label for topic 9",
}

def get_label(topic_id: int) -> str:
    return TOPIC_LABELS.get(topic_id, f"Topic {topic_id}")


# =============================================================================
# SECTION 6 — ASSIGN DOMINANT TOPIC TO EACH REVIEW
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 6: ASSIGNING DOMINANT TOPICS")
print("=" * 60)

def dominant_topic(bow) -> int:
    """Returns the topic id with the highest probability for a document."""
    topics = best_lda.get_document_topics(bow, minimum_probability=0)
    return max(topics, key=lambda x: x[1])[0]

print("Assigning dominant topics to reviews …")
df["topic_id"]    = [dominant_topic(bow) for bow in corpus]
df["topic_label"] = df["topic_id"].apply(get_label)
print("Done.")

# Topic frequency summary
print("\nReview counts per topic:")
print(df["topic_label"].value_counts().to_string())


# =============================================================================
# SECTION 7 — WORD CLOUDS PER TOPIC
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 7: WORD CLOUDS")
print("=" * 60)

n_cols = 4
n_rows = (best_k + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()

for topic_id in range(best_k):
    word_weights = dict(best_lda.show_topic(topic_id, topn=40))
    wc = WordCloud(
        width=400, height=300,
        background_color="white",
        colormap="cool",
        max_words=40,
    ).generate_from_frequencies(word_weights)
    axes[topic_id].imshow(wc, interpolation="bilinear")
    axes[topic_id].set_title(
        f"Topic {topic_id}: {get_label(topic_id)}",
        fontsize=10, fontweight="bold",
    )
    axes[topic_id].axis("off")

# Hide unused panels if best_k is not a multiple of n_cols
for i in range(best_k, len(axes)):
    axes[i].set_visible(False)

fig.suptitle("Word Clouds by LDA Topic", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("lda_wordclouds.png", bbox_inches="tight")
plt.show()
print("Saved: lda_wordclouds.png")


# =============================================================================
# SECTION 8 — TOPIC DISTRIBUTION BY AGE GROUP
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 8: TOPIC DISTRIBUTION BY AGE GROUP")
print("=" * 60)

topic_counts = (
    df.groupby(["age_group", "topic_label"])
    .size()
    .reset_index(name="count")
)
topic_totals = df.groupby("age_group").size().reset_index(name="total")
topic_counts = topic_counts.merge(topic_totals, on="age_group")
topic_counts["proportion"] = topic_counts["count"] / topic_counts["total"]

topic_order = [get_label(i) for i in range(best_k)]
x     = np.arange(len(topic_order))
width = 0.35

gz_props  = (topic_counts[topic_counts["age_group"] == "Gen Z (18–26)"]
             .set_index("topic_label"))
old_props = (topic_counts[topic_counts["age_group"] == "Older (27+)"]
             .set_index("topic_label"))

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(
    x - width/2,
    [gz_props.loc[t, "proportion"] if t in gz_props.index else 0 for t in topic_order],
    width, label="Gen Z (18–26)", color=GZ_COLOR, alpha=0.85, edgecolor="white",
)
ax.bar(
    x + width/2,
    [old_props.loc[t, "proportion"] if t in old_props.index else 0 for t in topic_order],
    width, label="Older (27+)", color=OLD_COLOR, alpha=0.85, edgecolor="white",
)
ax.set_xticks(x)
ax.set_xticklabels(topic_order, rotation=30, ha="right")
ax.set_ylabel("Proportion of Reviews")
ax.set_title("Topic Distribution by Age Group", fontweight="bold")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("lda_topic_distribution.png", bbox_inches="tight")
plt.show()
print("Saved: lda_topic_distribution.png")


# =============================================================================
# SECTION 9 — VADER SENTIMENT SCORES
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 9: VADER SENTIMENT PER REVIEW")
print("=" * 60)

# VADER (Hutto & Gilbert, 2014) produces a compound score in [-1, 1]:
#   +1 = maximally positive,  -1 = maximally negative,  0 = neutral.
# It is applied to the original (non-normalised) review text to preserve
# punctuation cues that VADER uses for intensity scoring.

analyser = SentimentIntensityAnalyzer()

print("Computing VADER compound scores …")
df["vader_compound"] = df[TEXT_COL].astype(str).apply(
    lambda t: analyser.polarity_scores(t)["compound"]
)
print("Done.")

# Aggregate mean sentiment per topic and age group
sentiment_agg = (
    df.groupby(["topic_label", "age_group"])["vader_compound"]
    .agg(mean_sentiment="mean", std_sentiment="std", n="count")
    .reset_index()
)

print("\nMean VADER compound sentiment per topic and age group:")
print(sentiment_agg.to_string(index=False))


# =============================================================================
# SECTION 10 — INDEPENDENT T-TESTS (Gen Z vs Older, per topic)
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 10: STATISTICAL SIGNIFICANCE TESTS")
print("=" * 60)
print("\nWelch's independent t-test — Gen Z vs Older, per topic:\n")

ttest_records = []

for topic in [get_label(i) for i in range(best_k)]:
    gz_scores  = df[(df["topic_label"] == topic) &
                    (df["age_group"] == "Gen Z (18–26)")]["vader_compound"]
    old_scores = df[(df["topic_label"] == topic) &
                    (df["age_group"] == "Older (27+)")]["vader_compound"]

    if len(gz_scores) > 1 and len(old_scores) > 1:
        # Welch's t-test (equal_var=False) does not assume equal group variances
        t_stat, p_val = ttest_ind(gz_scores, old_scores, equal_var=False)
        sig = ("***" if p_val < 0.001 else
               "**"  if p_val < 0.01  else
               "*"   if p_val < 0.05  else "ns")
        ttest_records.append({
            "Topic"        : topic,
            "GZ mean"      : round(gz_scores.mean(),  4),
            "Older mean"   : round(old_scores.mean(), 4),
            "Difference"   : round(gz_scores.mean() - old_scores.mean(), 4),
            "t-statistic"  : round(t_stat, 4),
            "p-value"      : round(p_val,  4),
            "Significance" : sig,
        })
        print(f"  {topic:<28}  t={t_stat:+.3f}  p={p_val:.4f}  {sig}")

ttest_df = pd.DataFrame(ttest_records).sort_values("p-value")

# Export full results table
ttest_df.to_csv("topic_sentiment_results.csv", index=False)
print("\nExported: topic_sentiment_results.csv")


# =============================================================================
# SECTION 11 — SENTIMENT HEATMAP
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 11: SENTIMENT HEATMAP")
print("=" * 60)

pivot = sentiment_agg.pivot(
    index="topic_label", columns="age_group", values="mean_sentiment"
)

fig, ax = plt.subplots(figsize=(8, 0.55 * best_k + 2))
sns.heatmap(
    pivot, annot=True, fmt=".3f", cmap="RdYlGn",
    center=0, vmin=-0.5, vmax=0.5,
    linewidths=0.5, linecolor="white",
    cbar_kws={"label": "Mean VADER Compound Score"},
    ax=ax,
)
ax.set_title("Mean Sentiment Score per Topic by Age Group", fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig("sentiment_heatmap.png", bbox_inches="tight")
plt.show()
print("Saved: sentiment_heatmap.png")

# ---------------------------------------------------------------------------
# Emotional intensity table (mean absolute compound score)
# A high |compound| score regardless of sign = strong emotional reaction.
# ---------------------------------------------------------------------------
intensity = (
    df.groupby(["topic_label", "age_group"])["vader_compound"]
    .apply(lambda x: x.abs().mean())
    .reset_index(name="mean_abs_sentiment")
)
pivot_intensity = intensity.pivot(
    index="topic_label", columns="age_group", values="mean_abs_sentiment"
).round(4)
pivot_intensity["Strongest reaction"] = pivot_intensity.idxmax(axis=1)

print("\nEmotional Intensity per Topic (mean |compound score|):")
print(pivot_intensity.to_string())


# =============================================================================
# SECTION 12 — FINAL COMPARISON BAR CHART
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 12: FINAL COMPARISON PLOT")
print("=" * 60)

topic_order_sorted = ttest_df["Topic"].tolist()   # sorted by p-value
x     = np.arange(len(topic_order_sorted))
width = 0.35

t_indexed  = ttest_df.set_index("Topic")
gz_means   = t_indexed.loc[topic_order_sorted, "GZ mean"]
old_means  = t_indexed.loc[topic_order_sorted, "Older mean"]
sigs       = t_indexed.loc[topic_order_sorted, "Significance"]

fig, ax = plt.subplots(figsize=(12, 5))
bars1 = ax.bar(x - width/2, gz_means,  width,
               label="Gen Z (18–26)", color=GZ_COLOR,  alpha=0.85, edgecolor="white")
bars2 = ax.bar(x + width/2, old_means, width,
               label="Older (27+)",   color=OLD_COLOR, alpha=0.85, edgecolor="white")

# Annotate significance stars above each bar pair
for i, (b1, b2, sig) in enumerate(zip(bars1, bars2, sigs)):
    if sig != "ns":
        y_top = max(b1.get_height(), b2.get_height()) + 0.01
        ax.text(i, y_top, sig, ha="center", va="bottom",
                fontsize=9, color="#333333")

ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(topic_order_sorted, rotation=30, ha="right")
ax.set_ylabel("Mean VADER Compound Score")
ax.set_title(
    "Mean Sentiment by Topic and Age Group\n(* p<.05  ** p<.01  *** p<.001)",
    fontweight="bold",
)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("sentiment_by_topic_group.png", bbox_inches="tight")
plt.show()
print("Saved: sentiment_by_topic_group.png")

print("\n" + "=" * 60)
print("LDA pipeline complete. Files produced:")
print("  lda_coherence.png              — coherence score sweep")
print("  lda_wordclouds.png             — word clouds per topic")
print("  lda_topic_distribution.png    — topic proportions by age group")
print("  sentiment_heatmap.png          — sentiment heatmap")
print("  sentiment_by_topic_group.png   — final bar chart with significance")
print("  topic_sentiment_results.csv    — t-test results table")
print("=" * 60)
