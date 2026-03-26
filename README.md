# NLP Group Project — Women's Fashion Review Analysis

> **For Claude Code:** This README is your complete implementation spec. Work through each phase in order. Each phase has clear inputs, outputs, and acceptance criteria. Do not skip phases or combine steps — each builds on the last.

---

## Project Overview

Aspect-based sentiment analysis on ~23,000 women's clothing reviews, comparing **Generation Z (ages 18–26)** vs **older consumers (27+)**. The goal is to identify which product attributes (fit, sizing, quality, price, style, delivery) trigger the strongest emotional reactions in each age group.

**Dataset:** [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) — download manually and place at `data/Womens Clothing E-Commerce Reviews.csv`.

**Candidate numbers:** XKKW4, WDYF5, SQBR7, SSSP0, TLWQ1  
**Members:** Hemakshi (XKKW4), Prishita (WDYF5), Charles (SQBR7), Yash (SSSP0), Diana (TLWQ1)  
**Module:** MSIN0221 — Natural Language Processing

---

## Repo Structure

```
nlp-project/
├── README.md
├── requirements.txt
├── data/                        ← put the dataset CSV here, cleaned CSVs saved here too
├── scripts/                     ← all Python scripts, one per person
│   ├── nlp_1_eda.py             ← Hemakshi
│   ├── nlp_2_sentiment.py       ← Prishita
│   ├── nlp_3_topic_modelling.py ← Charles
│   ├── nlp_4_linguistic.py      ← Yash
│   └── nlp_5_evaluation.py      ← Diana
└── outputs/                     ← all figures and tables saved here
```

---

## Who Does What

| Script | Owner | Branch |
|---|---|---|
| scripts/nlp_1_eda.py | Hemakshi | hemakshi/data-prep |
| scripts/nlp_2_sentiment.py | Prishita | prishita/sentiment |
| scripts/nlp_3_topic_modelling.py | Charles | charles/topic-modelling |
| scripts/nlp_4_linguistic.py | Yash | yash/linguistic-classifier |
| scripts/nlp_5_evaluation.py | Diana | diana/evaluation-report |

Only edit your own script. Run scripts in order — each one depends on the previous.

---

## Setup

```bash
pip install -r requirements.txt
```

**requirements.txt:**

```
pandas
numpy
matplotlib
seaborn
nltk
spacy
vaderSentiment
transformers
torch
gensim
pyLDAvis
scikit-learn
scipy
textblob
wordcloud
tqdm
```

Download required NLTK data at the top of any script that needs it:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
```

---

## Running the Project

```bash
python scripts/nlp_1_eda.py
python scripts/nlp_2_sentiment.py
python scripts/nlp_3_topic_modelling.py
python scripts/nlp_4_linguistic.py
python scripts/nlp_5_evaluation.py
```

Each script prints a `[DONE]` message at the end listing files it created. If a required input file is missing, it will print a clear error message telling you which script to run first.

---

## Phase 1 — Data Loading & EDA (`scripts/nlp_1_eda.py`)

**Owner:** Hemakshi (XKKW4) | **Branch:** `hemakshi/data-prep`  
**Input:** `data/Womens Clothing E-Commerce Reviews.csv`  
**Output:** `data/reviews_clean.csv`, `data/reviews_genz.csv`, `data/reviews_older.csv`

The dataset columns are: `Clothing ID`, `Age`, `Title`, `Review Text`, `Rating`, `Recommended IND`, `Positive Feedback Count`, `Division Name`, `Department Name`, `Class Name`. The primary text column is `Review Text` and the age column is `Age`.

### What to implement

1. Load the dataset from `data/Womens Clothing E-Commerce Reviews.csv` using `pandas.read_csv()`. Print the shape and column names on load.

2. **Age segmentation** — create a new column `age_group`:
   - `"gen_z"` for reviewers aged 18–26 (inclusive)
   - `"older"` for reviewers aged 27+
   - Drop rows where age is missing, null, or outside 18–100

3. **Text cleaning** — apply to the `Review Text` column:
   - Lowercase all text
   - Remove special characters (keep apostrophes for contractions)
   - Strip extra whitespace
   - Remove duplicate reviews (exact text match)
   - Drop reviews with fewer than 10 words

4. **Print a summary to console:**
   ```
   Total reviews after cleaning: XXXX
   Gen Z reviews: XXXX
   Older reviews: XXXX
   Avg review length (Gen Z): XX words
   Avg review length (Older): XX words
   ```

5. **Save three CSVs to `data/`:**
   - `reviews_clean.csv` — full cleaned dataset
   - `reviews_genz.csv` — Gen Z reviews only
   - `reviews_older.csv` — older reviews only
   - Include columns: `Review Text`, `Rating`, `Age`, `age_group`, `Clothing ID`, `Division Name`, `Department Name`, `Class Name`, `Recommended IND`

6. **Figures** — save to `outputs/`:
   - `eda_review_length_dist.png` — histogram of review word counts by age group (overlay, alpha=0.6)
   - `eda_rating_dist.png` — bar chart of star ratings (1–5) for each age group side by side
   - `eda_age_group_counts.png` — bar chart showing review count per age group
   - `eda_wordcloud_genz.png` — word cloud for Gen Z reviews
   - `eda_wordcloud_older.png` — word cloud for older reviews

### Acceptance criteria
- All three CSVs exist in `data/` and load without errors
- No nulls in `Review Text`, `age_group`, or `Rating`
- Both age groups have at least 500 reviews each
- All 5 figures exist in `outputs/`

---

## Phase 2 — Sentiment Analysis (`scripts/nlp_2_sentiment.py`)

**Owner:** Prishita (WDYF5) | **Branch:** `prishita/sentiment`  
**Input:** `data/reviews_clean.csv`  
**Output:** `data/reviews_sentiment.csv`, figures, stats table

### What to implement

1. **VADER baseline:**
   - Use `vaderSentiment.SentimentIntensityAnalyzer`
   - For each review compute: `vader_pos`, `vader_neg`, `vader_neu`, `vader_compound`
   - The compound score (−1 to +1) is the primary sentiment score

2. **DistilBERT sentiment:**
   - Load `distilbert-base-uncased-finetuned-sst-2-english` from HuggingFace `transformers`
   - Process in batches of 32; truncate reviews to 512 tokens
   - Output columns: `distilbert_label` (POSITIVE/NEGATIVE) and `distilbert_score` (confidence 0–1)
   - Add `distilbert_compound`: POSITIVE → score, NEGATIVE → −score (range −1 to +1)
   - Wrap the model call in try/except — if GPU/memory fails, fall back to CPU with batch size 8

3. **Statistical testing:**
   - Run independent t-tests (`scipy.stats.ttest_ind`) comparing Gen Z vs older for `vader_compound` and `distilbert_compound`
   - Save to `outputs/sentiment_ttest_results.csv` with columns: `metric`, `t_stat`, `p_value`, `significant` (True if p < 0.05), `gen_z_mean`, `older_mean`

4. **Figures** — save to `outputs/`:
   - `sentiment_vader_boxplot.png` — box plot of `vader_compound` by age group
   - `sentiment_distilbert_boxplot.png` — box plot of `distilbert_compound` by age group
   - `sentiment_compound_hist.png` — overlaid histogram of compound scores by age group (2-panel subplot)
   - `sentiment_rating_correlation.png` — scatter plot of `vader_compound` vs `Rating`, coloured by age group

5. Save the dataframe with all new sentiment columns to `data/reviews_sentiment.csv`

### Acceptance criteria
- All four sentiment columns present in output CSV
- t-test results CSV has exactly 2 rows
- All 4 figures exist
- Script completes in under 30 minutes on CPU

---

## Phase 3 — Topic Modelling (`scripts/nlp_3_topic_modelling.py`)

**Owner:** Charles (SQBR7) | **Branch:** `charles/topic-modelling`  
**Input:** `data/reviews_sentiment.csv`  
**Output:** `data/reviews_topics.csv`, figures, coherence table

### What to implement

1. **Preprocessing for LDA:**
   - Tokenise using NLTK `word_tokenize`
   - Remove stopwords (`nltk.corpus.stopwords`, English)
   - Lemmatise using `nltk.WordNetLemmatizer`
   - Remove tokens shorter than 3 characters
   - Build a Gensim `Dictionary`, filter extremes: `no_below=5`, `no_above=0.8`
   - Create a bag-of-words corpus

2. **LDA topic search:**
   - Train LDA models for `n_topics` in `[6, 8, 10, 12, 15]` using `gensim.models.LdaModel` with `random_state=42`
   - Compute coherence score for each using `gensim.models.CoherenceModel` (metric: `c_v`)
   - Save coherence scores to `outputs/lda_coherence_scores.csv`
   - Select the model with the highest coherence score
   - Print the top 10 words for each topic

3. **Topic labelling** — edit these labels at the top of the script after inspecting the top words:
   ```python
   TOPIC_LABELS = {
       0: "fit_and_sizing",
       1: "fabric_and_quality",
       2: "style_and_appearance",
       3: "price_and_value",
       4: "delivery_and_service",
       # add more as needed
   }
   ```
   Assign each review its dominant topic as `topic_id` and `topic_label`.

4. **Aspect-level sentiment comparison:**
   - Group by `topic_label` and `age_group`
   - Compute mean `vader_compound` and `distilbert_compound` per group
   - Save to `outputs/aspect_sentiment_by_group.csv`
   - Run a t-test per topic comparing Gen Z vs older on `vader_compound`; save to `outputs/aspect_ttest_results.csv`

5. **Figures** — save to `outputs/`:
   - `lda_coherence_plot.png` — line chart of coherence score vs number of topics
   - `topic_sentiment_heatmap.png` — heatmap: topics × age group, values = mean VADER compound
   - `topic_distribution_bar.png` — stacked bar chart of review proportion per topic by age group
   - `tfidf_top_terms_genz.png` — horizontal bar chart of top 20 TF-IDF terms for Gen Z
   - `tfidf_top_terms_older.png` — horizontal bar chart of top 20 TF-IDF terms for older group

6. Save the dataframe with `topic_id` and `topic_label` to `data/reviews_topics.csv`

### Acceptance criteria
- Coherence CSV has 5 rows (one per n_topics tested)
- Every review has a non-null `topic_label`
- Aspect sentiment CSV has rows for every topic × age group combination
- All 5 figures exist

---

## Phase 4 — Linguistic Analysis & Classifier (`scripts/nlp_4_linguistic.py`)

**Owner:** Yash (SSSP0) | **Branch:** `yash/linguistic-classifier`  
**Input:** `data/reviews_topics.csv`  
**Output:** figures, tables, classification reports

### What to implement

1. **Per-review feature extraction** — add these columns:
   - `word_count` — number of words
   - `char_count` — number of characters (no spaces)
   - `sentence_count` — number of sentences (`nltk.sent_tokenize`)
   - `avg_word_length` — mean word length in characters
   - `exclamation_count` — count of `!` characters
   - `caps_word_count` — count of ALL-CAPS words (≥2 chars, not stopwords)
   - `type_token_ratio` — unique words / total words (lexical diversity)
   - `question_count` — count of `?` characters

2. **Group-level summary:**
   - Compute mean and std for all features grouped by `age_group`
   - Save to `outputs/linguistic_features_summary.csv`
   - Run t-tests for each feature comparing Gen Z vs older
   - Save to `outputs/linguistic_ttest_results.csv` with columns: `feature`, `gen_z_mean`, `older_mean`, `t_stat`, `p_value`, `significant`

3. **TF-IDF vocabulary analysis:**
   - Fit a `TfidfVectorizer` separately on Gen Z and older reviews
   - Extract top 30 terms by mean TF-IDF score for each group
   - Save to `outputs/tfidf_top_terms.csv` with columns: `term`, `tfidf_score`, `age_group`

4. **Classification:**
   - Target: `age_group` (gen_z vs older)
   - Features: TF-IDF on `Review Text`, max 5000 features, unigrams and bigrams
   - Train/test split: 80/20, stratified, `random_state=42`
   - Model 1: `LogisticRegression(max_iter=1000, C=1.0)`
   - Model 2: `RandomForestClassifier(n_estimators=100, random_state=42)`
   - Save classification reports to `outputs/clf_logistic_report.csv` and `outputs/clf_rf_report.csv`
   - Save comparison to `outputs/clf_comparison.csv` with columns: `model`, `accuracy`, `precision_genz`, `recall_genz`, `f1_genz`, `precision_older`, `recall_older`, `f1_older`

5. **Figures** — save to `outputs/`:
   - `ling_feature_comparison.png` — grouped bar chart of Gen Z vs older means (z-score normalised)
   - `ling_exclamation_dist.png` — histogram of exclamation count by age group
   - `ling_ttr_boxplot.png` — box plot of type-token ratio by age group
   - `ling_wordcount_boxplot.png` — box plot of word count by age group
   - `clf_confusion_matrix_lr.png` — confusion matrix for Logistic Regression
   - `clf_confusion_matrix_rf.png` — confusion matrix for Random Forest
   - `clf_feature_importance_lr.png` — top 20 LR coefficients per class
   - `clf_roc_curve.png` — ROC curves for both models on same plot

### Acceptance criteria
- All feature columns present in output
- t-test CSV has one row per feature
- Both classifier reports exist
- All 8 figures exist

---

## Phase 5 — Evaluation, Error Analysis & Report (`scripts/nlp_5_evaluation.py`)

**Owner:** Diana (TLWQ1) | **Branch:** `diana/evaluation-report`  
**Input:** All CSVs in `data/` and `outputs/`  
**Output:** `outputs/error_analysis.csv`, `outputs/results_summary.md`

### What to implement

1. **Error analysis:**
   - Load the best performing model's test predictions (reload from `data/reviews_topics.csv` and rerun the best classifier from Phase 4 to get predictions)
   - Find all misclassified reviews; record: `true_label`, `predicted_label`, `vader_compound`, `topic_label`, `word_count`, `Review Text`
   - Save to `outputs/error_analysis.csv`
   - Print how many Gen Z reviews were predicted as older and vice versa
   - Print the 3 most common topic labels among misclassified reviews

2. **Full evaluation metrics:**
   - Load `outputs/clf_comparison.csv` and print a clean summary of all model results
   - Load `outputs/sentiment_ttest_results.csv` and print significance findings
   - Load `outputs/aspect_ttest_results.csv` and identify which topics had statistically significant differences between age groups

3. **Results summary:**
   - Write `outputs/results_summary.md` — a structured markdown file covering:
     - Dataset size and group breakdown
     - Key sentiment findings (mean scores, t-test results)
     - Top topics and which aspects differ most between groups
     - Classification results (accuracy and F1 for both models)
     - Key misclassification patterns
   - This file is for the write-up — bullet points are fine, no need for polished prose

### Acceptance criteria
- `error_analysis.csv` is non-empty
- `results_summary.md` exists and covers all sections above

---

## Coding Standards

- Use `pathlib.Path` for all file paths
- Set `random_state=42` and `np.random.seed(42)` at the top of every script
- Use `tqdm` for any loop over reviews
- All figures: `figsize=(10, 6)`, `dpi=150`, `bbox_inches='tight'`
- Use `sns.set_style("whitegrid")` for all plots
- Add a docstring at the top of each script describing its inputs and outputs
- Print `[DONE]` at the end of each script listing every file created

---

## Known Constraints

- Assume CPU-only. DistilBERT must use batching and truncation to stay within memory.
- The dataset is ~23k rows — use sparse matrices for TF-IDF throughout.
- LDA results vary between runs — always set `random_state=42`. Topic labels may need manual adjustment via the `TOPIC_LABELS` dict in Phase 3.
