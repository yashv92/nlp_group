# NLP Group Project — Women's Fashion Review Analysis

> **For Claude Code:** This README is your complete implementation spec. Work through each phase in order. Each phase has clear inputs, outputs, and acceptance criteria. Do not skip phases or combine steps — each builds on the last.

---

## Project Overview

Aspect-based sentiment analysis on ~23,000 women's clothing reviews, comparing **Generation Z (ages 18–26)** vs **older consumers (27+)**. The goal is to identify which product attributes (fit, sizing, quality, price, style, delivery) trigger the strongest emotional reactions in each age group.

**Dataset:** [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) — available on Kaggle. Download manually and place at `data/raw/Womens Clothing E-Commerce Reviews.csv`.

**Candidate numbers:** XKKW4, WDYF5, SQBR7, SSSP0, TLWQ1  
**Module:** MSIN0221 — Natural Language Processing

---

## Repo Structure

```
nlp-project/
├── README.md                  ← you are here
├── requirements.txt
├── data/
│   ├── raw/                   ← original downloaded data (do not modify)
│   └── processed/             ← outputs from phase 1
├── notebooks/                 ← optional exploratory notebooks
├── src/
│   ├── 01_data_prep.py
│   ├── 02_sentiment.py
│   ├── 03_topic_modelling.py
│   ├── 04_linguistic.py
│   └── 05_classification.py
├── outputs/
│   ├── figures/               ← all plots saved here as .png
│   ├── tables/                ← all result tables saved as .csv
│   └── models/                ← saved model artefacts
└── report/
    └── results_summary.md     ← auto-generated summary of key findings
```

---

## Setup

```bash
pip install -r requirements.txt
```

**requirements.txt** should contain:

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

## Phase 1 — Data Preparation (`src/01_data_prep.py`)

**Owner:** XKKW4  
**Input:** Raw Hugging Face dataset  
**Output:** `data/processed/reviews_clean.csv`

### What to implement

1. **Load the dataset** from `data/raw/Womens Clothing E-Commerce Reviews.csv` using `pandas.read_csv()`. The dataset has these columns: `Clothing ID`, `Age`, `Title`, `Review Text`, `Rating`, `Recommended IND`, `Positive Feedback Count`, `Division Name`, `Department Name`, `Class Name`. Print the shape and column names on load. The primary text column is `Review Text` and the age column is `Age`.

2. **Age segmentation** — create a new column `age_group`:
   - `"gen_z"` for reviewers aged 18–26 (inclusive)
   - `"older"` for reviewers aged 27+
   - Drop rows where age is missing, null, or outside 18–100 (treat as invalid)

3. **Text cleaning** — apply to the review text column:
   - Lowercase all text
   - Remove special characters (keep apostrophes for contractions)
   - Strip extra whitespace
   - Remove duplicate reviews (exact text match)
   - Drop reviews with fewer than 10 words

4. **EDA — generate and save the following figures** to `outputs/figures/`:
   - `eda_review_length_dist.png` — histogram of review word counts, split by age group (overlay, alpha=0.6)
   - `eda_rating_dist.png` — bar chart of star ratings (1–5) for each age group side by side
   - `eda_age_group_counts.png` — bar chart showing how many reviews are in each group
   - `eda_wordcloud_genz.png` — word cloud for Gen Z reviews
   - `eda_wordcloud_older.png` — word cloud for older reviews

5. **Print a summary to console:**
   ```
   Total reviews after cleaning: XXXX
   Gen Z reviews: XXXX
   Older reviews: XXXX
   Avg review length (Gen Z): XX words
   Avg review length (Older): XX words
   ```

6. **Save** the cleaned dataframe to `data/processed/reviews_clean.csv`. Include columns: `Review Text`, `Rating`, `Age`, `age_group`, `Clothing ID`, `Division Name`, `Department Name`, `Class Name`, `Recommended IND`.

### Acceptance criteria
- `reviews_clean.csv` exists and loads without errors
- No nulls in `review_text`, `age_group`, or `rating`
- Both age groups have at least 500 reviews each
- All 5 figures exist in `outputs/figures/`

---

## Phase 2 — Sentiment Analysis (`src/02_sentiment.py`)

**Owner:** WDYF5  
**Input:** `data/processed/reviews_clean.csv`  
**Output:** `data/processed/reviews_with_sentiment.csv`, figures, stats table

### What to implement

1. **VADER baseline:**
   - Use `vaderSentiment.SentimentIntensityAnalyzer`
   - For each review compute: `vader_pos`, `vader_neg`, `vader_neu`, `vader_compound`
   - The compound score (−1 to +1) is the primary sentiment score

2. **DistilBERT sentiment:**
   - Load `distilbert-base-uncased-finetuned-sst-2-english` from HuggingFace `transformers`
   - For efficiency, process in batches of 32; truncate reviews to 512 tokens
   - Output a column `distilbert_label` (POSITIVE/NEGATIVE) and `distilbert_score` (confidence 0–1)
   - Add a column `distilbert_compound`: map POSITIVE → score, NEGATIVE → −score (so range is −1 to +1, matching VADER)
   - **Important:** wrap the model call in a try/except — if GPU/memory fails, fall back to CPU with batch size 8

3. **Statistical testing:**
   - Run independent t-tests (use `scipy.stats.ttest_ind`) comparing Gen Z vs older for:
     - `vader_compound`
     - `distilbert_compound`
   - Save results to `outputs/tables/sentiment_ttest_results.csv` with columns: `metric`, `t_stat`, `p_value`, `significant` (True if p < 0.05), `gen_z_mean`, `older_mean`

4. **Figures** — save to `outputs/figures/`:
   - `sentiment_vader_boxplot.png` — box plot of `vader_compound` by age group
   - `sentiment_distilbert_boxplot.png` — box plot of `distilbert_compound` by age group
   - `sentiment_compound_hist.png` — overlaid histogram of compound scores by age group (both models, 2-panel subplot)
   - `sentiment_rating_correlation.png` — scatter plot of `vader_compound` vs `rating`, coloured by age group

5. **Save** the dataframe with new sentiment columns to `data/processed/reviews_with_sentiment.csv`

### Acceptance criteria
- All four sentiment columns present in output CSV
- t-test results CSV has exactly 2 rows (one per metric)
- All 4 figures exist
- Script completes in under 30 minutes on CPU

---

## Phase 3 — Topic Modelling & Aspect Extraction (`src/03_topic_modelling.py`)

**Owner:** SQBR7  
**Input:** `data/processed/reviews_with_sentiment.csv`  
**Output:** `data/processed/reviews_with_topics.csv`, figures, coherence table

### What to implement

1. **Preprocessing for LDA:**
   - Tokenise using NLTK `word_tokenize`
   - Remove stopwords (`nltk.corpus.stopwords`, English)
   - Lemmatise using `nltk.WordNetLemmatizer`
   - Remove tokens shorter than 3 characters
   - Build a Gensim `Dictionary` and filter extremes: `no_below=5`, `no_above=0.8`
   - Create a bag-of-words corpus

2. **LDA topic search:**
   - Train LDA models for `n_topics` in `[6, 8, 10, 12, 15]` using `gensim.models.LdaModel`
   - For each, compute coherence score using `gensim.models.CoherenceModel` (metric: `c_v`)
   - Save coherence scores to `outputs/tables/lda_coherence_scores.csv`
   - Select the `n_topics` with the highest coherence score as the final model
   - Print the top 10 words for each topic

3. **Topic labelling:**
   - After finding the best model, manually assign human-readable labels to topics based on their top words. Use this mapping logic — hardcode sensible defaults based on expected fashion review topics, but make the labels easy to edit at the top of the script:
     ```python
     # Edit these labels after inspecting top words from the model
     TOPIC_LABELS = {
         0: "fit_and_sizing",
         1: "fabric_and_quality",
         2: "style_and_appearance",
         3: "price_and_value",
         4: "delivery_and_service",
         # add more as needed based on n_topics
     }
     ```
   - Assign each review its dominant topic (highest probability) as `topic_id` and `topic_label`

4. **Aspect-level sentiment comparison:**
   - Group by `topic_label` and `age_group`
   - Compute mean `vader_compound` and `distilbert_compound` per group
   - Save to `outputs/tables/aspect_sentiment_by_group.csv`
   - For each topic, run a t-test comparing Gen Z vs older on `vader_compound`; save to `outputs/tables/aspect_ttest_results.csv`

5. **TF-IDF vocabulary analysis:**
   - Fit a `sklearn TfidfVectorizer` separately on Gen Z and older reviews
   - Extract the top 30 terms by mean TF-IDF score for each group
   - Save to `outputs/tables/tfidf_top_terms.csv` with columns: `term`, `tfidf_score`, `age_group`

6. **Figures** — save to `outputs/figures/`:
   - `lda_coherence_plot.png` — line chart of coherence score vs number of topics
   - `topic_sentiment_heatmap.png` — heatmap: topics (rows) × age group (columns), values = mean VADER compound score
   - `topic_distribution_bar.png` — stacked bar chart showing proportion of reviews per topic, split by age group
   - `tfidf_top_terms_genz.png` — horizontal bar chart of top 20 TF-IDF terms for Gen Z
   - `tfidf_top_terms_older.png` — horizontal bar chart of top 20 TF-IDF terms for older group

7. **Save** the dataframe with `topic_id` and `topic_label` columns to `data/processed/reviews_with_topics.csv`

### Acceptance criteria
- Final LDA model selected and coherence CSV has 5 rows
- Every review has a non-null `topic_label`
- Aspect sentiment CSV has rows for every combination of topic × age group
- All 5 figures exist

---

## Phase 4 — Linguistic Feature Analysis (`src/04_linguistic.py`)

**Owner:** SSSP0  
**Input:** `data/processed/reviews_with_topics.csv`  
**Output:** `outputs/tables/linguistic_features.csv`, figures

### What to implement

1. **Per-review feature extraction** — add these columns to the dataframe:
   - `word_count` — number of words
   - `char_count` — number of characters (no spaces)
   - `sentence_count` — number of sentences (use `nltk.sent_tokenize`)
   - `avg_word_length` — mean word length in characters
   - `exclamation_count` — count of `!` characters
   - `caps_word_count` — count of ALL-CAPS words (≥2 chars, not stopwords)
   - `type_token_ratio` — unique words / total words (lexical diversity)
   - `question_count` — count of `?` characters
   - `emotionally_charged_count` — count of words appearing in a predefined emotion lexicon (use the NRC Emotion Lexicon if accessible, otherwise use a simple wordlist of common positive/negative emotion words defined at the top of the script)

2. **Group-level summary** — compute mean and std for all features above, grouped by `age_group`. Save to `outputs/tables/linguistic_features_summary.csv`.

3. **Statistical tests:**
   - Run t-tests for each feature comparing Gen Z vs older
   - Save to `outputs/tables/linguistic_ttest_results.csv` with columns: `feature`, `gen_z_mean`, `older_mean`, `t_stat`, `p_value`, `significant`

4. **Figures** — save to `outputs/figures/`:
   - `ling_feature_comparison.png` — grouped bar chart showing Gen Z vs older means for each feature (normalise features to z-scores so they're on the same scale)
   - `ling_exclamation_dist.png` — histogram of exclamation count per review by age group
   - `ling_ttr_boxplot.png` — box plot of type-token ratio by age group
   - `ling_wordcount_boxplot.png` — box plot of word count by age group
   - `ling_caps_by_topic.png` — bar chart of mean caps word count per topic, split by age group

### Acceptance criteria
- All 9 feature columns present in the output
- t-test CSV has one row per feature
- All 5 figures exist

---

## Phase 5 — Classification & Evaluation (`src/05_classification.py`)

**Owner:** TLWQ1  
**Input:** `data/processed/reviews_with_topics.csv`  
**Output:** classification reports, error analysis, final summary

### What to implement

1. **Feature preparation:**
   - Target variable: `age_group` (binary: `gen_z` vs `older`)
   - Features: TF-IDF on `review_text` (max 5000 features, unigrams + bigrams)
   - Train/test split: 80/20, stratified by `age_group`, random state = 42
   - Print class balance in train and test sets

2. **Baseline — Logistic Regression:**
   - `sklearn.linear_model.LogisticRegression(max_iter=1000, C=1.0)`
   - Fit on train, predict on test
   - Print and save classification report to `outputs/tables/clf_logistic_report.csv`

3. **Random Forest:**
   - `sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=42)`
   - Fit on train, predict on test
   - Print and save classification report to `outputs/tables/clf_rf_report.csv`

4. **Model comparison:**
   - Save a summary table to `outputs/tables/clf_comparison.csv` with columns: `model`, `accuracy`, `precision_genz`, `recall_genz`, `f1_genz`, `precision_older`, `recall_older`, `f1_older`

5. **Error analysis:**
   - Find all misclassified test reviews from the best-performing model
   - For each misclassified review, record: `true_label`, `predicted_label`, `vader_compound`, `topic_label`, `word_count`, `review_text`
   - Save to `outputs/tables/error_analysis.csv`
   - Print: how many Gen Z reviews were predicted as older, and vice versa
   - Print the 3 most common topic labels among misclassified reviews

6. **Feature importance:**
   - For Logistic Regression: extract top 20 positive and top 20 negative coefficients (most predictive terms for each class)
   - For Random Forest: extract top 20 features by importance score
   - Save both to `outputs/tables/feature_importance.csv` with columns: `feature`, `importance`, `model`, `direction` (for LR: `gen_z` or `older`, for RF: `n/a`)

7. **Figures** — save to `outputs/figures/`:
   - `clf_confusion_matrix_lr.png` — confusion matrix for Logistic Regression
   - `clf_confusion_matrix_rf.png` — confusion matrix for Random Forest
   - `clf_feature_importance_lr.png` — horizontal bar chart of top 20 LR coefficients per class
   - `clf_feature_importance_rf.png` — horizontal bar chart of top 20 RF feature importances
   - `clf_roc_curve.png` — ROC curves for both models on the same plot

8. **Auto-generate results summary:**
   - Write `report/results_summary.md` — a structured markdown file with:
     - Key numbers from each phase (group sizes, best coherence score, t-test results, classification F1 scores)
     - A brief bullet-point interpretation of each finding
     - This file is for the write-up authors to copy from — it does not need to be polished prose

### Acceptance criteria
- Both classifier reports exist with accuracy, precision, recall, F1
- Error analysis CSV is non-empty
- Feature importance CSV has rows for both models
- `report/results_summary.md` exists and is populated
- All 5 figures exist

---

## Running Everything

Run phases in order:

```bash
python src/01_data_prep.py
python src/02_sentiment.py
python src/03_topic_modelling.py
python src/04_linguistic.py
python src/05_classification.py
```

Each script should:
- Print progress updates with `tqdm` where loops are involved
- Print a `[DONE]` message at the end listing files it created
- Handle file-not-found errors gracefully with a clear message (e.g. `"ERROR: Run 01_data_prep.py first"`)

---

## Coding Standards

- **Python 3.10+**
- Use `pathlib.Path` for all file paths (not `os.path`)
- Set random seeds where relevant: `random_state=42`, `np.random.seed(42)`
- Use `tqdm` for any loop over reviews
- All figures: `figsize=(10, 6)`, `dpi=150`, tight layout, saved with `bbox_inches='tight'`
- All figures should work in both light and dark backgrounds — use `seaborn` default style (`sns.set_style("whitegrid")`)
- No Jupyter notebooks required — all deliverables are `.py` scripts
- Add a docstring at the top of each script describing what it does, its inputs, and its outputs

---

## Known Constraints

- **Compute:** Assume CPU-only. DistilBERT inference must complete in reasonable time — use batching and truncation.
- **Memory:** The dataset is ~23k rows. Avoid loading the full TF-IDF matrix into memory for Random Forest — use sparse matrices throughout.
- **Time:** The full pipeline should run end-to-end in under 2 hours on a standard laptop.
- **LDA non-determinism:** LDA results vary between runs. Set `random_state=42` in `LdaModel`. Topic labels may need manual adjustment — the `TOPIC_LABELS` dict at the top of phase 3 is the right place to do this.
