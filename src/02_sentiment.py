"""
Phase 2 — Sentiment Analysis (Owner: WDYF5)

Inputs:
    data/processed/reviews_clean.csv

Outputs:
    data/processed/reviews_with_sentiment.csv
    outputs/tables/sentiment_ttest_results.csv
    outputs/figures/sentiment_vader_boxplot.png
    outputs/figures/sentiment_distilbert_boxplot.png
    outputs/figures/sentiment_compound_hist.png
    outputs/figures/sentiment_rating_correlation.png

Description:
    Runs two sentiment models on each review: VADER (rule-based baseline) and
    DistilBERT (fine-tuned on SST-2). Produces compound scores in the range [-1, +1]
    for both models, runs independent t-tests comparing Gen Z vs Older sentiment,
    and generates comparison figures.
"""
