"""
Phase 1 — Data Preparation (Owner: XKKW4)

Inputs:
    data/raw/Womens Clothing E-Commerce Reviews.csv

Outputs:
    data/processed/reviews_clean.csv
    outputs/figures/eda_review_length_dist.png
    outputs/figures/eda_rating_dist.png
    outputs/figures/eda_age_group_counts.png
    outputs/figures/eda_wordcloud_genz.png
    outputs/figures/eda_wordcloud_older.png

Description:
    Loads the raw Kaggle women's clothing reviews dataset, segments reviewers
    into Gen Z (18-26) and Older (27+) age groups, cleans review text (lowercase,
    remove special characters, deduplicate, drop short reviews), generates EDA
    figures, and saves a cleaned CSV for downstream phases.
"""
