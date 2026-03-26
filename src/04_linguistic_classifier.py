"""
Phase 4 — Linguistic Feature Analysis (Owner: SSSP0)

Inputs:
    data/processed/reviews_with_topics.csv

Outputs:
    outputs/tables/linguistic_features_summary.csv
    outputs/tables/linguistic_ttest_results.csv
    outputs/figures/ling_feature_comparison.png
    outputs/figures/ling_exclamation_dist.png
    outputs/figures/ling_ttr_boxplot.png
    outputs/figures/ling_wordcount_boxplot.png
    outputs/figures/ling_caps_by_topic.png

Description:
    Extracts per-review linguistic features (word count, sentence count, TTR,
    exclamation/question marks, ALL-CAPS words, emotionally charged words) for
    each review, runs t-tests comparing Gen Z vs Older on every feature, and
    produces comparison figures. Features feed into Phase 5 classification.
"""
