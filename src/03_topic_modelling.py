"""
Phase 3 — Topic Modelling & Aspect Extraction (Owner: SQBR7)

Inputs:
    data/processed/reviews_with_sentiment.csv

Outputs:
    data/processed/reviews_with_topics.csv
    outputs/tables/lda_coherence_scores.csv
    outputs/tables/aspect_sentiment_by_group.csv
    outputs/tables/aspect_ttest_results.csv
    outputs/tables/tfidf_top_terms.csv
    outputs/figures/lda_coherence_plot.png
    outputs/figures/topic_sentiment_heatmap.png
    outputs/figures/topic_distribution_bar.png
    outputs/figures/tfidf_top_terms_genz.png
    outputs/figures/tfidf_top_terms_older.png

Description:
    Trains Gensim LDA models for a range of topic counts (6, 8, 10, 12, 15),
    selects the best by c_v coherence, assigns each review a dominant topic label,
    computes aspect-level sentiment differences between Gen Z and Older reviewers,
    and extracts top TF-IDF vocabulary per group.
"""
