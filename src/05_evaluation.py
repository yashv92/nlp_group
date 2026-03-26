"""
Phase 5 — Classification & Evaluation (Owner: TLWQ1)

Inputs:
    data/processed/reviews_with_topics.csv

Outputs:
    outputs/tables/clf_logistic_report.csv
    outputs/tables/clf_rf_report.csv
    outputs/tables/clf_comparison.csv
    outputs/tables/error_analysis.csv
    outputs/tables/feature_importance.csv
    outputs/figures/clf_confusion_matrix_lr.png
    outputs/figures/clf_confusion_matrix_rf.png
    outputs/figures/clf_feature_importance_lr.png
    outputs/figures/clf_feature_importance_rf.png
    outputs/figures/clf_roc_curve.png
    report/results_summary.md

Description:
    Trains Logistic Regression and Random Forest classifiers to predict age group
    (Gen Z vs Older) from TF-IDF review features (unigrams + bigrams, max 5000).
    Evaluates both models, performs error analysis on misclassified reviews, extracts
    feature importances, and auto-generates a results_summary.md for the write-up.
"""
