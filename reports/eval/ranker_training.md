# LightGBM ranker vs. MVP baseline -- cross-validated

n = 980 labeled rows, 49 queries, 5-fold CV grouped by query
(no query's candidates are ever split across train and validation).

| Method | NDCG@5 | NDCG@10 |
|---|---|---|
| LightGBM LambdaRank | 0.982 | 0.984 |
| MVP weighted score | 0.998 | 0.992 |

**LightGBM ranker does NOT beat the MVP weighted-score baseline in cross-validation -- reported as-is, not tuned until it does. With ~1,000 rows this is a plausible, honest outcome: a simple weighted average of well-chosen features can beat a gradient-boosted model that doesn't have enough data to learn reliable splits.**

Features: mvp_score, avg_rating, has_rating, log_review_count, tfidf_sim, embedding_sim, has_reviews