# Ranking ablation: MVP weighted-aspect ranker vs. baselines

n = 49 queries with >=2 labeled candidates (2 skipped -- not enough labels yet).

All methods re-rank the same labeled candidate pool (the MVP ranker's own top-20 per
query) rather than each fetching its own candidates -- this measures re-ranking quality,
not recall. `tfidf_keyword` and `embedding` score against a pseudo-document per hotel
(name + area + the aggregated-quote explanation string), not the full review corpus.

| Method | NDCG@5 | NDCG@10 |
|---|---|---|
| mvp_ranker | 0.998 | 0.992 |
| rating_sort | 0.803 | 0.867 |
| tfidf_keyword | 0.696 | 0.753 |
| embedding | 0.757 | 0.798 |

If `mvp_ranker` doesn't beat `rating_sort` here, that's a real result to report as-is,
not a reason to keep tuning until it does -- see the project's own evaluation notes.