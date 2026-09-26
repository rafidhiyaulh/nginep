# Relevance labeling (ranking eval ground truth)

This is the one piece of the eval this project deliberately does NOT
automate: whether a hotel is actually a good match for a query is a real
judgment call, and the whole point of measuring NDCG against it is that the
ground truth is independent of the system being measured. If the same LLM
pipeline that ranks results also invented the "correct" ranking, the eval
would just measure self-agreement, not quality — see the same reasoning
already applied to extraction eval (`reports/eval/hoasa_extraction_eval.md`
uses IndoNLU's human labels, not LLM-generated ones, for exactly this reason).

## Workflow

1. `.venv/bin/python scripts/generate_eval_queries.py` — runs the 50 drafted
   queries (diverse on purpose: area, aspect mix, purpose, language, a few
   short/edge-case ones) through the real search pipeline and saves each
   query's top-20 candidates to `query_candidates.json`. The queries
   themselves are fine to have been drafted for coverage; only the
   relevance judgments below need to be independent.
2. Open `labeling/index.html` directly in a browser (no server needed),
   load `query_candidates.json`, and rate each candidate 0-3:
   - **0** — not relevant (wrong area, or none of the requested aspects hold up)
   - **1** — marginally relevant (right area, weak match on aspects)
   - **2** — good match
   - **3** — exactly what the query asked for
   Progress autosaves to the browser's localStorage, so it's fine to do
   this across several sittings. "Export" downloads the labeled JSON.
3. Save the exported file as `labeling/query_candidates_labeled.json`.
4. `.venv/bin/python scripts/eval_ranking.py` — computes NDCG@5/@10 for the
   MVP weighted-score ranker against these labels, plus baselines (sort by
   rating, keyword/BM25, embedding-only) for the ablation table the plan
   calls for.

~1,000 judgments (50 queries x ~20 candidates) is the target, but grading
in order means the most useful signal (top-ranked candidates) gets labeled
first — stopping partway through still gives a usable, if smaller, eval set.
