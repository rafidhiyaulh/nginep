"""Train a LightGBM LambdaRank reranker on the labeled queries and compare
it against the MVP weighted-score baseline via query-level cross-validation.

With ~50 queries x ~20 candidates (~1,000 rows), this is a genuinely small
dataset for a gradient-boosted ranker -- overfitting is the real risk, which
is exactly why this uses K-fold CV grouped by QUERY (never splitting one
query's candidates across train/val, which would leak) rather than a single
train/test split, and why the model is kept deliberately small (few leaves,
few trees). If cross-validated NDCG doesn't beat the MVP baseline, that is
reported as-is -- see reports/eval/ranker_training.md once this has run.

Requires labeling/query_candidates_labeled.json -- see labeling/README.md.

Run: .venv/bin/python scripts/train_ranker.py
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold

from nginep.extraction import make_client
from nginep.metrics import mean_ndcg_at_k
from nginep.ranking_features import embedding_similarities, tfidf_similarities

ROOT = Path(__file__).resolve().parents[1]
LABELED_JSON = ROOT / "labeling" / "query_candidates_labeled.json"
OUT_MD = ROOT / "reports" / "eval" / "ranker_training.md"
GCP_PROJECT = "nginep-tanyainap"
N_FOLDS = 5
FEATURE_NAMES = ["mvp_score", "avg_rating", "has_rating", "log_review_count", "tfidf_sim", "embedding_sim", "has_reviews"]


def build_features(queries: list[dict], client) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """Returns (X, y, query_ids_per_row, group_sizes) -- group_sizes in the
    same order queries appear, required by LightGBM's ranking objective."""
    rows, labels, query_id_per_row, group_sizes = [], [], [], []

    for q in queries:
        labeled = [c for c in q["candidates"] if c.get("relevance") is not None]
        if len(labeled) < 2:
            continue
        tfidf_sims = tfidf_similarities(q["query_text"], labeled)
        emb_sims = embedding_similarities(q["query_text"], labeled, client)
        for c, tf_sim, e_sim in zip(labeled, tfidf_sims, emb_sims):
            rating = c.get("avg_rating")
            rows.append([
                c.get("score", 0.0),
                rating if rating is not None else 0.0,
                1.0 if rating is not None else 0.0,
                np.log1p(c.get("review_count") or 0),
                tf_sim, e_sim,
                1.0 if c.get("has_reviews") else 0.0,
            ])
            labels.append(c["relevance"])
            query_id_per_row.append(q["query_id"])
        group_sizes.append(len(labeled))

    return np.array(rows), np.array(labels), query_id_per_row, group_sizes


def cross_validate(X: np.ndarray, y: np.ndarray, query_ids: list[str], n_folds: int) -> tuple[list, list]:
    unique_qids = sorted(set(query_ids))
    if len(unique_qids) < n_folds:
        n_folds = max(2, len(unique_qids))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    qid_arr = np.array(query_ids)

    ranker_rels, mvp_rels = [], []
    for train_qidx, val_qidx in kf.split(unique_qids):
        train_qids = {unique_qids[i] for i in train_qidx}
        val_qids = {unique_qids[i] for i in val_qidx}
        train_mask = np.array([q in train_qids for q in query_ids])
        val_mask = np.array([q in val_qids for q in query_ids])

        train_groups = _group_sizes_for(qid_arr[train_mask])
        model = lgb.LGBMRanker(
            objective="lambdarank", n_estimators=30, num_leaves=7,
            min_child_samples=5, learning_rate=0.1, verbosity=-1,
        )
        model.fit(X[train_mask], y[train_mask], group=train_groups)

        for qid in val_qids:
            row_mask = qid_arr == qid
            if row_mask.sum() < 2:
                continue
            true_rel = y[row_mask]
            mvp_score_order = np.argsort(-X[row_mask, 0])  # feature 0 = mvp_score
            pred_order = np.argsort(-model.predict(X[row_mask]))
            ranker_rels.append(true_rel[pred_order].tolist())
            mvp_rels.append(true_rel[mvp_score_order].tolist())

    return ranker_rels, mvp_rels


def _group_sizes_for(qid_arr: np.ndarray) -> list[int]:
    sizes = []
    seen = []
    for qid in qid_arr:
        if not seen or seen[-1] != qid:
            sizes.append(0)
            seen.append(qid)
        sizes[-1] += 1
    return sizes


def main() -> None:
    if not LABELED_JSON.exists():
        raise SystemExit(f"{LABELED_JSON} doesn't exist yet -- see labeling/README.md")
    queries = json.loads(LABELED_JSON.read_text())
    client = make_client(GCP_PROJECT)

    print("Building features (TF-IDF + embedding similarity per candidate)...")
    X, y, query_ids, _ = build_features(queries, client)
    n_queries = len(set(query_ids))
    print(f"{len(X)} labeled rows across {n_queries} queries with >=2 labels")

    if n_queries < 4:
        raise SystemExit(f"Only {n_queries} usable queries -- label more before training (need enough for CV folds).")

    print(f"Cross-validating ({N_FOLDS}-fold, grouped by query)...")
    ranker_rels, mvp_rels = cross_validate(X, y, query_ids, N_FOLDS)

    ranker_ndcg5 = mean_ndcg_at_k(ranker_rels, 5)
    ranker_ndcg10 = mean_ndcg_at_k(ranker_rels, 10)
    mvp_ndcg5 = mean_ndcg_at_k(mvp_rels, 5)
    mvp_ndcg10 = mean_ndcg_at_k(mvp_rels, 10)

    verdict = (
        "LightGBM ranker beats the MVP weighted-score baseline in cross-validation."
        if ranker_ndcg10 > mvp_ndcg10 else
        "LightGBM ranker does NOT beat the MVP weighted-score baseline in cross-validation "
        "-- reported as-is, not tuned until it does. With ~1,000 rows this is a plausible, "
        "honest outcome: a simple weighted average of well-chosen features can beat a "
        "gradient-boosted model that doesn't have enough data to learn reliable splits."
    )

    lines = [
        "# LightGBM ranker vs. MVP baseline -- cross-validated",
        "",
        f"n = {len(X)} labeled rows, {n_queries} queries, {N_FOLDS}-fold CV grouped by query",
        "(no query's candidates are ever split across train and validation).",
        "",
        "| Method | NDCG@5 | NDCG@10 |",
        "|---|---|---|",
        f"| LightGBM LambdaRank | {ranker_ndcg5:.3f} | {ranker_ndcg10:.3f} |",
        f"| MVP weighted score | {mvp_ndcg5:.3f} | {mvp_ndcg10:.3f} |",
        "",
        f"**{verdict}**",
        "",
        "Features: " + ", ".join(FEATURE_NAMES),
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nranker NDCG@10={ranker_ndcg10:.3f} vs mvp NDCG@10={mvp_ndcg10:.3f}")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
