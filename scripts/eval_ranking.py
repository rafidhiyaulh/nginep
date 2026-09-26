"""NDCG@5/@10 ablation: the MVP weighted-aspect-score ranker vs. baselines
(sort by rating, TF-IDF keyword match, embedding similarity), against
human-labeled relevance judgments.

Requires labeling/query_candidates_labeled.json to exist first -- see
labeling/README.md. Every baseline re-ranks the SAME labeled candidate pool
(the MVP ranker's own top-20 per query) rather than fetching its own
candidates: this measures re-ranking quality specifically, not recall, and
is a standard, disclosed simplification for an offline ablation at this
scale -- not a hidden one.

Run: .venv/bin/python scripts/eval_ranking.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nginep.extraction import make_client
from nginep.metrics import mean_ndcg_at_k

ROOT = Path(__file__).resolve().parents[1]
LABELED_JSON = ROOT / "labeling" / "query_candidates_labeled.json"
OUT_MD = ROOT / "reports" / "eval" / "ranking_ablation.md"
GCP_PROJECT = "nginep-tanyainap"
EMBED_MODEL = "gemini-embedding-001"


def _labeled_candidates(query: dict) -> list[dict]:
    return [c for c in query["candidates"] if c.get("relevance") is not None]


def order_mvp(candidates: list[dict]) -> list[dict]:
    return candidates  # already in the MVP ranker's predicted order


def order_rating(candidates: list[dict]) -> list[dict]:
    return sorted(candidates, key=lambda c: (c.get("avg_rating") or 0), reverse=True)


def order_tfidf(candidates: list[dict], query_text: str) -> list[dict]:
    docs = [f"{c['name']} {c['area']} {c.get('explanation', '')}" for c in candidates]
    vec = TfidfVectorizer().fit(docs + [query_text])
    doc_vecs = vec.transform(docs)
    q_vec = vec.transform([query_text])
    sims = cosine_similarity(q_vec, doc_vecs)[0]
    return [c for _, c in sorted(zip(sims, candidates), key=lambda x: x[0], reverse=True)]


def order_embedding(candidates: list[dict], query_text: str, client) -> list[dict]:
    docs = [f"{c['name']} {c['area']} {c.get('explanation', '')}" for c in candidates]
    resp = client.models.embed_content(model=EMBED_MODEL, contents=[query_text] + docs)
    vecs = np.array([e.values for e in resp.embeddings])
    q_vec, doc_vecs = vecs[0:1], vecs[1:]
    sims = cosine_similarity(q_vec, doc_vecs)[0]
    return [c for _, c in sorted(zip(sims, candidates), key=lambda x: x[0], reverse=True)]


def main() -> None:
    if not LABELED_JSON.exists():
        raise SystemExit(
            f"{LABELED_JSON} doesn't exist yet -- run scripts/generate_eval_queries.py, "
            f"then label relevance via labeling/index.html first. See labeling/README.md."
        )
    queries = json.loads(LABELED_JSON.read_text())
    client = make_client(GCP_PROJECT)

    methods = ["mvp_ranker", "rating_sort", "tfidf_keyword", "embedding"]
    per_query_rels = {m: [] for m in methods}
    n_skipped = 0

    for q in queries:
        labeled = _labeled_candidates(q)
        if len(labeled) < 2:  # need at least 2 to say anything about ranking
            n_skipped += 1
            continue
        orderings = {
            "mvp_ranker": order_mvp(labeled),
            "rating_sort": order_rating(labeled),
            "tfidf_keyword": order_tfidf(labeled, q["query_text"]),
            "embedding": order_embedding(labeled, q["query_text"], client),
        }
        for method, ordered in orderings.items():
            per_query_rels[method].append([c["relevance"] for c in ordered])

    n_used = len(queries) - n_skipped
    lines = [
        "# Ranking ablation: MVP weighted-aspect ranker vs. baselines",
        "",
        f"n = {n_used} queries with >=2 labeled candidates ({n_skipped} skipped -- not enough labels yet).",
        "",
        "All methods re-rank the same labeled candidate pool (the MVP ranker's own top-20 per",
        "query) rather than each fetching its own candidates -- this measures re-ranking quality,",
        "not recall. `tfidf_keyword` and `embedding` score against a pseudo-document per hotel",
        "(name + area + the aggregated-quote explanation string), not the full review corpus.",
        "",
        "| Method | NDCG@5 | NDCG@10 |",
        "|---|---|---|",
    ]
    for method in methods:
        ndcg5 = mean_ndcg_at_k(per_query_rels[method], k=5)
        ndcg10 = mean_ndcg_at_k(per_query_rels[method], k=10)
        lines.append(f"| {method} | {ndcg5:.3f} | {ndcg10:.3f} |")
        print(f"{method:15s} NDCG@5={ndcg5:.3f}  NDCG@10={ndcg10:.3f}")

    lines += [
        "",
        "If `mvp_ranker` doesn't beat `rating_sort` here, that's a real result to report as-is,",
        "not a reason to keep tuning until it does -- see the project's own evaluation notes.",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
