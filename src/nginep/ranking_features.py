"""Similarity-score helpers shared by the ranking ablation eval and the
LightGBM ranker's feature set, so both compute TF-IDF/embedding similarity
the same way instead of two copies drifting apart.

Embeddings are cached to disk by text hash: eval_ranking.py and
train_ranker.py both embed the exact same (query, candidate) pseudo-documents
independently, and without a cache the second script re-pays for and
re-waits on every embedding the first one already computed -- which is also
what tripped the embedding API's rate limit in practice, not just a cost
nicety.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

EMBED_MODEL = "gemini-embedding-001"
CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / ".cache" / "embeddings.json"
_cache: dict[str, list[float]] | None = None


def _load_cache() -> dict[str, list[float]]:
    global _cache
    if _cache is None:
        _cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    return _cache


def _save_cache() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(_cache))


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hotel_doc(c: dict) -> str:
    return f"{c['name']} {c['area']} {c.get('explanation', '')}"


def tfidf_similarities(query_text: str, candidates: list[dict]) -> list[float]:
    docs = [_hotel_doc(c) for c in candidates]
    vec = TfidfVectorizer().fit(docs + [query_text])
    doc_vecs = vec.transform(docs)
    q_vec = vec.transform([query_text])
    return cosine_similarity(q_vec, doc_vecs)[0].tolist()


def embedding_similarities(query_text: str, candidates: list[dict], client, max_retries: int = 5) -> list[float]:
    cache = _load_cache()
    texts = [query_text] + [_hotel_doc(c) for c in candidates]
    missing = [t for t in texts if _key(t) not in cache]

    if missing:
        for attempt in range(1, max_retries + 1):
            try:
                resp = client.models.embed_content(model=EMBED_MODEL, contents=missing)
                break
            except Exception:
                if attempt == max_retries:
                    raise
                time.sleep(min(5 * attempt, 30))
        for text, emb in zip(missing, resp.embeddings):
            cache[_key(text)] = emb.values
        _save_cache()

    vecs = np.array([cache[_key(t)] for t in texts])
    q_vec, doc_vecs = vecs[0:1], vecs[1:]
    return cosine_similarity(q_vec, doc_vecs)[0].tolist()
