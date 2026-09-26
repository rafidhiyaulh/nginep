"""Similarity-score helpers shared by the ranking ablation eval and the
LightGBM ranker's feature set, so both compute TF-IDF/embedding similarity
the same way instead of two copies drifting apart.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

EMBED_MODEL = "gemini-embedding-001"


def _hotel_doc(c: dict) -> str:
    return f"{c['name']} {c['area']} {c.get('explanation', '')}"


def tfidf_similarities(query_text: str, candidates: list[dict]) -> list[float]:
    docs = [_hotel_doc(c) for c in candidates]
    vec = TfidfVectorizer().fit(docs + [query_text])
    doc_vecs = vec.transform(docs)
    q_vec = vec.transform([query_text])
    return cosine_similarity(q_vec, doc_vecs)[0].tolist()


def embedding_similarities(query_text: str, candidates: list[dict], client) -> list[float]:
    docs = [_hotel_doc(c) for c in candidates]
    resp = client.models.embed_content(model=EMBED_MODEL, contents=[query_text] + docs)
    vecs = np.array([e.values for e in resp.embeddings])
    q_vec, doc_vecs = vecs[0:1], vecs[1:]
    return cosine_similarity(q_vec, doc_vecs)[0].tolist()
