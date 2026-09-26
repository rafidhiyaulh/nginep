"""Evaluation metrics: NDCG for the ranker, F1 for the aspect extractor.

Kept dependency-light and provider-agnostic on purpose — these don't know
about LightGBM, Gemini, or hotels. They take plain lists/labels so they can
be unit-tested with synthetic data now, before any real model exists.
"""
from __future__ import annotations

import math

from sklearn.metrics import classification_report, f1_score


def dcg_at_k(relevances: list[float], k: int) -> float:
    """Discounted cumulative gain for a ranked list (already in ranked order)."""
    return sum(
        (2 ** rel - 1) / math.log2(i + 2)  # i is 0-indexed -> position i+1, log2(pos+1)
        for i, rel in enumerate(relevances[:k])
    )


def ndcg_at_k(relevances: list[float], k: int) -> float:
    """Normalized DCG@k for one query's ranked relevance list.

    `relevances` must already be in the order the ranker produced (i.e.
    relevances[0] is whatever the ranker put first). Returns 0.0 for an
    empty list or when no item has positive relevance (IDCG would be 0).
    """
    if not relevances:
        return 0.0
    actual = dcg_at_k(relevances, k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    return actual / ideal if ideal > 0 else 0.0


def mean_ndcg_at_k(per_query_relevances: list[list[float]], k: int) -> float:
    """Mean NDCG@k across multiple queries — the number actually reported."""
    if not per_query_relevances:
        return 0.0
    scores = [ndcg_at_k(rels, k) for rels in per_query_relevances]
    return sum(scores) / len(scores)


def aspect_report(y_true: list[str], y_pred: list[str], labels: list[str] | None = None) -> dict:
    """Macro-F1 + per-class F1 for aspect-sentiment predictions.

    y_true/y_pred are parallel lists of label strings (e.g. "pos"/"neg"/
    "neut"/"neg_pos" for HoASA, or whatever label set the hand-labeled Bali
    eval sample ends up using).
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must be the same length, got {len(y_true)} vs {len(y_pred)}")
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {"macro_f1": macro_f1, "report": report}
