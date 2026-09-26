"""Fuzzy-match hotel names between the Mendeley review corpus and OSM.

Naive global fuzzy matching (score a name against every OSM hotel in Bali,
take the top hit) turned out to fail badly in practice: short/generic OSM
names ("Hotel", "Bali Hai") scored deceptively high against unrelated
queries, and there was nothing to catch it because a single threshold on
the top score can't tell a confident match from a lucky one.

This module fixes that with two standard entity-resolution techniques:
1. Blocking — only compare a hotel against OSM candidates within
   `radius_km` of its area's geocoded centroid, so hotels in Ubud are never
   scored against hotels in Nusa Dua.
2. A margin check — the top match must beat the runner-up by `margin`
   points, not just clear an absolute threshold. A high score that's only
   marginally ahead of the next-best candidate is ambiguous, not confident.
"""
from __future__ import annotations

from rapidfuzz import fuzz, process

from .osm import haversine_km


def match_one(
    name: str,
    candidates: list[dict],
    accept_threshold: float = 90.0,
    review_threshold: float = 75.0,
    margin: float = 5.0,
) -> dict:
    """Match a single hotel name against a (already-blocked) candidate list.

    Returns {"status": "matched"|"review"|"unmatched", "osm": dict|None,
    "score": float|None, "gap_to_runner_up": float|None, "n_candidates": int}.
    """
    if not candidates:
        return {"status": "unmatched", "osm": None, "score": None, "gap_to_runner_up": None, "n_candidates": 0}

    # Exact (case/whitespace-insensitive) matches short-circuit the fuzzy
    # scorer entirely. Without this, an exact match can *tie* a completely
    # wrong but generically-worded candidate — token_set_ratio scores 100
    # whenever one name's tokens are a subset of the other's (e.g. a bare
    # "Hotel" entry vs "Nusa Dua Beach Hotel & Spa"), and which of two
    # tied-at-100 candidates process.extract() returns first isn't something
    # to depend on. An exact string match has no such ambiguity.
    normalized = name.strip().lower()
    exact = [c for c in candidates if c["name"].strip().lower() == normalized]
    if exact:
        return {"status": "matched", "osm": exact[0], "score": 100.0, "gap_to_runner_up": 100.0, "n_candidates": len(candidates)}

    names = [c["name"] for c in candidates]
    results = process.extract(name, names, scorer=fuzz.token_set_ratio, limit=2)
    if not results:
        return {"status": "unmatched", "osm": None, "score": None, "gap_to_runner_up": None, "n_candidates": len(candidates)}

    top_name, top_score, top_idx = results[0]
    runner_up_score = results[1][1] if len(results) > 1 else 0.0
    gap = top_score - runner_up_score

    if top_score >= accept_threshold and gap >= margin:
        status = "matched"
    elif top_score >= review_threshold:
        status = "review"  # good enough to surface, not confident enough to auto-accept
    else:
        status = "unmatched"

    return {
        "status": status,
        "osm": candidates[top_idx],
        "score": top_score,
        "gap_to_runner_up": gap,
        "n_candidates": len(candidates),
    }


def match_hotels_blocked(
    hotels: list[dict],
    osm_candidates: list[dict],
    area_centroids: dict[str, tuple[float, float]],
    radius_km: float = 12.0,
) -> dict[str, dict]:
    """Match a list of {"name":..., "area":...} dicts against OSM, blocking
    candidates by distance from the hotel's area centroid first.
    """
    matches: dict[str, dict] = {}
    for h in hotels:
        centroid = area_centroids.get(h["area"])
        if centroid is not None:
            candidates = [
                c for c in osm_candidates
                if haversine_km(centroid[0], centroid[1], c["lat"], c["lon"]) <= radius_km
            ]
            if not candidates:  # geocode existed but nothing nearby — fall back to global
                candidates = osm_candidates
        else:
            candidates = osm_candidates
        matches[h["name"]] = match_one(h["name"], candidates)
    return matches
