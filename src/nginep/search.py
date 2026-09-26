"""Ties the pieces together: parse a free-text query, filter candidates by
area, score them against aggregated per-hotel aspect data, and generate an
explanation string built from real extracted quotes -- never from a fresh,
unverifiable LLM sentence. This is the MVP "simple weighted score" ranker
the project plan calls for before a learned LightGBM ranker replaces it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rapidfuzz import fuzz

from .query_parser import ParsedQuery, make_client_and_parse

AREA_MATCH_THRESHOLD = 60.0


@dataclass
class HotelResult:
    hotel_id: str
    name: str
    area: str
    has_reviews: bool
    score: float
    avg_rating: float | None
    review_count: int
    aspect_hits: dict = field(default_factory=dict)  # aspect -> aggregated stats used
    explanation: str = ""


def _area_matches(query_area: str | None, hotel_area: str) -> bool:
    if not query_area:
        return True
    return fuzz.token_set_ratio(query_area, hotel_area) >= AREA_MATCH_THRESHOLD


def _score_hotel(wanted_aspects: list[str], hotel_aspects: dict) -> tuple[float, dict]:
    """Average smoothed_pos_rate over the wanted aspects that have any
    review mentions; aspects with zero mentions for this hotel don't count
    for or against it (no evidence, not negative evidence)."""
    if not wanted_aspects:
        return 0.0, {}
    used = {}
    total = 0.0
    n = 0
    for aspect in wanted_aspects:
        stats = hotel_aspects.get(aspect)
        if stats and stats["mentioned"] > 0:
            total += stats["smoothed_pos_rate"]
            n += 1
            used[aspect] = stats
    if n == 0:
        return 0.0, {}
    return total / n, used


def _explain(aspect_hits: dict) -> str:
    """Built only from aggregated counts + real quotes -- no free generation."""
    if not aspect_hits:
        return "Belum ada cukup ulasan yang membahas aspek yang kamu cari."
    parts = []
    for aspect, stats in aspect_hits.items():
        pct_pos = stats["pos"] / stats["mentioned"] if stats["mentioned"] else 0
        line = f"{aspect}: dipuji di {stats['pos']} dari {stats['mentioned']} ulasan yang membahasnya ({pct_pos:.0%})"
        quote = (stats["pos_quotes"] or stats["neg_quotes"] or [None])[0]
        if quote:
            line += f'. Kutipan: "{quote}"'
        parts.append(line)
    return " | ".join(parts)


def search(
    client,
    query_text: str,
    hotels: list[dict],
    aspect_scores: dict,
    top_k: int = 10,
) -> tuple[ParsedQuery, list[HotelResult]]:
    """hotels: rows from data/processed/hotels.csv as dicts.
    aspect_scores: parsed data/processed/hotel_aspect_scores.json.
    """
    parsed = make_client_and_parse(client, query_text)

    reviewed, unreviewed = [], []
    for hotel in hotels:
        if not _area_matches(parsed.area, hotel.get("area") or ""):
            continue
        if hotel["has_reviews"]:
            reviewed.append(hotel)
        else:
            unreviewed.append(hotel)

    scored: list[HotelResult] = []
    for hotel in reviewed:
        hotel_aspects = aspect_scores.get(hotel["hotel_id"], {}).get("aspects", {})
        score, hits = _score_hotel(parsed.aspects_wanted, hotel_aspects)
        scored.append(HotelResult(
            hotel_id=hotel["hotel_id"], name=hotel["name"], area=hotel["area"],
            has_reviews=True, score=score, avg_rating=hotel.get("avg_rating"),
            review_count=int(hotel.get("review_count") or 0),
            aspect_hits=hits, explanation=_explain(hits),
        ))
    # secondary sort by avg_rating so equal-aspect-score hotels aren't ordered arbitrarily
    scored.sort(key=lambda r: (r.score, r.avg_rating or 0), reverse=True)

    results = scored[:top_k]
    if len(results) < top_k and unreviewed:
        # fill remaining slots with no-review-data hotels, clearly marked --
        # never blended into the ranked/scored list above.
        for hotel in unreviewed[: top_k - len(results)]:
            results.append(HotelResult(
                hotel_id=hotel["hotel_id"], name=hotel["name"], area=hotel["area"],
                has_reviews=False, score=0.0, avg_rating=None, review_count=0,
                explanation="Belum ada data ulasan untuk hotel ini.",
            ))
    return parsed, results
