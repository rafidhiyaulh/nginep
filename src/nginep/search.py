"""Ties the pieces together: parse a free-text query, filter candidates by
area, score them against aggregated per-hotel aspect data, and generate an
explanation string built from real extracted quotes -- never from a fresh,
unverifiable LLM sentence. This is the MVP "simple weighted score" ranker
the project plan calls for before a learned LightGBM ranker replaces it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from .aspects import ASPECT_LABELS_ID
from .query_parser import ParsedQuery, make_client_and_parse

# Filler words stripped from the query side before area matching -- NOT
# matched against the hotel side, since hotel area strings ("Nusa Dua Bali")
# legitimately contain "Bali" in every single one.
AREA_STOPWORDS = {"dekat", "pantai", "di", "area", "wilayah", "sekitar", "the", "near", "in", "at", "bali"}
TOKEN_MATCH_THRESHOLD = 85.0

# Colloquial area names that don't textually overlap with the area label
# actually present in the data, found via real search queries (e.g.
# "Uluwatu" matched only 1 hotel -- the one with a literal "Uluwatu" OSM
# address tag -- because every other nearby hotel's area field says
# "Pecatu Bali", the admin area Uluwatu sits inside). NOT an exhaustive
# gazetteer of Bali place names -- just the gaps actually observed.
AREA_SYNONYMS: dict[str, list[str]] = {
    "uluwatu": ["pecatu"],
}


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
    evidence: list = field(default_factory=list)  # structured form of explanation, for the API/frontend


def _area_matches(query_area: str | None, hotel_area: str) -> bool:
    """Every distinctive word in the query area must find a close match
    among the hotel area's words. A blended similarity score (like plain
    token_set_ratio) isn't enough here -- it was scoring "Nusa Dua" a 66.7
    match against "Nusa Ceningan Bali" (a different island entirely) purely
    because both share the generic word "Nusa" -- see the regression test.
    Requiring full coverage of the query's tokens, not just token overlap,
    fixes that without needing to special-case "Nusa"."""
    if not query_area:
        return True
    query_tokens = [t for t in re.findall(r"\w+", query_area.lower()) if t not in AREA_STOPWORDS]
    if not query_tokens:
        return True  # nothing distinctive left (e.g. query area was just "Bali") -- don't over-filter
    hotel_tokens = re.findall(r"\w+", hotel_area.lower())

    def token_ok(qt: str) -> bool:
        candidates = [qt, *AREA_SYNONYMS.get(qt, [])]
        return any(fuzz.ratio(c, ht) >= TOKEN_MATCH_THRESHOLD for c in candidates for ht in hotel_tokens)

    return all(token_ok(qt) for qt in query_tokens)


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


def aspect_evidence(aspect_hits: dict) -> list[dict]:
    """One structured row per requested aspect that has evidence -- built
    only from aggregated counts + real quotes, no free generation. Each row
    picks "dipuji"/"dikeluhkan" based on which way the sentiment actually
    leans, so the label never says "praised" while quoting a complaint.
    Kept separate from any string formatting so the API can hand the
    frontend real structure instead of one dense run-on sentence."""
    rows = []
    for aspect, stats in aspect_hits.items():
        label = ASPECT_LABELS_ID.get(aspect, aspect)
        pct_pos = stats["pos"] / stats["mentioned"] if stats["mentioned"] else 0
        positive = pct_pos >= 0.5
        quote = (stats["pos_quotes"] or stats["neg_quotes"] or [None])[0] if positive else (stats["neg_quotes"] or stats["pos_quotes"] or [None])[0]
        rows.append({
            "label": label,
            "positive": positive,
            "verb": "dipuji" if positive else "dikeluhkan",
            "pct": round(pct_pos * 100),
            "pos": stats["pos"],
            "mentioned": stats["mentioned"],
            "quote": quote,
        })
    rows.sort(key=lambda r: r["pct"], reverse=True)
    return rows


def _explain(aspect_hits: dict) -> str:
    """Plain-text join of aspect_evidence, for the CLI script."""
    rows = aspect_evidence(aspect_hits)
    if not rows:
        return "Belum ada cukup ulasan yang membahas apa yang kamu cari."
    parts = [
        f"{r['label']} {r['verb']} di {r['pos']} dari {r['mentioned']} ulasan ({r['pct']}%)"
        + (f": '{r['quote']}'" if r["quote"] else "")
        for r in rows
    ]
    return ", ".join(parts)


MAX_ALSO_NEARBY = 8


def search(
    client,
    query_text: str,
    hotels: list[dict],
    aspect_scores: dict,
    top_k: int = 10,
) -> tuple[ParsedQuery, list[HotelResult], list[dict]]:
    """hotels: rows from data/processed/hotels.csv as dicts.
    aspect_scores: parsed data/processed/hotel_aspect_scores.json.

    Returns (parsed_query, ranked_results, also_nearby). `also_nearby` is a
    short list of {hotel_id, name, area} for hotels in the matched area with
    no review data -- kept separate from `ranked_results` on purpose, rather
    than padded in as identical-looking cards with no evidence to show. One
    real hotel with no data is a useful "also exists here" mention; five of
    them as full repeated cards is just noise.
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
            aspect_hits=hits, explanation=_explain(hits), evidence=aspect_evidence(hits),
        ))
    # secondary sort by avg_rating so equal-aspect-score hotels aren't ordered arbitrarily
    scored.sort(key=lambda r: (r.score, r.avg_rating or 0), reverse=True)

    results = scored[:top_k]
    also_nearby = [
        {"hotel_id": h["hotel_id"], "name": h["name"], "area": h["area"]}
        for h in unreviewed[:MAX_ALSO_NEARBY]
    ]
    return parsed, results, also_nearby
