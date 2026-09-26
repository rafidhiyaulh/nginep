"""Build a stratified sample of real Bali reviews for a blind, hand-labeled
accuracy check of the aspect extractor on its actual production language
(English) -- HoASA (see eval_hoasa.py) is Indonesian, so it can't answer
"how accurate is this on the reviews the app actually uses."

The LLM's own extraction (data/processed/bali_aspect_extractions.jsonl) is
used ONLY to pick which reviews go into the sample, never written into the
sample file itself -- the file a human labels from must contain nothing but
raw review text, or the labeling isn't blind and the resulting F1 would be
worthless as an accuracy check.

Stratified two ways:
- by aspect: rare aspects (e.g. "bau" at 1.5% of reviews, "wifi" at 3.7%)
  would get almost no coverage under plain random sampling of ~100 reviews.
  Every aspect gets a minimum quota of reviews where the LLM's own output
  flagged it as likely-mentioned (sentiment != "neut"), rarest aspects
  filled first so they're not crowded out by common ones stealing the budget.
- by hotel: no single hotel (some have 1000+ reviews, one has 2) should
  dominate the sample or be entirely absent from it.

Run: .venv/bin/python scripts/sample_bali_labeling_set.py
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from nginep.aspects import ASPECTS

ROOT = Path(__file__).resolve().parents[1]
REVIEWS_CSV = ROOT / "data" / "raw" / "bali_hotel_review.csv"
EXTRACTIONS_JSONL = ROOT / "data" / "processed" / "bali_aspect_extractions.jsonl"
OUT_JSON = ROOT / "labeling" / "bali_english_sample.json"

MIN_PER_ASPECT = 15  # reviews per aspect where the LLM flagged it as likely-mentioned
MIN_PER_HOTEL = 2    # every hotel gets at least this many, capped by its own review count
MAX_PER_HOTEL = 12   # no single hotel dominates the sample, even one that skews toward rare aspects
TARGET_TOTAL = 100
SEED = 42


def main() -> None:
    reviews = pd.read_csv(REVIEWS_CSV, sep=";", encoding="utf-8-sig")
    reviews = reviews[reviews["Review"].notna() & (reviews["Review"].astype(str).str.strip() != "")]

    extractions: dict[str, dict] = {}
    with EXTRACTIONS_JSONL.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("error") is None and d.get("aspects"):
                extractions[d["id"]] = d["aspects"]

    row_ids = [rid for rid in reviews.index.astype(str) if rid in extractions]
    hotel_of = {rid: reviews.loc[int(rid), "Hotel"] for rid in row_ids}

    # aspect -> pool of row_ids where the LLM's own output looks like a mention
    # (sentiment != "neut"). This is a SAMPLING SIGNAL ONLY -- never exposed
    # to the labeler, and not assumed to be accurate (that's exactly what's
    # being measured).
    aspect_pool: dict[str, list[str]] = {a: [] for a in ASPECTS}
    for rid in row_ids:
        for aspect, v in extractions[rid].items():
            if v["sentiment"] != "neut":
                aspect_pool[aspect].append(rid)

    rng = random.Random(SEED)
    for pool in aspect_pool.values():
        rng.shuffle(pool)

    selected: list[str] = []
    selected_set: set[str] = set()
    hotel_counts: Counter = Counter()

    def add(rid: str) -> bool:
        if hotel_counts[hotel_of[rid]] >= MAX_PER_HOTEL:
            return False
        selected.append(rid)
        selected_set.add(rid)
        hotel_counts[hotel_of[rid]] += 1
        return True

    # Rarest aspects first, so common aspects (mentioned in most reviews
    # anyway) don't crowd the budget out before rare ones get a fair shot.
    # A hotel that happens to skew heavily toward a given aspect (e.g. an
    # eco retreat where guests specifically remark on AC/wifi/hot water)
    # is capped at MAX_PER_HOTEL rather than allowed to fill that aspect's
    # whole quota by itself -- hotel diversity wins over hitting the exact
    # aspect target when the two conflict; short-of-target aspects are
    # reported honestly in the printed summary below, not padded out.
    for aspect in sorted(aspect_pool, key=lambda a: len(aspect_pool[a])):
        pool = aspect_pool[aspect]
        have = sum(1 for rid in pool if rid in selected_set)
        candidates = [rid for rid in pool if rid not in selected_set]
        # prefer candidates whose hotel is currently under-represented, so
        # satisfying a rare aspect's quota doesn't accidentally also blow
        # the hotel-diversity budget
        candidates.sort(key=lambda rid: hotel_counts[hotel_of[rid]])
        for rid in candidates:
            if have >= MIN_PER_ASPECT:
                break
            if add(rid):
                have += 1

    # every hotel represented at least MIN_PER_HOTEL times, capped by how
    # many reviews that hotel actually has
    by_hotel: dict[str, list[str]] = {}
    for rid in row_ids:
        by_hotel.setdefault(hotel_of[rid], []).append(rid)
    for hotel, pool in by_hotel.items():
        rng.shuffle(pool)
        need = MIN_PER_HOTEL - hotel_counts[hotel]
        for rid in pool:
            if need <= 0:
                break
            if rid not in selected_set and add(rid):
                need -= 1

    # top up to TARGET_TOTAL, preferring hotels below their fair share
    remaining = [rid for rid in row_ids if rid not in selected_set]
    rng.shuffle(remaining)
    fair_share = TARGET_TOTAL / len(by_hotel)
    remaining.sort(key=lambda rid: hotel_counts[hotel_of[rid]] / fair_share)
    for rid in remaining:
        if len(selected) >= TARGET_TOTAL:
            break
        add(rid)

    rng.shuffle(selected)  # labeling order shouldn't cluster by hotel/aspect

    sample = [
        {
            "sample_id": i + 1,
            "row_id": rid,
            "hotel": hotel_of[rid],
            "review": str(reviews.loc[int(rid), "Review"]).strip(),
        }
        for i, rid in enumerate(selected)
    ]

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Sampled {len(sample)} reviews -> {OUT_JSON.relative_to(ROOT)}")
    print(f"\nHotels represented: {len(hotel_counts)} / {len(by_hotel)}")
    for hotel, n in hotel_counts.most_common():
        print(f"  {hotel:35s} {n:3d}")
    print("\nPer-aspect coverage in the sample (LLM-flagged as mentioned, sampling signal only):")
    for aspect in sorted(aspect_pool, key=lambda a: len(aspect_pool[a])):
        covered = sum(1 for rid in selected if rid in aspect_pool[aspect])
        print(f"  {aspect:15s} {covered:3d} / {MIN_PER_ASPECT} target  (pool size in full dataset: {len(aspect_pool[aspect])})")


if __name__ == "__main__":
    main()
