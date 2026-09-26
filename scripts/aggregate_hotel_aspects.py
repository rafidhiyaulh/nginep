"""Aggregate per-review aspect extractions into per-hotel aspect scores.

Smoothed so a hotel with 2 reviews mentioning wifi once doesn't get shown
as a confident 100% or 0% -- it's pulled toward a neutral prior (0.5) by
`ALPHA` pseudo-observations, weakly for a well-reviewed hotel, strongly for
a barely-reviewed one. Only hotels with review data appear in the output at
all; retrieval treats a missing hotel_id as "no evidence for any aspect",
not zero/negative -- those are different things.

Run: .venv/bin/python scripts/aggregate_hotel_aspects.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from nginep.aspects import ASPECTS

ROOT = Path(__file__).resolve().parents[1]
REVIEWS_CSV = ROOT / "data" / "raw" / "bali_hotel_review.csv"
EXTRACTIONS_JSONL = ROOT / "data" / "processed" / "bali_aspect_extractions.jsonl"
HOTELS_CSV = ROOT / "data" / "processed" / "hotels.csv"
OUT_JSON = ROOT / "data" / "processed" / "hotel_aspect_scores.json"

ALPHA = 4.0  # smoothing strength, in pseudo-observations pulling toward 0.5
MAX_QUOTES_PER_SIDE = 3


def main() -> None:
    reviews = pd.read_csv(REVIEWS_CSV, sep=";", encoding="utf-8-sig")
    hotels = pd.read_csv(HOTELS_CSV)
    name_to_hotel_id = dict(zip(hotels["name"], hotels["hotel_id"]))

    extractions = {}
    n_failed = 0
    with EXTRACTIONS_JSONL.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["aspects"] is None:
                n_failed += 1
                continue
            extractions[int(rec["id"])] = rec["aspects"]

    missing = set(reviews.index) - set(extractions)
    if missing:
        print(f"WARNING: {len(missing)}/{len(reviews)} reviews have no extraction yet "
              f"(extraction still running or incomplete) -- aggregating over what's available.")
    if n_failed:
        print(f"WARNING: {n_failed} reviews failed extraction after retries -- excluded.")

    # hotel_id -> aspect -> {pos, neg, neut, neg_pos, pos_quotes, neg_quotes}
    counts = defaultdict(lambda: defaultdict(lambda: {"pos": 0, "neg": 0, "neut": 0, "neg_pos": 0, "pos_quotes": [], "neg_quotes": []}))
    review_totals = defaultdict(int)

    for row_idx, row in reviews.iterrows():
        if row_idx not in extractions:
            continue
        hotel_id = name_to_hotel_id.get(row["Hotel"])
        if hotel_id is None:
            continue
        review_totals[hotel_id] += 1
        for aspect, result in extractions[row_idx].items():
            sentiment = result["sentiment"]
            bucket = counts[hotel_id][aspect]
            bucket[sentiment] += 1
            quote = result.get("quote")
            if quote:
                if sentiment == "pos" and len(bucket["pos_quotes"]) < MAX_QUOTES_PER_SIDE:
                    bucket["pos_quotes"].append(quote)
                elif sentiment == "neg" and len(bucket["neg_quotes"]) < MAX_QUOTES_PER_SIDE:
                    bucket["neg_quotes"].append(quote)

    output = {}
    for hotel_id, aspect_counts in counts.items():
        total_reviews = review_totals[hotel_id]
        output[hotel_id] = {"n_reviews_extracted": total_reviews, "aspects": {}}
        for aspect in ASPECTS:
            c = aspect_counts.get(aspect, {"pos": 0, "neg": 0, "neut": 0, "neg_pos": 0, "pos_quotes": [], "neg_quotes": []})
            mentioned = c["pos"] + c["neg"] + c["neg_pos"]
            smoothed_pos_rate = (c["pos"] + 0.5 * c["neg_pos"] + ALPHA * 0.5) / (mentioned + ALPHA)
            output[hotel_id]["aspects"][aspect] = {
                "pos": c["pos"], "neg": c["neg"], "neg_pos": c["neg_pos"], "neut": c["neut"],
                "mentioned": mentioned,
                "mention_rate": round(mentioned / total_reviews, 3) if total_reviews else 0.0,
                "smoothed_pos_rate": round(smoothed_pos_rate, 3),
                "pos_quotes": c["pos_quotes"],
                "neg_quotes": c["neg_quotes"],
            }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Aggregated {len(output)} hotels -> {OUT_JSON.relative_to(ROOT)}")
    for hotel_id, data in list(output.items())[:1]:
        print(f"  sample ({hotel_id}): n_reviews_extracted={data['n_reviews_extracted']}, "
              f"wifi={data['aspects']['wifi']}")


if __name__ == "__main__":
    main()
