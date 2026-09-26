"""Generate the query set + candidate pools for ranking evaluation.

The 50 queries below are drafted to be realistic and diverse (area
coverage, aspect mix, purpose, language, a few edge cases) -- that's a
reasonable thing to draft directly. What is NOT drafted here, on purpose:
relevance judgments. Deciding "is this hotel actually a good match for this
query" is a real human judgment call the ranking eval's credibility depends
on -- see labeling/README.md for that step, which is the user's own task.

Run (after the full extraction + aggregation is done, not before -- the
candidate pools and explanations should reflect real final data):
  .venv/bin/python scripts/generate_eval_queries.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nginep.extraction import make_client
from nginep.search import search

ROOT = Path(__file__).resolve().parents[1]
GCP_PROJECT = "nginep-tanyainap"
OUT_JSON = ROOT / "labeling" / "query_candidates.json"
TOP_K = 20

QUERIES = [
    # Ubud -- remote work / quiet, the flagship persona
    "hotel tenang di Ubud buat kerja remote, WiFi harus kencang, kamarnya bersih",
    "quiet hotel in Ubud for a digital nomad, need strong wifi",
    "penginapan di Ubud yang jauh dari keramaian",
    # Legian / Kuta -- beach/nightlife area, service & value focus
    "hotel di Legian dekat pantai, servicenya ramah",
    "budget hotel in Kuta, good value for money",
    "hotel murah di Kuta tapi kamar bersih",
    "hotel di Kuta dengan sarapan enak",
    "family-friendly hotel in Kuta, staff should be helpful with kids",
    # Sanur -- calmer beach area
    "hotel di Sanur yang tenang buat honeymoon",
    "quiet beachfront hotel in Sanur with good breakfast",
    "penginapan Sanur, wifi kencang buat kerja",
    # Nusa Dua -- resort area
    "hotel resort di Nusa Dua, servicenya bagus",
    "luxury hotel in Nusa Dua with excellent service",
    "hotel di Nusa Dua yang bersih dan nyaman",
    # Candidasa -- quieter east Bali
    "hotel di Candidasa yang sepi dan nyaman buat istirahat",
    "resort in Candidasa, good breakfast and clean rooms",
    # Jimbaran -- known for seafood/sunset
    "hotel di Jimbaran dekat pantai, wifi bagus",
    "hotel in Jimbaran for a honeymoon, quiet and romantic",
    # Uluwatu / Pecatu -- surf/cliff area
    "hotel di Uluwatu yang aksesnya gampang",
    "resort di Pecatu dengan pemandangan bagus dan servicenya oke",
    "hotel near Uluwatu, easy access, good wifi for work",
    # Tabanan -- rural/retreat area
    "eco retreat di Tabanan, tenang dan alami",
    "quiet eco hotel in Tabanan",
    # Nusa Ceningan -- island, villas
    "villa di Nusa Ceningan dekat Blue Lagoon",
    "quiet villa on Nusa Ceningan with clean rooms",
    # Buleleng -- north Bali
    "hotel butik di Buleleng, servicenya ramah",
    "boutique hotel in north Bali (Buleleng), good service",
    # Aspect-focused, area-agnostic
    "hotel dengan wifi paling kencang di Bali",
    "hotel yang sarapannya paling enak",
    "hotel paling bersih buat yang OCD",
    "hotel dengan AC dingin dan air panas lancar",
    "hotel yang tidak bau apek kamarnya",
    "hotel dengan TV kabel lengkap",
    "hotel dengan linen/sprei yang selalu bersih",
    "best value hotel in Bali, cheap but good",
    "hotel paling murah tapi tetap nyaman",
    # Purpose-driven
    "hotel buat bulan madu yang romantis dan tenang",
    "hotel buat solo traveler yang gampang kemana-mana",
    "hotel buat backpacker murah meriah",
    "business hotel in Bali with reliable wifi",
    "hotel keluarga dengan kamar bersih dan servis ramah",
    # Compound / multi-aspect, realistic messy phrasing
    "hotel dekat pantai di Kuta, sarapan enak, WiFi kencang, di bawah 500 ribu",
    "cari hotel di Ubud yang tenang, wifi kencang, sarapan enak, kamar bersih",
    "hotel in Sanur, quiet, good wifi, clean, friendly staff, breakfast included",
    "hotel budget di Legian, bersih, wifi lumayan, servis oke",
    # Short / minimal queries (edge cases)
    "hotel Ubud",
    "wifi kencang",
    "hotel murah",
    "quiet hotel",
    # No area at all -- pure aspect browse
    "hotel dengan pelayanan paling ramah di Bali",
    "hotel yang paling tenang buat tidur nyenyak",
]


def main() -> None:
    assert len(QUERIES) >= 45, f"only {len(QUERIES)} queries -- add more for a meaningful eval set"
    hotels = pd.read_csv(ROOT / "data" / "processed" / "hotels.csv").to_dict("records")
    aspect_scores = json.loads((ROOT / "data" / "processed" / "hotel_aspect_scores.json").read_text())
    client = make_client(GCP_PROJECT)

    out = []
    for i, q in enumerate(QUERIES):
        parsed, results = search(client, q, hotels, aspect_scores, top_k=TOP_K)
        out.append({
            "query_id": f"q{i:02d}",
            "query_text": q,
            "parsed": parsed.model_dump(),
            "candidates": [
                {
                    "hotel_id": r.hotel_id, "name": r.name, "area": r.area,
                    "has_reviews": r.has_reviews, "score": round(r.score, 3),
                    "avg_rating": r.avg_rating, "review_count": r.review_count,
                    "explanation": r.explanation,
                    "relevance": None,  # <- filled in by the labeling tool, not here
                }
                for r in results
            ],
        })
        print(f"  [{i+1}/{len(QUERIES)}] {q!r} -> {len(results)} candidates")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(out)} queries -> {OUT_JSON.relative_to(ROOT)}")
    print("Next: open labeling/index.html in a browser and label relevance for each candidate.")


if __name__ == "__main__":
    main()
