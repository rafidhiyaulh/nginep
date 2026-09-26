"""Quick CLI to try the search pipeline end to end without a web server.

Usage: .venv/bin/python scripts/search_cli.py "hotel tenang di Ubud, wifi kencang"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from nginep.extraction import make_client
from nginep.search import search

ROOT = Path(__file__).resolve().parents[1]
GCP_PROJECT = "nginep-tanyainap"


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: search_cli.py "<query text>"')
        raise SystemExit(1)
    query_text = sys.argv[1]

    hotels = pd.read_csv(ROOT / "data" / "processed" / "hotels.csv").to_dict("records")
    aspect_scores = json.loads((ROOT / "data" / "processed" / "hotel_aspect_scores.json").read_text())

    client = make_client(GCP_PROJECT)
    parsed, results, also_nearby = search(client, query_text, hotels, aspect_scores, top_k=5)

    print(f"Query: {query_text!r}")
    print(f"Parsed: {parsed.model_dump()}")
    print()
    for i, r in enumerate(results, 1):
        print(f"{i}. {r.name} ({r.area}) -- score={r.score:.2f}")
        print(f"   {r.explanation}")
        print()
    if also_nearby:
        print("Also in this area, no review data yet:")
        for h in also_nearby:
            print(f"  - {h['name']} ({h['area']})")


if __name__ == "__main__":
    main()
