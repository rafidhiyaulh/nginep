"""Build data/processed/hotels.csv by combining the Mendeley review corpus
with OpenStreetMap's hotel listings for Bali.

Matching approach (see src/nginep/matching.py for why): geocode each
Mendeley `Location` string to a centroid, block OSM candidates to within
`RADIUS_KM` of it, then fuzzy-match names within that shortlist and require
a score + a margin over the runner-up before auto-accepting. Anything that
doesn't clear that bar gets one direct Nominatim lookup on the hotel name
itself as a fallback, and is still labeled with its real match_status so
downstream code (and a human) can tell confident matches from guesses.

Run: .venv/bin/python scripts/build_hotel_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nginep.matching import match_hotels_blocked
from nginep.osm import fetch_hotels, geocode

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "bali_hotel_review.csv"
OUT_CSV = ROOT / "data" / "processed" / "hotels.csv"
MATCH_LOG = ROOT / "reports" / "eda" / "hotel_matching_log.md"
OSM_CACHE = ROOT / "data" / "raw" / ".cache" / "osm_bali_hotels.json"
RADIUS_KM = 12.0

# Manual corrections found by eyeballing the matching log — see reports/eda/
# hotel_matching_log.md for why each of these needed a human in the loop:
#  - "Merccure Bali Legian" is a typo in the source CSV for "Mercure".
#  - "The Stones Hotel Legian Bali" auto-matched to "The Legian Bali" at a
#    perfect token_set_ratio score, but those are two different real hotels —
#    token_set_ratio scores 100 whenever one name's tokens are a *subset* of
#    the other's, which a short, similarly-worded but distinct hotel name can
#    trigger. Forcing it through direct-name geocoding instead avoids
#    silently shipping a wrong pin on the map.
#  - "The Anvaya Beach Resort" isn't found under its full marketing name —
#    Nominatim only has it as "The ANVAYA Hotel" and only resolves it with a
#    short "<name> <area>" phrasing, not the longer official name.
#  - "Radisson Blu Resort Bali Uluwatu" (now officially "Radisson Blu Resort
#    & Villas, Bali Uluwatu") isn't geocodable via Nominatim under any
#    phrasing tried, including its real street address from the hotel's own
#    site (Jl. Pemutih, Uluwatu, Pecatu) — left for manual geocoding rather
#    than guessing coordinates with no source.
#  - "Adiwana Bisma Ubud" only resolves to its street ("Jalan Bisma, Ubud"),
#    not the building itself — kept as a street-level approximation, flagged
#    via coord_precision below rather than presented as exact.
GEOCODE_QUERY_OVERRIDES = {
    "Merccure Bali Legian": "Mercure Bali Legian, Indonesia",
    "The Stones Hotel Legian Bali": "The Stones Hotel Legian Bali Autograph Collection, Indonesia",
    "The Anvaya Beach Resort": "Anvaya Kuta Bali",
    "Adiwana Bisma Ubud": "Bisma Ubud Bali",
}
FORCE_DIRECT_GEOCODE = {"The Stones Hotel Legian Bali", "The Anvaya Beach Resort", "Adiwana Bisma Ubud"}
STREET_LEVEL_ONLY = {"Adiwana Bisma Ubud"}


def load_mendeley_hotel_stats() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV, sep=";", encoding="utf-8-sig")
    agg = (
        df.groupby("Hotel")
        .agg(
            review_count=("Review", "count"),
            avg_rating=("Rating", "mean"),
            area=("Location", lambda s: s.mode().iat[0]),
        )
        .reset_index()
        .rename(columns={"Hotel": "name"})
    )
    return agg


def slugify(name: str) -> str:
    return (
        name.lower().strip().replace("&", "and").replace("'", "")
        .replace(",", "").replace(".", "").replace("  ", " ").replace(" ", "-")
    )


def main() -> None:
    print("Loading Mendeley hotel stats...")
    mendeley = load_mendeley_hotel_stats()
    print(f"  {len(mendeley)} unique hotels, {int(mendeley['review_count'].sum())} reviews total")

    areas = sorted(mendeley["area"].unique())
    print(f"Geocoding {len(areas)} area centroids via Nominatim (rate-limited)...")
    area_centroids: dict[str, tuple[float, float]] = {}
    for area in areas:
        g = geocode(f"{area}, Indonesia")
        if g:
            area_centroids[area] = (g["lat"], g["lon"])
            print(f"  {area:35s} -> {g['lat']:.4f}, {g['lon']:.4f}  ({g['display_name'][:60]})")
        else:
            print(f"  {area:35s} -> NOT FOUND")

    if OSM_CACHE.exists():
        print(f"Loading OSM hotels from cache ({OSM_CACHE.relative_to(ROOT)})...")
        osm_hotels = json.loads(OSM_CACHE.read_text())
    else:
        print("Fetching OSM hotels for Bali (Overpass)...")
        osm_hotels = fetch_hotels("Bali")
        OSM_CACHE.parent.mkdir(parents=True, exist_ok=True)
        OSM_CACHE.write_text(json.dumps(osm_hotels))
        print(f"  cached raw response -> {OSM_CACHE.relative_to(ROOT)} (delete it to force a re-fetch)")
    seen, deduped = set(), []
    for h in osm_hotels:
        key = h["name"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(h)
    osm_hotels = deduped
    print(f"  {len(osm_hotels)} unique-named OSM hotel entries in Bali")

    print(f"Matching (blocked to {RADIUS_KM}km radius per area)...")
    hotel_dicts = mendeley[["name", "area"]].to_dict("records")
    matches = match_hotels_blocked(hotel_dicts, osm_hotels, area_centroids, radius_km=RADIUS_KM)

    match_log = [
        "# Hotel name matching log",
        "",
        "Blocked fuzzy match (geocoded area centroid, 12km radius, token_set_ratio,",
        "margin required over runner-up). `matched` = auto-accepted, `review` = flagged,",
        "not auto-trusted, `unmatched` = fell back to a direct name geocode.",
        "",
        "| Mendeley name | Area | Status | Matched OSM name | Score | Gap | Candidates in radius |",
        "|---|---|---|---|---|---|---|",
    ]

    matched_osm_keys = set()
    rows = []
    needs_direct_geocode = []

    for _, m in mendeley.iterrows():
        name, area = m["name"], m["area"]
        res = matches[name]
        status = "review" if name in FORCE_DIRECT_GEOCODE else res["status"]
        if status == "matched":
            osm = res["osm"]
            matched_osm_keys.add(osm["name"].strip().lower())
            rows.append({
                "hotel_id": slugify(name), "name": name, "area": area,
                "lat": osm["lat"], "lon": osm["lon"],
                "source": "mendeley+osm", "match_status": "matched",
                "review_count": int(m["review_count"]), "avg_rating": round(m["avg_rating"], 2),
                "has_reviews": True, "match_score": round(res["score"], 1),
                "coord_precision": "exact",
            })
            match_log.append(f"| {name} | {area} | matched | {osm['name']} | {res['score']:.1f} | {res['gap_to_runner_up']:.1f} | {res['n_candidates']} |")
        else:
            needs_direct_geocode.append((name, area, m, res))

    print(f"  {len(rows)}/{len(mendeley)} auto-matched with high confidence")
    print(f"  {len(needs_direct_geocode)} need a direct name geocode fallback...")

    BALI_BBOX = {"lat": (-9.0, -8.0), "lon": (114.3, 115.8)}

    for name, area, m, res in needs_direct_geocode:
        query = GEOCODE_QUERY_OVERRIDES.get(name, f"{name}, Bali, Indonesia")
        g = geocode(query)
        if g is None:
            # Retry unbounded in case the strict Bali viewbox excluded a real
            # point, but still validate the result actually lands in Bali —
            # an unbounded query returning some other country is worse than
            # no coordinate at all.
            g = geocode(query, bounded_to_bali=False)
            if g is not None:
                lat_ok = BALI_BBOX["lat"][0] <= g["lat"] <= BALI_BBOX["lat"][1]
                lon_ok = BALI_BBOX["lon"][0] <= g["lon"] <= BALI_BBOX["lon"][1]
                if not (lat_ok and lon_ok):
                    print(f"    rejecting out-of-Bali geocode for {name!r}: {g['display_name']}")
                    g = None
        if g:
            rows.append({
                "hotel_id": slugify(name), "name": name, "area": area,
                "lat": g["lat"], "lon": g["lon"],
                "source": "mendeley+nominatim_direct", "match_status": f"osm_{res['status']}_then_direct_geocode",
                "review_count": int(m["review_count"]), "avg_rating": round(m["avg_rating"], 2),
                "has_reviews": True, "match_score": None,
                "coord_precision": "street_level" if name in STREET_LEVEL_ONLY else "exact",
            })
            best_osm_name = res["osm"]["name"] if res["osm"] else "-"
            match_log.append(f"| {name} | {area} | {res['status']} -> direct geocode OK | {best_osm_name} (rejected) | {res['score'] or 0:.1f} | - | {res['n_candidates']} |")
        else:
            rows.append({
                "hotel_id": slugify(name), "name": name, "area": area,
                "lat": None, "lon": None,
                "source": "mendeley_no_coords", "match_status": "needs_manual_geocode",
                "review_count": int(m["review_count"]), "avg_rating": round(m["avg_rating"], 2),
                "has_reviews": True, "match_score": None,
                "coord_precision": None,
            })
            match_log.append(f"| {name} | {area} | **NO MATCH, NO GEOCODE — manual needed** | - | - | - | - |")

    for osm in osm_hotels:
        key = osm["name"].strip().lower()
        if key in matched_osm_keys:
            continue
        rows.append({
            "hotel_id": f"osm-{osm['osm_id']}", "name": osm["name"],
            "area": osm["addr_suburb"] or osm["addr_city"] or "",
            "lat": osm["lat"], "lon": osm["lon"],
            "source": "osm_only", "match_status": "n/a",
            "review_count": 0, "avg_rating": None,
            "has_reviews": False, "match_score": None,
            "coord_precision": "exact",
        })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    n_manual = (out["match_status"] == "needs_manual_geocode").sum()
    print(f"\nSaved {len(out)} hotels -> {OUT_CSV.relative_to(ROOT)}")
    print(f"  matched (high confidence):        {(out['match_status'] == 'matched').sum()}")
    print(f"  matched via direct geocode fallback: {out['match_status'].astype(str).str.contains('direct_geocode').sum()}")
    print(f"  still needing manual geocoding:   {n_manual}")
    print(f"  OSM-only, no review data yet:     {(~out['has_reviews']).sum()}")

    MATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    MATCH_LOG.write_text("\n".join(match_log), encoding="utf-8")
    print(f"Wrote matching log -> {MATCH_LOG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
