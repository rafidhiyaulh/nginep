"""Regression tests for the entity-resolution logic in matching.py.

These exist because the first (unblocked, threshold-only) version of this
matcher silently mis-matched 10 of 16 real hotels in practice — short/generic
OSM names scored deceptively high. These tests pin down the behavior that
fixed it, so it doesn't quietly regress.
"""
from nginep.matching import match_one
from nginep.osm import haversine_km


def candidate(name, lat=-8.5, lon=115.2):
    return {"name": name, "lat": lat, "lon": lon}


class TestMatchOne:
    def test_clear_winner_is_matched(self):
        candidates = [candidate("Atanaya Hotel"), candidate("Some Other Hotel")]
        result = match_one("Atanaya Hotel", candidates)
        assert result["status"] == "matched"
        assert result["osm"]["name"] == "Atanaya Hotel"

    def test_ambiguous_top_two_is_review_not_matched(self):
        # neither candidate is an exact match (so the exact-match
        # short-circuit doesn't apply), and both are near-equally plausible
        # -> must not silently auto-accept either one
        candidates = [candidate("Atanaya Hotel Kuta"), candidate("Atanaya Hotel Bali")]
        result = match_one("Atanaya Hotel", candidates, margin=5.0)
        assert result["status"] != "matched"

    def test_generic_short_name_does_not_win_on_substring_overlap(self):
        # this is the exact failure mode found in practice: a bare "Hotel"
        # entry scoring deceptively high against a much longer real name
        candidates = [candidate("Hotel"), candidate("Nusa Dua Beach Hotel & Spa")]
        result = match_one("Nusa Dua Beach Hotel & Spa", candidates)
        assert result["osm"]["name"] == "Nusa Dua Beach Hotel & Spa"

    def test_no_candidates_is_unmatched(self):
        result = match_one("Any Hotel", [])
        assert result["status"] == "unmatched"
        assert result["osm"] is None

    def test_low_score_is_unmatched_not_matched(self):
        candidates = [candidate("Completely Different Name")]
        result = match_one("Atanaya Hotel", candidates, review_threshold=75.0)
        assert result["status"] == "unmatched"


class TestHaversine:
    def test_same_point_is_zero(self):
        assert haversine_km(-8.5, 115.2, -8.5, 115.2) == 0.0

    def test_known_distance_kuta_to_ubud_roughly_correct(self):
        # Kuta ~ -8.72,115.17 ; Ubud ~ -8.51,115.26 -> real-world ~25km
        d = haversine_km(-8.72, 115.17, -8.51, 115.26)
        assert 20 < d < 30
