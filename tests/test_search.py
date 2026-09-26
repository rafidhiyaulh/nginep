from nginep.search import _area_matches, _explain, _score_hotel


class TestAreaMatches:
    def test_no_area_filter_matches_everything(self):
        assert _area_matches(None, "Ubud Bali") is True

    def test_substring_area_matches(self):
        assert _area_matches("Ubud", "Ubud Bali") is True

    def test_unrelated_area_does_not_match(self):
        assert _area_matches("Ubud", "Nusa Dua Bali") is False

    def test_phrase_with_landmark_still_matches_area_token(self):
        assert _area_matches("dekat pantai Kuta", "Kuta Bali") is True


class TestScoreHotel:
    def test_no_wanted_aspects_gives_zero(self):
        score, hits = _score_hotel([], {"wifi": {"mentioned": 10, "smoothed_pos_rate": 0.9}})
        assert score == 0.0
        assert hits == {}

    def test_averages_over_mentioned_aspects_only(self):
        hotel_aspects = {
            "wifi": {"mentioned": 10, "smoothed_pos_rate": 0.8, "pos": 8, "neg": 2, "pos_quotes": [], "neg_quotes": []},
            "service": {"mentioned": 5, "smoothed_pos_rate": 0.6, "pos": 3, "neg": 2, "pos_quotes": [], "neg_quotes": []},
            "tv": {"mentioned": 0, "smoothed_pos_rate": 0.5, "pos": 0, "neg": 0, "pos_quotes": [], "neg_quotes": []},
        }
        # tv has zero mentions -> excluded from the average, not counted as 0.5
        score, hits = _score_hotel(["wifi", "service", "tv"], hotel_aspects)
        assert score == (0.8 + 0.6) / 2
        assert set(hits) == {"wifi", "service"}

    def test_zero_mentions_for_all_wanted_aspects_gives_zero_not_crash(self):
        hotel_aspects = {"wifi": {"mentioned": 0, "smoothed_pos_rate": 0.5, "pos": 0, "neg": 0, "pos_quotes": [], "neg_quotes": []}}
        score, hits = _score_hotel(["wifi"], hotel_aspects)
        assert score == 0.0
        assert hits == {}

    def test_missing_aspect_key_is_treated_as_no_evidence(self):
        # hotel_aspects doesn't even have a "wifi" key at all
        score, hits = _score_hotel(["wifi"], {})
        assert score == 0.0
        assert hits == {}


class TestExplain:
    def test_empty_hits_gives_no_data_message(self):
        assert "Belum ada" in _explain({})

    def test_includes_a_real_quote_when_available(self):
        hits = {"wifi": {"mentioned": 10, "pos": 8, "pos_quotes": ["wifi kencang banget"], "neg_quotes": []}}
        explanation = _explain(hits)
        assert "wifi kencang banget" in explanation
        assert "8" in explanation and "10" in explanation

    def test_falls_back_to_negative_quote_if_no_positive_quote(self):
        hits = {"wifi": {"mentioned": 5, "pos": 1, "pos_quotes": [], "neg_quotes": ["wifi lemot"]}}
        explanation = _explain(hits)
        assert "wifi lemot" in explanation
