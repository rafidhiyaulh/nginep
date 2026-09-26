from nginep.search import _area_matches, _explain, _score_hotel, aspect_evidence


class TestAreaMatches:
    def test_no_area_filter_matches_everything(self):
        assert _area_matches(None, "Ubud Bali") is True

    def test_substring_area_matches(self):
        assert _area_matches("Ubud", "Ubud Bali") is True

    def test_unrelated_area_does_not_match(self):
        assert _area_matches("Ubud", "Nusa Dua Bali") is False

    def test_phrase_with_landmark_still_matches_area_token(self):
        assert _area_matches("dekat pantai Kuta", "Kuta Bali") is True

    def test_shared_generic_word_does_not_cause_false_match(self):
        # found via real end-to-end testing: "Nusa Dua" (mainland resort
        # area) was matching "Nusa Ceningan Bali" (a different island) at
        # 66.7 under plain token_set_ratio, purely because both share the
        # generic word "Nusa" -- same failure class as the OSM hotel-name
        # matching bug, just a different call site.
        assert _area_matches("Nusa Dua", "Nusa Ceningan Bali") is False
        assert _area_matches("Nusa Dua", "Nusa Dua Bali") is True

    def test_query_area_of_just_bali_matches_everything(self):
        # after stripping the stopword "bali" nothing distinctive is left
        # -- don't turn that into "matches nothing"
        assert _area_matches("Bali", "Ubud Bali") is True

    def test_known_synonym_matches(self):
        # "Uluwatu" doesn't textually overlap with "Pecatu Bali" at all,
        # but Uluwatu sits inside the Pecatu admin area in the real data
        assert _area_matches("Uluwatu", "Pecatu Bali") is True
        assert _area_matches("Uluwatu", "Labuan Sait Pecatu Bali") is True


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


class TestAspectEvidence:
    def test_empty_hits_gives_empty_list(self):
        assert aspect_evidence({}) == []

    def test_uses_friendly_label_not_raw_key(self):
        hits = {"sunrise_meal": {"mentioned": 10, "pos": 8, "pos_quotes": ["enak"], "neg_quotes": []}}
        rows = aspect_evidence(hits)
        assert rows[0]["label"] == "sarapan"

    def test_positive_flag_matches_the_verb(self):
        hits = {"wifi": {"mentioned": 10, "pos": 9, "pos_quotes": ["kencang"], "neg_quotes": []}}
        row = aspect_evidence(hits)[0]
        assert row["positive"] is True
        assert row["verb"] == "dipuji"

    def test_negative_leaning_uses_dikeluhkan(self):
        hits = {"wifi": {"mentioned": 10, "pos": 2, "pos_quotes": [], "neg_quotes": ["lemot banget"]}}
        row = aspect_evidence(hits)[0]
        assert row["positive"] is False
        assert row["verb"] == "dikeluhkan"
        assert row["quote"] == "lemot banget"

    def test_sorted_by_pct_descending(self):
        hits = {
            "wifi": {"mentioned": 10, "pos": 2, "pos_quotes": [], "neg_quotes": ["a"]},
            "service": {"mentioned": 10, "pos": 9, "pos_quotes": ["b"], "neg_quotes": []},
        }
        rows = aspect_evidence(hits)
        assert [r["label"] for r in rows] == ["pelayanan", "wifi"]
