import pytest
from pydantic import ValidationError

from nginep.query_parser import ParsedQuery


class TestParsedQuerySchema:
    def test_valid_aspects_accepted(self):
        pq = ParsedQuery(aspects_wanted=["wifi", "service", "sleep_quality"])
        assert pq.aspects_wanted == ["wifi", "service", "sleep_quality"]

    def test_invalid_aspect_name_rejected(self):
        # this is what stops the LLM from inventing an aspect outside the
        # taxonomy -- enforced by the schema, not just prompt wording
        with pytest.raises(ValidationError):
            ParsedQuery(aspects_wanted=["pool"])

    def test_defaults_are_all_none_or_empty(self):
        pq = ParsedQuery()
        assert pq.area is None
        assert pq.aspects_wanted == []
        assert pq.purpose is None
        assert pq.price_max_idr is None

    def test_price_accepts_plain_int(self):
        pq = ParsedQuery(price_max_idr=500000)
        assert pq.price_max_idr == 500000
