"""Turn a free-text search query (Indonesian or English) into a structured
filter, via the same Vertex AI Gemini setup as the aspect extractor.

This is the "query understanding" half of TanyaInap -- the same class of
problem Traveloka's own Universal Search team solved with NER + a trained
Product/Subproduct/Action classifier (see their Mar 2020 engineering blog
post). This takes an LLM-first approach instead: no training data, more
flexible for compound queries, but higher per-query cost and no guarantee
against a slightly-off area string -- retrieval has to handle that (fuzzy
matching against real hotel areas), not assume the parser is exact.
"""
from __future__ import annotations

from typing import Literal, Optional

from google.genai import types
from pydantic import BaseModel, Field

from .aspects import ASPECTS

# Literal[*ASPECTS] makes the LLM's choices schema-constrained to real
# aspect names -- it cannot invent one that doesn't exist in the taxonomy.
AspectName = Literal[tuple(ASPECTS)]  # type: ignore[valid-type]

MODEL = "gemini-3.1-flash-lite"


class ParsedQuery(BaseModel):
    area: Optional[str] = Field(None, description="Area/neighborhood in Bali mentioned, e.g. 'Ubud', 'Kuta', 'dekat pantai Sanur'. Null if none mentioned.")
    aspects_wanted: list[AspectName] = Field(default_factory=list, description="Which aspects the user explicitly cares about, from the fixed aspect list only.")
    purpose: Optional[str] = Field(None, description="Short free-text context if stated, e.g. 'kerja remote', 'bulan madu'. Null if not stated.")
    price_max_idr: Optional[int] = Field(None, description="Max price in Indonesian Rupiah if mentioned (e.g. 'di bawah 500 ribu' -> 500000). Null if not mentioned.")


_SYSTEM_INSTRUCTION = f"""Parse a hotel-search query (Indonesian or English) into structured filters.

aspects_wanted must only use these exact names: {", ".join(ASPECTS)}.
Only include an aspect if the user actually expressed a preference about it
(e.g. "wifi kencang" -> wifi, "sarapan enak" -> sunrise_meal, "tenang" ->
sleep_quality, "murah"/"budget" -> value, "dekat"/"strategis" -> accessibility).
Don't guess at aspects the query doesn't mention.

area should be the location phrase as the user wrote it (don't normalize/
translate it) -- the retrieval step handles fuzzy-matching it against real
hotel areas, not this parser.
"""


def make_client_and_parse(client, query_text: str) -> ParsedQuery:
    resp = client.models.generate_content(
        model=MODEL,
        contents=f"Query: {query_text}",
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=ParsedQuery,
            temperature=0.0,
        ),
    )
    if resp.parsed is None:
        raise ValueError(f"Model did not return parseable structured output: {resp.text!r}")
    return resp.parsed
