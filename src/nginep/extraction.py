"""LLM-based per-aspect sentiment extraction, via Vertex AI Gemini.

Each review is sent once, with a JSON response schema enforced by the API
(not just requested in the prompt) so parsing never fails on malformed
output. Every non-neutral aspect must come with a short verbatim quote from
the review as evidence — the product shows these quotes directly to end
users, so a hallucinated quote is worse than a missing one, and the prompt
says so explicitly.
"""
from __future__ import annotations

from typing import Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, create_model

from .aspects import ASPECT_MEANINGS, ASPECTS, Sentiment

MODEL = "gemini-3.1-flash-lite"
LOCATION = "global"


class AspectResult(BaseModel):
    sentiment: Sentiment
    quote: Optional[str] = None


# Built from ASPECTS rather than spelled out field-by-field so the schema
# can never drift out of sync with the taxonomy in aspects.py.
ReviewAspects = create_model(
    "ReviewAspects",
    **{aspect: (AspectResult, ...) for aspect in ASPECTS},
)


def _system_instruction() -> str:
    aspect_lines = "\n".join(f"- {a}: {ASPECT_MEANINGS[a]}" for a in ASPECTS)
    return f"""You label hotel reviews for a hotel-search product. For each of these {len(ASPECTS)} aspects, decide the sentiment expressed in the review.

Sentiment values:
- "pos": aspect mentioned positively
- "neg": aspect mentioned negatively
- "neut": aspect not mentioned, or mentioned without clear positive/negative sentiment
- "neg_pos": aspect mentioned with both positive and negative sentiment (e.g. different rooms/times)

For every aspect that is NOT "neut", also give a short quote copied VERBATIM
from the review text as evidence. This quote is shown directly to end users
as proof, so never invent or paraphrase it — if you can't find an exact
substring supporting a sentiment, use "neut" instead.

Aspects:
{aspect_lines}
"""


def make_client(project: str, location: str = LOCATION) -> genai.Client:
    return genai.Client(vertexai=True, project=project, location=location)


def extract_aspects(client: genai.Client, review_text: str) -> dict:
    """Returns {aspect: {"sentiment": "pos"|"neg"|"neut"|"neg_pos", "quote": str|None}}
    for every aspect in ASPECTS. Raises on API/parsing failure — callers own
    retry policy (see scripts/extract_aspects.py).
    """
    resp = client.models.generate_content(
        model=MODEL,
        contents=f"Review:\n{review_text}",
        config=types.GenerateContentConfig(
            system_instruction=_system_instruction(),
            response_mime_type="application/json",
            response_schema=ReviewAspects,
            temperature=0.0,
        ),
    )
    if resp.parsed is None:
        raise ValueError(f"Model did not return parseable structured output: {resp.text!r}")
    return resp.parsed.model_dump(mode="json")
