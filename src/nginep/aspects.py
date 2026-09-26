"""Shared aspect taxonomy for TanyaInap.

10 of the 13 aspects are named to match HoASA exactly (ac, air_panas, bau,
general, kebersihan, linen, service, sunrise_meal, tv, wifi) on purpose: it
lets the extractor be evaluated directly against HoASA's human-labeled test
set with zero relabeling or name-mapping. The other 3 (value, accessibility,
sleep_quality) aren't in HoASA but matter for the product — budget queries,
"near X", and "tenang/quiet" (the flagship example query is specifically
about a quiet Ubud stay for remote work).

Sentiment labels also reuse HoASA's own 4-class scheme rather than inventing
a new one, including its convention that "neut" covers both true-neutral
sentiment AND "aspect not mentioned" (HoASA doesn't distinguish these).
"""
from enum import Enum

HOASA_ASPECTS = [
    "ac", "air_panas", "bau", "general", "kebersihan",
    "linen", "service", "sunrise_meal", "tv", "wifi",
]
PRODUCT_ONLY_ASPECTS = ["value", "accessibility", "sleep_quality"]
ASPECTS = HOASA_ASPECTS + PRODUCT_ONLY_ASPECTS

ASPECT_MEANINGS = {
    "ac": "air conditioning",
    "air_panas": "hot water",
    "bau": "smell/odor",
    "general": (
        "an EXPLICIT overall/summary judgment about the hotel as a whole "
        "(e.g. \"overall great stay\", \"worth it\", \"won't come back\") — "
        "do not infer this from the sentiment of other specific aspects; "
        "use neut unless the review makes a standalone summary statement"
    ),
    "kebersihan": "cleanliness",
    "linen": "bedsheets/towels",
    "service": "staff service",
    "sunrise_meal": "breakfast",
    "tv": "television",
    "wifi": "wifi/internet",
    "value": "value for money / price fairness",
    "accessibility": "location — how easy to reach, how close to things",
    "sleep_quality": "quietness / noise / how well one could sleep",
}


# Friendly Indonesian labels for showing an aspect to an end user -- the
# raw keys above ("sunrise_meal", "sleep_quality") are internal taxonomy
# names, not words a user should ever see on screen.
ASPECT_LABELS_ID = {
    "ac": "AC",
    "air_panas": "air panas",
    "bau": "aroma kamar",
    "general": "kesan umum",
    "kebersihan": "kebersihan",
    "linen": "sprei & handuk",
    "service": "pelayanan",
    "sunrise_meal": "sarapan",
    "tv": "TV",
    "wifi": "wifi",
    "value": "harga",
    "accessibility": "lokasi & akses",
    "sleep_quality": "ketenangan",
}


class Sentiment(str, Enum):
    neg = "neg"
    neut = "neut"
    pos = "pos"
    neg_pos = "neg_pos"


SENTIMENT_LABELS = [s.value for s in Sentiment]
