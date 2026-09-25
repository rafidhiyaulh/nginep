# Data sources

## Bali Hotel Reviews Dataset (in use)
- File: `bali_hotel_review.csv`
- Source: Pramudya, Y. G. & Alamsyah, A. (2023). *Bali Hotel Reviews Dataset* (Version 2) [Data set]. Mendeley Data. https://doi.org/10.17632/s62ycm698z.2
- License: CC BY 4.0 — https://creativecommons.org/licenses/by/4.0/ (attribution required, satisfied by the citation above)
- Retrieved: 2026-09-26
- Contents: 5,798 reviews, 16 hotels, 12 sub-areas of Bali. Columns: `Location, Hotel, UserID, Title, Review, Rating, Value, Accessibility, Service, Room, Cleanliness, Sleep Quality`.
- Caveats found during audit:
  - The six aspect columns (`Value, Accessibility, Service, Room, Cleanliness, Sleep Quality`) are binary presence flags (0/1), not sentiment scores. Annotation method is undocumented upstream — treat as unverified until spot-checked by hand.
  - Reviews are ~100% English despite covering Bali hotels.
  - Ratings are heavily skewed positive (99.8% are 4-5 stars).
  - Hotel coverage is concentrated: Legian/Kuta alone is ~61% of all reviews; Ubud has exactly one hotel (Adiwana Bisma Ubud).

## HoASA (planned — evaluation benchmark only, not demo inventory)
- Source: IndoNLU benchmark, `indonlp/indonlu` on Hugging Face (HoASA subset)
- License: MIT
- Contents: 2,854 examples, 10 aspects, sourced from AiryRooms reviews, no hotel identifiers.
- Role: secondary cross-lingual robustness check for the aspect extractor. The primary eval set will be a hand-labeled sample of the Bali CSV above, since that's the actual production-language (English) corpus — HoASA alone would test the wrong language distribution.

## OpenStreetMap (planned — location/POI enrichment only)
- Source: Overpass API
- License: ODbL — https://opendatacommons.org/licenses/odbl/ (attribution / share-alike on derived data)
- Role: hotel coordinates and landmarks for distance-based filtering, and to supplement the candidate pool with hotels that have no review coverage in the Mendeley dataset (surfaced honestly as "no review data yet", not hidden).

## Datafiniti Hotel Reviews (backup, not currently used)
- Source: Kaggle, `datafiniti/hotel-reviews`
- ~1,000 US hotels, English-only. Only a fallback if the pipeline needs a larger corpus for engineering/testing; not part of the Bali product scope.

## Explicitly excluded
Traveloka and Google review/listing data are not used anywhere in this project — scraping either would violate their Terms of Service.
