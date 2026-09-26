# How these relevance labels were made

**This is Claude-assisted labeling, not independently human-verified.** Said
plainly: an AI assistant (Claude, not the Gemini models the search pipeline
itself runs on) read each query and candidate and assigned relevance. That
is a real limitation, not a footnote to skip. If these NDCG/ranker numbers
get cited anywhere that matters (an interview, for instance), say so
honestly: "the eval set was Claude-assisted and spot-checked, not fully
independently labeled." See the recommendation at the bottom before trusting
these numbers as final.

## Why this isn't as circular as it might sound, but still isn't gold-standard

The search pipeline's query parser, aspect extractor, and embeddings all run
on Gemini (Vertex AI). These labels were produced by Claude reading the
same underlying data (query intent + extracted aspect evidence) and judging
independently, deliberately not copying the pipeline's own `score` field.
A different model family judging the same evidence is a real, if partial,
reduction in circularity risk (it can't just be reproducing its own
reasoning), but it is not the same as a human who actually knows Bali
hotels or has stayed in one of them. Two sanity checks worth knowing:

- Every relevance label was assigned by reading the actual `explanation`
  text (area name + which review quotes support the requested aspects) and
  applying the rubric below, not by rank-ordering the existing `score`.
- Pearson correlation between the labels and the pipeline's own `score`
  (among the 360 candidates that had review data at all) is **0.81** --
  strongly positive, as expected since both are reading the same
  underlying evidence, but not 1.0. There's real, independent variance.

## Rubric

- **3** -- area genuinely matches the query's intent AND the explanation
  shows strong, specific positive evidence for the aspects actually asked
  for (e.g. high percentage, meaningful review count, an on-topic quote).
- **2** -- area matches AND there's decent supporting evidence for at least
  one requested aspect, or the evidence is positive but thin (small review
  count) or partial (one of several requested aspects has no evidence).
- **1** -- area matches (or is a plausible/adjacent match) but there's
  little or no evidence for what was actually asked. This is the default
  for every `has_reviews: false` candidate (620 of 980) -- plausible, not
  irrelevant, but genuinely unverified, and for reviewed hotels with only a
  handful of mentions or a percentage barely over half.
- **0** -- the available evidence actually contradicts what was asked (a
  minority-negative or clearly negative quote for the requested aspect), or
  the area doesn't really match the query's intent.

## What actually got judged

- 980 candidates across 51 queries (2 queries ended up with 0 candidates
  after area matching -- see below).
- 360 candidates were real hotels with review evidence -- read and judged
  individually against the rubric above.
- 620 candidates had no review data (`has_reviews: false`, OSM-listed
  hotels with no Mendeley review coverage) -- these all default to **1**
  rather than being individually judged, since there's no aspect evidence
  to differentiate them on; they're plausible (real hotel, right area) but
  unverified, which is exactly what a 1 should mean.

Score distribution: **0** x11, **1** x718, **2** x54, **3** x197.

Two queries ("villa di Nusa Ceningan dekat Blue Lagoon" and "boutique hotel
in north Bali (Buleleng)") returned 0 candidates after area matching -- the
parsed area string didn't clear the area-matching threshold against any
hotel in the dataset. Nothing to label there; worth knowing as a real gap
in area-matching coverage, separate from this labeling exercise.

## A judgment call worth flagging

Small-sample high percentages (e.g. "100% positive, but only 1 review
mentioned it") were deliberately scored lower (1-2, not 3) than a strong
percentage with a real sample size, since "wifi kencang" claimed off one
review isn't the same confidence as the same claim from fifty. This
tightens the correlation with the pipeline's own smoothed scoring (which
does the same thing deliberately) but is the right call either way -- a
human labeler applying a sane rubric would very likely make the same call,
not an artifact of matching the system under test.

## Recommendation

Before treating the NDCG/ranker numbers computed from this file as
something you can stand behind in an interview: spot-check a random 30-50
of the 360 individually-judged (has_reviews=true) rows yourself. If your
judgment mostly agrees, you can honestly say "Claude-assisted, spot-checked
by me." If it doesn't, that's worth knowing before the numbers get used for
anything, not after.
