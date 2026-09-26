# How the Bali-English extraction accuracy check works

**This closes a real gap, not a cosmetic one.** The extractor's only accuracy
number against independent human labels (see `reports/eval/hoasa_extraction_eval.md`)
was measured on HoASA, which is Indonesian-language. The app's actual
production corpus (Bali reviews) is English. Nothing before this had ever
checked whether the same accuracy holds in the language actually being used.

## Why blind, and why the author labels it personally

The LLM's own extraction (`data/processed/bali_aspect_extractions.jsonl`) is
used **only** to decide which reviews go into the sample (see the stratification
section below) -- it is never shown during labeling. `labeling/bali_english_sample.json`,
the file the labeling tool actually loads, contains nothing but
`sample_id`, `row_id`, `hotel`, and the raw review text. If the LLM's guess
were visible while labeling, the resulting F1 would just be measuring how
often the labeler agreed with what they were shown, not real accuracy.

This makes the labels the project author's own judgment, not a second AI's
(unlike the ranking relevance labels in `LABELING_METHOD.md`, which were
Claude-assisted) -- appropriate here specifically because this number is
meant to validate the *same class of system* (an LLM) doing the *same kind
of task* (aspect-sentiment extraction). Having another LLM label it would
reintroduce exactly the circularity this check exists to rule out. The
trade-off: it's one person's judgment, not independently cross-checked by a
second human. Worth disclosing plainly if this number gets cited anywhere
that matters.

## Sampling: stratified by hotel and by aspect

Plain random sampling of ~100 reviews would barely touch the rarer aspects
-- in the full 5,798-review dataset, `bau` (smell) is flagged by the
extractor in only 1.5% of reviews, `tv` in 2.0%, `air_panas` in 2.7%,
`wifi` in 3.7%. A 100-review random sample would carry only 1-4 examples of
each, not enough to say anything meaningful about accuracy on that aspect.

`scripts/sample_bali_labeling_set.py` instead:
- Uses the LLM's own per-aspect output as a *sampling signal only* (never
  shown to the labeler) to guarantee every one of the 13 aspects has at
  least 15 reviews in the sample where that aspect looks plausibly
  mentioned, rarest aspects filled first.
- Caps any single hotel at 12 of the 100 reviews (a hotel whose reviews
  happen to skew toward a rare aspect -- e.g. an eco-retreat where guests
  specifically remark on AC/wifi/hot water -- would otherwise dominate that
  aspect's quota by itself), and guarantees every one of the 16 hotels has
  at least 2.
- Fixed random seed (42) for reproducibility.

Result actually sampled (re-run the script to reproduce): 100 reviews, all
16 hotels represented (2-12 each), every aspect's 15-review minimum met or
exceeded (see the script's printed summary for the exact final counts).

## Label scheme

Same 4-class scheme as HoASA and the extractor itself (`neg`/`neut`/`pos`/`neg_pos`,
see `src/nginep/aspects.py`), so the numbers are directly comparable. `neut`
covers both "aspect not mentioned" and genuinely neutral sentiment, matching
HoASA's own convention -- every aspect defaults to `neut` in the labeling
UI, and the labeler only changes the ones actually discussed in that review.

## Workflow

1. `.venv/bin/python scripts/sample_bali_labeling_set.py` -- regenerates
   `labeling/bali_english_sample.json` (already committed; re-run only if
   the extraction data changes).
2. Open `labeling/aspect_labeling.html` in a browser, load that JSON file.
3. Label all 100 reviews (progress autosaves to the browser's localStorage,
   safe to close and resume).
4. Export -> save the download as `labeling/bali_aspect_labels.json`.
5. `.venv/bin/python scripts/eval_bali_extraction.py` -- joins the labels
   back to the LLM's predictions (by `row_id`) and writes
   `reports/eval/bali_extraction_eval.md`.

## Status

Sampling done, tooling built and smoke-tested. **Labeling itself not done
yet** -- `labeling/bali_aspect_labels.json` does not exist until step 3-4
above happens. Until then, the language-mismatch limitation stays in the
README as-is.
