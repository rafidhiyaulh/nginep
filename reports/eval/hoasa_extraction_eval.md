# LLM aspect extractor vs. HoASA (human-labeled, Indonesian)

Model: `gemini-3.1-flash-lite` (Vertex AI), zero-shot, temperature=0.
n = 286 reviews x 10 aspects = 2860 labeled instances.

**This is NOT the primary evaluation** — HoASA is Indonesian-language, while the actual
demo corpus (Bali reviews) is English (see data/raw/SOURCES.md). This measures the
extractor against real human-annotated ground truth (non-circular, since no LLM produced
these labels), and doubles as a cross-lingual robustness check.

## Headline: 91.9% raw agreement, 0.856 macro-F1

The naive macro-F1 over all 4 possible labels is only 0.642 — much lower than
raw agreement suggests. That's a metric artifact, not the real picture: `neg_pos` has
**zero examples in this test split's ground truth**, so any `neg_pos` prediction is a
guaranteed false positive with F1=0 for that class, which a macro average weights equally
against the other three. Restricting the macro-F1 to labels that actually appear in the
ground truth (neg/neut/pos) gives 0.856, which is the more honest headline number.
Separately: the model predicted `neg_pos` 10 times out of 2860 — a real,
if small, precision issue (mildly overzealous about detecting mixed sentiment), just not one
a support-blind macro-F1 represents fairly.

External reference point: a Sept 2025 IEEE paper comparing ML vs. LLMs on a comparable
Indonesian hotel-review ABSA task reported GPT-4o few-shot reaching ~96% accuracy — this
run is zero-shot (no few-shot examples) with a much smaller/cheaper model, so a few points
below that is expected, not a red flag.

## Per-aspect breakdown

| Aspect | Macro-F1 | Support (n reviews) |
|---|---|---|
| ac | 0.968 | 286 |
| air_panas | 0.836 | 286 |
| bau | 0.933 | 286 |
| general | 0.561 | 286 |
| kebersihan | 0.892 | 286 |
| linen | 0.616 | 286 |
| service | 0.904 | 286 |
| sunrise_meal | 0.829 | 286 |
| tv | 0.986 | 286 |
| wifi | 0.990 | 286 |

## Where it disagrees most

`general` started at 91.0% raw agreement on a 10-row smoke sample when the prompt asked
for "overall impression" — the model was inferring an overall sentiment from other
aspects, while HoASA's annotators only mark `general` non-neutral for an explicit summary
statement. Tightening the prompt to say so explicitly (see `src/nginep/aspects.py`) raised
a 20-row sample from 91% to 93% overall, 85% on `general` specifically. Full-set numbers
for that aspect are in the per-aspect table above.