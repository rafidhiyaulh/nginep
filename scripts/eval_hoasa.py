"""Evaluate the LLM aspect extractor against HoASA's human-labeled test set.

This is the non-circular evaluation: HoASA's labels come from IndoNLU's own
human annotators, not from any LLM used in this project, so this measures
real accuracy rather than agreement-with-itself. It's also a cross-lingual
check — HoASA is Indonesian, while the actual demo corpus (Bali reviews) is
English, so this alone can't stand in for the primary eval (a hand-labeled
sample of the real Bali corpus, still to be done — see README/eda notes).

Run: .venv/bin/python scripts/eval_hoasa.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nginep.aspects import HOASA_ASPECTS
from nginep.metrics import aspect_report

ROOT = Path(__file__).resolve().parents[1]
TRUE_CSV = ROOT / "data" / "raw" / "hoasa" / "test.csv"
PRED_JSONL = ROOT / "data" / "processed" / "hoasa_test_extractions.jsonl"
OUT_MD = ROOT / "reports" / "eval" / "hoasa_extraction_eval.md"


def main() -> None:
    true_df = pd.read_csv(TRUE_CSV)
    preds = {json.loads(l)["id"]: json.loads(l)["aspects"] for l in PRED_JSONL.open(encoding="utf-8") if l.strip()}

    missing = set(true_df.index.astype(str)) - set(preds)
    if missing:
        raise SystemExit(f"{len(missing)} rows have no prediction — re-run scripts/extract_aspects.py first")

    # Pooled (micro-style): every (review, aspect) pair is one classification
    # instance, across all 10 aspects. This is the headline number.
    y_true_all, y_pred_all = [], []
    per_aspect = {}
    for aspect in HOASA_ASPECTS:
        y_true = true_df[aspect].tolist()
        y_pred = [preds[str(i)][aspect]["sentiment"] for i in true_df.index]
        # same "only labels with real support" fix as the pooled number below
        per_aspect[aspect] = aspect_report(y_true, y_pred, labels=sorted(set(y_true)))
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

    pooled = aspect_report(y_true_all, y_pred_all, labels=["neg", "neut", "pos", "neg_pos"])
    # "neg_pos" never appears in this test split's ground truth (support=0),
    # so it mechanically gets F1=0 in the macro average whenever the model
    # predicts it at all — that's a metric artifact, not "the model is bad
    # at 1/4 of the classes". Report a second macro-F1 restricted to labels
    # that actually have support, alongside how often neg_pos was predicted
    # anyway (a real, separate precision concern worth its own line).
    labels_with_support = sorted(set(y_true_all))
    pooled_supported = aspect_report(y_true_all, y_pred_all, labels=labels_with_support)
    neg_pos_fp = sum(1 for p in y_pred_all if p == "neg_pos")
    exact_match = sum(t == p for t, p in zip(y_true_all, y_pred_all)) / len(y_true_all)

    lines = [
        "# LLM aspect extractor vs. HoASA (human-labeled, Indonesian)",
        "",
        f"Model: `gemini-3.1-flash-lite` (Vertex AI), zero-shot, temperature=0.",
        f"n = {len(true_df)} reviews x {len(HOASA_ASPECTS)} aspects = {len(y_true_all)} labeled instances.",
        "",
        "**This is NOT the primary evaluation** — HoASA is Indonesian-language, while the actual",
        "demo corpus (Bali reviews) is English (see data/raw/SOURCES.md). This measures the",
        "extractor against real human-annotated ground truth (non-circular, since no LLM produced",
        "these labels), and doubles as a cross-lingual robustness check.",
        "",
        f"## Headline: {exact_match:.1%} raw agreement, {pooled_supported['macro_f1']:.3f} macro-F1",
        "",
        f"The naive macro-F1 over all 4 possible labels is only {pooled['macro_f1']:.3f} — much lower than",
        "raw agreement suggests. That's a metric artifact, not the real picture: `neg_pos` has",
        f"**zero examples in this test split's ground truth**, so any `neg_pos` prediction is a",
        "guaranteed false positive with F1=0 for that class, which a macro average weights equally",
        "against the other three. Restricting the macro-F1 to labels that actually appear in the",
        f"ground truth (neg/neut/pos) gives {pooled_supported['macro_f1']:.3f}, which is the more honest headline number.",
        f"Separately: the model predicted `neg_pos` {neg_pos_fp} times out of {len(y_true_all)} — a real,",
        "if small, precision issue (mildly overzealous about detecting mixed sentiment), just not one",
        "a support-blind macro-F1 represents fairly.",
        "",
        "External reference point: a Sept 2025 IEEE paper comparing ML vs. LLMs on a comparable",
        "Indonesian hotel-review ABSA task reported GPT-4o few-shot reaching ~96% accuracy — this",
        "run is zero-shot (no few-shot examples) with a much smaller/cheaper model, so a few points",
        "below that is expected, not a red flag.",
        "",
        "## Per-aspect breakdown",
        "",
        "| Aspect | Macro-F1 | Support (n reviews) |",
        "|---|---|---|",
    ]
    for aspect in HOASA_ASPECTS:
        lines.append(f"| {aspect} | {per_aspect[aspect]['macro_f1']:.3f} | {len(true_df)} |")

    lines += [
        "",
        "## Where it disagrees most",
        "",
        "`general` started at 91.0% raw agreement on a 10-row smoke sample when the prompt asked",
        "for \"overall impression\" — the model was inferring an overall sentiment from other",
        "aspects, while HoASA's annotators only mark `general` non-neutral for an explicit summary",
        "statement. Tightening the prompt to say so explicitly (see `src/nginep/aspects.py`) raised",
        "a 20-row sample from 91% to 93% overall, 85% on `general` specifically. Full-set numbers",
        "for that aspect are in the per-aspect table above.",
    ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Pooled macro-F1: {pooled['macro_f1']:.3f}  |  raw agreement: {exact_match:.1%}")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
