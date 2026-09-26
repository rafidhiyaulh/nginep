"""Evaluate the LLM aspect extractor against a hand-labeled sample of the
real, English-language Bali reviews -- the corpus the app actually uses in
production, unlike HoASA (Indonesian, see eval_hoasa.py).

Run this AFTER labeling: open labeling/aspect_labeling.html, load
labeling/bali_english_sample.json, label all 100 reviews, export, and save
the result as labeling/bali_aspect_labels.json.

Run: .venv/bin/python scripts/eval_bali_extraction.py
"""
from __future__ import annotations

import json
from pathlib import Path

from nginep.aspects import ASPECTS
from nginep.metrics import aspect_report

ROOT = Path(__file__).resolve().parents[1]
LABELS_JSON = ROOT / "labeling" / "bali_aspect_labels.json"
PRED_JSONL = ROOT / "data" / "processed" / "bali_aspect_extractions.jsonl"
OUT_MD = ROOT / "reports" / "eval" / "bali_extraction_eval.md"


def main() -> None:
    if not LABELS_JSON.exists():
        raise SystemExit(
            f"{LABELS_JSON.relative_to(ROOT)} not found. Label the sample first: "
            "open labeling/aspect_labeling.html, load bali_english_sample.json, "
            "label all reviews, and export as bali_aspect_labels.json in this folder."
        )

    labels = json.loads(LABELS_JSON.read_text(encoding="utf-8"))
    preds = {json.loads(l)["id"]: json.loads(l)["aspects"] for l in PRED_JSONL.open(encoding="utf-8") if l.strip()}

    missing = [r["row_id"] for r in labels if r["row_id"] not in preds]
    if missing:
        raise SystemExit(f"{len(missing)} labeled row_ids have no LLM prediction: {missing[:5]}...")

    y_true_all, y_pred_all = [], []
    per_aspect = {}
    for aspect in ASPECTS:
        y_true = [r["labels"][aspect] for r in labels]
        y_pred = [preds[r["row_id"]][aspect]["sentiment"] for r in labels]
        per_aspect[aspect] = aspect_report(y_true, y_pred, labels=sorted(set(y_true) | set(y_pred)))
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

    labels_with_support = sorted(set(y_true_all))
    pooled = aspect_report(y_true_all, y_pred_all, labels=labels_with_support)
    exact_match = sum(t == p for t, p in zip(y_true_all, y_pred_all)) / len(y_true_all)

    lines = [
        "# LLM aspect extractor vs. hand-labeled Bali reviews (English, blind)",
        "",
        f"Model: `gemini-3.1-flash-lite` (Vertex AI), zero-shot, temperature=0.",
        f"n = {len(labels)} reviews x {len(ASPECTS)} aspects = {len(y_true_all)} labeled instances.",
        "",
        "**This is the primary accuracy number for the corpus the app actually uses.**",
        "HoASA (see hoasa_extraction_eval.md) is Indonesian-language and only a cross-lingual",
        "robustness check; this sample is English-language Bali reviews, hand-labeled blind",
        "(no LLM output visible during labeling) by the project's own author. See",
        "`labeling/BALI_LABELING_METHOD.md` for the full sampling and labeling methodology,",
        "including why it isn't the same as a fully independent third-party annotation.",
        "",
        f"## Headline: {exact_match:.1%} raw agreement, {pooled['macro_f1']:.3f} macro-F1",
        "",
        "## Per-aspect breakdown",
        "",
        "| Aspect | Macro-F1 | Support (n reviews) |",
        "|---|---|---|",
    ]
    for aspect in ASPECTS:
        lines.append(f"| {aspect} | {per_aspect[aspect]['macro_f1']:.3f} | {len(labels)} |")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Pooled macro-F1: {pooled['macro_f1']:.3f}  |  raw agreement: {exact_match:.1%}")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
