"""Generate EDA charts + a summary report for the Bali hotel reviews corpus.

Chart styling follows the project's dataviz conventions: single-hue blue for
magnitude/ordinal encodings (no decorative rainbow on single-series bars),
light chart surface, muted gridlines, thin bars, direct value labels instead
of a legend where there's only one series.

Run: .venv/bin/python scripts/run_eda.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "bali_hotel_review.csv"
OUT_DIR = ROOT / "reports" / "eda"

# dataviz skill palette (light mode)
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
BLUE_ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # steps 250/350/450/550/650, rating 1..5

ASPECT_COLS = ["Value", "Accessibility", "Service", "Room", "Cleanliness", "Sleep Quality"]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "font.size": 10,
    "font.family": "sans-serif",
    "axes.grid": True,
    "grid.color": GRIDLINE,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
})


def hbar(series: pd.Series, title: str, xlabel: str, fname: str, color=BLUE, fmt="{:.0f}"):
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.4 * len(series))))
    series = series.sort_values()
    bars = ax.barh(series.index.astype(str), series.values, color=color, height=0.65)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="y", visible=False)
    for bar, val in zip(bars, series.values):
        ax.text(bar.get_width() + series.values.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                 fmt.format(val), va="center", ha="left", fontsize=9, color=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_CSV, sep=";", encoding="utf-8-sig")
    df["review_len_words"] = df["Review"].astype(str).str.split().str.len()

    print(f"Loaded {len(df)} reviews, {df['Hotel'].nunique()} hotels, {df['Location'].nunique()} areas")

    # 1. Reviews per hotel
    hbar(df["Hotel"].value_counts(), "Reviews per hotel", "Number of reviews", "reviews_per_hotel.png")

    # 2. Reviews per area
    hbar(df["Location"].value_counts(), "Reviews per area", "Number of reviews", "reviews_per_area.png")

    # 3. Rating distribution (ordinal blue shading, lightest=1 star, darkest=5 star)
    rating_counts = df["Rating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [BLUE_ORDINAL[int(r) - 1] for r in rating_counts.index]
    bars = ax.bar(rating_counts.index.astype(str), rating_counts.values, color=colors, width=0.6)
    ax.set_xlabel("Rating (stars)")
    ax.set_ylabel("Number of reviews")
    ax.set_title("Rating distribution — heavily skewed positive", loc="left", fontsize=12, color=INK, pad=12)
    ax.grid(axis="x", visible=False)
    total = rating_counts.sum()
    for bar, val in zip(bars, rating_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                 f"{val}\n({val/total:.1%})", ha="center", va="bottom", fontsize=9, color=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rating_distribution.png", dpi=150)
    plt.close(fig)

    # 4. Review length distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["review_len_words"].clip(upper=300), bins=40, color=BLUE, edgecolor=SURFACE, linewidth=0.3)
    ax.set_xlabel("Review length (words, clipped at 300)")
    ax.set_ylabel("Number of reviews")
    ax.set_title("Review length distribution", loc="left", fontsize=12, color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "review_length_distribution.png", dpi=150)
    plt.close(fig)

    # 5. Aspect flag frequency (these are presence flags, not sentiment — see SOURCES.md)
    aspect_freq = df[ASPECT_COLS].apply(pd.to_numeric, errors="coerce").fillna(0).sum().sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(aspect_freq.index, aspect_freq.values, color=BLUE, height=0.6)
    ax.set_xlabel("Number of reviews flagging this aspect")
    ax.set_title("Aspect-flag frequency (presence, not sentiment)", loc="left", fontsize=12, color=INK, pad=12)
    ax.grid(axis="y", visible=False)
    for bar, val in zip(bars, aspect_freq.values):
        ax.text(bar.get_width() + aspect_freq.values.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{int(val)} ({val/len(df):.1%})", va="center", ha="left", fontsize=9, color=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "aspect_frequency.png", dpi=150)
    plt.close(fig)

    # Summary numbers for the report
    stats = {
        "n_reviews": len(df),
        "n_hotels": df["Hotel"].nunique(),
        "n_areas": df["Location"].nunique(),
        "pct_4_5_star": (df["Rating"] >= 4).mean(),
        "median_len": int(df["review_len_words"].median()),
        "reviews_top2_areas_pct": df["Location"].value_counts().head(2).sum() / len(df),
    }

    report = f"""# EDA — Bali Hotel Reviews (Mendeley)

Generated by `scripts/run_eda.py`. Source data: `data/raw/bali_hotel_review.csv`
(see `data/raw/SOURCES.md` for provenance/license).

## Headline numbers

- **{stats['n_reviews']:,} reviews** across **{stats['n_hotels']} hotels** in **{stats['n_areas']} areas** of Bali.
- **{stats['pct_4_5_star']:.1%}** of reviews are 4–5 stars — very little negative signal to learn from.
- Median review length: **{stats['median_len']} words**.
- Top 2 areas (Legian Kuta Bali + Kuta Bali) account for **{stats['reviews_top2_areas_pct']:.1%}** of all reviews — coverage is concentrated, not evenly spread across Bali.

## Reviews per hotel

![Reviews per hotel](reviews_per_hotel.png)

Highly uneven: the top hotel alone (Merccure Bali Legian / a Mendeley-CSV typo for
Mercure) has more reviews than the bottom ~10 hotels combined. Any per-hotel
aspect score needs smoothing so low-review hotels don't get noisy extreme scores.

## Reviews per area

![Reviews per area](reviews_per_area.png)

Ubud — the area used in this project's example query ("hotel tenang di Ubud
buat kerja remote") — has exactly one hotel in this dataset. See
`hotel_matching_log.md` and the project decision to supplement candidates
with OSM-listed hotels that have no review coverage, rather than pretending
Ubud has depth it doesn't.

## Rating distribution

![Rating distribution](rating_distribution.png)

TripAdvisor-style positivity skew, as expected. Only a handful of reviews are
3 stars or below — an aspect *sentiment* classifier trained/evaluated on this
corpus alone will barely see negative examples. The LLM extraction pipeline
should be evaluated with that in mind (e.g. don't claim strong negative-class
recall from a handful of examples).

## Review length

![Review length distribution](review_length_distribution.png)

## Aspect-flag frequency

![Aspect frequency](aspect_frequency.png)

**Caveat (see `data/raw/SOURCES.md`): these six columns are binary
presence flags, not sentiment scores**, and the annotation method isn't
documented upstream. Treat this chart as "what reviewers tend to mention",
not "what hotels score well on" — that distinction matters for how far this
data can be trusted as eval ground truth versus the LLM extraction pipeline's
own output (which will be evaluated against a hand-labeled sample instead,
per the data audit notes).
"""
    (OUT_DIR / "eda_report.md").write_text(report, encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'eda_report.md'}")
    print("Charts:", ", ".join(p.name for p in sorted(OUT_DIR.glob("*.png"))))


if __name__ == "__main__":
    main()
