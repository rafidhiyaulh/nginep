"""Run the LLM aspect extractor over a CSV of reviews -> JSONL of results.

Resumable: writes one JSON line per review as soon as it's done, and skips
any row IDs already present in the output file on a re-run — an interrupted
~20-minute job (thousands of reviews, each its own API call) doesn't have to
restart from zero. Concurrent (thread pool; these are I/O-bound API calls),
with retry-with-backoff on transient failures (rate limits, 5xx).

Usage:
  .venv/bin/python scripts/extract_aspects.py \
      --input data/raw/bali_hotel_review.csv --csv-sep ";" \
      --text-col Review --id-col _row \
      --output data/processed/bali_aspect_extractions.jsonl

  .venv/bin/python scripts/extract_aspects.py \
      --input data/raw/hoasa/test.csv \
      --text-col review --id-col _row \
      --output data/processed/hoasa_test_extractions.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import pandas as pd

from nginep.extraction import extract_aspects, make_client

GCP_PROJECT = "nginep-tanyainap"
MAX_RETRIES = 5


def load_rows(input_path: Path, csv_sep: str, text_col: str, id_col: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, sep=csv_sep, encoding="utf-8-sig")
    if id_col == "_row":
        df["_row"] = df.index.astype(str)
    df = df[df[text_col].notna() & (df[text_col].astype(str).str.strip() != "")]
    return df


def already_done(output_path: Path) -> set[str]:
    """IDs that succeeded. A row that exhausted retries and recorded an
    error is deliberately NOT counted as done -- otherwise a transient
    failure (e.g. a 429 rate-limit) would be permanently stuck, silently
    skipped on every future re-run instead of retried."""
    if not output_path.exists():
        return set()
    done = set()
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("error") is None:
                    done.add(rec["id"])
            except (json.JSONDecodeError, KeyError):
                continue  # tolerate a truncated last line from a killed run
    return done


def extract_one(client, row_id: str, text: str) -> dict:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            aspects = extract_aspects(client, text)
            return {"id": row_id, "aspects": aspects, "error": None}
        except Exception as e:  # noqa: BLE001 — genuinely want to retry any transient failure
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(min(2 ** attempt, 30))
    return {"id": row_id, "aspects": None, "error": str(last_err)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--csv-sep", default=",")
    ap.add_argument("--text-col", required=True)
    ap.add_argument("--id-col", default="_row")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="process only the first N rows (smoke test)")
    args = ap.parse_args()

    df = load_rows(args.input, args.csv_sep, args.text_col, args.id_col)
    if args.limit:
        df = df.head(args.limit)

    done = already_done(args.output)
    todo = [(str(r[args.id_col]), str(r[args.text_col])) for _, r in df.iterrows() if str(r[args.id_col]) not in done]
    print(f"{len(df)} rows total, {len(done)} already done, {len(todo)} to process (max_workers={args.max_workers})")

    if not todo:
        print("Nothing to do.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = make_client(GCP_PROJECT)
    write_lock = Lock()
    n_ok, n_err = 0, 0
    t0 = time.monotonic()

    with args.output.open("a", encoding="utf-8") as out_f, ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(extract_one, client, row_id, text): row_id for row_id, text in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            with write_lock:
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
            if result["error"] is None:
                n_ok += 1
            else:
                n_err += 1
                print(f"  [{result['id']}] FAILED: {result['error'][:200]}", file=sys.stderr)
            if i % 100 == 0 or i == len(todo):
                elapsed = time.monotonic() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(f"  {i}/{len(todo)} done ({n_ok} ok, {n_err} failed) — {rate:.1f}/s, {elapsed:.0f}s elapsed")

    print(f"Done. {n_ok} succeeded, {n_err} failed. Output -> {args.output}")


if __name__ == "__main__":
    main()
