"""TanyaInap web API. Loads hotel + aspect data once at startup (not per
request), and reuses one Vertex AI client for the process lifetime.

Run: .venv/bin/uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from nginep.extraction import make_client
from nginep.search import search

from .rate_limit import rate_limit

ROOT = Path(__file__).resolve().parents[1]
GCP_PROJECT = "nginep-tanyainap"

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["client"] = make_client(GCP_PROJECT)
    _state["hotels"] = pd.read_csv(ROOT / "data" / "processed" / "hotels.csv").to_dict("records")
    _state["aspect_scores"] = json.loads((ROOT / "data" / "processed" / "hotel_aspect_scores.json").read_text())
    n_reviewed = sum(1 for h in _state["hotels"] if h["has_reviews"])
    print(f"Loaded {len(_state['hotels'])} hotels ({n_reviewed} with review data, "
          f"{len(_state['aspect_scores'])} aggregated so far).")
    yield
    _state.clear()


app = FastAPI(title="TanyaInap API", lifespan=lifespan)


@app.get("/api/search", dependencies=[Depends(rate_limit)])
def api_search(q: str = Query(..., min_length=1, max_length=300), top_k: int = Query(10, ge=1, le=30)):
    parsed, results, also_nearby = search(_state["client"], q, _state["hotels"], _state["aspect_scores"], top_k=top_k)
    return {
        "query": q,
        "parsed": parsed.model_dump(),
        "results": [
            {
                "hotel_id": r.hotel_id, "name": r.name, "area": r.area,
                "score": round(r.score, 3),
                "avg_rating": r.avg_rating, "review_count": r.review_count,
                "evidence": r.evidence,
            }
            for r in results
        ],
        "also_nearby": also_nearby,
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "hotels_loaded": len(_state.get("hotels", [])),
        "hotels_with_aggregated_aspects": len(_state.get("aspect_scores", {})),
    }


app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")
