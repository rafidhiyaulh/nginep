"""Fixed-window rate limiting per client IP, same pattern as this author's
other project (Charissa) -- protects the Vertex AI budget and the service
from abuse now that this is a publicly reachable URL, not just localhost.

In-memory, so it's only correct with a single running instance -- the Cloud
Run deploy for this service pins --max-instances=1 for exactly this reason
(also a hard cap on worst-case concurrent cost, independent of this limiter).
A real multi-instance deployment would need shared state (Redis, Firestore)
instead; not needed at this project's traffic scale.
"""
from __future__ import annotations

import time
from collections import defaultdict

from fastapi import HTTPException, Request

WINDOW_SECONDS = 3600
MAX_REQUESTS_PER_WINDOW = 20

_hits: dict[str, list[float]] = defaultdict(list)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request) -> None:
    ip = _client_ip(request)
    now = time.monotonic()
    window_start = now - WINDOW_SECONDS
    hits = [t for t in _hits[ip] if t > window_start]
    if len(hits) >= MAX_REQUESTS_PER_WINDOW:
        raise HTTPException(
            status_code=429,
            detail=f"Terlalu banyak permintaan. Maksimal {MAX_REQUESTS_PER_WINDOW} pencarian per jam, coba lagi nanti ya.",
        )
    hits.append(now)
    _hits[ip] = hits
