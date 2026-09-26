# Technical details

Moved out of the main README to keep that one focused on the business side. See `README.md` for the why and the results.

## Architecture

Offline, run once to prepare the data (see `scripts/`):

```
Mendeley Bali hotel reviews ──┐
                               ├──► LLM aspect extraction ──► per-hotel score aggregation
OpenStreetMap (location) ─────┘              │
                                              ▼
                                    data/processed/*.json, *.csv
                                    (baked into the deployed app)
```

Live, on every search:

```
Browser
  │  HTTPS
  ▼
FastAPI + a plain HTML/JS frontend (Cloud Run, scales to zero when idle)
  │
  ▼
Vertex AI (Gemini 3.1 Flash-Lite): parses the query into a structured filter
  │
  ▼
Ranker scores the pre-computed hotel data, no LLM call needed per hotel
```

- **LLM layer**: Vertex AI, not a plain API key. See `docs/vertex_ai_setup.md` for why.
- **Data**: pre-computed offline, not queried live. No database, a redeploy is how the live data updates.
- **Live deploy details** (service account, rate limiting, budget alert, redeploy command): see `docs/deployment.md`.

## Setup (local development)

```
git clone https://github.com/rafidhiyaulh/nginep.git
cd nginep
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .

gcloud auth application-default login
gcloud auth application-default set-quota-project <your-gcp-project-with-vertex-ai-enabled>

.venv/bin/uvicorn app.main:app --reload --port 8000
```

Then open http://localhost:8000.

To rebuild the data from scratch, see the scripts in order: `scripts/build_hotel_table.py`, `scripts/extract_aspects.py`, `scripts/aggregate_hotel_aspects.py`.

## API

- `GET /api/search?q=<query>&top_k=10`: search, returns the parsed query plus ranked results with an evidence based explanation for each.
- `GET /api/health`: liveness check, also reports how much data is loaded.

Rate limited to 20 requests per hour per visitor on the live deployment (see `docs/deployment.md`).
