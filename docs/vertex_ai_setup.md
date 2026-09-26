# Vertex AI setup (for the aspect extractor)

Uses Vertex AI, not a plain Gemini API key, because Google excluded Gemini
Developer API / AI Studio usage from free trial credit for GCP accounts
created after 2026-03-02 — Vertex AI still accepts it. See the decision
discussion in chat / commit history for the full reasoning.

## One-time setup (already done for this project)

- GCP project `nginep-tanyainap`, under the `rafidhiyaulh@gmail.com` account,
  linked to the billing account holding the free trial credit.
- Vertex AI API (`aiplatform.googleapis.com`) enabled on that project.
- Auth is via Application Default Credentials (ADC), **not an API key** —
  nothing secret lives in this repo or in `.env` for this part:
  ```
  gcloud auth application-default login          # pick rafidhiyaulh@gmail.com
  gcloud auth application-default set-quota-project nginep-tanyainap
  ```
- Model: `gemini-3.1-flash-lite` (current cheap/fast tier; the older/cheaper
  Gemini 2.5 Flash-Lite retires 2026-10-16, so not used here), region
  `global`, via the unified `google-genai` SDK with `vertexai=True`.

## Reproducing this on another machine

1. `gcloud auth application-default login` with an account that has access
   to a GCP project with the Vertex AI API enabled and billing linked.
2. Either reuse `nginep-tanyainap` (if you have access) or change
   `GCP_PROJECT` in `scripts/extract_aspects.py` / wherever
   `nginep.extraction.make_client()` is called, to your own project id.
3. `gcloud auth application-default set-quota-project <your-project-id>`.

## Measured cost

From real `usage_metadata` on live calls: roughly 350-500 input tokens and
100-250 output tokens per review (fixed prompt/schema overhead dominates
for short reviews). At Gemini 3.1 Flash-Lite's $0.25 / $1.50 per million
input/output tokens, extracting all 5,798 Bali reviews costs on the order
of **$2-5 total** — trial credit covers this many times over.
