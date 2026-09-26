# Deployment (Cloud Run)

Live: https://tanyainap-610631276830.asia-southeast2.run.app

## Why Cloud Run

Same reasoning as the Vertex AI choice itself (`docs/vertex_ai_setup.md`):
staying inside the GCP project already set up means the deployed service can
call Vertex AI using its **attached service account's identity directly** --
no API key, no service-account JSON file, nothing secret to protect in this
deployment at all. Scales to zero when idle (near-$0 when nobody's using it)
and the free tier (2M requests/month) comfortably covers a personal-project
demo's traffic.

## What's in place

- **Service account**: `tanyainap-cloudrun@nginep-tanyainap.iam.gserviceaccount.com`,
  granted only `roles/aiplatform.user` (not broader project access) --
  least-privilege, scoped to exactly what the app calls.
- **Rate limiting** (`app/rate_limit.py`): 20 requests/hour per IP, in-memory.
  Only correct with a single instance, which is why:
- **`--max-instances=1`**: also a hard cap on worst-case concurrent Vertex AI
  cost, independent of the rate limiter. Fine at this project's traffic
  scale; a real multi-instance deployment would need shared rate-limit state
  (Redis/Firestore) instead.
- **Budget alert**: $20 threshold on the billing account (50%/100% of budget),
  email notification -- a coarse safety net (scoped to the whole billing
  account, which also holds two unrelated older projects, not just this one)
  on top of the rate limit + instance cap, not instead of them.
- **Lean image**: `requirements-app.txt` (not the full `requirements.txt`)
  installs only what the running web app actually imports -- pandas,
  rapidfuzz, google-genai, fastapi, uvicorn. scikit-learn/lightgbm/matplotlib
  (offline-only, used by the `scripts/` pipeline and eval, never imported by
  `app/`) stay out of the deployed container.
- **Static data baked into the image**: `data/processed/hotels.csv` and
  `hotel_aspect_scores.json` are copied in at build time (see `Dockerfile`).
  The dataset doesn't change live, so there's no runtime database -- a
  redeploy is how the live data updates, same as any other code change.

## Redeploying after a data/code change

```
gcloud run deploy tanyainap --source . --project=nginep-tanyainap \
  --region=asia-southeast2 \
  --service-account=tanyainap-cloudrun@nginep-tanyainap.iam.gserviceaccount.com \
  --allow-unauthenticated --max-instances=1 --min-instances=0 \
  --memory=512Mi --cpu=1
```

## Known limitation

`--allow-unauthenticated` makes this genuinely public -- anyone with the URL
can use it (that's the point, for friend/community testing), which also
means it's discoverable/shareable beyond who it's directly sent to. The rate
limit + instance cap + budget alert above are the mitigations, not a claim
that it's invite-only.
