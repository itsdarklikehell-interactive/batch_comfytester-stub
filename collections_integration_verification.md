# Civitai Collections Integration — Verification Report

Task: fd412358-b45c-4b79-bf25-4f7c77961070 (ml-engineer)
Board: batch-comfytester
Code under test: /home/rizzo/Work/batch_comfytester/comfytester/
  - comfytester/civitai.py — fetch_civitai_collections(), civitai_api_get(), get_download_url()
  - comfytester/batch.py — collect_model_ids(), process_model(), _dest_dir_for(), _download_if_needed()
  - comfytester/config.py — civitai_token_file, civitai_api_base, pacing/retries
  - comfytester/downloader.py — download_file()
Auth: bearer token from /home/rizzo/.civitai/comfy-api-key (32 bytes, loaded by civitai.init())
Verification date: 2026-10-17 (CEST)

============================================================
1. LIVE API RESULTS (executed against civitai.com)
============================================================

1.1 GET /api/v1/collections?userId=me — PASS
  - HTTP 200 with Authorization: Bearer <token> (token accepted, no 401).
  - Cursor-paginated: response includes metadata.nextCursor and metadata.nextPage
    (observed nextCursor values 18364142 and 18396287 across probes).
  - limit up to 100 per page; collection items include id + name (observed:
    "U1H", "Sci-Fi", "Upcoming models", "characters", "Krea 2", …).
  - User owns/follows 100+ collections (first page saturated at 100 with a
    nextCursor, so the total is >100).
  => The "list ALL user collections" half of the integration works end-to-end
     with bearer auth, exactly as fetch_civitai_collections() assumes.

1.2 GET /api/v1/collections/{id}?limit=100 (per-collection models) — NOT VERIFIED LIVE
  - Blocked: Civitai returned HTTP 429 "Rate limit exceeded" on every request
    from this egress IP from ~23:20 CEST onward, including the very first request
    of each attempt, with exponential backoff up to 60s exhausted (6 retries).
    Brief 200 windows were observed only before 23:20.
  - Could not sweep all collections or confirm the live `models` array shape.

1.3 Signed download URL spot-checks (civitai.py download-url + ranged GET) — NOT VERIFIED LIVE
  - Same 429 blocker; not reached.

============================================================
2. STATIC CODE REVIEW OF THE INTEGRATION (deterministic, executed by reading
   the checked-in code — no network required)
============================================================

2.1 fetch_civitai_collections(token) — comfytester/civitai.py:216
  Step A — list collections (cursor paginated):
    GET /collections?userId=me&limit=100 [+cursor], loop on metadata.nextCursor.
  Step B — per collection, list models (cursor paginated):
    GET /collections/{cid}?limit=100 [+cursor], reads data.get("models", []).
  Step C — type filter (CIVITAI_COLLECTION_TYPES, line 207):
    CHECKPOINT   -> models/checkpoints  (i.e. "models")
    LORA         -> models/loras        (LoRAs; LoCon/Lycoris/DoRA normalized
                                            to LORA downstream in batch.py:152)
    TEXTUALINVERSION -> models/embeddings ("embeddings"; Civitai API label for
                                            TextualInversion embeddings)
    VAE          -> models/vae
    CONTROLNET   -> models/controlnet
  All other types (Other, etc.) are skipped — matches the task scope of
  "LoRAs, embeddings, and models".
  Step D — dedupe preserving order; logs "N unique generation-capable model IDs
  across M collection(s)".
  => Code covers LoRAs (LORA), embeddings (TEXTUALINVERSION), and models
     (CHECKPOINT) from every user collection, cursor-paginated on both endpoints.

  Civitai developer docs (developer.civitai.com/site/reference/collections)
  confirm the model type enum includes Checkpoint, TextualInversion, LORA,
  Controlnet — consistent with the filter set above.

2.2 Per-type download routing — comfytester/batch.py process_model()
  mtype CHECKPOINT  -> _process_checkpoint  -> models/checkpoints/  (with
                                                     per-family GB guard)
  mtype LORA        -> _process_lora        -> models/loras/
  mtype VAE         -> _process_vae         -> models/vae/
  mtype CONTROLNET  -> _process_controlnet  -> models/controlnet/
  mtype TEXTUALINVERSION -> _process_download_only -> models/embeddings/
                                                     (download-only, no gen)
  _dest_dir_for() (batch.py:197) maps TEXTUALINVERSION/AESTHETICGRADIENT to
  models/embeddings, LORA to models/loras, CHECKPOINT to models/checkpoints.
  => LoRAs, embeddings, and checkpoints are each routed to the correct
     ComfyUI model directory.

2.3 Download mechanics
  - get_download_url() (civitai.py:68): calls civitai.py download-url
    <version_id> -> https://civitai.com/api/download/models/{id}[?selector params]
    then appends ?token=/&token= from CIVITAI_TOKEN -> avoids 401 on
    Civitai signed-storage redirects (per skill auth rules).
  - _download_if_needed() (batch.py:231): skips if dest exists AND size > 0;
    otherwise download_file(); marks failed on URL/download failure; never
    kills the batch on a single bad model (process_model wrapped in try/except,
    batch.py:93-194).
  - download_file() (downloader.py:21): curl -L follows 302 to signed storage,
    atomic .part -> rename, retries w/ backoff, non-empty check, optional
    size/SHA256 verification.
  => Download integration is type-aware, authenticated, resumable, and
     failure-isolated per model.

============================================================
3. BLOCKER — LIVE SWEEP INCOMPLETE
============================================================

Civitai API rate limiting: every request to api.civitai.com from this egress IP
has returned HTTP 429 {"error":"Rate limit exceeded"} continuously since about
23:20 CEST. The list endpoint was observed returning HTTP 200 earlier the same
evening (multiple probes), proving auth + pagination work; the per-collection
models fetch and full >100-collection sweep could not be completed live.

Civitai's own docs describe /collections endpoints as "public, edge-cached,
rate-limited" — consistent with the observed behavior.

Re-run command once the rate-limit window clears (from /home/rizzo/Work/batch_comfytester):
  python3 /home/rizzo/.hermes/profiles/ml-engineer/cache/scratch/verify_collections.py
  (script also at that path; writes JSON to
   ~/.hermes/profiles/ml-engineer/cache/scratch/collections_verify.json and a
   progress log to .../verify_collections.log)

Expected full verification when unblocked:
  - paginate all collections (>100; cursor loop)
  - per collection, paginate models (cursor loop), tally by type
  - confirm CHECKPOINT/LORA/TEXTUALINVERSION appear across collections
  - signed download-URL + ranged GET spot check for one sample of each type

============================================================
4. CONCLUSION
============================================================

The Civitai collections integration as implemented in comfytester/civitai.py +
comfytester/batch.py is designed to download ALL of the user's collections and
covers LoRAs (LORA), embeddings (TEXTUALINVERSION), and models/checkpoints
(CHECKPOINT), with cursor pagination on both API endpoints, bearer-auth
download URLs, per-type ComfyUI directory routing (models/loras/,
models/embeddings/, models/checkpoints/), and per-model failure isolation.

Live-confirmed: bearer auth, /collections?userId=me listing with cursor
pagination (>100 collections).
Not yet live-confirmed (429 rate limit on this egress IP): per-collection
models fetch across all collections, and signed download spot-checks. Re-run the
verification script when the rate limit clears; no code change is indicated by
the static review.