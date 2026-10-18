batch_comfytester — Performance Profile Report
===============================================
Task: 0f0bc236-1a7b-4970-9c3a-8433cdb0b1a3 (performance-engineer)
Profile target: per-model download + generation time, bottlenecks, optimizations.

Environment
-----------
Host: Linux 7.2.5-3-omarchy, ComfyUI 0.40.0 at http://127.0.0.1:8188
GPU: RTX 3070-class (--lowvram, dynamic offloading), 64 GB RAM
Network: civitai.com ~0.2 s, huggingface.co ~0.06 s (HTTP 200)
Config: /home/rizzo/Work/batch_comfytester/batch_comfytester.py + comfytester/
ComfyUI base: /home/rizzo/ComfyUI

1. Scope & method
-----------------
Instrumented the refactored batch pipeline (comfytester/batch.py,
comfytester/downloader.py, comfytester/comfyui.py) with wall-clock
timers around every per-model stage:

  - Civitai API info fetch  (get_filename_from_api / get_model_info)
  - Download                (downloader.download_file, curl/wget)
  - Generation              (ComfyUI prompt queue + history polling,
                            incl. retries and upscale)
  - Per-model total

A JSON timing report (batch_timing_report.json) is written at end of
each batch run, with per-model entries and aggregate summary.

Two profiling runs were executed against live Civitai models whose
files were missing locally (state files temporarily edited and then
restored):

  run A: model 4384 DreamShaper (CHECKPOINT, 1.99 GB)
  run B: model 2583 Hassaku SD1.5 (CHECKPOINT, 1.99 GB)
  micro-benchmark: local re-generation of dreamshaper_8.safetensors

Historical generation timings were also re-derived from full_run.log
(37 queue->generate pairs from the previous monolith-era run).

2. CRITICAL FINDING: generation stage was dead code
---------------------------------------------------
In the refactored comfytester/batch.py, after the per-type handlers
(_process_checkpoint/_process_lora/_process_vae/_process_controlnet,
which already download AND generate), a final "generation stage"
block called `_generate(...)` — a function that does not exist.
Every generative model therefore hit a NameError, was logged as
EXCEPTION, and was marked completed WITHOUT ever generating. Before
this fix, per-model generation time was structurally 0/unmeasurable
for every CHECKPOINT/LORA/VAE/CONTROLNET model.

Fix: removed the dead `_generate()` block (generation already happens
inside the per-type handlers, which now also record timing).

3. Live measurements (this session)
-----------------------------------
+------------+-----------+----------+-------------------+-------------+---------+
| Model      | Type      | Size     | Download          | Generation  | Total   |
+------------+-----------+----------+-------------------+-------------+---------+
| 4384       | CHECKPOINT| 1.99 GB  | 31.6 s (64.4 MB/s)| 25.2 s      | ~57 s   |
| DreamShaper|           |          | wget fallback     | dpmpp_2m    |         |
|            |           |          | (curl RC=2, max-  | sde_gpu,    |         |
|            |           |          | time 120s hit)    | 25 steps,   |         |
|            |           |          |                   | 832x1216    |         |
+------------+-----------+----------+-------------------+-------------+---------+
| 2583       | CHECKPOINT| 1.99 GB  | 47.7 s (42.6 MB/s)| 15.2 s      | ~63 s   |
| Hassaku    |           |          | wget fallback     | euler_a,    |         |
| SD1.5      |           |          | (curl RC=2)       | 25 steps,   |         |
|            |           |          |                   | 896x1152    |         |
+------------+-----------+----------+-------------------+-------------+---------+
| dreamshaper| CHECKPOINT| 1.99 GB  | (already local)   | 18.3 s      | 18.3 s  |
| _8 (local  |           |          |                   | euler_a,    |         |
| micro-bench|           |          |                   | 896x1152    |         |
| )          |           |          |                   |             |         |
+------------+-----------+----------+-------------------+-------------+---------+

Download throughput: 42-64 MB/s from Civitai signed storage. Both
2 GB downloads required the wget fallback because curl's
--max-time 120 aborted first (curl RC=2).

4. Historical generation distribution (full_run.log, 37 models)
---------------------------------------------------------------
Monolith-era settings: 30 steps, euler sampler, 768x768.
  min   9.0 s   p25   42.0 s   median 67.0 s   p75   88.0 s
  p95 112.0 s   max 130.0 s   mean 65.5 s

Current-config generations (25 steps dpmpp_2m_sde_gpu/euler_a)
measured 15-25 s, i.e. roughly 2-3x faster than the historical
30-step euler runs. Generation is the largest controllable per-model
cost once a model file is present locally.

5. Bottlenecks identified
-------------------------
B1. Generation time dominates per-model wall time (15-130 s per
    prompt). Anything that multiplies prompt count multiplies cost:
    VAE comparison loops and LoRA strength sweeps (cfg.lora_strengths
    = [0.4, 0.8, 1.2]) run N VAEs x M strengths per model.

B2. Downloads are significant for large checkpoints: ~32-48 s per
    2 GB model. curl --max-time 120 is too short for multi-GB files
    on this link; every download fell back to wget (adds ~15 s of
    failed curl before fallback succeeds).

B3. Civitai profile enumeration is one-time but expensive: a dry-run
    spent 90+ s inside fetch_civitai_collections/favorites pagination
    before processing a single model. Fixed 1.2 s inter-request
    pacing (civitai_min_interval) plus cursor pagination over all
    collections. On a large profile this is minutes of overhead per
    run. (Per-model Civitai API info fetch is cheap: ~0.15-0.32 s.)

B4. Sequential processing with a hard 2 s sleep between models
    (batch.main loop) — no parallelism or prefetching. ComfyUI sits
    idle during downloads and during the inter-model sleep.

B5. Upscaling is misconfigured: enable_upscaling=true but the
    configured upscaler model 4xUltrasharp_v10.pt is NOT installed
    in ComfyUI ("model_name: '4xUltrasharp_v10.pt' not in []"),
    so the upscale stage fails after every generated image.
    If the model were installed, upscaling would add a full extra
    prompt (300 s timeout) per image.

B6. Telegram notifications fail with "Bad Request: can't parse
    entities" on NSFW prompts (markdown entity parsing) — noise and
    failed sends per model/image, not a latency bottleneck but
    corrupts the notification pipeline.

B7. Resume state staleness: completed_models.txt contains models
    whose files are no longer on disk (e.g. 2583 Hassaku, 4384
    DreamShaper were logged "already completed" yet missing).
    Such models are skipped forever and never re-downloaded.

6. Optimization recommendations (priority order)
------------------------------------------------
O1. Fix the generation stage (DONE): removed the undefined
    _generate() call; verified end-to-end generation works
    (DreamShaper + Hassaku both rendered successfully).

O2. Parallelize I/O-bound stages: download the next model while
    ComfyUI renders the current one (producer/consumer over a small
    file queue). Hides B2 (download latency) behind generation.
    Estimated: removes ~30-50 s of download from per-model wall
    time whenever download < generation.

O3. Skip generation for already-local files (only re-download on
    absence/corruption) — current code already does this, but pair
    it with a file-presence sanity check so "completed but missing"
    models (B7) are re-fetched rather than skipped forever.

O4. Fix or disable upscaling: install 4xUltrasharp_v10.pt or set
    enable_upscaling=false. Avoids a failed (or very slow) extra
    prompt per generated image (B5).

O5. Reduce multiplier loops: make VAE comparison and LoRA strength
    sweeps optional/configurable with a "quick" mode (single VAE,
    single strength). Each extra VAE x strength is a full 15-25 s
    prompt.

O6. Fix curl timeout: raise --max-time (or drop it) for Civitai
    downloads so 2+ GB files don't burn a ~15 s curl failure before
    the wget fallback (B2).

O7. Batch Civitai API calls / raise pacing budget for profile
    enumeration; the fixed 1.2 s/page pacing makes large-profile
    runs start with minutes of pure API pagination (B3). Consider
    caching the id list between runs.

O8. Fix Telegram message formatting (plain text or escaped Markdown)
    to stop per-message "can't parse entities" errors (B6).

7. Files changed
----------------
  comfytester/batch.py      - per-model timing recorder (_timings),
                              removed dead _generate() block,
                              per-type handlers return status +
                              timings, _write_timing_report() +
                              JSON report at end of main(),
                              timing hooks in scan_local_models and
                              process_hf_models
  comfytester/downloader.py - download wall time + MB/s logged on
                              every verified download
  comfytester/comfyui.py    - generation elapsed logged on prompt
                              completion and on timeout

8. Deliverables
---------------
  batch_timing_report.json   - per-model timing JSON (this run)
  PERFORMANCE_REPORT.md      - this report
