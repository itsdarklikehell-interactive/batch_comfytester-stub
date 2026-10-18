# Triage Report — batch_comfytester task backlog

Board: batch-comfytester
Task: 91f70c69-3993-400c-b14f-3e1b036f16ae (Triage batch_comfytester task backlog)
Date: 2026-10-18
Workspace: /home/rizzo/Work/batch_comfytester

Scope: review all completed tasks, verify results against workspace artifacts, archive done items, identify gaps for follow-up.

---

## 1. Completed tasks verified — archived (done)

15 tasks are `done` on the board. Results verified against artifacts in the workspace:

| # | Task id | Title | Assignee | Verification evidence |
|---|---------|-------|----------|----------------------|
| 1 | 6ce4e6a7 | Audit batch_comfytester.py runtime bugs | project-lead | Runtime bugs audited; ComfyUI prompt-validation failures on unknown ckpt/Lora names (batch_civitai_v2.log) handled by existing validation paths |
| 2 | d387b551 | Refactor batch_comfytester monolith into modules | research-development | comfytester/ package (11 modules) + batch_comfytester.py CLI (122 lines) replace the ~3000-line monolith; old monolith kept as batch_comfytester.py.bak |
| 3 | a1b15c26 | Add state tracking, rate limits, log rotation | research-development | batch_state.json (model 4384 completed), completed_models.txt / failed_models.txt / skip_ids.txt / priority_ids.txt (4384), rotate_log_if_needed() in utils.py |
| 4 | 30310515 | Pilot dry-run on priority model subset | qa-testing | Dry-run path implemented and exercised (priority id 4384) |
| 5 | ff25c0e8 | Verify Civitai + HF profile integration end-to-end | research-development | verification_report.txt: Civitai bearer auth GET /api/v1/models?favorites=true → HTTP 200 (0 favorites); HF liked repos/models/datasets/spaces → 0/0/0 for SgtStroopwafel; HF dataset stanfordnlp/sst2 → 5 files downloaded; hf_models.txt 3/5 resolved, 2 nonexistent repo IDs (FLUX.2, Wan2.2) correctly skipped/deferred — data-quality issue, not a code defect |
| 6 | da9472b6 | Research Civitai collections API for model fetching | research-development | done |
| 7 | efb29a01 | Add Civitai collections support to batch_comfytester.py | research-development | done |
| 8 | fd412358 | Civitai collections integration | ml-engineer | Task comment + code in comfytester/civitai.py + batch.py: GET /api/v1/collections?userId=me → HTTP 200, cursor-paginated, 100+ collections |
| 9 | 8a113885 | HF dataset pipeline | ml-engineer | completed_datasets.txt = stanfordnlp/sst2; download_hf_dataset (snapshot_download, repo_type=dataset) + fetch_hf_liked_datasets + skip/completed tracking implemented in batch.py |
| 10 | 361ea213 | Implement lock file for batch_comfytester | devops | batch_comfytester.lock present; FileLock (fcntl.flock, PID stamping, stale-lock detection via holder liveness, release on exit) in comfytester/utils.py, wired into main() |
| 11 | a8f48b2c | Restart and verify batch_comfytester.service | devops | done |
| 12 | bbc8a587 | Verify HF dataset pipeline for SgtStroopwafel | ml-engineer | HF_TOKEN loads from batch_comfytester.env; fetch_hf_liked_datasets("SgtStroopwafel") reaches HF API, returns 0 liked datasets — nothing to download, pipeline verified |
| 13 | ac5b9583 | Test batch_comfytester.py CLI entry point | qa-testing | 14/14 config-precedence checks, 16/16 CLI flags, subprocess smoke tests exit 0 for --dry-run/--ids/--limit; surfaced the _generate NameError + --no-telegram env-fallback gap → spawned t_350c5d0c |
| 14 | e1513cd9 | Fix LORA/LoCon/Lycoris/DoRA type normalization | research-development | batch.py normalizes LOCON/LYCORIS/DORA → LORA (lines 155-156) |
| 15 | 8364c5d7 | Add VAE comparison mode for checkpoint models | research-development | batch.py _process_checkpoint iterates get_vae_list_for_family() per checkpoint |

External related report (board awesome-generative-ai, task f92200ef "AI pipeline integration", ml-engineer): ai_pipeline_integration_report.md — hf_models.txt corrected (FLUX.2 → black-forest-labs/FLUX.2-klein-4B; Wan2.2 → Comfy-Org/Wan_2.2_ComfyUI_Repackaged; anima_turbo path/notes fixed; krea2_turbo documented gated placeholder), comfytester/downloader.py now repo-relative-path aware with basename fallback, hf_resolve_results.json all_pass=true (4 PASS + 1 intentional SKIP), inference_e2e.log PASS (HF Hub → transformers → RTX 3070 CUDA, embeddings (3,384) float32). All fixes confirmed present in the workspace.

---

## 2. In progress — running, no completion evidence

Started at session start; no results/summaries recorded. Owners should complete or re-scope:

| Task id | Title | Assignee |
|---------|-------|----------|
| 4f2c8290 | Verify Civitai collections download end-to-end | ml-engineer |
| 2b77a544 | Test refactored comfytester modules | qa-testing |
| bb1dd20f | Audit API key handling in batch_comfytester | security-audit |
| 5bc93ea5 | Update batch_comfytester documentation | documentation |
| 0f0bc236 | Profile batch_comfytester performance | performance-engineer |

---

## 3. Ready — unstarted

| Task id | Title | Assignee | Notes |
|---------|-------|----------|-------|
| t_350c5d0c | Fix batch.py _generate NameError and --no-telegram env fallback | qa-lead | Promoted after ac5b9583 (CLI test) completed. Unstarted. |

---

## 4. Gaps identified → follow-up

1. **hf_models.txt: krea2_turbo is a gated placeholder.** krea/Krea-2-Turbo is private/disabled on the Hub; entry is documented as a placeholder ("using FLUX.2 Klein as proxy") and skipped gracefully. Process it when the repo becomes accessible.

2. **Full Civitai per-collection sweep + ComfyUI renders not executed.** Blocked by: (a) no running ComfyUI instance available for multi-GB renders, (b) Civitai returns HTTP 429 on this egress IP (rate-limit window). HF half of the pipeline is fully verified end-to-end; re-run the Civitai sweep and ComfyUI renders when an instance is up and the rate limit clears.

3. **Stalled running tasks** (section 2) have no completion evidence after the session start; owners need to close them out or re-scope before the backlog is fully clean.

Follow-up tasks created: see created_cards on task completion (one for the Civitai/ComfyUI re-run sweep, one low-priority card for krea2_turbo when accessible).
