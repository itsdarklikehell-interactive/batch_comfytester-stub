Security Audit — API key handling in batch_comfytester
=======================================================
Task: bb1dd20f-af0f-4f1d-a271-6bfcd03fa4c8
Workspace: /home/rizzo/Work/batch_comfytester
Profile: security-audit

Scope
-----
Review how CIVITAI_TOKEN_FILE, HF_TOKEN, KILOCODE_API_KEY are loaded and used;
ensure no secrets leak to logs or Telegram.

Files inspected
---------------
batch_comfytester.py, batch_comfytester.env, batch_comfytester.py.bak,
comfytester/{config,civitai,telegram,utils,batch,downloader,comfyui,state}.py,
full_run.log, comfytester.log, verification_report.txt

1. CIVITAI_TOKEN_FILE / Civitai API token
-----------------------------------------
- config.civitai_token_file defaults to /home/rizzo/.civitai/comfy-api-key and is
  read at civitai.py:init() into module-global CIVITAI_TOKEN.
- Token file permissions: 600 (owner-only) — GOOD. Token is 32 bytes.
- The Civitai token is appended to download URLs as `token={CIVITAI_TOKEN}`
  (civitai.get_download_url). The URL itself is never logged (only the filename),
  and downloader.py does not log URLs — GOOD.
- GAP: CIVITAI_TOKEN_FILE is NOT in config.ENV_OVERRIDES, so the CIVITAI_TOKEN_FILE
  environment variable has no effect. The token file path cannot be overridden via
  env. FIXED: added "CIVITAI_TOKEN_FILE": ("civitai_token_file", str) to ENV_OVERRIDES.
- The Civitai token does NOT appear in any log file (searched; 0 matches) — GOOD.

2. HF_TOKEN
-----------
- Loaded via os.getenv("HF_TOKEN", "") in batch.py (3 places) and passed to
  huggingface_hub (download_file_hf, fetch_hf_liked_datasets, snapshot_download).
- batch.py logs only "HF_TOKEN set"/"no HF_TOKEN" (presence, not value) — GOOD.
- batch_comfytester.env contains a real (partially masked in-file) HF token.
  verification_report.txt confirms "real hf_ token, 37 chars" loaded from the env.
- HF_TOKEN is not redacted by utils.redact_token (was Telegram-only). FIXED:
  redact_token now also redacts HF_TOKEN (from env/config) and the Civitai token.

3. KILOCODE_API_KEY
-------------------
- Zero matches anywhere in the codebase. Not configured, not loaded, not used.
- The task body lists it, but it is absent from batch_comfytester. No action
  required other than noting it; if it becomes needed, add it to ENV_OVERRIDES
  and cover it in redact_token.

4. TELEGRAM_BOT_TOKEN
---------------------
- Loaded from config/env. Used in telegram.py sendPhoto/sendMessage via curl URLs
  `https://api.telegram.org/bot{token}/...`.
- Refactored telegram.py redacts the token from all error/description output via
  redact_token() and never logs the curl command or URL — GOOD (regression fixed
  vs. old monolith).
- send_telegram_message logs text[:60]; send_telegram_photo does not log captions.
  Risk: if a secret were ever passed in message text it would leak to logs AND
  Telegram. Current callers pass only model names/statuses — no secrets.

5. Secret storage — batch_comfytester.env (CRITICAL)
----------------------------------------------------
- Mode 644 (world-readable). Contains TELEGRAM_BOT_TOKEN and HF_TOKEN.
- FIXED: chmod 600 applied so only the owner can read it.
- No .gitignore exists — if VCS is added, *.env and *.log would be committed with
  secrets. Recommended: add .gitignore excluding batch_comfytester.env, *.log,
  __pycache__/, .worktrees/.

6. Historical secret leak in full_run.log (CRITICAL)
----------------------------------------------------
- full_run.log:310 (old monolith, 2026-10-17 00:55:04) logged the FULL curl command
  including the complete Telegram bot token
  `8998492813:AAE4_TPipdAfGiHPiNTgax8sqGlqZVvOb2A` plus chat_id, caption and
  photo path — a live secret written to a log file still on disk.
- FIXED: redacted the token to [REDACTED]:[REDACTED] in full_run.log.
- The Telegram bot token MUST be considered compromised and rotated (revoke and
  re-issue via @BotFather), since it was recorded in plaintext in full_run.log.
- Refactored telegram.py no longer logs the command, so this regression is fixed
  in current code; the historical log was the only carrier.

7. Token redaction coverage (MEDIUM)
------------------------------------
- utils.redact_token() previously only redacted TELEGRAM_BOT_TOKEN. HF_TOKEN and
  the Civitai token could leak via exception text (e.g. downloader.py:151 logs
  raw exception e from huggingface_hub).
- FIXED: redact_token now gathers TELEGRAM_BOT_TOKEN, HF_TOKEN (env + config) and
  the Civitai token (read from civitai_token_file) and redacts all of them,
  longest-first to avoid partial replacement.

Summary of changes made
-----------------------
1. chmod 600 batch_comfytester.env (protect secrets on disk)
2. comfytester/config.py: added CIVITAI_TOKEN_FILE to ENV_OVERRIDES
3. comfytester/utils.py: broadened redact_token to cover Telegram/HF/Civitai tokens
4. full_run.log: redacted the leaked full Telegram bot token

Recommendations
---------------
1. Rotate the Telegram bot token immediately (it was logged in full in full_run.log).
   Consider also rotating the HF token given it is stored in plaintext in .env.
2. Move secrets out of batch_comfytester.env into a 600-protected secrets file or
   a secrets manager; keep the env file out of any VCS (add .gitignore).
3. Treat logs as potentially sensitive: retain log rotation and avoid writing URLs
   that embed tokens (already the case; keep it).
4. Add CIVITAI_TOKEN_FILE, HF_TOKEN, and any future KILOCODE_API_KEY to ENV_OVERRIDES
   consistently so env overrides are honored.
5. Re-run this audit after token rotation to confirm no plaintext secrets remain
   in logs or committed files.