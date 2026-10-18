"""Civitai downloader: model info, download URLs, file download, profile fetches."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.parse
import urllib.request
import urllib.error

from .config import AppConfig
from .downloader import download_file
from .utils import log

cfg: AppConfig | None = None
CIVITAI_TOKEN = None
_civitai_last_request: float = 0.0


def init(config: AppConfig):
    global cfg, CIVITAI_TOKEN
    cfg = config
    # Civitai API token for authenticated downloads (avoids 401 on signed-storage redirects)
    CIVITAI_TOKEN = None
    if os.path.exists(config.civitai_token_file):
        try:
            with open(config.civitai_token_file) as f:
                CIVITAI_TOKEN = f.read().strip()
        except Exception:
            pass


def civitai_pace():
    """Enforce a minimum interval between Civitai API requests to avoid rate limits."""
    global _civitai_last_request
    elapsed = time.time() - _civitai_last_request
    if elapsed < cfg.civitai_min_interval:
        time.sleep(cfg.civitai_min_interval - elapsed)


def run_civitai(args):
    try:
        result = subprocess.run(
            ["python3", cfg.civitai_script] + args,
            capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            log(f"Error running civitai {' '.join(args)}: {result.stderr[:200]}")
            return None
        return result.stdout.strip()
    except Exception as e:
        log(f"Exception running civitai: {e}")
        return None


def get_model_info(model_id):
    out = run_civitai(["model", str(model_id)])
    if out is None:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        log(f"Failed to parse JSON for model {model_id}")
        return None


def get_download_url(version_id):
    out = run_civitai(["download-url", str(version_id)])
    if out is None:
        return None
    # Append Civitai API token to authenticated download URL (avoids 401 on signed-storage redirects)
    if CIVITAI_TOKEN:
        separator = "&" if "?" in out else "?"
        out = f"{out}{separator}token={CIVITAI_TOKEN}"
    return out


def get_filename_from_api(model_id):
    info = get_model_info(model_id)
    if info is None:
        return None, None, None, None
    model_name = info.get("name", f"model_{model_id}")
    model_tags = info.get("tags", [])
    versions = info.get("modelVersions", [])
    if not versions:
        return model_name, None, None, model_tags
    version = versions[0]
    version_id = version.get("id")
    files = version.get("files", [])
    if not files:
        return model_name, version_id, None, model_tags
    main_file = None
    for f in files:
        if f.get("type") in ("Main", "Model"):
            main_file = f
            break
    if main_file is None:
        main_file = files[0]
    return model_name, version_id, main_file.get("name"), model_tags


def civitai_api_get(path, token=None, params=None):
    """Make a GET request to the Civitai API with optional bearer auth.

    Paces requests to stay under Civitai's rate limit and retries with
    exponential backoff on 429 (rate-limited) and 5xx responses.
    """
    url = f"{cfg.civitai_api_base}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    last_err = None
    for attempt in range(cfg.civitai_max_retries):
        civitai_pace()
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                _civitai_last_request = time.time()
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503, 504) and attempt < cfg.civitai_max_retries - 1:
                backoff = min(2 ** attempt, cfg.civitai_backoff_cap)
                log(f"  Civitai API {path} HTTP {e.code} — backing off {backoff}s (attempt {attempt + 1}/{cfg.civitai_max_retries})")
                time.sleep(backoff)
                continue
            log(f"  Civitai API error {path}: {e}")
            return None
        except Exception as e:
            last_err = e
            log(f"  Civitai API error {path}: {e}")
            return None
    log(f"  Civitai API {path} failed after {cfg.civitai_max_retries} attempts: {last_err}")
    return None


def fetch_civitai_favorites(token):
    """Fetch model IDs from the caller's Civitai favorites/bookmarks (auth required)."""
    if not token:
        log("  Civitai: no token — skipping favorites fetch")
        return []
    ids = []
    cursor = None
    while True:
        params = {"favorites": "true", "limit": 100}
        if cursor:
            params["cursor"] = cursor
        data = civitai_api_get("/models", token=token, params=params)
        if data is None:
            break
        items = data.get("items", [])
        for item in items:
            mid = item.get("id")
            if mid:
                ids.append(str(mid))
        cursor = data.get("metadata", {}).get("nextCursor")
        if not cursor or not items:
            break
    log(f"  Civitai favorites: {len(ids)} models found")
    return ids


def fetch_civitai_followed(token):
    """Fetch model IDs from creators the user follows on Civitai (auth required)."""
    if not token:
        log("  Civitai: no token — skipping followed creators fetch")
        return []
    # Get authenticated user profile to find followed creators
    me = civitai_api_get("/me", token=token)
    if me is None:
        log("  Civitai: could not fetch /me — cannot determine followed creators")
        return []
    followed = me.get("followed") or me.get("following") or []
    if not followed:
        log("  Civitai: /me returned no followed/following list — creator follow feed unavailable via this API version")
        return []
    ids = []
    for creator in followed:
        username = creator.get("username") if isinstance(creator, dict) else str(creator)
        if not username:
            continue
        cursor = None
        while True:
            params = {"username": username, "limit": 100}
            if cursor:
                params["cursor"] = cursor
            data = civitai_api_get("/models", token=token, params=params)
            if data is None:
                break
            for item in data.get("items", []):
                mid = item.get("id")
                if mid:
                    ids.append(str(mid))
            cursor = data.get("metadata", {}).get("nextCursor")
            if not cursor or not data.get("items"):
                break
    log(f"  Civitai followed creators: {len(ids)} models found across {len(followed)} creator(s)")
    return ids


# Civitai model types pulled from collections. Civitai labels embeddings as
# "TextualInversion". LoCon/Lycoris/DoRA are normalized to LORA downstream in
# batch.process_model(), so we keep the raw "LORA" here.
CIVITAI_COLLECTION_TYPES = {
    "CHECKPOINT",        # checkpoints
    "LORA",              # LoRA (and kin)
    "VAE",               # VAE
    "TEXTUALINVERSION",  # Embeddings (Civitai API label)
    "CONTROLNET",        # ControlNet
}


def fetch_civitai_collections(token):
    """Fetch model IDs from all collections the user owns/follows on Civitai (auth required).

    Uses GET /api/v1/collections?userId=me to list collections (cursor paginated),
    then GET /api/v1/collections/{id}?limit=100 for each collection's models
    (cursor paginated). Only generation-capable types are kept.
    """
    if not token:
        log("  Civitai: no token — skipping collections fetch")
        return []
    if cfg.civitai_skip_collections:
        log("  Civitai: collections fetch disabled (civitai_skip_collections) — skipping to avoid rate limits")
        return []
    # List collections owned/followed by the authenticated user
    collections = []
    cursor = None
    while True:
        params = {"userId": "me", "limit": 100}
        if cursor:
            params["cursor"] = cursor
        data = civitai_api_get("/collections", token=token, params=params)
        if data is None:
            break
        items = data.get("items", [])
        for item in items:
            cid = item.get("id")
            if cid:
                collections.append(str(cid))
        cursor = data.get("metadata", {}).get("nextCursor")
        if not cursor or not items:
            break
    log(f"  Civitai collections: {len(collections)} collection(s) found")
    ids = []
    for cid in collections:
        cursor = None
        while True:
            params = {"limit": 100}
            if cursor:
                params["cursor"] = cursor
            data = civitai_api_get(f"/collections/{cid}", token=token, params=params)
            if data is None:
                break
            models = data.get("models", [])
            for m in models:
                mtype = (m.get("type") or "").upper()
                if mtype not in CIVITAI_COLLECTION_TYPES:
                    continue
                mid = m.get("id")
                if mid:
                    ids.append(str(mid))
            cursor = data.get("metadata", {}).get("nextCursor")
            if not cursor or not models:
                break
    # Dedupe preserving order
    seen = set()
    uniq = []
    for mid in ids:
        if mid not in seen:
            uniq.append(mid)
            seen.add(mid)
    log(f"  Civitai collections: {len(uniq)} unique generation-capable model IDs across {len(collections)} collection(s)")
    return uniq
