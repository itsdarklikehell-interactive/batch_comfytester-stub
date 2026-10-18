"""Generic file downloaders: Civitai signed-storage downloads and HuggingFace Hub."""

from __future__ import annotations

import hashlib
import os
import subprocess
import time

from .config import AppConfig
from .utils import log

cfg: AppConfig | None = None


def init(config: AppConfig):
    global cfg
    cfg = config


def download_file(url, dest_path, expected_size=None, expected_sha256=None, max_retries=3):
    """Download a file with retries, atomic write (.part -> rename), and optional size/SHA256 verification.

    Uses curl -L as primary downloader (follows Civitai's 302 redirects to signed storage),
    wget as fallback. Never leaves a truncated final file if interrupted.
    """
    dest_dir = os.path.dirname(dest_path)
    part_path = dest_path + ".part"
    os.makedirs(dest_dir, exist_ok=True)
    dl_start = time.time()

    for attempt in range(max_retries):
        try:
            # Primary: curl -L follows redirects to Civitai's signed-storage host reliably
            result = subprocess.run(
                ["curl", "-L", "--max-time 120", "--retry", "3", "-s", "-o", part_path, url],
                capture_output=True, text=True, timeout=300)

            # Fallback: wget if curl fails
            if result.returncode != 0:
                log(f"  Download attempt {attempt + 1}/{max_retries} failed (curl RC={result.returncode})")
                result2 = subprocess.run(
                    ["wget", "--timeout=60", "--tries=3", "-q", "-O", part_path, url],
                    capture_output=True, text=True, timeout=300)
                if result2.returncode != 0:
                    log(f"  wget fallback also failed (RC={result2.returncode})")
                    if os.path.exists(part_path):
                        os.remove(part_path)
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return False
                log("  wget fallback succeeded")

            # Verify downloaded file exists and is non-empty
            if not os.path.exists(part_path) or os.path.getsize(part_path) == 0:
                log(f"  Downloaded file is missing or empty")
                if os.path.exists(part_path):
                    os.remove(part_path)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return False

            # Verify expected size if provided
            if expected_size is not None:
                actual_size = os.path.getsize(part_path)
                if actual_size != expected_size:
                    log(f"  Size mismatch: expected {expected_size}, got {actual_size}")
                    os.remove(part_path)
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return False

            # Verify SHA256 if provided
            if expected_sha256 is not None:
                sha256 = hashlib.sha256()
                with open(part_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        sha256.update(chunk)
                if sha256.hexdigest().lower() != expected_sha256.lower():
                    log(f"  SHA256 mismatch: expected {expected_sha256[:16]}..., got {sha256.hexdigest()[:16]}...")
                    os.remove(part_path)
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return False

            # Atomic rename to final destination
            os.replace(part_path, dest_path)
            elapsed = time.time() - dl_start
            size_bytes = os.path.getsize(dest_path)
            mbps = (size_bytes / (1024 ** 2)) / elapsed if elapsed > 0 else 0.0
            log(f"  Download verified OK ({size_bytes / (1024**3):.2f} GB in {elapsed:.1f}s, {mbps:.1f} MB/s)")
            return True

        except Exception as e:
            log(f"  Download exception: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            if os.path.exists(part_path):
                os.remove(part_path)
            return False

    return False


def download_file_hf(hf_repo, local_path, token=None):
    """Download a file from HuggingFace using huggingface_hub (built-in retries, token support).

    local_path is repo-relative and may include subdirectories (e.g.
    "diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors").
    The hub request uses local_path as-is and falls back to its basename for repos
    that only expose the file at root. The file lands at
    COMFYUI_BASE/models/<local_path>, preserving the subdirectory layout.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        log("  huggingface_hub not available — falling back to wget")
        filename = os.path.basename(local_path)
        dest_file = os.path.join(cfg.comfyui_base, "models", local_path)
        url = f"https://huggingface.co/{hf_repo}/resolve/main/{filename}"
        return download_file(url, dest_file)

    dest_file = os.path.join(cfg.comfyui_base, "models", local_path)
    dest_dir = os.path.dirname(dest_file)
    candidates = [local_path, os.path.basename(local_path)]
    last_error = None

    for filename in candidates:
        try:
            os.makedirs(dest_dir, exist_ok=True)
            downloaded = hf_hub_download(
                repo_id=hf_repo,
                filename=filename,
                token=token,
                local_dir=dest_dir,
                local_dir_use_symlinks=False,
            )
            # hf_hub_download caches in dest_dir; ensure file is at the right path
            cached_path = os.path.join(dest_dir, filename)
            if os.path.exists(cached_path) and cached_path != dest_file:
                os.replace(cached_path, dest_file)
            if os.path.exists(dest_file) and os.path.getsize(dest_file) > 0:
                log(f"  HF download OK ({os.path.getsize(dest_file) / (1024**3):.2f} GB)")
                return True
            log("  HF download produced empty file")
            return False
        except Exception as e:
            last_error = e
            log(f"  HF download attempt for '{filename}' failed: {e}")

    log(f"  HF download failed for {hf_repo} ({local_path}): {last_error}")
    return False


def download_hf_dataset(repo_id, local_dir_name=None, token=None):
    """Download a whole HuggingFace dataset (repo_type='dataset') via snapshot_download.

    Files land in cfg.hf_datasets_dir/<local_dir_name> (default: repo id with '/' -> '_').
    snapshot_download resumes gracefully if files already exist. Returns True if the
    destination directory ends up non-empty.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log("  huggingface_hub not available — cannot download HF datasets")
        return False
    local_dir_name = local_dir_name or repo_id.rsplit("/", 1)[-1]
    dest_dir = os.path.join(cfg.hf_datasets_dir, local_dir_name)
    os.makedirs(dest_dir, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            local_dir=dest_dir,
        )
        file_count = sum(len(files) for _, _, files in os.walk(dest_dir))
        if file_count:
            log(f"  HF dataset OK ({file_count} files): {repo_id} -> {dest_dir}")
            return True
        log(f"  HF dataset download produced no files: {repo_id}")
        return False
    except Exception as e:
        log(f"  HF dataset download failed: {repo_id}: {e}")
        return False
