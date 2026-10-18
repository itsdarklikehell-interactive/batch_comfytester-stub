"""ComfyUI client: prompt queue, health checks, restarts, history polling, upscaling."""

from __future__ import annotations

import json
import os
import subprocess
import time

from .config import AppConfig
from .civitai import get_download_url, run_civitai
from .downloader import download_file_hf
from .telegram import send_telegram_photo
from .utils import log

cfg: AppConfig | None = None


def init(config: AppConfig):
    global cfg
    cfg = config


def queue_prompt(workflow):
    data = json.dumps({"prompt": workflow})
    resp = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{cfg.comfyui_url}/prompt",
         "-H", "Content-Type: application/json",
         "-d", data],
        capture_output=True, text=True, timeout=15)
    try:
        resp_json = json.loads(resp.stdout)
        if "prompt_id" in resp_json:
            return resp_json["prompt_id"]
        else:
            log(f"  ComfyUI error: {resp.stdout[:300]}")
            return None
    except json.JSONDecodeError:
        log(f"  Failed to parse prompt response: {resp.stdout[:200]}")
        return None


def check_comfyui_health():
    """Return True if ComfyUI is responding, False otherwise."""
    try:
        resp = subprocess.run(
            ["curl", "-s", f"{cfg.comfyui_url}/system_stats"],
            capture_output=True, text=True, timeout=10)
        return resp.returncode == 0 and bool(resp.stdout.strip())
    except Exception:
        return False


def restart_comfyui():
    """Attempt to restart ComfyUI server."""
    log("  ComfyUI appears down — attempting restart...")
    try:
        subprocess.run(["pkill", "-f", "ComfyUI/main.py"], capture_output=True, timeout=15)
    except Exception:
        pass
    time.sleep(3)
    clean_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    clean_env["HOME"] = os.environ.get("HOME", "/home/rizzo")
    clean_env["USER"] = os.environ.get("USER", "rizzo")
    log_file = open(cfg.restart_log_file, "a")
    subprocess.Popen(
        [cfg.comfyui_venv_python, cfg.comfyui_main_py,
         "--enable-manager", "--listen", "0.0.0.0", "--port", "8188", "--lowvram"],
        stdout=log_file, stderr=subprocess.STDOUT,
        env=clean_env, cwd=cfg.comfyui_base)
    log_file.close()
    for i in range(20):
        time.sleep(3)
        if check_comfyui_health():
            log("  ComfyUI restarted successfully")
            return True
    log("  ComfyUI restart failed after 60s")
    return False


def wait_for_completion(prompt_id, timeout=300):
    """Returns (True, None) on success, (False, corrupt_path, False) on corrupt file,
    (False, None, True) on execution error (retryable via redownload + retry),
    (False, None, False) on crash/timeout."""
    import re

    start = time.time()
    consecutive_failures = 0
    while time.time() - start < timeout:
        try:
            if not check_comfyui_health():
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    if restart_comfyui():
                        consecutive_failures = 0
                    else:
                        log("  ComfyUI unavailable, aborting this model")
                        return False, None, False
                time.sleep(5)
                continue
            consecutive_failures = 0
            resp = subprocess.run(
                ["curl", "-s", f"{cfg.comfyui_url}/history/{prompt_id}"],
                capture_output=True, text=True, timeout=10)
            hist = json.loads(resp.stdout)
            if prompt_id in hist:
                status = hist[prompt_id].get("status", {})
                if status.get("status_str") == "success":
                    elapsed = time.time() - start
                    log(f"  Prompt {prompt_id} completed in {elapsed:.1f}s")
                    return True, None, False
                if status.get("status_str") == "error":
                    msgs = status.get("messages", [])
                    log(f"  ComfyUI execution error: {msgs}")
                    # All execution errors are retryable via redownload
                    log("  Execution error detected — will attempt redownload and retry")
                    return False, None, True
                corrupt_path = None
                for msg in msgs:
                    msg_str = str(msg)
                    match = re.search(r"File path: ([^\\s\\\\]+)", msg_str)
                    if match:
                        corrupt_path = match.group(1)
                        log(f"  Corrupt file detected: {corrupt_path}")
                        break
                return False, corrupt_path, False
        except Exception as e:
            log(f"  History poll error: {e}")
        time.sleep(3)
    log(f"  Timeout waiting for completion ({timeout}s, elapsed {time.time() - start:.1f}s)")
    return False, None, False


def redownload_model(version_id, dest_file):
    """Try to redownload a model file from Civitai. Returns True if successful."""
    if version_id is None:
        log("  No version_id available — cannot redownload")
        return False
    download_url = get_download_url(version_id)
    if download_url is None:
        log("  Failed to get download URL — cannot redownload")
        return False

    # Fetch version info for SHA256 + size verification
    expected_size = None
    expected_sha256 = None
    version_info = run_civitai(["version", str(version_id)])
    if version_info:
        try:
            vinfo = json.loads(version_info)
            files = vinfo.get("files", [])
            if files:
                main_file = None
                for f in files:
                    if f.get("type") in ("Main", "Model"):
                        main_file = f
                        break
                if main_file is None:
                    main_file = files[0]
                expected_size = main_file.get("sizeKB", 0) * 1024  # sizeKB -> bytes
                hashes = main_file.get("hashes", {})
                expected_sha256 = hashes.get("SHA256")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            log(f"  Could not parse version info for verification: {e}")

    log(f"  Re-downloading: {os.path.basename(dest_file)}")
    if download_file(download_url, dest_file, expected_size=expected_size, expected_sha256=expected_sha256):
        log(f"  Re-download OK ({os.path.getsize(dest_file) / (1024**3):.2f} GB)")
        return True
    log("  Re-download failed — file missing or empty")
    return False


def redownload_hf(hf_repo, local_path):
    """Try to redownload a model file from HuggingFace. Returns True if successful."""
    log(f"  Re-downloading from HF: {os.path.basename(local_path)}")
    if download_file_hf(hf_repo, local_path):
        dest_file = os.path.join(cfg.comfyui_base, "models", local_path)
        log(f"  HF re-download OK ({os.path.getsize(dest_file) / (1024**3):.2f} GB)")
        return True
    log("  HF re-download failed — file missing or empty")
    return False


def generate_with_retry(workflow, timeout=300, max_retries=1):
    """Queue a prompt and wait for completion, retrying if ComfyUI crashes mid-generation."""
    for attempt in range(max_retries + 1):
        prompt_id = queue_prompt(workflow)
        if prompt_id is None:
            if attempt < max_retries:
                if restart_comfyui():
                    continue
            return None, None
        success, corrupt_path, execution_error = wait_for_completion(prompt_id, timeout=timeout)
        if success:
            return prompt_id, None
        if corrupt_path:
            return None, corrupt_path
        if execution_error:
            log("  Execution error — retrying once without ComfyUI restart")
            prompt_id = queue_prompt(workflow)
            if prompt_id is not None:
                log(f"  Prompt ID (retry): {prompt_id}")
                success2, corrupt_path2, execution_error2 = wait_for_completion(prompt_id, timeout=timeout)
                if success2:
                    return prompt_id, None
                if corrupt_path2:
                    return None, corrupt_path2
            return None, None
        # Prompt did not complete - ComfyUI may have crashed mid-generation
        if attempt < max_retries:
            log(f"  Prompt did not complete, attempting ComfyUI restart and retry (attempt {attempt + 1}/{max_retries + 1})")
            if restart_comfyui():
                continue
        return None, None
    return None, None


def get_output_image(prompt_id):
    try:
        resp = subprocess.run(
            ["curl", "-s", f"{cfg.comfyui_url}/history/{prompt_id}"],
            capture_output=True, text=True, timeout=10)
        hist = json.loads(resp.stdout)
        if prompt_id in hist:
            outputs = hist[prompt_id].get("outputs", {})
            for node_id, out in outputs.items():
                if "images" in out and isinstance(out["images"], list) and len(out["images"]) > 0:
                    img = out["images"][0]
                    filename = img.get("filename")
                    subfolder = img.get("subfolder", "")
                    if subfolder:
                        path = os.path.join(cfg.comfyui_base, "output", subfolder, filename)
                    else:
                        path = os.path.join(cfg.comfyui_base, "output", filename)
                    if os.path.exists(path):
                        return path
    except Exception as e:
        log(f"  Error extracting image: {e}")
    return None


def upscale_image(image_path, upscaler_model=None):
    """Upscale a generated image using the specified upscaler model via ComfyUI."""
    if not cfg.enable_upscaling:
        return None

    upscaler_model = upscaler_model or cfg.upscaler_model
    filename = os.path.basename(image_path)

    # ComfyUI's LoadImage node reads from the input/ directory, but generated images
    # are in output/. Copy the image to input/ so LoadImage can find it.
    input_dir = os.path.join(cfg.comfyui_base, "input")
    os.makedirs(input_dir, exist_ok=True)
    input_copy = os.path.join(input_dir, filename)
    if os.path.exists(image_path):
        import shutil
        shutil.copy2(image_path, input_copy)

    # Build upscale workflow: LoadImage -> UpscaleModelLoader -> ImageUpscaleWithModel -> SaveImage
    workflow = {
        "1": {"inputs": {"image": filename, "image_upload": True}, "class_type": "LoadImage"},
        "2": {"inputs": {"model_name": upscaler_model}, "class_type": "UpscaleModelLoader"},
        "3": {"inputs": {"image": ["1", 0], "upscale_model": ["2", 0]}, "class_type": "ImageUpscaleWithModel"},
        "4": {"inputs": {"images": ["3", 0], "filename_prefix": f"upscaled/{filename}"}, "class_type": "SaveImage"},
    }

    if not check_comfyui_health():
        if not restart_comfyui():
            log(f"  ComfyUI unavailable, skipping upscale for {filename}")
            return None

    prompt_id = queue_prompt(workflow)
    if prompt_id is None:
        log(f"  Failed to queue upscale prompt for {filename}")
        return None

    log(f"  Upscaling {filename} with {upscaler_model} (prompt {prompt_id})")
    success, corrupt_path, execution_error = wait_for_completion(prompt_id, timeout=300)
    if execution_error:
        log("  Upscale execution error — retrying once")
        prompt_id = queue_prompt(workflow)
        if prompt_id is not None:
            log(f"  Prompt ID (retry): {prompt_id}")
            success, corrupt_path, execution_error = wait_for_completion(prompt_id, timeout=300)
            if execution_error:
                log("  Upscale failed again — skipping upscale for this image")
                return None
        if not success:
            return None
    if not success and not corrupt_path:
        # ComfyUI may have crashed mid-generation - restart and retry once
        if restart_comfyui():
            log("  ComfyUI restarted, retrying generation...")
            prompt_id = queue_prompt(workflow)
            if prompt_id is not None:
                log(f"  Prompt ID (retry): {prompt_id}")
                success, corrupt_path, execution_error = wait_for_completion(prompt_id, timeout=300)
    if success:
        upscaled_path = get_output_image(prompt_id)
        if upscaled_path and os.path.exists(upscaled_path):
            log(f"  Upscaled image: {upscaled_path}")
            # Send upscaled to Telegram too
            try:
                send_telegram_photo(
                    upscaled_path,
                    f"Upscaled: {os.path.basename(image_path)}"
                )
            except Exception as e:
                log(f"  Telegram upscale send error: {e}")
            return upscaled_path
        else:
            log(f"  No upscaled output image found")
    else:
        log(f"  Upscale did not complete for {filename}")
    return None
