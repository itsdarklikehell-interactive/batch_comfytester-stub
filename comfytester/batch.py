"""batch_comfytester orchestration: per-model processing, id collection, main loop.

All paths/behaviour come from AppConfig (config module) — no hardcoded
/home/rizzo literals. Delegates to comfytester modules:
  civitai, downloader, families, state, telegram, utils, comfyui, workflows.
"""

from __future__ import annotations

import fcntl
import json
import os
import random
import re
import time
import traceback
from datetime import datetime

from . import families
from . import config as _config_mod
from .civitai import (
    fetch_civitai_collections,
    fetch_civitai_favorites,
    fetch_civitai_followed,
    get_download_url,
    get_filename_from_api,
    get_model_info,
    run_civitai,
)
from .comfyui import (
    check_comfyui_health,
    get_output_image,
    queue_prompt,
    redownload_hf,
    redownload_model,
    restart_comfyui,
    upscale_image,
    wait_for_completion,
)
from .downloader import download_file, download_file_hf, download_hf_dataset
from .families import (
    detect_family,
    get_family_dialect,
    get_max_checkpoint_gb,
    get_random_prompt,
    get_settings,
    get_vae_list_for_family,
    inject_lora_triggers,
    load_family_map,
    load_prompt_pools,
    vae_file_exists,
)
from .state import (
    load_completed_datasets,
    load_completed_models,
    load_failed_models,
    load_priority_skip_lists,
    mark_completed,
    mark_completed_dataset,
    mark_failed,
    save_metadata_sidecar,
)
from .telegram import send_telegram_message, send_telegram_photo, send_telegram_status
from .utils import find_control_hint_image, log, rotate_log_if_needed
from .workflows import (
    build_workflow_anima,
    build_workflow_ckpt,
    build_workflow_ckpt_combo,
    build_workflow_controlnet,
    build_workflow_lora,
    build_workflow_qwen_image_2_1,
    build_workflow_vae,
    build_workflow_wan22_video,
)

cfg: _config_mod.AppConfig | None = None

# Date-based output folder, computed once per run when output_date_folder is enabled.
_date_folder: str | None = None

# Per-model performance timing, accumulated during the run and written to a
# JSON report at the end of batch.main(). Keyed by model id (or filename for
# local/HF scans).
_timings: dict[str, dict] = {}

# Architecture compatibility groups for combination testing.
# LoRAs and embeddings can only be combined with checkpoints of the same
# architecture family. Families not listed here won't auto-combine (e.g.
# anime is ambiguous between SD1.x and SDXL).
COMBO_ARCH_GROUPS: dict[str, set[str]] = {
    "sd1x": {"sd15", "sd21", "photorealistic", "segmind_vega", "stable_cascade", "supr", "depthfm"},
    "sdxl": {"pony", "illustrious", "furry"},
    "flux": {"flux_dev", "flux_klein"},
}


def init(config: _config_mod.AppConfig):
    """Bind the active config and init every sub-module (called once from main)."""
    global cfg
    cfg = config
    _config_mod.load_config  # keep reference; config already loaded by caller
    from . import civitai, comfyui, downloader, families, state, telegram, utils, workflows

    civitai.init(cfg)
    comfyui.init(cfg)
    downloader.init(cfg)
    families.init(cfg)
    state.init(cfg)
    telegram.init(cfg)
    utils.init(cfg)
    workflows.init(cfg)


# ---------------------------------------------------------------------------
# Per-model processing
# ---------------------------------------------------------------------------

def process_model(model_id, completed_models=None, priority=None, skip=None):
    """Process a single Civitai model: info -> family -> download -> generate.

    Mirrors the old monolith's process_model but delegates to the
    comfytester modules. Returns None (side effects: downloads, renders,
    state files, Telegram messages).
    """
    completed_models = completed_models or set()
    skip = skip or set()

    model_name = family = mtype = None
    file_gb = 0.0
    status = "completed"
    timings: dict = {}
    t_model = time.time()
    try:
        log(f"Processing model {model_id}")
        # Defensive: Civitai model IDs must be integers; reject anything else
        if not str(model_id).isdigit():
            log(f"  Rejecting non-numeric model ID: {model_id}")
            status = "invalid_id"
            return
        # Dry-run mode: list what would be processed without downloading/generating
        if cfg.dry_run:
            log(f"  [DRY-RUN] Would process model {model_id}")
            status = "dry_run"
            return
        # Skip list check
        if model_id in skip:
            log(f"  Skipping model {model_id} (on skip list)")
            send_telegram_status(f"skipping model {model_id}")
            status = "skipped"
            return
        # Completed/resume check
        if model_id in completed_models:
            log(f"  Skipping model {model_id} (already completed)")
            status = "cached"
            return

        t_api = time.time()
        result = get_filename_from_api(model_id)
        timings["api_fetch_s"] = time.time() - t_api
        if result is None or result[1] is None:
            log(f"  Could not determine filename")
            status = "api_failed"
            return
        if len(result) == 4:
            model_name, version_id, fname, model_tags = result
        else:
            model_name, version_id, fname = result
            model_tags = []

        if fname is None:
            log(f"  Could not determine filename")
            return
        log(f"  Model name: {model_name}")
        log(f"  Filename: {fname}, Version ID: {version_id}")
        send_telegram_status(f"processing {model_name} ({model_id})")

        # Detect family and get settings/prompt
        family = detect_family(model_name, model_tags)
        # Family filter: only process models of the requested family
        if cfg.family_filter and family != cfg.family_filter:
            log(f"  Skipping model {model_id} (family {family} != filter {cfg.family_filter})")
            return
        settings = get_settings(family)
        prompt, negative = get_random_prompt(
            family,
            category_order=cfg.prompt_category_order or None,
            negative_category=cfg.prompt_negative_category,
            nsfw_mode=cfg.nsfw_mode,
        )
        log(f"  Family: {family} | Prompt: {prompt[:80]}... | Settings: {settings['steps']}steps "
            f"CFG={settings['cfg']} {settings['sampler']} {settings['resolution'][0]}x{settings['resolution'][1]}")

        mtype = ""
        info = get_model_info(model_id)
        if info is not None:
            mtype = info.get("type", "").upper()
        # Normalize LoCon / Lycoris / DoRA to LORA — ComfyUI's LoraLoader handles all of them.
        # Civitai labels these "LoCon", "Lycoris", "DoRA" (uppercased: LOCON, LYCORIS, DORA).
        if mtype in ("LOCON", "LYCORIS", "DORA"):
            mtype = "LORA"
        log(f"  Type: {mtype}")

        vae_name = None
        strength = 0.8

        if mtype == "CHECKPOINT":
            status = _process_checkpoint(timings, model_id, model_name, fname, version_id, family,
                                         mtype, settings, prompt, negative, vae_name, strength)
        elif mtype == "LORA":
            status = _process_lora(timings, model_id, model_name, fname, version_id, family, mtype,
                                   settings, prompt, negative, vae_name, strength)
        elif mtype == "VAE":
            status = _process_vae(timings, model_id, model_name, fname, version_id, family, mtype,
                                  settings, prompt, negative)
        elif mtype == "CONTROLNET":
            status = _process_controlnet(timings, model_id, model_name, fname, version_id, family,
                                         mtype, settings, prompt, negative)
        elif mtype in ("TEXTUALINVERSION", "HYPERNETWORK", "UPSCALER", "AESTHETICGRADIENT",
                       "MOTIONMODULE", "POSES", "WILDCARDS", "DETECTION", "WORKFLOWS"):
            status = _process_download_only(timings, model_id, model_name, fname, version_id, mtype)
        elif mtype == "OTHER":
            log(f"  Model type OTHER - skipping download (manual review needed): {fname}")
            mark_failed(model_id)
            status = "manual_review"
            log(f"  Finished model {model_id}")
            return
        else:
            log(f"  Unknown type {mtype}, skipping")
            status = "skipped"
            return

    except Exception as e:
        status = "error"
        log(f"  EXCEPTION processing model {model_id}: {e}")
        traceback.print_exc()
    finally:
        _timings[model_id] = {
            "model_id": model_id,
            "model_name": model_name,
            "family": family,
            "mtype": mtype,
            "file_gb": round(file_gb, 3),
            "status": status,
            "api_fetch_s": round(timings.get("api_fetch_s", 0.0), 3),
            "download_s": round(timings.get("download_s", 0.0), 3),
            "generation_s": round(timings.get("generation_s", 0.0), 3),
            "upscale_s": round(timings.get("upscale_s", 0.0), 3),
            "total_s": round(time.time() - t_model, 3),
        }
    mark_completed(model_id)
    log(f"  Finished model {model_id}")


def _dest_dir_for(mtype, fname):
    """Return the ComfyUI models subdirectory and dest file for a model type."""
    if mtype == "CHECKPOINT" and fname.lower().endswith(".gguf"):
        dest_dir = os.path.join(cfg.comfyui_base, "models", "checkpoints", "gguf")
    elif mtype == "CHECKPOINT":
        dest_dir = os.path.join(cfg.comfyui_base, "models", "checkpoints")
    elif mtype == "LORA":
        dest_dir = os.path.join(cfg.comfyui_base, "models", "loras")
    elif mtype == "VAE":
        dest_dir = os.path.join(cfg.comfyui_base, "models", "vae")
    elif mtype == "CONTROLNET":
        dest_dir = os.path.join(cfg.comfyui_base, "models", "controlnet")
    elif mtype == "TEXTUALINVERSION" or mtype == "AESTHETICGRADIENT":
        dest_dir = os.path.join(cfg.comfyui_base, "models", "embeddings")
    elif mtype == "HYPERNETWORK":
        dest_dir = os.path.join(cfg.comfyui_base, "models", "hypernetworks")
    elif mtype == "UPSCALER":
        dest_dir = os.path.join(cfg.comfyui_base, "models", "upscale_models")
    elif mtype == "MOTIONMODULE":
        dest_dir = os.path.join(cfg.comfyui_base, "models", "animatediff")
    elif mtype == "POSES":
        dest_dir = os.path.join(cfg.comfyui_base, "models", "controlnet")
    elif mtype == "WILDCARDS":
        dest_dir = os.path.join(cfg.comfyui_base, "models", "wildcards")
    elif mtype == "DETECTION":
        dest_dir = os.path.join(cfg.comfyui_base, "models", "detection")
    elif mtype == "WORKFLOWS":
        dest_dir = os.path.join(cfg.comfyui_base, "workflows")
    else:
        dest_dir = os.path.join(cfg.comfyui_base, "models")
    os.makedirs(dest_dir, exist_ok=True)
    return os.path.join(dest_dir, fname)


def _download_if_needed(model_id, fname, version_id, mtype, dest_file):
    """Download a Civitai model file if missing/empty. Returns dest_file or None."""
    if os.path.exists(dest_file) and os.path.getsize(dest_file) > 0:
        log(f"  {mtype} already exists: {fname}")
        return dest_file
    download_url = get_download_url(version_id)
    if download_url is None:
        log(f"  Failed to get download URL")
        mark_failed(model_id)
        return None
    log(f"  Downloading {mtype.lower()} to {fname}")
    if not download_file(download_url, dest_file):
        log(f"  Download failed for {dest_file}")
        mark_failed(model_id)
        return None
    send_telegram_status(f"downloaded {fname}")
    if not os.path.exists(dest_file) or os.path.getsize(dest_file) == 0:
        log(f"  File missing or empty: {dest_file}")
        mark_failed(model_id)
        return None
    return dest_file


def _timed_download(timings: dict, model_id, fname, version_id, mtype, dest_file):
    """Download a model file if needed, recording wall time into timings['download_s']."""
    t0 = time.time()
    result = _download_if_needed(model_id, fname, version_id, mtype, dest_file)
    timings["download_s"] = timings.get("download_s", 0.0) + (time.time() - t0)
    return result


def _process_checkpoint(timings: dict, model_id, model_name, fname, version_id, family, mtype,
                        settings, prompt, negative, vae_name, strength):
    """Download + generate a checkpoint. Returns a status string."""
    dest_file = _dest_dir_for(mtype, fname)
    dest_file = _timed_download(timings, model_id, fname, version_id, mtype, dest_file)
    if dest_file is None:
        return "download_failed"
    file_gb = os.path.getsize(dest_file) / (1024 ** 3)
    max_gb = get_max_checkpoint_gb(family)
    if file_gb > max_gb:
        log(f"  Warning: oversized checkpoint {file_gb:.1f}GB > {max_gb}GB VRAM limit for {family} — generating anyway (Dynamic VRAM will offload): {fname}")
    # VAE comparison mode: generate with each compatible VAE, or once with checkpoint's own VAE
    generated_any = False
    for vae_name in get_vae_list_for_family(family):
        vae_label = vae_name if vae_name else "checkpoint_default"
        if not vae_file_exists(vae_name):
            log(f"  VAE file not found locally: {vae_label}, skipping")
            continue
        log(f"  Generating with VAE: {vae_label}")
        workflow, seed = build_workflow_ckpt(fname, prompt, negative, settings,
                                             vae_name=vae_name, family=family,
                                             date_folder=_date_folder)
        log(f"  Using seed: {seed}")
        t0 = time.time()
        _run_generation(model_id, model_name, mtype, family, settings, prompt, negative,
                        seed, version_id, workflow, timings,
                        extra={"civitai_version_id": version_id, "vae": vae_label})
        timings["generation_s"] = timings.get("generation_s", 0.0) + (time.time() - t0)
        generated_any = True
    return "generated" if generated_any else "skipped"


def _process_lora(timings: dict, model_id, model_name, fname, version_id, family, mtype,
                  settings, prompt, negative, vae_name, strength):
    """Download + generate a LoRA. Returns a status string."""
    dest_file = _dest_dir_for(mtype, fname)
    dest_file = _timed_download(timings, model_id, fname, version_id, mtype, dest_file)
    if dest_file is None:
        return "download_failed"
    # VAE comparison + LoRA strength sweep: generate for each VAE x strength
    generated_any = False
    for vae_name in get_vae_list_for_family(family):
        vae_label = vae_name if vae_name else "checkpoint_default"
        if not vae_file_exists(vae_name):
            log(f"  VAE file not found locally: {vae_label}, skipping")
            continue
        for strength in cfg.lora_strengths:
            log(f"  Generating with VAE: {vae_label} | LoRA strength: {strength}")
            if cfg.dry_run:
                log(f"  [DRY-RUN] Would generate LoRA image with VAE={vae_label}, strength={strength}")
                continue
            workflow, seed = build_workflow_lora(fname, prompt, negative, settings,
                                                 strength=strength, vae_name=vae_name, family=family,
                                                 date_folder=_date_folder)
            log(f"  Using seed: {seed}")
            t0 = time.time()
            _run_generation(model_id, model_name, mtype, family, settings, prompt, negative,
                            seed, version_id, workflow, timings,
                            extra={"civitai_version_id": version_id, "lora_strength": strength,
                                   "vae": vae_label})
            timings["generation_s"] = timings.get("generation_s", 0.0) + (time.time() - t0)
            generated_any = True
    return "generated" if generated_any else "skipped"


def _process_vae(timings: dict, model_id, model_name, fname, version_id, family, mtype,
                 settings, prompt, negative):
    """Download + generate a VAE. Returns a status string."""
    dest_file = _dest_dir_for(mtype, fname)
    dest_file = _timed_download(timings, model_id, fname, version_id, mtype, dest_file)
    if dest_file is None:
        return "download_failed"
    workflow, seed = build_workflow_vae(fname, prompt, negative, settings, family=family,
                                        date_folder=_date_folder)
    log(f"  Using seed: {seed}")
    t0 = time.time()
    _run_generation(model_id, model_name, mtype, family, settings, prompt, negative,
                    seed, version_id, workflow, timings, extra={"civitai_version_id": version_id})
    timings["generation_s"] = timings.get("generation_s", 0.0) + (time.time() - t0)
    return "generated"


def _process_controlnet(timings: dict, model_id, model_name, fname, version_id, family, mtype,
                        settings, prompt, negative):
    """Download + generate with ControlNet. Returns a status string."""
    if not cfg.controlnet_enabled:
        log(f"  ControlNet generation disabled, skipping: {fname}")
        mark_failed(model_id)
        return "skipped"
    dest_file = _dest_dir_for(mtype, fname)
    dest_file = _timed_download(timings, model_id, fname, version_id, mtype, dest_file)
    if dest_file is None:
        return "download_failed"
    hint_path = find_control_hint_image()
    if not hint_path:
        log("  Control hint image not found in input/ or output/, skipping generation")
        mark_completed(model_id)
        return "skipped"
    log(f"  Generating with ControlNet: {fname} (hint: {os.path.basename(hint_path)})")
    if cfg.dry_run:
        log(f"  [DRY-RUN] Would generate ControlNet image with {fname}")
        mark_completed(model_id)
        return "skipped"
    workflow, seed = build_workflow_controlnet(fname, prompt, negative, settings,
                                               control_image=hint_path, family=family,
                                               date_folder=_date_folder)
    log(f"  Using seed: {seed}")
    t0 = time.time()
    _run_generation(model_id, model_name, mtype, family, settings, prompt, negative,
                    seed, version_id, workflow, timings,
                    extra={"civitai_version_id": version_id, "controlnet": fname,
                           "control_hint": hint_path})
    timings["generation_s"] = timings.get("generation_s", 0.0) + (time.time() - t0)
    return "generated"


def _process_download_only(timings: dict, model_id, model_name, fname, version_id, mtype):
    dest_file = _dest_dir_for(mtype, fname)
    _timed_download(timings, model_id, fname, version_id, mtype, dest_file)
    log(f"  Finished model {model_id}")
    return "download_only"


def _run_generation(model_id, model_name, mtype, family, settings, prompt, negative,
                    seed, version_id, workflow, timings: dict | None = None, extra=None):
    """Queue a prompt, wait, handle errors/retries, save outputs + Telegram."""
    if extra is None:
        extra = {}
    if not check_comfyui_health():
        if not restart_comfyui():
            log("  ComfyUI unavailable, skipping model")
            mark_completed(model_id)
            return
    prompt_id = queue_prompt(workflow)
    if prompt_id is None:
        log(f"  Failed to queue prompt")
        return
    log(f"  Prompt ID: {prompt_id}")
    success, corrupt_path, execution_error = wait_for_completion(prompt_id, timeout=300)

    if execution_error:
        log("  Execution error — attempting redownload + retry")
        # redownload only applies to Civitai-sourced checkpoints/loras/vaes
        if mtype in ("CHECKPOINT", "LORA", "VAE") and version_id is not None:
            # re-fetch dest file for redownload
            fname = None
            res = get_filename_from_api(model_id)
            if res and len(res) >= 3:
                fname = res[2]
            if fname:
                dest_file = _dest_dir_for(mtype, fname)
                if redownload_model(version_id, dest_file):
                    workflow = _rebuild_workflow(mtype, fname, prompt, negative, settings,
                                                 family, extra)
                    prompt_id = queue_prompt(workflow)
                    if prompt_id is not None:
                        log(f"  Prompt ID (retry): {prompt_id}")
                        success, corrupt_path, execution_error = wait_for_completion(prompt_id, timeout=300)
        if execution_error:
            log(f"  Still failing after redownload — skipping this model")
            mark_completed(model_id)
            return
        if not success:
            mark_completed(model_id)
            return

    if not success and not corrupt_path:
        if restart_comfyui():
            log("  ComfyUI restarted, retrying generation...")
            prompt_id = queue_prompt(workflow)
            if prompt_id is not None:
                log(f"  Prompt ID (retry): {prompt_id}")
                success, corrupt_path, execution_error = wait_for_completion(prompt_id, timeout=300)

    if success:
        img_path = get_output_image(prompt_id)
        if img_path and os.path.exists(img_path):
            log(f"  Generated image: {img_path}")
            save_metadata_sidecar(img_path, model_id, model_name, mtype, family,
                                  prompt, negative, seed, settings, extra=extra)
            send_telegram_photo(
                img_path,
                f"Batch render: {os.path.basename(img_path)}\\nFamily: {family}\\n"
                f"Model ID: {model_id}\\nPrompt: {prompt[:200]}...\\nSeed: {seed}")
            send_telegram_status(f"rendered {os.path.basename(img_path)}")
            if cfg.enable_upscaling:
                t0 = time.time()
                upscaled = upscale_image(img_path)
                if timings is not None:
                    timings["upscale_s"] = timings.get("upscale_s", 0.0) + (time.time() - t0)
                if upscaled:
                    save_metadata_sidecar(upscaled, model_id, model_name, mtype, family,
                                          prompt, negative, seed, settings,
                                          extra={**extra, "upscaled_from": os.path.basename(img_path),
                                                 "upscaler": cfg.upscaler_model})
        else:
            log(f"  No output image found")
    elif corrupt_path:
        log(f"  File corrupt: {corrupt_path}. Deleting and re-downloading...")
        try:
            os.remove(corrupt_path)
            log(f"  Deleted corrupt file: {os.path.basename(corrupt_path)}")
        except OSError as e:
            log(f"  Failed to delete corrupt file: {e}")
            mark_completed(model_id)
            return
        # Re-download + retry for Civitai-sourced types
        res = get_filename_from_api(model_id)
        fname = res[2] if res and len(res) >= 3 else None
        if mtype in ("CHECKPOINT", "LORA", "VAE") and fname and version_id is not None:
            download_url = get_download_url(version_id)
            if download_url is None:
                log(f"  Failed to get download URL for re-download")
                mark_completed(model_id)
                return
            dest_file = _dest_dir_for(mtype, fname)
            log(f"  Re-downloading {mtype.lower()}: {fname}")
            if not download_file(download_url, dest_file):
                log(f"  Download failed for {dest_file}")
                mark_completed(model_id)
                return
            log(f"  Re-download OK, retrying generation")
            workflow = _rebuild_workflow(mtype, fname, prompt, negative, settings,
                                         family, extra)
            prompt_id = queue_prompt(workflow)
            if prompt_id is None:
                log(f"  Failed to queue prompt on retry")
                mark_completed(model_id)
                return
            log(f"  Prompt ID (retry): {prompt_id}")
            success2, _, _ = wait_for_completion(prompt_id, timeout=300)
            if success2:
                img_path = get_output_image(prompt_id)
                if img_path and os.path.exists(img_path):
                    log(f"  Generated image (retry): {img_path}")
                    save_metadata_sidecar(img_path, model_id, model_name, mtype, family,
                                          prompt, negative, seed, settings,
                                          extra={**extra, "retry": True})
                    send_telegram_photo(
                        img_path,
                        f"Batch render (retry): {os.path.basename(img_path)}\\nFamily: {family}\\n"
                        f"Model ID: {model_id}\\nPrompt: {prompt[:200]}...\\nSeed: {seed}")
                    send_telegram_status(f"rendered {os.path.basename(img_path)}")
                    if cfg.enable_upscaling:
                        t0 = time.time()
                        upscaled = upscale_image(img_path)
                        if timings is not None:
                            timings["upscale_s"] = timings.get("upscale_s", 0.0) + (time.time() - t0)
                        if upscaled:
                            save_metadata_sidecar(upscaled, model_id, model_name, mtype, family,
                                                  prompt, negative, seed, settings,
                                                  extra={**extra, "retry": True,
                                                         "upscaled_from": os.path.basename(img_path),
                                                         "upscaler": cfg.upscaler_model})
                else:
                    log(f"  No output image found (retry)")
            else:
                log(f"  Prompt did not complete on retry (model {model_id})")
        else:
            log(f"  Corrupt file detected but type {mtype} not supported for re-download")
    else:
        log(f"  Prompt did not complete")


def _rebuild_workflow(mtype, fname, prompt, negative, settings, family, extra):
    """Re-build the workflow for a retry (mirrors the original per-type builder)."""
    vae_name = extra.get("vae")
    strength = extra.get("lora_strength", 0.8)
    if mtype == "CHECKPOINT":
        return build_workflow_ckpt(fname, prompt, negative, settings,
                                   vae_name=vae_name, family=family,
                                   date_folder=_date_folder)[0]
    if mtype == "LORA":
        return build_workflow_lora(fname, prompt, negative, settings,
                                   strength=strength, vae_name=vae_name, family=family,
                                   date_folder=_date_folder)[0]
    if mtype == "VAE":
        return build_workflow_vae(fname, prompt, negative, settings, family=family,
                                  date_folder=_date_folder)[0]
    if mtype == "CONTROLNET":
        return build_workflow_controlnet(fname, prompt, negative, settings,
                                         control_image=extra.get("control_hint"),
                                         family=family, date_folder=_date_folder)[0]
    return None


# ---------------------------------------------------------------------------
# Model id collection
# ---------------------------------------------------------------------------

def _parse_id_list_file(path):
    """Parse the monolith's id-list file: digits after 'models/' segments."""
    ids = set()
    if not path or not os.path.exists(path):
        return ids
    with open(path) as f:
        for line in f:
            if "models/" in line:
                parts = line.strip().split("models/")
                for part in parts[1:]:
                    seg = part.split("/")[0]
                    if seg.isdigit():
                        ids.add(seg)
    return ids


def collect_model_ids(cfg):
    """Collect the ordered list of model IDs to process.

    Sources (deduped):
      1. CLI --ids filter (if given, only those ids)
      2. id-list file (cfg.id_list_path)
      3. Civitai profile: favorites + followed creators + collections (if token)
      4. Priority file (cfg.priority_file) — ordered first
    Ordering: priority ids (in file order), then previously-failed ids (retry),
    then remaining unique ids sorted numerically. Skip-list and completed ids
    are excluded from the run (process_model also guards them).
    Returns (ordered_ids, skip_set, completed_set, priority_list).
    """
    civitai_token = None
    if os.path.exists(cfg.civitai_token_file):
        try:
            with open(cfg.civitai_token_file) as f:
                civitai_token = f.read().strip()
        except Exception:
            pass

    ids = set()
    # Civitai profile sources (favorites + followed creators + collections)
    if civitai_token:
        fav_ids = fetch_civitai_favorites(civitai_token)
        followed_ids = fetch_civitai_followed(civitai_token)
        collection_ids = fetch_civitai_collections(civitai_token)
        profile_ids = fav_ids + followed_ids + collection_ids
        seen = set()
        deduped = []
        for mid in profile_ids:
            if mid not in seen:
                deduped.append(mid)
                seen.add(mid)
        ids.update(deduped)
        log(f"Civitai profile sources: {len(fav_ids)} favorites + {len(followed_ids)} "
            f"followed-creator + {len(collection_ids)} collection models = {len(deduped)} unique IDs")
    else:
        log("Civitai: no CIVITAI_TOKEN — skipping favorites/followed/collections fetch")

    # id-list file (legacy document list)
    ids.update(_parse_id_list_file(cfg.id_list_path))

    # CLI id filter overrides: only process the given ids
    if cfg.id_filter:
        ids = {mid for mid in ids if mid in set(cfg.id_filter)}
        # also make sure explicitly-requested ids are present even if not in other sources
        ids.update(cfg.id_filter)

    # --- Validate: reject non-numeric Civitai IDs (API requires int) ---
    invalid_ids = sorted(mid for mid in ids if not mid.isdigit())
    if invalid_ids:
        log(f"  Rejecting {len(invalid_ids)} non-numeric model ID(s): {invalid_ids}")
        log("  Civitai model IDs must be integers — non-numeric IDs cannot be resolved")
        ids = {mid for mid in ids if mid.isdigit()}
    log(f"Total model IDs to process: {len(ids)} (including Civitai profile IDs)")

    completed_models = load_completed_models()
    priority, skip = load_priority_skip_lists()
    failed_models = load_failed_models()
    # Filter failed_models to numeric IDs only (Civitai API requires int; HF names like
    # 'qwen_image_2_1' or 'anima_turbo' from deferred downloads would crash key=int)
    failed_models = {mid for mid in failed_models if mid.isdigit()}
    log(f"Found {len(ids)} unique model IDs | {len(completed_models)} already completed | "
        f"{len(priority)} priority | {len(skip)} skip | {len(failed_models)} previously failed (retry)")

    ordered = []
    seen = set()
    for mid in priority:
        if mid in ids and mid not in seen:
            ordered.append(mid)
            seen.add(mid)
    # Retry previously failed models next
    for mid in sorted(failed_models, key=int):
        if mid in ids and mid not in seen:
            ordered.append(mid)
            seen.add(mid)
    for mid in sorted(ids, key=int):
        if mid not in seen and mid not in skip:
            ordered.append(mid)
            seen.add(mid)

    if cfg.limit is not None:
        ordered = ordered[: cfg.limit]
        log(f"Limit applied: processing first {len(ordered)} model IDs")

    return ordered, skip, completed_models, priority


# ---------------------------------------------------------------------------
# Local scan + HuggingFace processing
# ---------------------------------------------------------------------------

def scan_local_models():
    """Scan local model directories for manually-placed models and test them."""
    generatable_dirs = [
        (os.path.join(cfg.comfyui_base, "models", "checkpoints"), "CHECKPOINT"),
        (os.path.join(cfg.comfyui_base, "models", "checkpoints", "gguf"), "CHECKPOINT"),
        (os.path.join(cfg.comfyui_base, "models", "loras"), "LORA"),
        (os.path.join(cfg.comfyui_base, "models", "vae"), "VAE"),
    ]
    log_only_dirs = [
        (os.path.join(cfg.comfyui_base, "models", "controlnet"), "CONTROLNET"),
        (os.path.join(cfg.comfyui_base, "models", "embeddings"), "TEXTUALINVERSION"),
        (os.path.join(cfg.comfyui_base, "models", "upscale_models"), "UPSCALER"),
        (os.path.join(cfg.comfyui_base, "models", "hypernetworks"), "HYPERNETWORK"),
        (os.path.join(cfg.comfyui_base, "models", "animatediff"), "MOTIONMODULE"),
        (os.path.join(cfg.comfyui_base, "models", "wildcards"), "WILDCARDS"),
        (os.path.join(cfg.comfyui_base, "models", "detection"), "DETECTION"),
        (os.path.join(cfg.comfyui_base, "workflows"), "WORKFLOWS"),
    ]

    log("Scanning for locally-placed models...")
    total_found = 0

    for scan_dir, mtype in log_only_dirs:
        if not os.path.exists(scan_dir):
            continue
        files = [f for f in sorted(os.listdir(scan_dir))
                 if f.endswith((".safetensors", ".ckpt", ".pt", ".json"))]
        if files:
            log(f"  Local {mtype}: {len(files)} file(s) in {os.path.basename(scan_dir)}/")
            for f in files[:10]:
                fpath = os.path.join(scan_dir, f)
                sz = os.path.getsize(fpath) / (1024 ** 2)
                log(f"    - {f} ({sz:.1f} MB)")
            if len(files) > 10:
                log(f"    ... and {len(files) - 10} more")
            total_found += len(files)

    for scan_dir, mtype in generatable_dirs:
        if not os.path.exists(scan_dir):
            continue
        found = []
        if scan_dir.endswith(os.path.join("models", "checkpoints")):
            walker = os.walk(scan_dir)
        else:
            walker = [(scan_dir, [], sorted(os.listdir(scan_dir)))]
        for root, dirs, files in walker:
            for fname in sorted(files):
                if not fname.endswith((".safetensors", ".ckpt", ".pt")):
                    continue
                fpath = os.path.join(root, fname)
                file_gb = os.path.getsize(fpath) / (1024 ** 3)
                local_family = detect_family(fname)
                max_gb = get_max_checkpoint_gb(local_family)
                if mtype == "CHECKPOINT" and file_gb > max_gb:
                    log(f"  Warning: local checkpoint {file_gb:.1f}GB > {max_gb}GB VRAM limit for {local_family} — generating anyway (Dynamic VRAM will offload): {fname}")
                found.append((fname, fpath))

        if not found:
            continue
        log(f"  Local {mtype}: {len(found)} file(s) to test")
        for fname, fpath in found:
            try:
                family = detect_family(fname)
                settings = get_settings(family)
                prompt, negative = get_random_prompt(
                    family,
                    category_order=cfg.prompt_category_order or None,
                    negative_category=cfg.prompt_negative_category,
                    nsfw_mode=cfg.nsfw_mode,
                )
                log(f"    Testing {fname} | Family: {family} | {settings['steps']}steps "
                    f"CFG={settings['cfg']} {settings['sampler']} {settings['resolution'][0]}x{settings['resolution'][1]}")
                if mtype == "CHECKPOINT":
                    workflow, seed = build_workflow_ckpt(fname, prompt, negative, settings, family=family,
                                                         date_folder=_date_folder)
                elif mtype == "LORA":
                    workflow, seed = build_workflow_lora(fname, prompt, negative, settings, family=family,
                                                         date_folder=_date_folder)
                elif mtype == "VAE":
                    workflow, seed = build_workflow_vae(fname, prompt, negative, settings, family=family,
                                                        date_folder=_date_folder)
                else:
                    continue
                log(f"      Using seed: {seed}")
                success = False
                corrupt_path = None
                t_local = time.time()
                if not check_comfyui_health():
                    if not restart_comfyui():
                        log("      ComfyUI unavailable, skipping")
                        continue
                prompt_id = queue_prompt(workflow)
                if prompt_id is None:
                    log(f"      Failed to queue prompt for {fname}")
                    continue
                log(f"      Prompt ID: {prompt_id}")
                success, corrupt_path, execution_error = wait_for_completion(prompt_id, timeout=300)
                if execution_error:
                    log("  Execution error — retrying once with existing local file")
                    prompt_id = queue_prompt(workflow)
                    if prompt_id is not None:
                        log(f"  Prompt ID (retry): {prompt_id}")
                        success, corrupt_path, execution_error = wait_for_completion(prompt_id, timeout=300)
                        if execution_error or not success:
                            log("  Local model still failing after retry — skipping")
                            continue
                    else:
                        continue
                if not success and not corrupt_path:
                    if restart_comfyui():
                        log("  ComfyUI restarted, retrying generation...")
                        prompt_id = queue_prompt(workflow)
                        if prompt_id is not None:
                            log(f"  Prompt ID (retry): {prompt_id}")
                            success, corrupt_path, execution_error = wait_for_completion(prompt_id, timeout=300)
                if success:
                    img_path = get_output_image(prompt_id)
                    if img_path and os.path.exists(img_path):
                        log(f"      Generated image: {img_path}")
                        save_metadata_sidecar(img_path, fname, fname, mtype, family,
                                              prompt, negative, seed, settings,
                                              extra={"source": "local_scan"})
                        send_telegram_photo(
                            img_path,
                            f"Local {mtype} render: {fname}\\nFamily: {family}\\n"
                            f"Prompt: {prompt[:200]}...\\nSeed: {seed}")
                        send_telegram_status(f"rendered {os.path.basename(img_path)}")
                        if cfg.enable_upscaling:
                            upscaled = upscale_image(img_path)
                            if upscaled:
                                save_metadata_sidecar(upscaled, fname, fname, mtype, family,
                                                      prompt, negative, seed, settings,
                                                      extra={"source": "local_scan",
                                                           "upscaled_from": os.path.basename(img_path),
                                                           "upscaler": cfg.upscaler_model})
                    else:
                        log(f"      No output image found for {fname}")
                elif corrupt_path:
                    log(f"      Corrupt file for {fname}: {corrupt_path}")
                else:
                    log(f"      Prompt did not complete for {fname}")
            except Exception as e:
                log(f"      EXCEPTION processing local {mtype} {fname}: {e}")
                traceback.print_exc()
            _timings[fname] = {
                "model_id": fname,
                "model_name": fname,
                "family": family,
                "mtype": mtype,
                "file_gb": round(file_gb, 3),
                "status": "generated" if success else "local_failed",
                "api_fetch_s": 0.0,
                "download_s": 0.0,
                "generation_s": round(time.time() - t_local, 3),
                "upscale_s": 0.0,
                "total_s": round(time.time() - t_local, 3),
            }
            log(f"      Finished local {fname}")
            time.sleep(2)
        total_found += len(found)

    if total_found == 0:
        log("  No local models found to test")
    else:
        log(f"Local scan complete — {total_found} model file(s) found across all directories")


def process_hf_models():
    """Process HuggingFace models already present in ComfyUI models directories.

    Reads hf_models.txt and generates images/video for models whose files are local
    (or downloadable via HF Hub).
    """
    if not cfg.hf_list_path or not os.path.exists(cfg.hf_list_path):
        log(f"HF model list not found: {cfg.hf_list_path}")
        return
    log("Processing HuggingFace models from local files...")
    hf_token = os.getenv("HF_TOKEN", "")
    with open(cfg.hf_list_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) < 5:
                log(f"  Skipping malformed HF entry: {line[:80]}")
                continue
            model_id, family, mtype, hf_repo, local_path = parts[:5]
            notes = parts[5] if len(parts) > 5 else ""
            t_hf = time.time()

            if model_id in load_completed_models():
                log(f"  Skipping {model_id} (already completed)")
                continue

            full_path = os.path.join(cfg.comfyui_base, "models", local_path)
            if not os.path.exists(full_path):
                log(f"  {model_id}: file not found locally, attempting HF download from {hf_repo}...")
                if download_file_hf(hf_repo, local_path, token=hf_token):
                    log(f"  HF download OK for {model_id}")
                else:
                    log(f"  {model_id}: HF download failed — deferring.")
                    mark_failed(model_id)
                    continue

            file_gb = os.path.getsize(full_path) / (1024 ** 3)
            max_gb = get_max_checkpoint_gb(family)
            if mtype == "CHECKPOINT" and file_gb > max_gb:
                log(f"  Warning: {model_id} {file_gb:.1f}GB > {max_gb}GB limit for {family} — generating anyway (Dynamic VRAM will offload)")

            log(f"  Processing HF model: {model_id} ({family}, {mtype}) — {file_gb:.1f}GB")
            settings = get_settings(family)
            prompt, negative = get_random_prompt(
                family,
                category_order=cfg.prompt_category_order or None,
                negative_category=cfg.prompt_negative_category,
                nsfw_mode=cfg.nsfw_mode,
            )
            log(f"    Family: {family} | {settings['steps']}steps CFG={settings['cfg']} "
                f"{settings['sampler']} {settings['resolution'][0]}x{settings['resolution'][1]}")

            fname = os.path.basename(local_path)
            if not check_comfyui_health():
                if not restart_comfyui():
                    log("  ComfyUI unavailable, skipping HF model")
                    return

            if family == "qwen_image":
                workflow, seed = build_workflow_qwen_image_2_1(local_path, prompt, negative, settings, family=family,
                                                               date_folder=_date_folder)
            elif family == "anima":
                workflow, seed = build_workflow_anima(local_path, prompt, negative, settings, family=family,
                                                      date_folder=_date_folder)
            elif family == "wan22_video":
                workflow, seed = build_workflow_wan22_video(local_path, prompt, negative, settings, family=family,
                                                            date_folder=_date_folder)
            elif mtype == "CHECKPOINT":
                workflow, seed = build_workflow_ckpt(fname, prompt, negative, settings, family=family,
                                                     date_folder=_date_folder)
            else:
                log(f"  Unsupported HF model type {mtype} for {model_id}")
                mark_completed(model_id)
                continue

            log(f"    Using seed: {seed}")
            prompt_id = queue_prompt(workflow)
            if prompt_id is None:
                log(f"    Failed to queue prompt for {model_id}")
                mark_completed(model_id)
                continue
            log(f"    Prompt ID: {prompt_id}")
            success, corrupt_path, execution_error = wait_for_completion(prompt_id, timeout=300)

            if execution_error:
                log(f"    Execution error for {model_id} — attempting HF redownload + retry")
                if redownload_hf(hf_repo, local_path):
                    prompt_id = queue_prompt(workflow)
                    if prompt_id is not None:
                        log(f"    Prompt ID (retry): {prompt_id}")
                        success, corrupt_path, execution_error = wait_for_completion(prompt_id, timeout=300)
                        if execution_error or not success:
                            log(f"    HF model still failing after redownload + retry — marking completed")
                            mark_completed(model_id)
                            continue
                else:
                    log(f"    HF re-download failed — retrying once with existing local file")
                    prompt_id = queue_prompt(workflow)
                    if prompt_id is not None:
                        log(f"    Prompt ID (retry): {prompt_id}")
                        success, corrupt_path, execution_error = wait_for_completion(prompt_id, timeout=300)
                        if execution_error or not success:
                            log(f"    HF model still failing after retry — marking completed")
                            mark_completed(model_id)
                            continue
                    else:
                        mark_completed(model_id)
                        continue

            if success:
                img_path = get_output_image(prompt_id)
                if img_path and os.path.exists(img_path):
                    log(f"    Generated: {img_path}")
                    save_metadata_sidecar(img_path, model_id, model_id, mtype, family,
                                          prompt, negative, seed, settings,
                                          extra={"hf_repo": hf_repo, "local_path": local_path, "notes": notes})
                    send_telegram_photo(
                        img_path,
                        f"HF render: {model_id}\\nFamily: {family}\\nType: {mtype}\\n"
                        f"Prompt: {prompt[:200]}...\\nSeed: {seed}")
                    if cfg.enable_upscaling and mtype == "CHECKPOINT":
                        upscaled = upscale_image(img_path)
                        if upscaled:
                            save_metadata_sidecar(upscaled, model_id, model_id, mtype, family,
                                                  prompt, negative, seed, settings,
                                                  extra={"hf_repo": hf_repo,
                                                         "upscaled_from": os.path.basename(img_path),
                                                         "upscaler": cfg.upscaler_model})
                else:
                    log(f"    No output image found for {model_id}")
            elif corrupt_path:
                log(f"    Corrupt file for {model_id}: {corrupt_path}")
            else:
                log(f"    Prompt did not complete for {model_id}")

            _timings[model_id] = {
                "model_id": model_id,
                "model_name": model_id,
                "family": family,
                "mtype": mtype,
                "file_gb": round(file_gb, 3),
                "status": "generated" if success else "hf_failed",
                "api_fetch_s": 0.0,
                "download_s": 0.0,
                "generation_s": round(time.time() - t_hf, 3),
                "upscale_s": 0.0,
                "total_s": round(time.time() - t_hf, 3),
            }
            mark_completed(model_id)
            log(f"    Finished HF model {model_id}")
            time.sleep(2)

    log("HF model processing complete.")


# --------------------------------------------------------------------------- #
# Combination-testing mode: checkpoint x LoRA(s) x embedding(s)
# --------------------------------------------------------------------------- #

def get_architecture_group(family: str) -> str | None:
    """Return the architecture group for a model family, or None if ambiguous."""
    for group, families in COMBO_ARCH_GROUPS.items():
        if family in families:
            return group
    return None


def scan_local_loras() -> list[tuple[str, str]]:
    """Return [(filename, detected_family)] for local LoRA files."""
    loras_dir = os.path.join(cfg.comfyui_base, "models", "loras")
    if not os.path.isdir(loras_dir):
        return []
    loras = []
    for fname in sorted(os.listdir(loras_dir)):
        if fname.endswith((".safetensors", ".ckpt", ".pt")):
            family = detect_family(fname)
            loras.append((fname, family))
    return loras


def scan_local_embeddings() -> list[str]:
    """Return list of local embedding filenames."""
    emb_dir = os.path.join(cfg.comfyui_base, "models", "embeddings")
    if not os.path.isdir(emb_dir):
        return []
    return [f for f in sorted(os.listdir(emb_dir))
            if f.endswith((".safetensors", ".pt", ".bin"))]


def _process_checkpoint_combo(
    timings: dict,
    model_id: str,
    model_name: str,
    fname: str,
    version_id,
    mtype: str,
    family: str,
    settings: dict,
    prompt: str,
    negative: str,
    vae_name,
    strength: float,
    local_loras: list[tuple[str, str]],
    local_embeddings: list[str],
) -> None:
    """Test a checkpoint with compatible LoRA(s) and embedding(s) combinations.

    Only combines models within the same architecture group (e.g. SDXL LoRA
    with SDXL-family checkpoint). Never mixes SD1.x with SDXL or FLUX.
    """
    arch_group = get_architecture_group(family)
    if arch_group is None:
        log(f"  Skipping combo for {fname}: ambiguous architecture family '{family}'")
        return

    # Find compatible LoRAs (same architecture group)
    compatible_loras = [
        (lf, lf_family) for lf, lf_family in local_loras
        if get_architecture_group(lf_family) == arch_group
    ]
    if not compatible_loras:
        log(f"  No compatible LoRAs for {fname} (arch={arch_group})")
        return

    # Limit LoRAs per combo
    max_loras = min(cfg.combo_max_loras_per_ckpt, len(compatible_loras))
    selected_loras = compatible_loras[:max_loras]

    # Select embeddings to append to negative prompt
    embed_negatives = cfg.combo_embed_negatives
    if embed_negatives:
        selected_embeddings = [e for e in local_embeddings if e in embed_negatives][:3]
    else:
        selected_embeddings = local_embeddings[:3]

    lora_names = [lf for lf, _ in selected_loras]
    lora_strengths = [cfg.combo_lora_strength] * len(lora_names)

    if cfg.prompt_lora_trigger_injection:
        dialect = get_family_dialect(family)
        prompt = inject_lora_triggers(prompt, lora_names, dialect)

    # Build combo list: (loras, strengths, negative_prompt, label)
    combos: list[tuple[list, list, str, str]] = []
    combos.append((lora_names, lora_strengths, negative, "no_embedding"))
    for emb in selected_embeddings:
        emb_negative = f"{negative}, {emb}" if negative else emb
        combos.append((lora_names, lora_strengths, emb_negative, emb))

    combos = combos[:cfg.combo_max_combos_per_ckpt]

    generated_any = False
    for lora_names_combo, lora_strengths_combo, combo_negative, emb_label in combos:
        lora_desc = ",".join(lora_names_combo)
        log(f"  Combo: {fname} + LoRAs({lora_desc}) + emb({emb_label})")

        for vae_name_loop in get_vae_list_for_family(family):
            vae_label = vae_name_loop if vae_name_loop else "checkpoint_default"
            if not vae_file_exists(vae_name_loop):
                log(f"  VAE file not found locally: {vae_label}, skipping")
                continue

            workflow, seed = build_workflow_ckpt_combo(
                fname, lora_names_combo, lora_strengths_combo,
                prompt, combo_negative, settings,
                vae_name=vae_name_loop, family=family,
                output_dir=cfg.combo_output_dir, arch_group=arch_group,
                date_folder=_date_folder,
            )
            log(f"  Using seed: {seed}")
            t0 = time.time()
            extra = {
                "vae": vae_label,
                "loras": lora_names_combo,
                "lora_strengths": lora_strengths_combo,
                "embedding": emb_label,
                "combo_mode": True,
            }
            _run_generation(
                model_id, model_name, mtype, family, settings, prompt, combo_negative,
                seed, version_id, workflow, timings, extra=extra,
            )
            timings["generation_s"] = timings.get("generation_s", 0.0) + (time.time() - t0)
            generated_any = True

    if not generated_any:
        log(f"  No combos generated for {fname}")


def run_combo_mode(config: _config_mod.AppConfig) -> None:
    """Run combination-testing mode: local checkpoints x compatible LoRAs x embeddings."""
    if not config.combo_mode:
        return

    log("=== Combination-testing mode ===")
    log(f"  Output folder: {cfg.comfyui_base}/output/{cfg.combo_output_dir}/<arch>/<checkpoint>/")
    local_loras = scan_local_loras()
    local_embeddings = scan_local_embeddings()
    log(f"  Found {len(local_loras)} local LoRAs, {len(local_embeddings)} local embeddings")

    checkpoints_dir = os.path.join(config.comfyui_base, "models", "checkpoints")
    if not os.path.isdir(checkpoints_dir):
        log("  No checkpoints directory — skipping combo mode")
        return

    found = []
    for root, _dirs, files in os.walk(checkpoints_dir):
        for fname in sorted(files):
            if not fname.endswith((".safetensors", ".ckpt", ".pt")):
                continue
            fpath = os.path.join(root, fname)
            file_gb = os.path.getsize(fpath) / (1024 ** 3)
            local_family = detect_family(fname)
            max_gb = get_max_checkpoint_gb(local_family)
            if file_gb > max_gb:
                log(f"  Warning: oversized checkpoint {file_gb:.1f}GB > {max_gb}GB — generating anyway (Dynamic VRAM will offload): {fname}")
            found.append((fname, fpath, local_family))

    log(f"  Found {len(found)} local checkpoints for combo testing")

    for fname, _fpath, family in found:
        settings = get_settings(family)
        prompt, negative = get_random_prompt(family, nsfw_mode=cfg.nsfw_mode)
        log(f"  Combo checkpoint: {fname} | Family: {family}")

        model_id = f"local:{fname}"
        version_id = None

        _process_checkpoint_combo(
            _timings, model_id, fname, fname, version_id, "CHECKPOINT",
            family, settings, prompt, negative, None,
            config.combo_lora_strength, local_loras, local_embeddings,
        )
        time.sleep(2)

    log("Combination-testing mode complete.")


def fetch_hf_liked_datasets(username, token=None):
    """Return the dataset repo IDs liked by the given HF user.

    Uses huggingface_hub HfApi.list_liked_repos. Public likes are visible
    without a token; a token is required to see private likes.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        log("  huggingface_hub not available — cannot list HF liked repos")
        return []
    try:
        likes = HfApi().list_liked_repos(user=username, token=token)
        datasets = list(getattr(likes, "datasets", []) or [])
        log(f"HF liked datasets for {username}: {len(datasets)}")
        return datasets
    except Exception as e:
        log(f"HF liked-repos fetch failed for {username}: {e}")
        return []


def process_hf_datasets():
    """Download HF datasets liked by cfg.hf_username into cfg.hf_datasets_dir.

    Skip/completed tracking: datasets already recorded in
    completed_datasets.txt, or already present on disk, are skipped; newly
    downloaded datasets are marked completed so they are not re-fetched on
    subsequent runs. Returns the number of datasets downloaded this run.
    """
    hf_token = os.getenv("HF_TOKEN", "")
    if not cfg.hf_username:
        log("HF: no hf_username configured — skipping HF dataset downloads")
        return 0

    log(f"Processing HF datasets liked by user {cfg.hf_username} ...")
    completed = load_completed_datasets()
    downloaded = 0
    for repo_id in fetch_hf_liked_datasets(cfg.hf_username, token=hf_token or None):
        if repo_id in completed:
            log(f"  Skipping dataset {repo_id} (already completed)")
            continue
        local_dir_name = repo_id.replace("/", "_")
        dest_dir = os.path.join(cfg.hf_datasets_dir, local_dir_name)
        if os.path.isdir(dest_dir) and any(os.scandir(dest_dir)):
            log(f"  Skipping dataset {repo_id} (already present at {dest_dir})")
            mark_completed_dataset(repo_id)
            continue
        if download_hf_dataset(repo_id, local_dir_name=local_dir_name, token=hf_token or None):
            mark_completed_dataset(repo_id)
            downloaded += 1
        else:
            log(f"  {repo_id}: HF dataset download failed — will retry on next run")
    log(f"HF dataset processing complete — {downloaded} dataset(s) downloaded.")
    return downloaded


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def _write_timing_report():
    """Write per-model download/generation timings to a JSON report + log a summary."""
    if not _timings:
        return
    models = list(_timings.values())
    n = len(models)
    by_status: dict[str, int] = {}
    total_download = total_generation = total_api = total_upscale = total_wall = 0.0
    gen_times = []
    dl_times = []
    for m in models:
        by_status[m["status"]] = by_status.get(m["status"], 0) + 1
        total_download += m["download_s"]
        total_generation += m["generation_s"]
        total_api += m["api_fetch_s"]
        total_upscale += m["upscale_s"]
        total_wall += m["total_s"]
        if m["generation_s"] > 0:
            gen_times.append(m["generation_s"])
        if m["download_s"] > 0:
            dl_times.append(m["download_s"])

    def avg(xs):
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "comfyui_base": cfg.comfyui_base,
            "dry_run": cfg.dry_run,
            "family_filter": cfg.family_filter or None,
            "limit": cfg.limit,
            "lora_strengths": cfg.lora_strengths,
            "enable_upscaling": cfg.enable_upscaling,
        },
        "models": models,
        "summary": {
            "model_count": n,
            "total_wall_s": round(total_wall, 3),
            "by_status": by_status,
            "api_fetch_s": round(total_api, 3),
            "download_s": round(total_download, 3),
            "generation_s": round(total_generation, 3),
            "upscale_s": round(total_upscale, 3),
            "download_avg_s": avg(dl_times),
            "generation_avg_s": avg(gen_times),
            "slowest_by_total": sorted(models, key=lambda m: m["total_s"], reverse=True)[:10],
            "slowest_generations": sorted(models, key=lambda m: m["generation_s"], reverse=True)[:10],
        },
    }
    report_dir = os.path.dirname(cfg.log_file) or "."
    path = os.path.join(report_dir, "batch_timing_report.json")
    try:
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        log(f"Timing report written: {path} ({n} model(s))")
        s = report["summary"]
        log(f"Timing summary: {n} models | wall {s['total_wall_s']}s | download {s['download_s']}s | "
            f"generation {s['generation_s']}s | api {s['api_fetch_s']}s | upscale {s['upscale_s']}s | "
            f"avg gen {s['generation_avg_s']}s | by status {s['by_status']}")
    except OSError as e:
        log(f"Failed to write timing report: {e}")


def main(config=None):
    """Run the full batch: collect ids, process Civitai models, local scan, HF models."""
    global cfg
    if config is not None:
        cfg = config
        init(cfg)
    if cfg is None:
        raise RuntimeError("batch.main: no config bound — call init() or pass config")

    rotate_log_if_needed()
    os.makedirs(os.path.join(cfg.comfyui_base, "output"), exist_ok=True)
    os.makedirs(cfg.metadata_dir, exist_ok=True)
    os.makedirs(cfg.hf_datasets_dir, exist_ok=True)

    # Date-based output folder: YYYY-MM-DD/ prepended to all output prefixes.
    global _date_folder
    if cfg.output_date_folder:
        _date_folder = datetime.now().strftime("%Y-%m-%d")
        log(f"Output date folder enabled: {_date_folder}/")

    families.load_prompt_pools(cfg)
    log(
        f"Prompt pools loaded: {len(families.PROMPT_POOLS)} family(ies) | "
        f"nsfw={cfg.nsfw_mode} | category order: {cfg.prompt_category_order or '<per family>'}"
    )

    # --- Civitai + HuggingFace profile integration ---
    hf_token = os.getenv("HF_TOKEN", "")
    if hf_token:
        log(f"HF: HF_TOKEN set (user {cfg.hf_username}) — HF models + datasets pipeline enabled")
    else:
        log("HF: no HF_TOKEN — using public HF access only (datasets still downloadable)")

    ordered, skip, completed_models, priority = collect_model_ids(cfg)

    for idx, mid in enumerate(ordered, 1):
        process_model(mid, completed_models=completed_models, priority=priority, skip=skip)
        if idx % 10 == 0 or idx == len(ordered):
            log(f"  Progress: {idx}/{len(ordered)} models processed")
        time.sleep(2)
    log("Civitai batch processing complete.")

    # Scan and test locally-placed models
    scan_local_models()

    # Process HuggingFace models from local files
    process_hf_models()

    # Download HF datasets liked by the configured HF user (SgtStroopwafel)
    hf_datasets_downloaded = process_hf_datasets()

    # Combination-testing mode: local checkpoints x LoRAs x embeddings
    run_combo_mode(cfg)

    _write_timing_report()
    log("All processing complete.")
    send_telegram_message(
        f"*Batch complete* — processed {len(ordered)} Civitai models + local checkpoints, "
        f"{hf_datasets_downloaded} HF dataset(s). "
        f"Check `{cfg.comfyui_base}/output/` for `batch_*.png` renders.")
