"""Application configuration: defaults, JSON config file, env var overrides.

All paths and behaviours that were previously hardcoded in the monolith live
here. Precedence (low -> high):

  1. dataclass field defaults (mirror the old hardcoded values)
  2. JSON config file (--config, default batch_comfytester.config.json)
  3. batch_comfytester.env (simple KEY=value parser, env already set wins)
  4. Environment variables (TELEGRAM_BOT_TOKEN, HF_TOKEN, COMFYUI_BASE, ...)
  5. CLI arguments (--comfyui-base, --dry-run, --family, --ids, --limit, ...)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields

DEFAULT_MAX_CHECKPOINT_GB = {
    "default": 24.0,
    "flux_klein": 24.0,
    "flux_dev": 24.0,
    "wan22_video": 24.0,
    "qwen_image": 24.0,
    "anima": 12.0,
    "krea2": 24.0,
    "pixart": 12.0,
    "ltxv": 24.0,
    "sd15": 12.0,
    "sd21": 12.0,
    "illustrious": 12.0,
    "pony": 12.0,
    "furry": 12.0,
    "anime": 12.0,
    "photorealistic": 12.0,
    "segmind_vega": 12.0,
    "stable_cascade": 12.0,
    "supr": 12.0,
    "depthfm": 12.0,
}

DEFAULT_VAE_COMPATIBILITY = {
    "wan22_video": ["wan2.2_vae.safetensors"],
    "anima": ["anima-turbo-v1.0.safetensors"],
    "qwen_image": ["qwen-image-2.1-vae.safetensors"],
}

# (env var name) -> (config attribute, converter)
ENV_OVERRIDES = {
    "TELEGRAM_BOT_TOKEN": ("telegram_bot_token", str),
    "TELEGRAM_HOME_CHANNEL": ("telegram_home_channel", str),
    "HF_TOKEN": ("hf_token", str),
    "CIVITAI_TOKEN_FILE": ("civitai_token_file", str),
    "CIVITAI_SCRIPT": ("civitai_script", str),
    "COMFYUI_BASE": ("comfyui_base", str),
    "LOG_FILE": ("log_file", str),
    "NSFW_MODE": ("nsfw_mode", bool),
    "DRY_RUN": ("dry_run", bool),
    "MAX_CHECKPOINT_GB": ("max_checkpoint_gb", json.loads),
}

# CLI argument -> config attribute (handled in __main__.build_argparser)


def _parse_bool(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class AppConfig:
    # --- Paths (formerly hardcoded /home/rizzo/... literals) ---
    comfyui_base: str = "/home/rizzo/ComfyUI"
    civitai_script: str = "/home/rizzo/.hermes/skills/civitai-api/scripts/civitai.py"
    log_file: str = "/home/rizzo/Work/batch_comfytester/comfytester.log"
    log_max_bytes: int = 5 * 1024 * 1024  # 5 MB
    log_backup_count: int = 5
    prompt_pools_dir: str = "/home/rizzo/ComfyUI/prompt_pools"
    model_family_map: str = "/home/rizzo/ComfyUI/prompt_pools/model_family_map.json"
    env_file: str = "/home/rizzo/Work/batch_comfytester/batch_comfytester.env"
    civitai_token_file: str = "/home/rizzo/.civitai/comfy-api-key"
    civitai_api_base: str = "https://civitai.com/api/v1"
    civitai_min_interval: float = 1.2  # seconds between Civitai API requests (~50/min)
    civitai_max_retries: int = 3       # max retries with backoff on 429/5xx (reduced from 5)
    civitai_backoff_cap: int = 15      # max backoff seconds per retry (reduced from 30)
    civitai_skip_collections: bool = True  # skip collections fetch on persistent 429 (they're the main rate-limit source)
    hf_datasets_dir: str = "/home/rizzo/ComfyUI/models/datasets"
    hf_username: str = "SgtStroopwafel"
    completed_file: str = "/home/rizzo/Work/batch_comfytester/completed_models.txt"
    priority_file: str = "/home/rizzo/Work/batch_comfytester/priority_ids.txt"
    skip_file: str = "/home/rizzo/Work/batch_comfytester/skip_ids.txt"
    failed_file: str = "/home/rizzo/Work/batch_comfytester/failed_models.txt"
    state_file: str = "/home/rizzo/Work/batch_comfytester/batch_state.json"
    completed_datasets_file: str = "/home/rizzo/Work/batch_comfytester/completed_datasets.txt"
    metadata_dir: str = "/home/rizzo/ComfyUI/output/metadata"
    lock_file: str = "/home/rizzo/Work/batch_comfytester/batch_comfytester.lock"
    id_list_path: str = "/home/rizzo/.hermes/cache/documents/doc_8418378d742b_list.txt"
    hf_list_path: str = "/home/rizzo/Work/batch_comfytester/hf_models.txt"
    restart_log_file: str = "/home/rizzo/comfyui_restart.log"
    comfyui_venv_python: str = "/home/rizzo/ComfyUI/venv/bin/python"
    comfyui_main_py: str = "/home/rizzo/ComfyUI/main.py"
    comfyui_url: str = "http://127.0.0.1:8188"

    # --- Behaviour / limits ---
    max_checkpoint_gb: dict = None  # populated in __post_init__
    base_ckpt: str = "sdXL_base_v1.0.safetensors"
    upscaler_model: str = "4xUltrasharp_v10.pt"
    enable_upscaling: bool = True
    upscale_factor: int = 4
    lora_strengths: list = None  # populated in __post_init__
    vae_compatibility: dict = None  # populated in __post_init__
    controlnet_enabled: bool = True
    dry_run: bool = False
    nsfw_mode: bool = False

    # --- Combination testing ---
    combo_mode: bool = False
    combo_max_loras_per_ckpt: int = 2
    combo_lora_strength: float = 0.8
    combo_max_combos_per_ckpt: int = 5
    combo_embed_negatives: list = None  # embedding filenames to append to negative prompt
    combo_output_dir: str = "combos"  # base output subfolder for combo generations

    # --- Normal testing output subfolders (per model type) ---
    ckpt_output_dir: str = "checkpoints"
    lora_output_dir: str = "loras"
    vae_output_dir: str = "vaes"
    cn_output_dir: str = "controlnet"
    qwen_output_dir: str = "qwen"
    anima_output_dir: str = "anima"
    wan22_output_dir: str = "wan22"

    # --- Prompt pools ---
    prompt_pool_categories: dict = None  # family -> ordered category list (defaults per dialect)
    prompt_template: str = None          # global template override
    prompt_negative_category: str = "negative"
    prompt_lora_trigger_injection: bool = True
    prompt_pool_auto_create: bool = True
    prompt_category_order: list = None   # global category order override for all families
    known_lora_triggers: dict = None     # lora_name -> [trigger words], overrides filename-stem guess

    # --- Output sorting ---
    output_date_folder: bool = False  # prepend YYYY-MM-DD/ to all output prefixes

    # --- CLI-driven filters (no defaults; set via --family / --ids / --limit) ---
    family_filter: str = ""
    id_filter: list = None  # populated in __post_init__
    limit: int = None

    # --- Telegram ---
    telegram_bot_token: str = ""
    telegram_home_channel: str = "-1004470891735"
    telegram_max_bytes: int = 8 * 1024 * 1024

    # --- HuggingFace ---
    hf_token: str = ""

    # --- Fallback prompts ---
    default_prompt: str = (
        "a small young petite skinny android girl walking through a cyberpunk cityscape, "
        "neon signs, rain-slicked streets, futuristic, high detail, 8k"
    )
    default_negative: str = (
        "blurry, low quality, distorted, worst quality, lowres, bad anatomy, bad hands, "
        "text, error, missing fingers, extra digit, fewer, cropped, worst quality, low "
        "quality, normal quality, jpeg artifacts, signature, watermark, username, bad signature"
    )
    default_prompt_nsfw: str = (
        "photorealistic, photorealistic photo, nude, topless, bare breasts, highly detailed "
        "skin, skin texture, pores, 8k UHD, cinematic lighting, masterpiece, best quality, "
        "film grain, raw photo, detailed face, suggestive pose, adult, uncensored"
    )
    default_negative_nsfw: str = (
        "blurry, low quality, distorted, worst quality, lowres, bad anatomy, bad hands, "
        "text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, "
        "signature, watermark, username, artist name, censored, clothed, dressed, covered, "
        "bra, shirt, SFW, safe, modest, underwear, panties, swimsuit, bikini"
    )

    def __post_init__(self):
        if self.max_checkpoint_gb is None:
            self.max_checkpoint_gb = dict(DEFAULT_MAX_CHECKPOINT_GB)
        if self.lora_strengths is None:
            self.lora_strengths = [0.4, 0.8, 1.2]
        if self.vae_compatibility is None:
            self.vae_compatibility = dict(DEFAULT_VAE_COMPATIBILITY)
        if self.id_filter is None:
            self.id_filter = []
        if self.combo_embed_negatives is None:
            self.combo_embed_negatives = []
        if self.prompt_pool_categories is None:
            self.prompt_pool_categories = {}
        if self.prompt_category_order is None:
            self.prompt_category_order = []
        if self.known_lora_triggers is None:
            self.known_lora_triggers = {
                # Scottzilla Systems MCNL v1 — Multi Concept NSFW LoRA for Qwen Image Edit
                # Matches both the HF repo name and the likely downloaded filename stem.
                "scottzillasystems/qwen-image-edit-plus-nsfw-lora": [
                    "nsfw", "nipples", "vagina", "penis", "missionary",
                    "cowgirl", "reversecowgirlpov", "blowjob", "cum_on_face",
                    "creamp1e", "l1ck",
                ],
                "qwen-image-edit-plus-nsfw-lora": [
                    "nsfw", "nipples", "vagina", "penis", "missionary",
                    "cowgirl", "reversecowgirlpov", "blowjob", "cum_on_face",
                    "creamp1e", "l1ck",
                ],
            }


def load_config(path=None):
    """Load AppConfig: defaults, optional JSON config file, .env, env overrides."""
    cfg = AppConfig()

    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = json.load(f)
        known = {field.name for field in fields(AppConfig)}
        for key, value in data.items():
            if key in known:
                setattr(cfg, key, value)
            else:
                print(f"Ignoring unknown config key: {key}")

    # batch_comfytester.env (setdefault semantics: an already-set env var wins)
    if os.path.exists(cfg.env_file):
        with open(cfg.env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

    # Environment variable overrides
    for env_name, (attr, conv) in ENV_OVERRIDES.items():
        value = os.environ.get(env_name)
        if value is not None and value != "":
            converted = _parse_bool(value) if conv is bool else conv(value)
            setattr(cfg, attr, converted)

    return cfg
