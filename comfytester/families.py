"""Model family detection, prompt pools, generation settings and VRAM limits."""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass

from .config import AppConfig
from .utils import log

cfg: AppConfig | None = None

FAMILY_SETTINGS = {
    "photorealistic": {
        "steps": 25,
        "cfg": 7,
        "sampler": "dpmpp_2m_sde_gpu",
        "scheduler": "karras",
        "resolution": [832, 1216],
        "aspect_ratio": "portrait",
        "seed_range": [1, 999999],
    },
    "anime": {
        "steps": 25,
        "cfg": 5,
        "sampler": "euler_ancestral",
        "scheduler": "normal",
        "resolution": [896, 1152],
        "aspect_ratio": "portrait",
        "seed_range": [1, 999999],
    },
    "pony": {
        "steps": 25,
        "cfg": 7,
        "sampler": "euler_ancestral",
        "scheduler": "normal",
        "resolution": [1024, 1024],
        "aspect_ratio": "square",
        "seed_range": [1, 999999],
    },
    "illustrious": {
        "steps": 25,
        "cfg": 5,
        "sampler": "euler_a",
        "scheduler": "normal",
        "resolution": [1536, 1536],
        "aspect_ratio": "square",
        "seed_range": [1, 999999],
    },
    "furry": {
        "steps": 25,
        "cfg": 4,
        "sampler": "dpmpp_2m_sde_gpu",
        "scheduler": "karras",
        "resolution": [1024, 1024],
        "aspect_ratio": "square",
        "seed_range": [1, 999999],
    },
    "qwen_image": {
        "steps": 40,
        "cfg": 1,
        "sampler": "euler",
        "scheduler": "simple",
        "resolution": [2048, 2048],
        "aspect_ratio": "square",
        "seed_range": [1, 999999],
    },
    "anima": {
        "steps": 12,
        "cfg": 1,
        "sampler": "er_sde",
        "scheduler": "simple",
        "resolution": [1024, 1024],
        "aspect_ratio": "square",
        "seed_range": [1, 999999],
    },
    "krea2": {
        "steps": 8,
        "cfg": 1,
        "sampler": "er_sde",
        "scheduler": "simple",
        "resolution": [1280, 720],
        "aspect_ratio": "landscape",
        "seed_range": [1, 999999],
    },
    "flux_klein": {
        "steps": 4,
        "cfg": 1,
        "sampler": "euler",
        "scheduler": "normal",
        "resolution": [1024, 1024],
        "aspect_ratio": "square",
        "seed_range": [1, 999999],
    },
    "flux_dev": {
        "steps": 30,
        "cfg": 1,
        "sampler": "euler",
        "scheduler": "normal",
        "resolution": [1024, 1024],
        "aspect_ratio": "square",
        "seed_range": [1, 999999],
    },
    "wan22_video": {
        "steps": 50,
        "cfg": 5,
        "sampler": "unipc",
        "scheduler": "normal",
        "resolution": [1280, 704],
        "aspect_ratio": "landscape",
        "seed_range": [1, 999999],
    },
    "pixart": {
        "steps": 20,
        "cfg": 2,
        "sampler": "euler_a",
        "scheduler": "simple",
        "resolution": [1024, 1024],
        "aspect_ratio": "square",
        "seed_range": [1, 999999],
    },
    "ltxv": {
        "steps": 30,
        "cfg": 3,
        "sampler": "euler",
        "scheduler": "normal",
        "resolution": [768, 512],
        "aspect_ratio": "landscape",
        "seed_range": [1, 999999],
    },
    "sd15": {
        "steps": 25,
        "cfg": 7,
        "sampler": "dpmpp_2m_sde_gpu",
        "scheduler": "karras",
        "resolution": [512, 512],
        "aspect_ratio": "square",
        "seed_range": [1, 999999],
    },
    "sd21": {
        "steps": 25,
        "cfg": 7,
        "sampler": "dpmpp_2m_sde_gpu",
        "scheduler": "karras",
        "resolution": [512, 512],
        "aspect_ratio": "square",
        "seed_range": [1, 999999],
    },
    "segmind_vega": {
        "steps": 15,
        "cfg": 1,
        "sampler": "euler",
        "scheduler": "simple",
        "resolution": [1024, 1024],
        "aspect_ratio": "square",
        "seed_range": [1, 999999],
    },
    "stable_cascade": {
        "steps": 20,
        "cfg": 7,
        "sampler": "euler",
        "scheduler": "normal",
        "resolution": [1024, 1024],
        "aspect_ratio": "square",
        "seed_range": [1, 999999],
    },
    "supr": {
        "steps": 20,
        "cfg": 7,
        "sampler": "euler",
        "scheduler": "normal",
        "resolution": [1024, 1024],
        "aspect_ratio": "square",
        "seed_range": [1, 999999],
    },
    "depthfm": {
        "steps": 20,
        "cfg": 7,
        "sampler": "euler",
        "scheduler": "normal",
        "resolution": [512, 512],
        "aspect_ratio": "square",
        "seed_range": [1, 999999],
    },
}

# Model name -> family hints (keywords in model name/tags)
FAMILY_KEYWORDS = {
    "anime": ["anime", "manga", "2d", "cel", "niji", "anything", "counterfeit", "meinamix", "dreamshaper", "nsfw-gen-anime"],
    "pony": ["pony", "mlp", "my little pony"],
    "illustrious": ["illustrious", "ilxl", "novelai"],
    "furry": ["furry", "anthro", "yiff"],
    "photorealistic": ["realistic", "photo", "cinematic", "portrait", "real", "photography", "heartsync", "nsfw-uncensored", "nsfw-gen", "newrealityxl", "nsfwmodel", "nsfwv2", "z-image", "ideogram"],
    "qwen_image": ["qwen", "qwen_image", "qwen-image", "qwen-image-2.1"],
    "anima": ["anima", "circle-stone"],
    "krea2": ["krea2", "krea-2", "krea_2"],
    "flux_klein": ["klein", "flux2", "flux.2"],
    "flux_dev": ["flux_dev", "flux1-dev", "flux.1-dev", "flux1_dev", "heartsync", "lustly"],
    "wan22_video": ["wan2.2", "wan22", "ti2v"],
    "pixart": ["pixart", "pixart-sigma", "pixart_sigma"],
    "ltxv": ["ltxv", "ltx-video", "ltx_video", "pinkcherry"],
    "sd15": ["sd1.5", "sd15", "stable diffusion 1.5"],
    "sd21": ["sd2.1", "sd21", "stable diffusion 2.1"],
    "segmind_vega": ["segmind", "vega", "segmind_vega", "segmind-vega"],
    "stable_cascade": ["stable_cascade", "stable-cascade", "cascade"],
    "supr": ["supr", "supirr"],
    "depthfm": ["depthfm", "depth-fm", "depth_fm"],
}

# --- Prompt dialects ------------------------------------------------------------------
# How each family's text encoder wants to be addressed. CLIP-based families
# (SD1.x, SDXL, Pony, Illustrious, anime...) prefer comma-separated tags with
# important terms first and ComfyUI emphasis syntax (word:weight). T5-based
# families (Flux, Qwen, Anima) read full natural-language sentences and ignore
# weights entirely.
DIALECT_CLIP_TAG = "clip_tag"
DIALECT_T5 = "t5_natural"
DIALECT_VIDEO = "video_natural"

FAMILY_DIALECTS: dict[str, str] = {
    "sd15": DIALECT_CLIP_TAG,
    "sd21": DIALECT_CLIP_TAG,
    "pony": DIALECT_CLIP_TAG,
    "illustrious": DIALECT_CLIP_TAG,
    "furry": DIALECT_CLIP_TAG,
    "anime": DIALECT_CLIP_TAG,
    "photorealistic": DIALECT_CLIP_TAG,
    "segmind_vega": DIALECT_CLIP_TAG,
    "stable_cascade": DIALECT_CLIP_TAG,
    "supr": DIALECT_CLIP_TAG,
    "depthfm": DIALECT_CLIP_TAG,
    "krea2": DIALECT_CLIP_TAG,
    "pixart": DIALECT_T5,
    "qwen_image": DIALECT_T5,
    "anima": DIALECT_T5,
    "flux_dev": DIALECT_T5,
    "flux_klein": DIALECT_T5,
    "wan22_video": DIALECT_VIDEO,
    "ltxv": DIALECT_VIDEO,
}

# Default category order per dialect. Categories missing from a pool are
# skipped, so partial pools never break a run.
DEFAULT_CATEGORY_ORDER: dict[str, list[str]] = {
    DIALECT_CLIP_TAG: ["subject", "medium", "style", "quality", "lighting", "color"],
    DIALECT_T5: ["subject", "setting", "lighting", "optics", "quality"],
    DIALECT_VIDEO: ["subject", "action", "setting", "camera", "lighting", "quality"],
}

DEFAULT_TEMPLATE: dict[str, str] = {
    DIALECT_CLIP_TAG: "{subject}, {medium}, {style}, {quality}, {lighting}, {color}",
    DIALECT_T5: "{subject} {setting} {lighting} {optics} {quality}.",
    DIALECT_VIDEO: "{subject}. {action} in {setting}, {camera} work, {lighting}, {quality}.",
}

NEGATIVE_CATEGORY = "negative"


@dataclass(frozen=True)
class PromptEntry:
    """One prompt building block: the text plus an emphasis weight.

    `weight` is rendered as ComfyUI emphasis `(text:weight)` for CLIP dialects
    and ignored for T5/video dialects, whose encoders do not support it.
    """

    text: str
    weight: float = 1.0


@dataclass
class PromptPool:
    """A family's prompt pool: versioned, dialect-tagged, categorized.

    `nsfw_mode` controls whether library entries tagged "nsfw" are eligible
    for selection. When False, NSFW-tagged entries are excluded so SFW pools
    never emit explicit content.
    """

    version: int
    family: str
    dialect: str
    template: str
    categories: dict[str, list[PromptEntry]]
    source: str | None = None
    nsfw_mode: bool = False


FAMILY_MAP: dict = {}
PROMPT_POOLS: dict[str, PromptPool] = {}
PROMPT_LIBRARIES: dict[str, dict] = {}  # library_name -> {"category": ..., "entries": [...]}


def init(config: AppConfig):
    global cfg, FAMILY_MAP
    cfg = config
    if config.model_family_map:
        loaded = load_family_map(config.model_family_map)
        if loaded:
            FAMILY_MAP = loaded


def load_family_map(path):
    """Load the model family map JSON (family -> keyword list)."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        log(f"Could not load model family map {path}: {e}")
        return {}


def load_prompt_libraries(config: AppConfig | None = None) -> None:
    """Load all category library files from prompt_pools/libraries/.

    Each library file is a JSON object:
        {"category": "subjects", "entries": [{"text": "...", "tags": [...], "weight": 1.0}, ...]}

    Libraries are shared across families; entries carry optional tags so a
    family pool can filter which entries apply to it.
    """
    global PROMPT_LIBRARIES, cfg
    if config is not None:
        cfg = config
    if cfg is None:
        raise RuntimeError("load_prompt_libraries: no config bound — call families.init() first")
    PROMPT_LIBRARIES = {}
    lib_dir = os.path.join(cfg.prompt_pools_dir, "libraries")
    if not os.path.isdir(lib_dir):
        log(f"Prompt library directory not found: {lib_dir}")
        return
    for fname in sorted(os.listdir(lib_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(lib_dir, fname)
        data = _load_pool_file(path)
        if data is None:
            continue
        lib_name = data.get("category", os.path.splitext(fname)[0])
        PROMPT_LIBRARIES[lib_name] = data
        log(f"Prompt library loaded: {lib_name} ({len(data.get('entries', []))} entries)")


def get_family_dialect(family: str) -> str:
    """Return the prompt dialect for a family (CLIP tags by default)."""
    return FAMILY_DIALECTS.get(family, DIALECT_CLIP_TAG)


def _entry_from_value(value) -> PromptEntry:
    """Parse a pool entry: PromptEntry passthrough, plain string, or {"text": ..., "weight": ...}."""
    if isinstance(value, PromptEntry):
        return value
    if isinstance(value, dict):
        return PromptEntry(text=str(value.get("text", "")), weight=float(value.get("weight", 1.0)))
    return PromptEntry(text=str(value))


def _parse_category_items(items, family: str) -> list:
    """Parse a list of raw category items.

    Inline entries (strings or {"text":..., "weight":...} dicts) become
    PromptEntry objects. Library reference dicts ({"library":..., "tags":...})
    are kept as dicts and resolved later by _resolve_category_value.
    """
    parsed: list = []
    for v in items:
        if isinstance(v, dict) and "library" in v:
            parsed.append(v)
        else:
            parsed.append(_entry_from_value(v))
    return parsed


def render_entry(entry: PromptEntry, dialect: str) -> str:
    """Render a single entry for the target dialect.

    CLIP dialects get ComfyUI emphasis syntax `(text:weight)`. T5 and video
    dialects receive plain text — their encoders do not support attention
    weights, and stuffing `(word:1.2)` into a T5 prompt only adds noise.
    """
    text = entry.text.strip()
    if not text:
        return ""
    if dialect == DIALECT_CLIP_TAG and entry.weight != 1.0:
        return f"({text}:{entry.weight})"
    return text


def _resolve_library_category(lib_ref: dict, family: str, nsfw_mode: bool = False) -> list[PromptEntry]:
    """Resolve a library category reference to a list of PromptEntry.

    `lib_ref` is a dict like {"library": "subjects", "tags": ["photorealistic"]}.
    Entries are included when they carry no tags, the "all" tag, or a tag
    matching the target family.

    Entries tagged "nsfw" are excluded unless `nsfw_mode` is True, so SFW
    pools never emit explicit content.

    If `lib_ref` carries additional tags (e.g. {"nsfw"}), entries must share
    at least one tag with the ref to pass.
    """
    lib_name = lib_ref.get("library", "")
    ref_tags = set(lib_ref.get("tags", []))
    lib_data = PROMPT_LIBRARIES.get(lib_name)
    if not lib_data:
        log(f"Prompt library '{lib_name}' not found — skipping category")
        return []
    entries: list[PromptEntry] = []
    for raw in lib_data.get("entries", []):
        entry_tags = set(raw.get("tags", []))
        # NSFW-gated entries: only eligible when nsfw_mode is enabled.
        if "nsfw" in entry_tags and not nsfw_mode:
            continue
        if "all" in entry_tags or family in entry_tags or not entry_tags:
            # Ref tag intersection: entry must share at least one tag with ref.
            if ref_tags and not entry_tags & ref_tags:
                continue
            entries.append(_entry_from_value(raw))
    return entries


def _resolve_category_value(value, family: str, nsfw_mode: bool = False) -> list[PromptEntry]:
    """Resolve a category value: inline list, library reference dict, or single entry.

    A list may contain a mix of inline entries (strings/dicts) and library
    reference dicts ({"library": "...", "tags": [...]});
    each item is resolved individually.
    """
    if isinstance(value, list):
        entries: list[PromptEntry] = []
        for v in value:
            if isinstance(v, dict) and "library" in v:
                entries.extend(_resolve_library_category(v, family, nsfw_mode))
            else:
                entry = _entry_from_value(v)
                if entry.text.strip():
                    entries.append(entry)
        return entries
    if isinstance(value, dict):
        if "library" in value:
            return _resolve_library_category(value, family, nsfw_mode)
        entry = _entry_from_value(value)
        return [entry] if entry.text.strip() else []
    entry = _entry_from_value(value)
    return [entry] if entry.text.strip() else []


def _load_pool_file(path: str) -> dict | None:
    """Load one pool JSON file. Returns None if missing or unreadable."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError) as e:
        log(f"Invalid prompt pool {path}: {e}")
        return None


def _adapt_v1_pool(positive, negative, dialect: str) -> PromptPool:
    """Wrap the legacy flat {"positive": [...], "negative": [...]} format."""
    categories = {
        "subject": [_entry_from_value(v) for v in (positive or []) if str(v).strip()],
        NEGATIVE_CATEGORY: [_entry_from_value(v) for v in (negative or []) if str(v).strip()],
    }
    template = DEFAULT_TEMPLATE.get(dialect, DEFAULT_TEMPLATE[DIALECT_CLIP_TAG])
    return PromptPool(
        version=1,
        family="",
        dialect=dialect,
        template=template,
        categories=categories,
    )


def default_pool_for_family(family: str, nsfw: bool = False) -> PromptPool:
    """Build the default v2 pool for a family when no pool file exists."""
    dialect = get_family_dialect(family)
    template = DEFAULT_TEMPLATE.get(dialect, DEFAULT_TEMPLATE[DIALECT_CLIP_TAG])
    categories: dict[str, list[PromptEntry]] = {
        "subject": [
            PromptEntry(cfg.default_prompt_nsfw if (cfg and cfg.nsfw_mode) else cfg.default_prompt)
        ],
        NEGATIVE_CATEGORY: [
            PromptEntry(
                cfg.default_negative_nsfw if (cfg and cfg.nsfw_mode) else cfg.default_negative
            )
        ],
    }
    return PromptPool(
        version=2,
        family=family,
        dialect=dialect,
        template=template,
        categories=categories,
        nsfw_mode=nsfw,
    )


def _write_pool_file(path: str, pool: PromptPool) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "version": pool.version,
        "family": pool.family,
        "dialect": pool.dialect,
        "template": pool.template,
        "categories": {
            name: [{"text": entry.text, "weight": entry.weight} for entry in entries]
            for name, entries in pool.categories.items()
        },
    }
    with open(path, "w") as af:
        json.dump(payload, af, indent=2)


def load_prompt_pools(config: AppConfig | None = None) -> None:
    """Load prompt pools for every known family.

    Supports the legacy flat format {"positive": [...], "negative": [...]} and
    the versioned categorized format:
        {"version": 2, "family": ..., "dialect": ..., "template": "...",
         "categories": {"subject": [...], "negative": [...], ...}}
    Missing or invalid pools are auto-created from the config defaults so
    every family always has a pool to draw from.
    """
    global PROMPT_POOLS, cfg
    if config is not None:
        cfg = config
    if cfg is None:
        raise RuntimeError("load_prompt_pools: no config bound — call families.init() first")
    load_prompt_libraries(cfg)
    PROMPT_POOLS = {}
    pool_dir = cfg.prompt_pools_dir
    pool_suffix = "_nsfw" if cfg.nsfw_mode else ""
    for family in sorted(FAMILY_SETTINGS):
        pool_path = os.path.join(pool_dir, f"{family}{pool_suffix}.json")
        data = _load_pool_file(pool_path)
        dialect = get_family_dialect(family)
        if data is None:
            if cfg.prompt_pool_auto_create:
                pool = default_pool_for_family(family, cfg.nsfw_mode)
                _write_pool_file(pool_path, pool)
                log(f"Auto-created prompt pool: {pool_path}")
            else:
                log(f"No prompt pool found for family: {family}{pool_suffix}")
                continue
        else:
            version = int(data.get("version", 1))
            if version >= 2:
                categories = {
                    name: _parse_category_items(cats or [], family)
                    for name, cats in (data.get("categories") or {}).items()
                }
                pool = PromptPool(
                    version=version,
                    family=str(data.get("family", family)),
                    dialect=str(data.get("dialect", dialect)),
                    template=data.get("template")
                    or DEFAULT_TEMPLATE.get(dialect, DEFAULT_TEMPLATE[DIALECT_CLIP_TAG]),
                    categories=categories,
                    source=pool_path,
                    nsfw_mode=bool(pool_suffix),
                )
            else:
                pool = _adapt_v1_pool(data.get("positive", []), data.get("negative", []), dialect)
                pool.source = pool_path
            if not any(entries for entries in pool.categories.values()):
                log(f"Prompt pool {pool_path} is empty — falling back to defaults")
                pool = default_pool_for_family(family, cfg.nsfw_mode)
        PROMPT_POOLS[family] = pool


def detect_family(model_name, model_tags=None, family_keywords=FAMILY_KEYWORDS):
    """Detect model family from name and optional tags."""
    name_lower = (model_name or "").lower()
    tags_str = " ".join(model_tags or []).lower() if model_tags else ""
    combined = f"{name_lower} {tags_str}"

    scores = {}
    for family, keywords in family_keywords.items():
        score = sum(1 for kw in keywords if kw in combined)
        if score > 0:
            scores[family] = score

    if scores:
        return max(scores, key=scores.get)
    return "photorealistic"  # default fallback


def build_prompt_from_pool(
    pool: PromptPool,
    category_order: list[str] | None = None,
    negative_category: str = NEGATIVE_CATEGORY,
    nsfw_mode: bool = False,
) -> tuple[str, str]:
    """Assemble (positive, negative) by sampling one entry per category.

    `category_order` overrides the pool's dialect default. Missing categories
    are skipped so a partial pool never breaks a run; unreplaced placeholders
    are stripped from the template.

    `nsfw_mode` controls whether library entries tagged "nsfw" are eligible
    for selection.
    """
    dialect = pool.dialect
    order = category_order or DEFAULT_CATEGORY_ORDER.get(
        dialect, DEFAULT_CATEGORY_ORDER[DIALECT_CLIP_TAG]
    )
    template = pool.template or DEFAULT_TEMPLATE.get(dialect, "{subject}")

    parts: dict[str, str] = {}
    for category in order:
        raw = pool.categories.get(category)
        if not raw:
            continue
        entries = _resolve_category_value(raw, pool.family, nsfw_mode)
        if not entries:
            continue
        entry = random.choice(entries)
        rendered = render_entry(entry, dialect)
        if rendered:
            parts[category] = rendered

    for category, text in parts.items():
        template = template.replace("{" + category + "}", text)
    positive = re.sub(r"\{[^}]*\}", "", template)
    positive = re.sub(r"\s+", " ", positive).strip(", ")

    negative_entries = pool.categories.get(negative_category)
    negative = ""
    if negative_entries:
        resolved_negative = _resolve_category_value(negative_entries, pool.family, nsfw_mode)
        if resolved_negative:
            negative = render_entry(random.choice(resolved_negative), dialect)

    return positive, negative


def get_random_prompt(
    family: str = "photorealistic",
    category_order: list[str] | None = None,
    negative_category: str = NEGATIVE_CATEGORY,
    nsfw_mode: bool = False,
) -> tuple[str, str]:
    """Get a (positive, negative) prompt for a family from its pool.

    Falls back to AppConfig.default_prompt / default_negative when the pool
    is missing or empty.

    `nsfw_mode` controls whether library entries tagged "nsfw" are eligible
    for selection.
    """
    pool = PROMPT_POOLS.get(family)
    if pool and any(pool.categories.values()):
        return build_prompt_from_pool(pool, category_order, negative_category,
                                      nsfw_mode=nsfw_mode or pool.nsfw_mode)
    return (
        cfg.default_prompt_nsfw if (cfg and cfg.nsfw_mode) else cfg.default_prompt,
        cfg.default_negative_nsfw if (cfg and cfg.nsfw_mode) else cfg.default_negative,
    )


def inject_lora_triggers(prompt: str, lora_names: list[str], dialect: str = DIALECT_CLIP_TAG) -> str:
    """Append LoRA trigger words to a prompt.

    Checks `cfg.known_lora_triggers` for documented trigger words per LoRA
    name (e.g. Scottzilla MCNL v1 NSFW triggers). Falls back to deriving the
    trigger from the LoRA filename stem. CLIP dialects weight the trigger
    `(word:1.2)`; T5/video dialects get plain text at the end of the prompt.
    """
    if not lora_names:
        return prompt
    triggers = []
    known = cfg.known_lora_triggers if cfg else None
    if not known:
        known = {}
    for name in lora_names:
        stem = os.path.splitext(os.path.basename(name))[0]
        documented = known.get(name) or known.get(stem)
        if documented:
            trigger_words = documented
        else:
            trigger_words = [stem.replace("_", " ").replace("-", " ")]
        for trigger in trigger_words:
            if dialect == DIALECT_CLIP_TAG:
                triggers.append(f"({trigger}:1.2)")
            else:
                triggers.append(trigger)
    return prompt + ", " + ", ".join(triggers)


def get_settings(family):
    """Get generation settings for a family."""
    return FAMILY_SETTINGS.get(family, FAMILY_SETTINGS["photorealistic"])


def get_max_checkpoint_gb(family):
    """Get the VRAM limit for a specific family.

    Returns float('inf') if the limit is <= 0 (size check disabled).
    """
    limit = cfg.max_checkpoint_gb.get(family, cfg.max_checkpoint_gb.get("default", float("inf")))
    if limit <= 0:
        return float("inf")
    return limit


def get_vae_list_for_family(family):
    """Return the list of VAEs to try for a given model family."""
    if family in cfg.vae_compatibility:
        return cfg.vae_compatibility[family]
    # Default: use checkpoint's own VAE
    return [None]


def vae_file_exists(vae_name):
    """Check if a VAE file exists in ComfyUI's VAE directory."""
    if vae_name is None:
        return True  # checkpoint's own VAE always "exists"
    vae_path = os.path.join(cfg.comfyui_base, "models", "vae", vae_name)
    return os.path.exists(vae_path)
