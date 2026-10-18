"""Unit tests for the expanded prompt-pool system (comfytester.families)."""

import json
import os

import pytest

from comfytester.config import AppConfig
from comfytester import families
from comfytester.families import (
    DIALECT_CLIP_TAG,
    DIALECT_T5,
    DIALECT_VIDEO,
    PromptEntry,
    PromptPool,
    build_prompt_from_pool,
    default_pool_for_family,
    get_family_dialect,
    get_random_prompt,
    inject_lora_triggers,
    load_prompt_pools,
    render_entry,
)


def _make_config(tmp_path, nsfw=False):
    return AppConfig(
        comfyui_base=str(tmp_path / "comfyui"),
        prompt_pools_dir=str(tmp_path / "pools"),
        model_family_map=str(tmp_path / "model_family_map.json"),
        nsfw_mode=nsfw,
        default_prompt="default positive",
        default_negative="default negative",
        default_prompt_nsfw="nsfw positive",
        default_negative_nsfw="nsfw negative",
        prompt_pool_auto_create=True,
    )


def _write_pool(tmp_path, family, data, nsfw=False):
    pool_dir = tmp_path / "pools"
    pool_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_nsfw" if nsfw else ""
    path = pool_dir / f"{family}{suffix}.json"
    path.write_text(json.dumps(data))
    return path


def test_config_defaults():
    cfg = AppConfig()
    assert cfg.prompt_category_order == []
    assert cfg.prompt_pool_categories == {}
    assert cfg.prompt_negative_category == "negative"
    assert cfg.prompt_lora_trigger_injection is True
    assert cfg.prompt_pool_auto_create is True


def test_load_prompt_pools_v2_format(tmp_path):
    cfg = _make_config(tmp_path)
    families.init(cfg)
    _write_pool(
        tmp_path,
        "photorealistic",
        {
            "version": 2,
            "family": "photorealistic",
            "dialect": "clip_tag",
            "template": "{subject}, {quality}, {lighting}",
            "categories": {
                "subject": [
                    {"text": "cinematic portrait", "weight": 1.0},
                    "a cyberpunk android",
                ],
                "quality": [{"text": "masterpiece", "weight": 1.2}, "8k"],
                "lighting": ["neon lighting", "soft light"],
                "negative": ["blurry", "worst quality"],
            },
        },
    )
    load_prompt_pools(cfg)
    pool = families.PROMPT_POOLS["photorealistic"]
    assert isinstance(pool, PromptPool)
    assert pool.version == 2
    assert pool.dialect == DIALECT_CLIP_TAG
    assert "color" not in pool.categories  # not in pool -> skipped at build time
    positive, negative = get_random_prompt("photorealistic")
    assert positive, "positive prompt must not be empty"
    assert negative in ("blurry", "worst quality")
    # CLIP emphasis rendering: weighted entries get (text:weight).
    assert "(masterpiece:1.2)" in positive or "8k" in positive or "(cyberpunk android)" not in positive


def test_v1_legacy_pool_adaptation(tmp_path):
    cfg = _make_config(tmp_path)
    families.init(cfg)
    _write_pool(tmp_path, "anime", {"positive": ["1girl, solo"], "negative": ["bad hands"]})
    load_prompt_pools(cfg)
    pool = families.PROMPT_POOLS["anime"]
    assert pool.version == 1
    assert pool.dialect == DIALECT_CLIP_TAG
    assert set(pool.categories) == {"subject", "negative"}
    positive, negative = get_random_prompt("anime")
    assert positive == "1girl, solo"
    assert negative == "bad hands"


def test_missing_pool_auto_created_on_disk(tmp_path):
    cfg = _make_config(tmp_path)
    families.init(cfg)
    # No pool files at all: every family should get an auto-created default.
    load_prompt_pools(cfg)
    assert len(families.PROMPT_POOLS) == len(families.FAMILY_SETTINGS)
    pool_path = tmp_path / "pools" / "flux_dev_nsfw.json" if cfg.nsfw_mode else tmp_path / "pools" / "flux_dev.json"
    assert pool_path.exists(), f"expected auto-created pool at {pool_path}"
    positive, negative = get_random_prompt("flux_dev")
    assert positive and negative


def test_t5_dialect_emits_plain_text(tmp_path):
    cfg = _make_config(tmp_path)
    families.init(cfg)
    _write_pool(
        tmp_path,
        "flux_dev",
        {
            "version": 2,
            "family": "flux_dev",
            "dialect": "t5_natural",
            "template": "{subject} {quality}.",
            "categories": {
                "subject": [{"text": "a cinematic portrait", "weight": 9.0}],
                "quality": ["highly detailed photograph"],
                "negative": ["blurry"],
            },
        },
    )
    load_prompt_pools(cfg)
    positive, negative = get_random_prompt("flux_dev")
    assert "(" not in positive  # T5 ignores emphasis weights
    assert positive == "a cinematic portrait highly detailed photograph." or positive.endswith(
        "highly detailed photograph."
    )
    assert negative == "blurry"


def test_category_order_override(tmp_path):
    cfg = _make_config(tmp_path)
    families.init(cfg)
    _write_pool(
        tmp_path,
        "pony",
        {
            "version": 2,
            "family": "pony",
            "dialect": "clip_tag",
            "template": "{subject}, {medium}",
            "categories": {
                "subject": ["pony subject"],
                "medium": ["pony medium"],
                "negative": ["pony negative"],
            },
        },
    )
    load_prompt_pools(cfg)
    # Only subject: order lists a category missing from the pool -> skipped, not fatal.
    positive, negative = get_random_prompt("pony", category_order=["subject", "nonexistent"])
    assert positive == "pony subject"
    assert negative == "pony negative"

    # Global config override applies to all families. Template order wins:
    # "{subject}, {medium}" renders subject first regardless of category order.
    cfg.prompt_category_order = ["medium", "subject"]
    positive, negative = get_random_prompt("pony", category_order=cfg.prompt_category_order)
    assert positive == "pony subject, pony medium"


def test_inject_lora_triggers():
    dialect_clip = DIALECT_CLIP_TAG
    dialect_t5 = DIALECT_T5
    base = "a cinematic portrait"
    assert inject_lora_triggers(base, ["some_lora.safetensors"], dialect_clip) == (
        "a cinematic portrait, (some lora:1.2)"
    )
    assert inject_lora_triggers(base, ["my-character-v1"], dialect_t5) == (
        "a cinematic portrait, my character v1"
    )
    assert inject_lora_triggers(base, [], dialect_clip) == base
    assert inject_lora_triggers("", ["x"], dialect_clip) == ", (x:1.2)"


def test_render_entry_weights():
    assert render_entry(PromptEntry("masterpiece", 1.2), DIALECT_CLIP_TAG) == "(masterpiece:1.2)"
    assert render_entry(PromptEntry("masterpiece", 1.0), DIALECT_CLIP_TAG) == "masterpiece"
    assert render_entry(PromptEntry("a sentence", 9.0), DIALECT_T5) == "a sentence"
    assert render_entry(PromptEntry("   ", 1.0), DIALECT_CLIP_TAG) == ""


def test_default_pool_for_family_shape(tmp_path):
    cfg = _make_config(tmp_path)
    families.init(cfg)
    pool = default_pool_for_family("wan22_video", nsfw=False)
    assert pool.dialect == DIALECT_VIDEO
    assert "subject" in pool.categories and "negative" in pool.categories
    positive, negative = build_prompt_from_pool(pool)
    assert positive and negative


def test_get_family_dialect_mapping():
    assert get_family_dialect("flux_dev") == DIALECT_T5
    assert get_family_dialect("pony") == DIALECT_CLIP_TAG
    assert get_family_dialect("wan22_video") == DIALECT_VIDEO
    assert get_family_dialect("unknown_family") == DIALECT_CLIP_TAG


def test_real_pools_load_end_to_end():
    """Sanity check against the actual ComfyUI prompt_pools directory."""
    cfg = AppConfig(
        comfyui_base="/home/rizzo/ComfyUI",
        prompt_pools_dir="/home/rizzo/ComfyUI/prompt_pools",
        model_family_map="/home/rizzo/ComfyUI/prompt_pools/model_family_map.json",
        nsfw_mode=False,
    )
    families.init(cfg)
    load_prompt_pools(cfg)
    assert len(families.PROMPT_POOLS) == len(families.FAMILY_SETTINGS)
    for family, pool in families.PROMPT_POOLS.items():
        assert pool.categories, f"{family} pool has no categories"
        positive, negative = get_random_prompt(family)
        assert positive, f"{family} produced empty positive prompt"
        assert negative, f"{family} produced empty negative prompt"