"""Unit tests for comfytester.config: AppConfig defaults and load_config() precedence."""

from dataclasses import fields
import json
import os
import builtins

import pytest

from comfytester.config import AppConfig, ENV_OVERRIDES, load_config, _parse_bool

REAL_ENV_FILE = "/home/rizzo/Work/batch_comfytester/batch_comfytester.env"


def test_parse_bool():
    for true_val in ("1", "true", "True", "TRUE", "yes", "Yes", "on", "ON", "  on  "):
        assert _parse_bool(true_val) is True, true_val
    for false_val in ("0", "false", "False", "no", "NO", "off", "", "   ", "maybe", "42"):
        assert _parse_bool(false_val) is False, false_val


def test_app_config_defaults():
    cfg = AppConfig()
    assert cfg.comfyui_base == "/home/rizzo/ComfyUI"
    assert cfg.civitai_api_base == "https://civitai.com/api/v1"
    assert cfg.dry_run is False
    assert cfg.nsfw_mode is False
    assert cfg.civitai_max_retries == 3  # default reduced from 5 (see AppConfig comment)
    assert cfg.civitai_min_interval == 1.2
    assert cfg.telegram_home_channel == "-1004470891735"
    assert cfg.telegram_max_bytes == 8 * 1024 * 1024
    assert cfg.max_checkpoint_gb["default"] == 24.0
    assert cfg.max_checkpoint_gb["wan22_video"] == 24.0
    assert cfg.lora_strengths == [0.4, 0.8, 1.2]
    assert cfg.id_filter == []
    assert cfg.family_filter == ""
    assert cfg.limit is None


def test_app_config_explicit_overrides():
    cfg = AppConfig(dry_run=True, nsfw_mode=True, civitai_max_retries=9, telegram_max_bytes=1234)
    assert cfg.dry_run is True
    assert cfg.nsfw_mode is True
    assert cfg.civitai_max_retries == 9
    assert cfg.telegram_max_bytes == 1234


def test_mutable_defaults_are_copies_per_instance():
    c1 = AppConfig()
    c2 = AppConfig()
    assert c1.max_checkpoint_gb is not c2.max_checkpoint_gb
    assert c1.lora_strengths is not c2.lora_strengths
    assert c1.id_filter is not c2.id_filter
    assert c1.vae_compatibility is not c2.vae_compatibility
    c1.max_checkpoint_gb["weird-family"] = 99.0
    assert "weird-family" not in c2.max_checkpoint_gb
    c1.lora_strengths.append(9.9)
    assert c2.lora_strengths == [0.4, 0.8, 1.2]


def test_env_overrides_map_to_known_config_attributes():
    known = {f.name for f in fields(AppConfig)}
    for attr, _conv in ENV_OVERRIDES.values():
        assert attr in known, attr


@pytest.fixture
def hide_real_env_file(monkeypatch):
    """Pretend the repo's real batch_comfytester.env does not exist, for hermetic load_config tests."""
    real = os.path.realpath(REAL_ENV_FILE)
    orig_exists = os.path.exists

    def fake_exists(path):
        if os.path.realpath(path) == real:
            return False
        return orig_exists(path)

    monkeypatch.setattr(os.path, "exists", fake_exists)


def test_load_config_defaults_only(hide_real_env_file, monkeypatch):
    for key in ENV_OVERRIDES:
        monkeypatch.delenv(key, raising=False)
    cfg = load_config()
    assert cfg.dry_run is False
    assert cfg.nsfw_mode is False
    assert cfg.civitai_max_retries == 3  # default reduced from 5 (see AppConfig comment)
    assert cfg.telegram_bot_token == ""
    assert cfg.hf_token == ""


def test_load_config_json_overrides_defaults(tmp_path, hide_real_env_file, monkeypatch):
    for key in ENV_OVERRIDES:
        monkeypatch.delenv(key, raising=False)
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"dry_run": True, "civitai_max_retries": 9, "bogus_key": 123}))
    cfg = load_config(path=str(cfg_file))
    assert cfg.dry_run is True
    assert cfg.civitai_max_retries == 9
    assert not hasattr(cfg, "bogus_key")  # unknown keys are ignored, not stored


def test_load_config_env_var_wins_over_json(tmp_path, hide_real_env_file, monkeypatch):
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"dry_run": False}))
    monkeypatch.setenv("DRY_RUN", "true")
    cfg = load_config(path=str(cfg_file))
    assert cfg.dry_run is True  # environment overrides the JSON file


def test_load_config_env_file_applies_when_environ_unset(monkeypatch, tmp_path):
    # Real deployment env file lives at AppConfig().env_file
    # (/home/rizzo/Work/batch_comfytester/batch_comfytester.env); that path is absent
    # in this checkout, so serve equivalent content for that path only.
    # Nothing is set in environ, so .env values (via setdefault) flow into the
    # env-override stage.
    for key in ENV_OVERRIDES:
        monkeypatch.delenv(key, raising=False)
    target = os.path.realpath(AppConfig().env_file)
    env_file = tmp_path / "batch_comfytester.env"
    env_file.write_text("NSFW_MODE=true\nTELEGRAM_HOME_CHANNEL=-1004470891735\n")
    real_exists = os.path.exists
    real_open = builtins.open

    def fake_exists(path):
        return target == os.path.realpath(path) or real_exists(path)

    def fake_open(path, *args, **kwargs):
        if target == os.path.realpath(path):
            return env_file.open("r")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os.path, "exists", fake_exists)
    monkeypatch.setattr(builtins, "open", fake_open)
    cfg = load_config()
    assert cfg.nsfw_mode is True  # .env has NSFW_MODE=true
    assert cfg.telegram_home_channel == "-1004470891735"  # from real .env


def test_load_config_already_set_env_wins_over_env_file(monkeypatch):
    # Real .env has NSFW_MODE=true, but environ is already set -> setdefault is a no-op.
    monkeypatch.setenv("NSFW_MODE", "false")
    monkeypatch.setenv("DRY_RUN", "true")
    cfg = load_config()
    assert cfg.nsfw_mode is False  # environ wins over the .env file
    assert cfg.dry_run is True


def test_load_config_bool_env_conversion(hide_real_env_file, monkeypatch):
    for key in ENV_OVERRIDES:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NSFW_MODE", "true")
    monkeypatch.setenv("DRY_RUN", "1")
    cfg = load_config()
    assert cfg.nsfw_mode is True
    assert cfg.dry_run is True


def test_load_config_missing_file_raises(hide_real_env_file, monkeypatch):
    for key in ENV_OVERRIDES:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(path="/tmp/nonexistent-config-12345.json")
