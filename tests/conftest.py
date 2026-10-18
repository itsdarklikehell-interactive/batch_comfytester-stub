"""Shared fixtures and import setup for comfytester unit tests."""
import sys
from pathlib import Path

# Make the workspace root importable so 'import comfytester' works
# no matter how pytest is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402
from comfytester.config import AppConfig  # noqa: E402


@pytest.fixture
def config(tmp_path):
    """AppConfig whose every path lives inside tmp_path (hermetic, no real /home/rizzo writes)."""
    return AppConfig(
        comfyui_base=str(tmp_path / "comfyui"),
        log_file=str(tmp_path / "comfytester.log"),
        log_max_bytes=100,
        env_file=str(tmp_path / "test.env"),
        civitai_token_file=str(tmp_path / "civitai-token"),
        civitai_api_base="https://civitai.example/api/v1",
        civitai_min_interval=0.01,
        civitai_max_retries=3,
        hf_datasets_dir=str(tmp_path / "hf" / "datasets"),
        state_file=str(tmp_path / "batch_state.json"),
        completed_file=str(tmp_path / "completed_models.txt"),
        failed_file=str(tmp_path / "failed_models.txt"),
        skip_file=str(tmp_path / "skip_ids.txt"),
        priority_file=str(tmp_path / "priority_ids.txt"),
        completed_datasets_file=str(tmp_path / "completed_datasets.txt"),
        lock_file=str(tmp_path / "batch.lock"),
        telegram_max_bytes=100,
    )


@pytest.fixture
def bound(config):
    """Bind config into every module that keeps a module-level cfg (init() pattern)."""
    from comfytester import civitai, config as cfg_mod, downloader, state, telegram, utils

    cfg_mod.init(config) if hasattr(cfg_mod, "init") else None
    state.init(config)
    telegram.init(config)
    civitai.init(config)
    downloader.init(config)
    utils.init(config)
    return config
