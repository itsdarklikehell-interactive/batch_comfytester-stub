"""Unit tests for comfytester.state: completed/failed/priority/skip lists and metadata sidecars."""
import json
import os
from pathlib import Path

import pytest

from comfytester import state


def test_load_completed_models_missing_file(bound):
    assert state.load_completed_models() == set()


def test_mark_and_load_completed_models(bound):
    state.mark_completed("model-1")
    state.mark_completed("model-2")
    state.mark_completed("model-1")  # duplicate append should not break loading
    assert state.load_completed_models() == {"model-1", "model-2"}


def test_load_completed_models_skips_blank_lines(tmp_path, bound):
    state.cfg.completed_file = str(tmp_path / "completed_models.txt")
    Path(state.cfg.completed_file).write_text("id-a\n\nid-b\n   \n")
    assert state.load_completed_models() == {"id-a", "id-b"}


def test_load_failed_models_missing_file(bound):
    assert state.load_failed_models() == set()


def test_mark_failed_records_without_crashing_when_telegram_off(bound):
    # telegram is bound with no token -> send_telegram_status skips delivery
    state.mark_failed("bad-model")
    assert state.load_failed_models() == {"bad-model"}


def test_completed_datasets_roundtrip(bound):
    assert state.load_completed_datasets() == set()
    state.mark_completed_dataset("user/ds-1")
    state.mark_completed_dataset("user/ds-2")
    assert state.load_completed_datasets() == {"user/ds-1", "user/ds-2"}


def test_load_priority_skip_lists_missing_files(bound):
    priority, skip = state.load_priority_skip_lists()
    assert priority == []
    assert skip == set()


def test_load_priority_skip_lists_parses_files(tmp_path, bound):
    state.cfg.priority_file = str(tmp_path / "priority_ids.txt")
    state.cfg.skip_file = str(tmp_path / "skip_ids.txt")
    Path(state.cfg.priority_file).write_text("id-1\n\nid-2\n# not-a-comment-here\nid-1\n")
    Path(state.cfg.skip_file).write_text("skip-1\nskip-2\n")
    priority, skip = state.load_priority_skip_lists()
    # priority preserves order and duplicates; blanks are skipped but '#' lines are kept (current behavior)
    assert priority == ["id-1", "id-2", "# not-a-comment-here", "id-1"]
    assert skip == {"skip-1", "skip-2"}


def test_save_metadata_sidecar_writes_json(tmp_path, bound):
    image = tmp_path / "gen.png"
    image.write_bytes(b"PNG-DATA")
    settings = {"steps": 30, "cfg": 7, "sampler": "euler"}
    state.save_metadata_sidecar(
        str(image),
        "123",
        "Cool Model",
        "checkpoint",
        "anime",
        "prompt text",
        "negative",
        42,
        settings,
        extra={"foo": "bar"},
    )
    sidecar = tmp_path / "gen.meta.json"
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text())
    assert meta["image"] == "gen.png"
    assert meta["model_id"] == "123"
    assert meta["model_name"] == "Cool Model"
    assert meta["model_type"] == "checkpoint"
    assert meta["family"] == "anime"
    assert meta["prompt"] == "prompt text"
    assert meta["negative_prompt"] == "negative"
    assert meta["seed"] == 42
    assert meta["settings"] == settings
    assert meta["foo"] == "bar"
    assert "timestamp" in meta


def test_save_metadata_sidecar_extra_none(tmp_path, bound):
    image = tmp_path / "gen.jpg"
    image.write_bytes(b"JPG")
    state.save_metadata_sidecar(str(image), "1", "n", "t", "f", "p", "n", 1, {})
    meta = json.loads((tmp_path / "gen.meta.json").read_text())
    assert "extra" not in meta  # no 'extra' key when None


def test_save_metadata_sidecar_unwritable_dir_does_not_crash(bound):
    # Directory does not exist and is not created -> open fails -> caught and logged, no crash.
    state.save_metadata_sidecar(
        "/nonexistent-dir/no-such/gen.png", "1", "n", "t", "f", "p", "n", 1, {}
    )
