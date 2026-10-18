"""Unit tests for comfytester.utils: redaction, pid probe, file lock, log rotation, control-hint scan."""
import os
import time

import pytest

from comfytester import utils


def test_redact_token(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "secret123")
    assert utils.redact_token("msg with secret123 inside") == "msg with [REDACTED] inside"
    assert utils.redact_token("no token here") == "no token here"
    assert utils.redact_token(None) is None
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    assert utils.redact_token("msg secret123 inside") == "msg secret123 inside"


def test_pid_alive(monkeypatch):
    assert utils._pid_alive(0) is False
    assert utils._pid_alive(-5) is False
    assert utils._pid_alive(os.getpid()) is True  # our own process is alive
    assert utils._pid_alive(999999999) is False  # no such pid

    monkeypatch.setattr(os, "kill", lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError))
    assert utils._pid_alive(999999) is False

    monkeypatch.setattr(os, "kill", lambda pid, sig: (_ for _ in ()).throw(PermissionError))
    assert utils._pid_alive(999999) is True  # exists but we cannot signal it


def test_file_lock_acquire_release(tmp_path):
    lock = utils.FileLock(str(tmp_path / "batch.lock"))
    assert lock.acquire() is True
    assert lock.acquired is True
    lock.release()
    assert lock.acquired is False
    # Re-acquire after release works (fcntl locks die with the fd).
    assert lock.acquire() is True
    lock.release()


def test_file_lock_context_manager(tmp_path):
    path = str(tmp_path / "batch.lock")
    with utils.FileLock(path) as lock:
        assert lock.acquired is True
    assert lock.acquired is False


def test_file_lock_held_by_live_process_is_rejected(tmp_path):
    path = str(tmp_path / "batch.lock")
    first = utils.FileLock(path)
    assert first.acquire() is True
    second = utils.FileLock(path)
    assert second.acquire(blocking=False) is False
    first.release()


def test_file_lock_stale_holder_is_cleared(tmp_path, monkeypatch):
    path = str(tmp_path / "batch.lock")
    first = utils.FileLock(path)
    assert first.acquire() is True  # stamps our live pid into the lock file
    monkeypatch.setattr(utils, "_pid_alive", lambda pid: False)  # holder now looks dead -> stale
    second = utils.FileLock(path)
    assert second.acquire(blocking=False) is True  # stale lock cleared and re-acquired
    second.release()
    first.release()


def test_log_writes_to_log_file(tmp_path, bound, monkeypatch):
    log_file = tmp_path / "comfytester.log"
    monkeypatch.setattr(utils._config, "log_file", str(log_file))
    utils.log("hello test line")
    assert "hello test line" in log_file.read_text()


def test_rotate_log_if_needed_small_file_not_rotated(tmp_path, bound, monkeypatch):
    log_file = tmp_path / "comfytester.log"
    log_file.write_text("small")
    monkeypatch.setattr(utils._config, "log_file", str(log_file))
    utils.rotate_log_if_needed()
    assert log_file.read_text() == "small"
    assert not os.path.exists(str(log_file) + ".1")


def test_rotate_log_if_needed_rotates_oversized_log(tmp_path, bound, monkeypatch):
    log_file = tmp_path / "comfytester.log"
    log_file.write_text("x" * 1000)  # log_max_bytes is 100 in the fixture config
    monkeypatch.setattr(utils._config, "log_file", str(log_file))
    utils.rotate_log_if_needed()
    assert (tmp_path / "comfytester.log.1").read_text() == "x" * 1000


def test_rotate_log_if_needed_shifts_backups(tmp_path, bound, monkeypatch):
    log_file = tmp_path / "comfytester.log"
    log_file.write_text("x" * 1000)
    (tmp_path / "comfytester.log.1").write_text("old-1")
    (tmp_path / "comfytester.log.2").write_text("old-2")
    monkeypatch.setattr(utils._config, "log_file", str(log_file))
    utils.rotate_log_if_needed()
    assert (tmp_path / "comfytester.log.1").read_text() == "x" * 1000
    assert (tmp_path / "comfytester.log.2").read_text() == "old-1"
    assert (tmp_path / "comfytester.log.3").read_text() == "old-2"


def test_find_control_hint_image_no_config(monkeypatch):
    monkeypatch.setattr(utils, "_config", None)
    assert utils.find_control_hint_image() is None


def test_find_control_hint_image_no_dirs(tmp_path, monkeypatch, bound):
    monkeypatch.setattr(utils._config, "comfyui_base", str(tmp_path / "does-not-exist"))
    assert utils.find_control_hint_image() is None


def test_find_control_hint_image_ignores_non_images(tmp_path, monkeypatch, bound):
    base = tmp_path / "comfyui"
    (base / "input").mkdir(parents=True)
    (base / "input" / "readme.txt").write_text("not an image")
    monkeypatch.setattr(utils._config, "comfyui_base", str(base))
    assert utils.find_control_hint_image() is None


def test_find_control_hint_image_picks_newest_by_mtime(tmp_path, monkeypatch, bound):
    base = tmp_path / "comfyui"
    (base / "input").mkdir(parents=True)
    (base / "output").mkdir(parents=True)
    old = base / "input" / "old.png"
    new = base / "input" / "new.png"
    old.write_bytes(b"old-image")
    new.write_bytes(b"new-image")
    os.utime(old, (1000, 1000))
    os.utime(new, (2000, 2000))
    monkeypatch.setattr(utils._config, "comfyui_base", str(base))
    assert utils.find_control_hint_image() == str(new)


def test_find_control_hint_image_output_only_when_input_empty(tmp_path, monkeypatch, bound):
    base = tmp_path / "comfyui"
    (base / "input").mkdir(parents=True)
    (base / "output").mkdir(parents=True)
    new = base / "output" / "new.png"
    new.write_bytes(b"new-image")
    os.utime(new, (2000, 2000))
    monkeypatch.setattr(utils._config, "comfyui_base", str(base))
    assert utils.find_control_hint_image() == str(new)
