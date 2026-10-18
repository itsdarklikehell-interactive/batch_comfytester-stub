"""Unit tests for comfytester.telegram: image prep, send retries, redaction, skip-without-token."""

import os
from types import SimpleNamespace

import pytest
from PIL import Image

from comfytester import telegram, utils


class FakeResult:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture
def png_under_limit(tmp_path):
    img = tmp_path / "small.png"
    Image.new("RGB", (1, 1)).save(img, "PNG")
    return str(img)


@pytest.fixture
def png_over_limit(tmp_path):
    img = tmp_path / "big.png"
    Image.new("RGB", (200, 200)).save(img, "PNG")
    assert os.path.getsize(img) > 100  # telegram_max_bytes fixture is 100
    return str(img)


def test_prepare_telegram_image_under_limit(bound, png_under_limit, monkeypatch):
    monkeypatch.setattr(telegram.cfg, "telegram_max_bytes", 10**6)
    path, cleanup = telegram.prepare_telegram_image(png_under_limit)
    assert path == png_under_limit
    assert cleanup is False


def test_prepare_telegram_image_missing_file(bound):
    path, cleanup = telegram.prepare_telegram_image("/tmp/does-not-exist-12345.png")
    assert path == "/tmp/does-not-exist-12345.png"
    assert cleanup is False


def test_prepare_telegram_image_over_limit_converts_to_jpeg(bound, png_over_limit):
    path, cleanup = telegram.prepare_telegram_image(png_over_limit)
    assert cleanup is True
    assert path != png_over_limit
    assert path.endswith("_tg.jpg")
    assert os.path.exists(path)
    os.remove(path)  # caller cleanup


def test_send_telegram_photo_no_token_skips(monkeypatch, bound):
    called = []
    monkeypatch.setattr(
        telegram, "subprocess", SimpleNamespace(run=lambda *a, **k: called.append(1) or FakeResult(0))
    )
    monkeypatch.setattr(telegram.cfg, "telegram_bot_token", "")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    img = "/tmp/some-image.png"
    telegram.send_telegram_photo(img, "caption")
    assert called == []


def test_send_telegram_photo_success(monkeypatch, bound, png_under_limit):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return FakeResult(0, '{"ok": true}')

    monkeypatch.setattr(telegram, "subprocess", SimpleNamespace(run=fake_run))
    monkeypatch.setattr(telegram.cfg, "telegram_bot_token", "bot123:SECRET")
    long_caption = "c" * 2000
    telegram.send_telegram_photo(png_under_limit, long_caption)
    assert len(calls) == 1
    assert calls[0][0] == "curl"
    assert "-X" in calls[0] and "POST" in calls[0]
    caption_arg = [a for a in calls[0] if a.startswith("caption=")][0]
    assert caption_arg == "caption=" + long_caption[:1000]  # caption truncated to 1000 chars


def test_send_telegram_photo_api_error_retries_and_redacts(monkeypatch, bound, png_under_limit):
    calls = []
    sleeps = []
    logs = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) == 1:
            return FakeResult(0, '{"ok": false, "description": "token bot123:SECRET is invalid"}')
        return FakeResult(0, '{"ok": true}')

    monkeypatch.setattr(telegram, "subprocess", SimpleNamespace(run=fake_run))
    monkeypatch.setattr(telegram.cfg, "telegram_bot_token", "bot123:SECRET")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot123:SECRET")  # redact_token reads environ
    monkeypatch.setattr(telegram, "log", lambda msg: logs.append(msg))
    monkeypatch.setattr(time_module(), "sleep", lambda d: sleeps.append(d))

    telegram.send_telegram_photo(png_under_limit, "cap")
    assert len(calls) == 2
    assert sleeps == [1]  # 2**0 backoff before retry
    joined = "\n".join(logs)
    assert "[REDACTED]" in joined
    assert "bot123:SECRET" not in joined


def test_send_telegram_photo_invalid_json_with_rc0_counts_as_sent(monkeypatch, bound, png_under_limit):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return FakeResult(0, "not-json-at-all")

    monkeypatch.setattr(telegram, "subprocess", SimpleNamespace(run=fake_run))
    monkeypatch.setattr(telegram.cfg, "telegram_bot_token", "bot123:SECRET")
    telegram.send_telegram_photo(png_under_limit, "cap")
    assert len(calls) == 1  # curl 0 -> treated as delivered despite unparseable body


def test_send_telegram_message_ok(monkeypatch, bound):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return FakeResult(0, '{"ok": true}')

    monkeypatch.setattr(telegram, "subprocess", SimpleNamespace(run=fake_run))
    monkeypatch.setattr(telegram.cfg, "telegram_bot_token", "bot123:SECRET")
    telegram.send_telegram_message("hello world")
    assert len(calls) == 1
    assert any(a.startswith("text=") for a in calls[0])


def test_send_telegram_message_no_token_skips(monkeypatch, bound):
    called = []
    monkeypatch.setattr(
        telegram, "subprocess", SimpleNamespace(run=lambda *a, **k: called.append(1) or FakeResult(0))
    )
    monkeypatch.setattr(telegram.cfg, "telegram_bot_token", "")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    telegram.send_telegram_message("hello")
    assert called == []


def test_send_telegram_status_uses_bullet_prefix(monkeypatch, bound):
    captured = []
    monkeypatch.setattr(telegram, "send_telegram_message", lambda text: captured.append(text))
    telegram.send_telegram_status("deploy started")
    assert captured == ["• deploy started"]


def time_module():
    import time

    return time
