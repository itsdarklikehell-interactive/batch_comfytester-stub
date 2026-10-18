"""Unit tests for comfytester.civitai: API paging, retry/backoff, URL signing, file parsing.

All network and subprocess calls are mocked; nothing hits the real Civitai API.
"""
import json
import time
import urllib.error
import urllib.request

import pytest

from comfytester import civitai


class FakeHTTPResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def make_http_error(code, url="https://civitai.example/api/v1/models"):
    return urllib.error.HTTPError(url, code, "msg", {}, None)


@pytest.fixture
def reset_state(monkeypatch):
    monkeypatch.setattr(civitai, "_civitai_last_request", 0.0)
    monkeypatch.setattr(civitai, "CIVITAI_TOKEN", None)


def test_civitai_init_reads_token_file(tmp_path, monkeypatch):
    from comfytester.config import AppConfig

    token_file = tmp_path / "civitai-token"
    token_file.write_text("  abc123  \n")
    cfg = AppConfig(civitai_token_file=str(token_file))
    civitai.init(cfg)
    assert civitai.CIVITAI_TOKEN == "abc123"


def test_civitai_init_missing_token_file(tmp_path, monkeypatch):
    from comfytester.config import AppConfig

    cfg = AppConfig(civitai_token_file=str(tmp_path / "missing"))
    civitai.init(cfg)
    assert civitai.CIVITAI_TOKEN is None


def test_civitai_pace_sleeps_when_under_interval(bound, monkeypatch):
    clock = [0.0]
    sleeps = []
    monkeypatch.setattr(time, "time", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", lambda d: sleeps.append(d))
    civitai._civitai_last_request = 0.0
    civitai.civitai_pace()  # elapsed 0.0 < interval -> sleep full interval
    assert sleeps == [civitai.cfg.civitai_min_interval]
    clock[0] = civitai.cfg.civitai_min_interval / 2
    civitai.civitai_pace()  # remaining half-interval
    assert sleeps[-1] == civitai.cfg.civitai_min_interval / 2
    clock[0] = 100.0
    civitai.civitai_pace()  # elapsed >= interval -> no sleep
    assert len(sleeps) == 2  # no third sleep


def test_get_model_info_parses_json(monkeypatch, bound):
    monkeypatch.setattr(civitai, "run_civitai", lambda args: '{"name": "Awesome", "id": 42}')
    assert civitai.get_model_info(42) == {"name": "Awesome", "id": 42}


def test_get_model_info_invalid_json_returns_none(monkeypatch, bound):
    monkeypatch.setattr(civitai, "run_civitai", lambda args: "{not json")
    assert civitai.get_model_info(42) is None


def test_get_model_info_none_returns_none(monkeypatch, bound):
    monkeypatch.setattr(civitai, "run_civitai", lambda args: None)
    assert civitai.get_model_info(42) is None


def test_get_download_url_appends_token_with_ampersand(monkeypatch, bound, reset_state):
    monkeypatch.setattr(civitai, "run_civitai", lambda args: "https://storage.example/file?sig=abc")
    monkeypatch.setattr(civitai, "CIVITAI_TOKEN", "secret-token")
    assert civitai.get_download_url(9) == "https://storage.example/file?sig=abc&token=secret-token"


def test_get_download_url_appends_token_with_question_mark(monkeypatch, bound, reset_state):
    monkeypatch.setattr(civitai, "run_civitai", lambda args: "https://storage.example/file")
    monkeypatch.setattr(civitai, "CIVITAI_TOKEN", "secret-token")
    assert civitai.get_download_url(9) == "https://storage.example/file?token=secret-token"


def test_get_download_url_without_token_unchanged(monkeypatch, bound, reset_state):
    monkeypatch.setattr(civitai, "run_civitai", lambda args: "https://storage.example/file?sig=abc")
    monkeypatch.setattr(civitai, "CIVITAI_TOKEN", None)
    assert civitai.get_download_url(9) == "https://storage.example/file?sig=abc"


INFO_MAIN = {
    "name": "Cool Model",
    "tags": ["tag1", "tag2"],
    "modelVersions": [
        {
            "id": 77,
            "files": [
                {"type": "Pruned", "name": "cool.pruned.safetensors"},
                {"type": "Main", "name": "cool.safetensors"},
                {"type": "VAE", "name": "cool.vae.pt"},
            ],
        }
    ],
}


def test_get_filename_from_api_picks_main_file(monkeypatch, bound):
    monkeypatch.setattr(civitai, "get_model_info", lambda mid: INFO_MAIN)
    name, vid, fname, tags = civitai.get_filename_from_api(3)
    assert (name, vid, fname, tags) == ("Cool Model", 77, "cool.safetensors", ["tag1", "tag2"])


def test_get_filename_from_api_falls_back_to_first_file(monkeypatch, bound):
    info = dict(INFO_MAIN)
    info["modelVersions"] = [{"id": 77, "files": [{"type": "VAE", "name": "vae.pt"}]}]
    monkeypatch.setattr(civitai, "get_model_info", lambda mid: info)
    name, vid, fname, tags = civitai.get_filename_from_api(3)
    assert (name, vid, fname, tags) == ("Cool Model", 77, "vae.pt", ["tag1", "tag2"])


def test_get_filename_from_api_no_versions(monkeypatch, bound):
    monkeypatch.setattr(civitai, "get_model_info", lambda mid: {"name": "Solo", "tags": []})
    assert civitai.get_filename_from_api(3) == ("Solo", None, None, [])


def test_get_filename_from_api_version_without_files(monkeypatch, bound):
    monkeypatch.setattr(civitai, "get_model_info", lambda mid: {"name": "Solo", "modelVersions": [{"id": 5}]})
    assert civitai.get_filename_from_api(3) == ("Solo", 5, None, [])


def test_get_filename_from_api_info_none(monkeypatch, bound):
    monkeypatch.setattr(civitai, "get_model_info", lambda mid: None)
    assert civitai.get_filename_from_api(3) == (None, None, None, None)


def test_civitai_api_get_retries_on_429_then_succeeds(monkeypatch, bound, reset_state):
    calls = []
    sleeps = []

    def fake_urlopen(req, timeout=30):
        calls.append(req.full_url)
        if len(calls) == 1:
            raise make_http_error(429)
        return FakeHTTPResponse({"items": [{"id": 1}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda d: sleeps.append(d))

    result = civitai.civitai_api_get("/models", token="tok", params={"limit": 10})
    assert result == {"items": [{"id": 1}]}
    assert len(calls) == 2
    assert sleeps == [1]  # backoff min(2**0, 30) = 1s
    assert calls[0] == "https://civitai.example/api/v1/models?limit=10"


def test_civitai_api_get_bearer_token_header(monkeypatch, bound, reset_state):
    captured = {}

    def fake_urlopen(req, timeout=30):
        captured["headers"] = dict(req.header_items())
        return FakeHTTPResponse({"ok": True})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda d: None)
    civitai.civitai_api_get("/models", token="tok-123")
    assert captured["headers"]["Authorization"] == "Bearer tok-123"
    assert "Mozilla/5.0" in captured["headers"]["User-agent"]


def test_civitai_api_get_404_is_not_retried(monkeypatch, bound, reset_state):
    calls = []

    def fake_urlopen(req, timeout=30):
        calls.append(1)
        raise make_http_error(404)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    slept = []

    def boom(d):
        slept.append(d)

    monkeypatch.setattr(time, "sleep", boom)
    assert civitai.civitai_api_get("/models") is None
    assert len(calls) == 1
    assert slept == []


def test_civitai_api_get_network_error_returns_none(monkeypatch, bound, reset_state):
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=30: (_ for _ in ()).throw(OSError("boom")))
    monkeypatch.setattr(time, "sleep", lambda d: (_ for _ in ()).throw(AssertionError("sleep must not be called")))
    assert civitai.civitai_api_get("/models") is None


def test_fetch_civitai_favorites_pagination(monkeypatch, bound, reset_state):
    pages = [
        {"items": [{"id": 10}, {"id": 20}], "metadata": {"nextCursor": "c1"}},
        {"items": [{"id": 30}], "metadata": {}},
    ]
    seen = []

    def fake_get(path, token=None, params=None):
        seen.append((path, token, dict(params or {})))
        return pages.pop(0)

    monkeypatch.setattr(civitai, "civitai_api_get", fake_get)
    ids = civitai.fetch_civitai_favorites("tok")
    assert ids == ["10", "20", "30"]
    assert seen[0][1] == "tok"
    assert seen[0][2] == {"favorites": "true", "limit": 100}
    assert seen[1][2] == {"favorites": "true", "limit": 100, "cursor": "c1"}


def test_fetch_civitai_favorites_no_token(monkeypatch, bound, reset_state):
    called = []
    monkeypatch.setattr(civitai, "civitai_api_get", lambda *a, **k: called.append(1) or [])
    assert civitai.fetch_civitai_favorites(None) == []
    assert civitai.fetch_civitai_favorites("") == []
    assert called == []


def test_fetch_civitai_followed(monkeypatch, bound, reset_state):
    def fake_get(path, token=None, params=None):
        if path == "/me":
            return {"followed": [{"username": "alice"}, {"username": "bob"}]}
        if path == "/models":
            user = params.get("username")
            return {"items": [{"id": 100 if user == "alice" else 200}], "metadata": {}}
        raise AssertionError(f"unexpected path {path}")

    monkeypatch.setattr(civitai, "civitai_api_get", fake_get)
    assert civitai.fetch_civitai_followed("tok") == ["100", "200"]


def test_fetch_civitai_followed_me_fails(monkeypatch, bound, reset_state):
    monkeypatch.setattr(civitai, "civitai_api_get", lambda *a, **k: None)
    assert civitai.fetch_civitai_followed("tok") == []


def test_fetch_civitai_followed_no_followed_list(monkeypatch, bound, reset_state):
    monkeypatch.setattr(civitai, "civitai_api_get", lambda *a, **k: {"followed": []})
    assert civitai.fetch_civitai_followed("tok") == []


def test_fetch_civitai_followed_skips_creators_without_username(monkeypatch, bound, reset_state):
    def fake_get(path, token=None, params=None):
        if path == "/me":
            return {"followed": [{"username": "alice"}, {"nope": True}]}
        if path == "/models":
            return {"items": [{"id": 5}], "metadata": {}}
        raise AssertionError(path)

    monkeypatch.setattr(civitai, "civitai_api_get", fake_get)
    assert civitai.fetch_civitai_followed("tok") == ["5"]


def test_fetch_civitai_followed_no_token(monkeypatch, bound, reset_state):
    called = []
    monkeypatch.setattr(civitai, "civitai_api_get", lambda *a, **k: called.append(1) or [])
    assert civitai.fetch_civitai_followed(None) == []
    assert called == []


def test_fetch_civitai_collections_filters_and_dedupes(monkeypatch, bound, reset_state):
    def fake_get(path, token=None, params=None):
        if path == "/collections":
            return {"items": [{"id": "c1"}, {"id": "c2"}], "metadata": {}}
        if path == "/collections/c1":
            return {
                "models": [
                    {"id": 1, "type": "CHECKPOINT"},
                    {"id": 2, "type": "LORA"},
                    {"id": 3, "type": "ARTICLE"},  # excluded: not generation-capable
                ],
                "metadata": {},
            }
        if path == "/collections/c2":
            return {
                "models": [
                    {"id": 1, "type": "CHECKPOINT"},  # duplicate across collections
                    {"id": 4, "type": "VAE"},
                ],
                "metadata": {},
            }
        raise AssertionError(path)

    monkeypatch.setattr(civitai, "civitai_api_get", fake_get)
    assert civitai.fetch_civitai_collections("tok") == ["1", "2", "4"]


def test_fetch_civitai_collections_paginates_within_collection(monkeypatch, bound, reset_state):
    def fake_get(path, token=None, params=None):
        if path == "/collections":
            return {"items": [{"id": "c1"}], "metadata": {}}
        if path == "/collections/c1":
            if params.get("cursor"):
                return {"models": [{"id": 2, "type": "CHECKPOINT"}], "metadata": {}}
            return {"models": [{"id": 1, "type": "CHECKPOINT"}], "metadata": {"nextCursor": "x"}}
        raise AssertionError(path)

    monkeypatch.setattr(civitai, "civitai_api_get", fake_get)
    assert civitai.fetch_civitai_collections("tok") == ["1", "2"]


def test_fetch_civitai_collections_no_token(monkeypatch, bound, reset_state):
    called = []
    monkeypatch.setattr(civitai, "civitai_api_get", lambda *a, **k: called.append(1) or [])
    assert civitai.fetch_civitai_collections(None) == []
    assert called == []
