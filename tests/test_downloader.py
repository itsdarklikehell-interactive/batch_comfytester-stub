"""Unit tests for comfytester.downloader: retries, size/SHA verification, atomic rename, HF fallback."""

import os
import sys

import pytest

from comfytester import downloader


class FakeResult:
    def __init__(self, returncode, stderr=""):
        self.returncode = returncode
        self.stderr = stderr


def make_fake_run(sequence):
    """sequence: list of (returncode, bytes_or_None) written to the -o/-O path per call."""
    calls = []
    seq = list(sequence)

    def run(cmd, **kwargs):
        calls.append(list(cmd))
        rc, content = seq.pop(0)
        for flag in ("-o", "-O"):
            if flag in cmd:
                path = cmd[cmd.index(flag) + 1]
                if content is not None:
                    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                    with open(path, "wb") as f:
                        f.write(content)
                break
        return FakeResult(rc)

    return run, calls


def test_download_file_success(monkeypatch, tmp_path):
    dest = str(tmp_path / "model.safetensors")
    run, calls = make_fake_run([(0, b"model-bytes")])
    monkeypatch.setattr(downloader.subprocess, "run", run)
    ok = downloader.download_file("https://example.com/model.safetensors", dest)
    assert ok is True
    with open(dest, "rb") as f:
        assert f.read() == b"model-bytes"
    assert not os.path.exists(dest + ".part")  # atomic rename consumed the part file
    assert "--max-time 120" in calls[0] and "-o" in calls[0]


def test_download_file_empty_download_retries(monkeypatch, tmp_path):
    dest = str(tmp_path / "m.safetensors")
    run, calls = make_fake_run([(0, b""), (0, b"ok")])
    import time as time_module

    monkeypatch.setattr(downloader.subprocess, "run", run)
    monkeypatch.setattr(time_module, "sleep", lambda d: None)
    ok = downloader.download_file("https://example.com/m", dest, max_retries=3)
    assert ok is True
    assert len(calls) == 2


def test_download_file_size_mismatch_then_retry(monkeypatch, tmp_path):
    import time as time_module

    dest = str(tmp_path / "m.safetensors")
    sleeps = []
    run, calls = make_fake_run([(0, b"short"), (0, b"correct-content-length")])
    monkeypatch.setattr(downloader.subprocess, "run", run)
    monkeypatch.setattr(time_module, "sleep", lambda d: sleeps.append(d))
    ok = downloader.download_file("https://example.com/m", dest, expected_size=22, max_retries=3)
    assert ok is True
    with open(dest, "rb") as f:
        assert f.read() == b"correct-content-length"
    assert sleeps == [1]  # 2**0 backoff
    assert len(calls) == 2


def test_download_file_sha256_mismatch(monkeypatch, tmp_path):
    import time as time_module

    dest = str(tmp_path / "m.safetensors")
    run, calls = make_fake_run([(0, b"wrong-content")] * 3)
    monkeypatch.setattr(downloader.subprocess, "run", run)
    monkeypatch.setattr(time_module, "sleep", lambda d: None)
    ok = downloader.download_file("https://example.com/m", dest, expected_sha256="deadbeef", max_retries=3)
    assert ok is False
    assert not os.path.exists(dest)
    assert not os.path.exists(dest + ".part")
    assert len(calls) == 3  # retried until max_retries exhausted


def test_download_file_curl_fails_wget_succeeds(monkeypatch, tmp_path):
    import time as time_module

    dest = str(tmp_path / "m.safetensors")
    run, calls = make_fake_run([(1, None), (0, b"wget-content")])
    monkeypatch.setattr(downloader.subprocess, "run", run)
    monkeypatch.setattr(time_module, "sleep", lambda d: None)
    ok = downloader.download_file("https://example.com/m", dest, max_retries=3)
    assert ok is True
    with open(dest, "rb") as f:
        assert f.read() == b"wget-content"
    assert calls[1][0] == "wget"


def test_download_file_all_downloaders_fail(monkeypatch, tmp_path):
    import time as time_module

    dest = str(tmp_path / "m.safetensors")
    run, calls = make_fake_run([(1, None), (1, None)] * 3)
    monkeypatch.setattr(downloader.subprocess, "run", run)
    monkeypatch.setattr(time_module, "sleep", lambda d: None)
    ok = downloader.download_file("https://example.com/m", dest, max_retries=3)
    assert ok is False
    assert not os.path.exists(dest)
    assert not os.path.exists(dest + ".part")
    assert len(calls) == 6  # curl + wget per attempt


def test_download_file_subprocess_exception(monkeypatch, tmp_path):
    import time as time_module

    dest = str(tmp_path / "m.safetensors")

    def run(cmd, **kwargs):
        raise OSError("curl missing")

    monkeypatch.setattr(downloader.subprocess, "run", run)
    monkeypatch.setattr(time_module, "sleep", lambda d: None)
    ok = downloader.download_file("https://example.com/m", dest, max_retries=2)
    assert ok is False


def test_download_file_hf_import_error_falls_back_to_download_file(monkeypatch, bound, tmp_path):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("huggingface_hub hidden for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)

    captured = []

    def fake_download_file(url, dest_path, **kwargs):
        captured.append((url, dest_path))
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(b"hf-fallback")
        return True

    monkeypatch.setattr(downloader, "download_file", fake_download_file)

    ok = downloader.download_file_hf("User/Repo", "subdir/model.safetensors", token="hf_x")
    assert ok is True
    assert len(captured) == 1
    url, dest = captured[0]
    assert url == "https://huggingface.co/User/Repo/resolve/main/model.safetensors"
    assert dest == os.path.join(downloader.cfg.comfyui_base, "models", "subdir/model.safetensors")


def test_download_file_hf_success(monkeypatch, bound):
    import huggingface_hub

    def fake_hf_download(repo_id, filename, token, local_dir, local_dir_use_symlinks):
        target = os.path.join(local_dir, filename)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as f:
            f.write(b"hf-bytes")
        return target

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_download)
    ok = downloader.download_file_hf("User/Repo", "subdir/model.safetensors")
    assert ok is True
    dest = os.path.join(downloader.cfg.comfyui_base, "models", "subdir/model.safetensors")
    assert os.path.exists(dest)
    with open(dest, "rb") as f:
        assert f.read() == b"hf-bytes"


def test_download_hf_dataset_import_error(monkeypatch, bound):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("huggingface_hub hidden for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)
    assert downloader.download_hf_dataset("User/ds") is False


def test_download_hf_dataset_success(monkeypatch, bound):
    import huggingface_hub

    def fake_snapshot(repo_id, repo_type, token, local_dir):
        os.makedirs(local_dir, exist_ok=True)
        with open(os.path.join(local_dir, "file.txt"), "w") as f:
            f.write("data")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    assert downloader.download_hf_dataset("User/ds") is True


def test_download_hf_dataset_no_files(monkeypatch, bound):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: None)
    assert downloader.download_hf_dataset("User/ds") is False
