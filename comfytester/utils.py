"""Shared utilities: logging, token redaction, small filesystem helpers."""

from __future__ import annotations

import fcntl
import os
import time

from .config import AppConfig

_config: AppConfig | None = None


def init(config: AppConfig):
    """Bind the active config (called once from batch.main before anything runs)."""
    global _config
    _config = config


def log(msg):
    """Append a timestamped line to the log file and print it to stdout."""
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    if _config is not None and _config.log_file:
        try:
            with open(_config.log_file, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass
    print(line, flush=True)


def _candidate_tokens():
    """Collect secret token strings currently in effect so they can be redacted from log text.

    Covers the Telegram bot token, the HuggingFace token, and the Civitai API
    token (read from civitai_token_file). Tokens are returned longest-first so a
    longer token is replaced before any shorter token it may contain.
    """
    tokens = []
    for env_name in ("TELEGRAM_BOT_TOKEN", "HF_TOKEN"):
        value = os.getenv(env_name)
        if value:
            tokens.append(value)
    if _config is not None and _config.hf_token:
        tokens.append(_config.hf_token)
    if _config is not None and _config.civitai_token_file:
        try:
            with open(_config.civitai_token_file) as f:
                value = f.read().strip()
        except OSError:
            value = ""
        if value:
            tokens.append(value)
    # Deduplicate, longest first.
    return sorted(set(tokens), key=len, reverse=True)


def redact_token(text):
    """Redact configured secret tokens (Telegram bot token, HF token, Civitai token)
    from any text before logging."""
    if not text:
        return text
    for token in _candidate_tokens():
        text = text.replace(token, "[REDACTED]")
    return text


def find_control_hint_image():
    """Scan input/ then output/ for the newest image to use as ControlNet hint."""
    if _config is None:
        return None
    for scan_dir in [os.path.join(_config.comfyui_base, "input"), os.path.join(_config.comfyui_base, "output")]:
        if not os.path.exists(scan_dir):
            continue
        images = []
        for f in os.listdir(scan_dir):
            if f.lower().endswith(( ".png", ".jpg", ".jpeg", ".webp")):
                images.append(os.path.join(scan_dir, f))
        if images:
            return max(images, key=os.path.getmtime)
    return None


def rotate_log_if_needed():
    """Rotate the log file when it exceeds log_max_bytes, keeping backups."""
    if not _config or not _config.log_file:
        return
    log_file = _config.log_file
    if not os.path.exists(log_file):
        return
    if os.path.getsize(log_file) <= _config.log_max_bytes:
        return
    for i in range(_config.log_backup_count - 1, 0, -1):
        src = f"{log_file}.{i}"
        dst = f"{log_file}.{i + 1}"
        if os.path.exists(src):
            os.rename(src, dst)
    os.rename(log_file, f"{log_file}.1")


def _pid_alive(pid: int) -> bool:
    """Return True if *pid* exists on this system (os.kill(pid, 0) probe)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we lack permission to signal it.
        return True
    except OSError:
        return False
    return True


class FileLock:
    """Cross-run exclusive lock on a file, using fcntl.flock.

    fcntl advisory locks are released automatically by the kernel when the
    holding process dies or closes its file descriptor, so a crash never
    leaves a permanently stuck lock. A stale lock file (left behind by a
    wedged process or a different locking scheme) is detected by checking
    whether the PID stamped inside is still alive; if the holder is dead,
    the file is removed and the lock is retried.
    """

    def __init__(self, path: str):
        self.path = path
        self._fd = None
        self._acquired = False

    # -- internals -------------------------------------------------------

    def _read_holder_pid(self):
        try:
            with open(self.path, "r") as f:
                data = f.read().strip()
        except OSError:
            return None
        try:
            return int(data.split()[0]) if data else None
        except ValueError:
            return None

    def _write_pid(self):
        if self._fd is None:
            return
        try:
            self._fd.seek(0)
            self._fd.truncate()
            self._fd.write(f"{os.getpid()}\n")
            self._fd.flush()
        except OSError:
            pass

    # -- public API ------------------------------------------------------

    def acquire(self, blocking: bool = False) -> bool:
        """Acquire the lock. Returns True on success.

        Returns False when a *live* process already holds the lock. A stale
        lock (holder PID dead or missing) is cleared and retried.
        """
        flags = fcntl.LOCK_EX
        if not blocking:
            flags |= fcntl.LOCK_NB
        while True:
            try:
                self._fd = open(self.path, "a+")
            except OSError as e:
                log(f"Could not open lock file {self.path}: {e}")
                return False
            try:
                fcntl.flock(self._fd, flags)
            except BlockingIOError:
                holder = self._read_holder_pid()
                self._fd.close()
                self._fd = None
                if holder is not None and _pid_alive(holder):
                    log(
                        f"batch_comfytester lock held by PID {holder} "
                        f"({self.path}) — another run is in progress, exiting."
                    )
                    return False
                # Stale lock: holder is dead or no PID — clear and retry.
                log(f"Removing stale lock file {self.path} (holder PID {holder}).")
                try:
                    os.remove(self.path)
                except OSError:
                    pass
                continue
            # Lock acquired — stamp our PID so other runs can identify us.
            self._write_pid()
            self._acquired = True
            log(f"Acquired batch lock: {self.path}")
            return True

    def release(self):
        """Release the lock and close the file descriptor."""
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                self._fd.close()
            except OSError:
                pass
            self._fd = None
        self._acquired = False

    @property
    def acquired(self) -> bool:
        return self._acquired

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc_info):
        self.release()
        return False
