"""Telegram notifier: sendPhoto/sendMessage via Bot API with retries and token redaction."""

from __future__ import annotations

import json
import os
import subprocess
import time

from .config import AppConfig
from .utils import log, redact_token

cfg: AppConfig | None = None


def init(config: AppConfig):
    global cfg
    cfg = config


def _escape_markdown(text: str) -> str:
    """Escape Telegram MarkdownV2 special characters so plain text never breaks parsing."""
    if not text:
        return text
    for ch in r"_*[]()~`>#+-=|{}.!":
        text = text.replace(ch, f"\\{ch}")
    return text


def prepare_telegram_image(image_path):
    """Return a path to an image under ~8MB suitable for sendPhoto.

    If the original PNG is too large, convert to JPEG q90 using PIL if available.
    Returns (path_to_send, should_cleanup) — caller should delete the converted
    file after sending.
    """
    try:
        size = os.path.getsize(image_path)
    except OSError:
        return image_path, False
    if size <= cfg.telegram_max_bytes:
        return image_path, False
    # Too large — try PIL conversion to JPEG q90
    try:
        from PIL import Image

        out_path = os.path.splitext(image_path)[0] + "_tg.jpg"
        with Image.open(image_path) as im:
            if im.mode in ("RGBA", "P", "LA"):
                im = im.convert("RGB")
            im.save(out_path, "JPEG", quality=90, optimize=True)
        new_size = os.path.getsize(out_path)
        log(
            f"  Telegram: image {os.path.basename(image_path)} {size/(1024*1024):.1f}MB -> JPEG {new_size/(1024*1024):.1f}MB"
        )
        return out_path, True
    except ImportError:
        log(
            f"  Telegram: image {os.path.basename(image_path)} {size/(1024*1024):.1f}MB exceeds limit and PIL not available — skipping send"
        )
        return None, False
    except Exception as e:
        log(f"  Telegram: failed to convert {os.path.basename(image_path)}: {e}")
        return None, False


def send_telegram_photo(image_path, caption):
    """Send a generated image to the Telegram home channel via Bot API sendPhoto.

    Hardened: 120s timeout, up to 3 retries with backoff, size capping via JPEG
    conversion, and token redaction in all log output.
    """
    token = cfg.telegram_bot_token or ""
    chat_id = cfg.telegram_home_channel or os.getenv("TELEGRAM_HOME_CHANNEL", "-1004470891735")
    if not token:
        log(f"  Telegram: TELEGRAM_BOT_TOKEN not set, skipping photo delivery")
        return
    # Prepare image (may convert oversized PNG to JPEG)
    send_path, cleanup = prepare_telegram_image(image_path)
    if send_path is None:
        return
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    last_err = None
    for attempt in range(3):
        try:
            resp = subprocess.run(
                ["curl", "-s", "-X", "POST", url,
                 "-F", f"chat_id={chat_id}",
                 "-F", f"caption={_escape_markdown(caption[:1000])}",
                 "-F", f"photo=@{send_path}"],
                capture_output=True, text=True, timeout=120)
            try:
                result = json.loads(resp.stdout)
                if result.get("ok"):
                    log(f"  Telegram: photo sent ({os.path.basename(send_path)})")
                    if cleanup:
                        try:
                            os.remove(send_path)
                        except OSError:
                            pass
                    return
                else:
                    desc = redact_token(result.get("description", "unknown error"))
                    log(f"  Telegram: API error: {desc}")
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                    return
            except (json.JSONDecodeError, ValueError):
                if resp.returncode == 0:
                    log(f"  Telegram: photo sent ({os.path.basename(send_path)})")
                    if cleanup:
                        try:
                            os.remove(send_path)
                        except OSError:
                            pass
                    return
                else:
                    last_err = redact_token(resp.stderr[:200])
                    log(f"  Telegram: send failed (curl RC={resp.returncode}): {last_err}")
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                    return
        except Exception as e:
            last_err = redact_token(str(e))
            log(f"  Telegram: exception: {last_err}")
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
    if cleanup:
        try:
            os.remove(send_path)
        except OSError:
            pass


def send_telegram_message(text):
    """Send a plain text message to the Telegram home channel via Bot API sendMessage.

    Hardened: 120s timeout, up to 3 retries with backoff, token redaction,
    Markdown-escaped text to prevent entity parse errors.
    """
    token = cfg.telegram_bot_token or ""
    chat_id = cfg.telegram_home_channel or os.getenv("TELEGRAM_HOME_CHANNEL", "-1004470891735")
    if not token:
        log(f"  Telegram: TELEGRAM_BOT_TOKEN not set, skipping message delivery")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    last_err = None
    for attempt in range(3):
        try:
            resp = subprocess.run(
                ["curl", "-s", "-X", "POST", url,
                 "-F", f"chat_id={chat_id}",
                 "-F", f"text={_escape_markdown(text[:4000])}",
                 "-F", "parse_mode=MarkdownV2"],
                capture_output=True, text=True, timeout=120)
            try:
                result = json.loads(resp.stdout)
                if result.get("ok"):
                    log(f"  Telegram: message sent ({text[:60]}...)")
                    return
                else:
                    desc = redact_token(result.get("description", "unknown error"))
                    log(f"  Telegram: API error: {desc}")
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                    return
            except (json.JSONDecodeError, ValueError):
                if resp.returncode == 0:
                    log(f"  Telegram: message sent ({text[:60]}...)")
                    return
                else:
                    last_err = redact_token(resp.stderr[:200])
                    log(f"  Telegram: send failed (curl RC={resp.returncode}): {last_err}")
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                    return
        except Exception as e:
            last_err = redact_token(str(e))
            log(f"  Telegram: exception: {last_err}")
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue


def send_telegram_status(text):
    """Send a short status message to the Telegram home channel."""
    send_telegram_message(f"• {text}")
