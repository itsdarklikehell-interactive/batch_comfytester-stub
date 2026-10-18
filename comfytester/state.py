"""State tracker: completed / failed / priority / skip lists and metadata sidecars."""

from __future__ import annotations

import json
import os
import time

from .config import AppConfig
from .telegram import send_telegram_status
from .utils import log

cfg: AppConfig | None = None


def init(config: AppConfig):
    global cfg
    cfg = config


def load_completed_models():
    """Load set of already-completed model IDs for resume tracking."""
    completed = set()
    if os.path.exists(cfg.completed_file):
        with open(cfg.completed_file) as f:
            for line in f:
                mid = line.strip()
                if mid:
                    completed.add(mid)
    return completed


def mark_completed(model_id):
    """Mark a model ID as completed in the resume tracking file."""
    with open(cfg.completed_file, "a") as f:
        f.write(f"{model_id}\n")


def mark_failed(model_id):
    """Mark a model ID as failed (download/generation) so it can be retried on next run."""
    with open(cfg.failed_file, "a") as f:
        f.write(f"{model_id}\n")
    send_telegram_status(f"failed model {model_id}")


def load_failed_models():
    """Load set of previously-failed model IDs for retry on next run."""
    failed = set()
    if os.path.exists(cfg.failed_file):
        with open(cfg.failed_file) as f:
            for line in f:
                mid = line.strip()
                if mid:
                    failed.add(mid)
    return failed


def load_completed_datasets():
    """Load set of already-downloaded dataset repo IDs for resume tracking."""
    completed = set()
    if os.path.exists(cfg.completed_datasets_file):
        with open(cfg.completed_datasets_file) as f:
            for line in f:
                did = line.strip()
                if did:
                    completed.add(did)
    return completed


def mark_completed_dataset(dataset_id):
    """Mark a dataset repo ID as downloaded/completed in the tracking file."""
    with open(cfg.completed_datasets_file, "a") as f:
        f.write(f"{dataset_id}\n")


def load_priority_skip_lists():
    """Load priority and skip model ID lists."""
    priority = []
    skip = set()
    if os.path.exists(cfg.priority_file):
        with open(cfg.priority_file) as f:
            for line in f:
                mid = line.strip()
                if mid:
                    priority.append(mid)
    if os.path.exists(cfg.skip_file):
        with open(cfg.skip_file) as f:
            for line in f:
                mid = line.strip()
                if mid:
                    skip.add(mid)
    return priority, skip


def save_metadata_sidecar(image_path, model_id, model_name, mtype, family, prompt, negative, seed, settings, extra=None):
    """Write a JSON sidecar next to a generated image with full generation metadata."""
    meta_path = os.path.splitext(image_path)[0] + ".meta.json"
    meta = {
        "image": os.path.basename(image_path),
        "model_id": str(model_id),
        "model_name": model_name,
        "model_type": mtype,
        "family": family,
        "prompt": prompt,
        "negative_prompt": negative,
        "seed": seed,
        "settings": settings,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        meta.update(extra)
    try:
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        log(f"  Metadata sidecar: {os.path.basename(meta_path)}")
    except Exception as e:
        log(f"  Failed to write metadata sidecar: {e}")
