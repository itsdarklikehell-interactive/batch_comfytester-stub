"""batch_comfytester — thin CLI entry point.

Delegates all work to the comfytester modules (config, civitai, downloader,
families, state, telegram, utils, comfyui, workflows, batch). Replaces the
old 3000-line monolith.

Paths and behaviour come from AppConfig with this precedence (low -> high):
  1. dataclass defaults            (mirror the old hardcoded /home/rizzo values)
  2. JSON config file (--config)
  3. batch_comfytester.env (KEY=value)
  4. Environment variables
  5. CLI arguments (--dry-run, --family, --ids, --limit, ...)
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running both as `python3 batch_comfytester.py` (cwd on sys.path) and
# as `python3 -m comfytester.__main__` from an installed package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from comfytester import batch
from comfytester.config import load_config
from comfytester import civitai, comfyui, downloader, families, state, telegram, utils, workflows
from comfytester.utils import FileLock


def build_argparser():
    p = argparse.ArgumentParser(
        prog="batch_comfytester.py",
        description="Batch Civitai/ComfyUI model tester — refactored into comfytester modules.",
    )
    p.add_argument("--config", default=None, help="JSON config file path "
                        "(default: batch_comfytester.config.json next to this script)")
    p.add_argument("--comfyui-base", default=None,
                   help="ComfyUI base directory (overrides config/env)")
    p.add_argument("--dry-run", action="store_true",
                   help="List what would be processed without downloading/generating")
    p.add_argument("--family", default=None,
                   help="Only process models of this family (family filter)")
    p.add_argument("--ids", default=None,
                   help="Comma-separated model IDs to process (ID filter)")
    p.add_argument("--limit", type=int, default=None,
                   help="Max number of models to process")
    p.add_argument("--combo", action="store_true",
                   help="Enable combination-testing mode (checkpoint x LoRA x embedding)")
    p.add_argument("--nsfw", action="store_true", help="Enable NSFW mode")
    p.add_argument(
        "--prompt-categories",
        default=None,
        help="Comma-separated prompt category order override (applies to all families)",
    )
    p.add_argument(
        "--show-prompt-pools",
        action="store_true",
        help="Print the loaded prompt pools and exit without generating anything",
    )
    p.add_argument("--no-telegram", action="store_true", help="Disable Telegram notifications")
    p.add_argument("--log-file", default=None, help="Log file path")
    p.add_argument("--state-file", default=None, help="JSON state file path")
    return p


def main(argv=None):
    parser = build_argparser()
    args = parser.parse_args(argv)

    # --- Config: defaults -> JSON -> .env -> env vars -> CLI overrides ---
    config_path = args.config
    if config_path is None:
        default_config = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "batch_comfytester.config.json")
        if os.path.exists(default_config):
            config_path = default_config
    cfg = load_config(config_path)

    # CLI overrides (highest precedence)
    if args.comfyui_base:
        cfg.comfyui_base = args.comfyui_base
    if args.dry_run:
        cfg.dry_run = True
    if args.family:
        cfg.family_filter = args.family
    if args.ids:
        raw_ids = [mid.strip() for mid in args.ids.split(",") if mid.strip()]
        invalid = [mid for mid in raw_ids if not mid.isdigit()]
        if invalid:
            parser.error(
                f"--ids contains non-numeric model ID(s): {invalid}. "
                "Civitai model IDs must be integers."
            )
        cfg.id_filter = raw_ids
    if args.limit is not None:
        cfg.limit = args.limit
    if args.combo:
        cfg.combo_mode = True
    if args.nsfw:
        cfg.nsfw_mode = True
    if args.no_telegram:
        cfg.telegram_bot_token = ""
    if args.log_file:
        cfg.log_file = args.log_file
    if args.state_file:
        cfg.state_file = args.state_file

    # --- Bind config into every module ---
    civitai.init(cfg)
    comfyui.init(cfg)
    downloader.init(cfg)
    families.init(cfg)
    state.init(cfg)
    telegram.init(cfg)
    utils.init(cfg)
    workflows.init(cfg)
    batch.init(cfg)

    families.load_prompt_pools(cfg)
    if args.show_prompt_pools:
        for family in sorted(families.PROMPT_POOLS):
            pool = families.PROMPT_POOLS[family]
            cats = ", ".join(
                f"{name}: {len(entries)}" for name, entries in pool.categories.items()
            )
            print(
                f"{family} [{pool.dialect}] v{pool.version} template={pool.template!r} | {cats}"
            )
        return 0

    if args.prompt_categories:
        cfg.prompt_category_order = [
            c.strip() for c in args.prompt_categories.split(",") if c.strip()
        ]

    # --- Ensure runtime directories exist ---
    os.makedirs(os.path.join(cfg.comfyui_base, "output"), exist_ok=True)
    os.makedirs(cfg.metadata_dir, exist_ok=True)
    os.makedirs(cfg.hf_datasets_dir, exist_ok=True)

    log = utils.log
    log(f"batch_comfytester refactor: dry_run={cfg.dry_run} family_filter={cfg.family_filter or '-'} "
        f"id_filter={cfg.id_filter or '-'} limit={cfg.limit} combo_mode={cfg.combo_mode}")
    log(f"ComfyUI base: {cfg.comfyui_base} | log: {cfg.log_file} | state: {cfg.state_file}")

    # --- Single-instance lock: acquire before processing, release on exit ---
    lock = FileLock(cfg.lock_file)
    if not lock.acquire():
        return 1
    try:
        batch.main(cfg)
    finally:
        lock.release()

    return 0


if __name__ == "__main__":
    sys.exit(main())
