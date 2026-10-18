"""comfytester package: refactored batch_comfytester modules.

Modules:
  config        - AppConfig + load_config() (paths/behaviour, no hardcoded /home/rizzo)
  civitai       - Civitai downloader: model info, download URLs, profile fetches
  downloader    - Generic file downloaders (curl/wget + huggingface_hub)
  families      - Model family detection, prompt pools, generation settings, VRAM limits
  state         - State tracker: completed/failed/priority/skip lists, metadata sidecars
  telegram      - Telegram notifier (sendPhoto/sendMessage with retries + redaction)
  utils         - Shared utilities: logging, token redaction, helpers
  comfyui       - ComfyUI client: prompt queue, health checks, restarts, history polling, upscaling
  workflows     - ComfyUI workflow builders per model family/type
  batch         - Orchestration: process_model, id collection, main batch loop

Entry point: batch_comfytester.py (thin CLI, argparse, delegates to these modules).
"""

__all__ = [
    "config",
    "civitai",
    "downloader",
    "families",
    "state",
    "telegram",
    "utils",
    "comfyui",
    "workflows",
    "batch",
]
