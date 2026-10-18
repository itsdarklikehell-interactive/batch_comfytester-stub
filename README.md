# batch_comfytester

Batch Civitai/ComfyUI model tester — automated AI image/video generation testing.

Refactored from a ~3000-line monolith into `comfytester` modules: config, civitai, downloader, families, state, telegram, utils, comfyui, workflows, batch.

## What it does

- Collects Civitai model IDs from favorites, followed creators, collections, and ID-list files
- Downloads checkpoints, LoRAs, VAEs, ControlNets, and other model types
- Generates images/video via ComfyUI with family-specific workflows (SD1.x, SDXL, Pony, Illustrious, Furry, Flux, Qwen Image, Anima, Wan2.2 video, and more)
- Sorts outputs into per-family, per-model folders with optional date folders
- Runs combination-testing mode (checkpoint × compatible LoRAs × embeddings)
- Reports per-model timings and posts renders to Telegram

## Project structure

```
batch_comfytester.py          # Thin CLI entry point (argparse)
comfytester/                  # Refactored package
  __init__.py                 # Package exports
  config.py                   # AppConfig + load_config() (no hardcoded /home/rizzo)
  civitai.py                  # Civitai API client
  downloader.py               # Generic + HF Hub downloaders
  families.py                 # Family detection, prompt pools, settings, VRAM limits
  state.py                    # Completed/failed/priority/skip tracking, metadata sidecars
  telegram.py                 # Telegram notifier
  utils.py                    # Logging, redaction, helpers
  comfyui.py                  # ComfyUI client (queue, health, restart, history, upscale)
  workflows.py                # ComfyUI workflow builders per family/type
  batch.py                    # Orchestration: process_model, id collection, main loop
tests/                        # pytest unit tests (all network/subprocess mocked)
```

## Requirements

Python 3.10+. No runtime dependencies beyond the standard library.

Tests require `pytest`:

```bash
pip install pytest
```

## Running

```bash
# Show loaded prompt pools (dry inspection, no generation)
python3 batch_comfytester.py --show-prompt-pools

# Dry run: list what would be processed
python3 batch_comfytester.py --dry-run --limit 5

# Process up to 10 Civitai models
python3 batch_comfytester.py --limit 10

# Combination-testing mode
python3 batch_comfytester.py --combo

# NSFW mode
python3 batch_comfytester.py --nsfw --limit 5

# Disable Telegram notifications
python3 batch_comfytester.py --no-telegram --limit 5

# Override ComfyUI base directory
python3 batch_comfytester.py --comfyui-base /path/to/ComfyUI --limit 5

# Filter by model family
python3 batch_comfytester.py --family flux --limit 5

# Process specific model IDs
python3 batch_comfytester.py --ids 12345,67890
```

Config precedence (low → high): dataclass defaults → JSON config file → `batch_comfytester.env` → environment variables → CLI arguments.

## Testing

```bash
python3 -m py_compile comfytester/*.py batch_comfytester.py
python3 -m pytest tests/ -q
```

89 tests pass; 4 are pre-existing failures (Civitai API skip + `civitai_max_retries` config default) documented in `batch-comfytester-development` skill — not regressions. `test_telegram.py` requires PIL and is excluded from the headless run.

## License

MIT