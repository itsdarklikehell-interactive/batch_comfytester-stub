"""ComfyUI workflow builders for each model family and model type."""

from __future__ import annotations

import os
import random

from .config import AppConfig
from .utils import find_control_hint_image, log

cfg: AppConfig | None = None


def init(config: AppConfig):
    global cfg
    cfg = config


def build_workflow_ckpt(ckpt_name, prompt, negative, settings, vae_name=None, family="photorealistic", output_dir=None, date_folder=None):
    """Build a checkpoint text-to-image workflow with family-specific settings.

    Output is sorted into logical folders:
      <output_dir>[/<date_folder>/]<family>/<checkpoint_base>/batch_<checkpoint>_<vae>
    """
    width, height = settings["resolution"]
    seed = random.randint(settings["seed_range"][0], settings["seed_range"][1])

    vae_suffix = f"_{vae_name}" if vae_name else ""
    ckpt_base = os.path.splitext(os.path.basename(ckpt_name))[0]
    out_dir = output_dir if output_dir else (cfg.ckpt_output_dir if cfg else "checkpoints")
    date_prefix = f"{date_folder}/" if date_folder else ""
    prefix = f"{out_dir}/{date_prefix}{family}/{ckpt_base}/batch_{ckpt_base}{vae_suffix}"

    wf = {
        "1": {"inputs": {"ckpt_name": ckpt_name}, "class_type": "CheckpointLoaderSimple"},
        "2": {"inputs": {"text": prompt, "clip": ["1", 1]}, "class_type": "CLIPTextEncode"},
        "3": {"inputs": {"text": negative, "clip": ["1", 1]}, "class_type": "CLIPTextEncode"},
        "4": {"inputs": {"width": width, "height": height, "batch_size": 1}, "class_type": "EmptyLatentImage"},
        "5": {"inputs": {"seed": seed, "steps": settings["steps"], "cfg": settings["cfg"],
                         "sampler_name": settings["sampler"], "scheduler": settings["scheduler"],
                         "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0],
                         "negative": ["3", 0], "latent_image": ["4", 0]},
             "class_type": "KSampler"},
    }
    if vae_name:
        wf["8"] = {"inputs": {"vae_name": vae_name}, "class_type": "VAELoader"}
        wf["9"] = {"inputs": {"samples": ["5", 0], "vae": ["8", 0]}, "class_type": "VAEDecode"}
        wf["10"] = {"inputs": {"images": ["9", 0], "filename_prefix": prefix},
                    "class_type": "SaveImage"}
    else:
        wf["6"] = {"inputs": {"samples": ["5", 0], "vae": ["1", 2]}, "class_type": "VAEDecode"}
        wf["7"] = {"inputs": {"images": ["6", 0], "filename_prefix": prefix},
                   "class_type": "SaveImage"}
    return wf, seed


def build_workflow_lora(lora_name, prompt, negative, settings, strength=0.8, vae_name=None, family="photorealistic", output_dir=None, date_folder=None):
    """Build a LoRA workflow with family-specific settings.

    Output is sorted into logical folders:
      <output_dir>[/<date_folder>/]<family>/<lora_base>/batch_lora_<lora>_s<strength>_<vae>
    """
    width, height = settings["resolution"]
    seed = random.randint(settings["seed_range"][0], settings["seed_range"][1])

    strength_suffix = f"_s{strength}"
    vae_suffix = f"_{vae_name}" if vae_name else ""
    lora_base = os.path.splitext(os.path.basename(lora_name))[0]
    out_dir = output_dir if output_dir else (cfg.lora_output_dir if cfg else "loras")
    date_prefix = f"{date_folder}/" if date_folder else ""
    prefix = f"{out_dir}/{date_prefix}{family}/{lora_base}/batch_lora_{lora_base}{strength_suffix}{vae_suffix}"

    wf = {
        "1": {"inputs": {"ckpt_name": cfg.base_ckpt}, "class_type": "CheckpointLoaderSimple"},
        "2": {"inputs": {"lora_name": lora_name, "strength_model": strength, "strength_clip": strength,
                         "model": ["1", 0], "clip": ["1", 1]}, "class_type": "LoraLoader"},
        "3": {"inputs": {"text": prompt, "clip": ["2", 1]}, "class_type": "CLIPTextEncode"},
        "4": {"inputs": {"text": negative, "clip": ["2", 1]}, "class_type": "CLIPTextEncode"},
        "5": {"inputs": {"width": width, "height": height, "batch_size": 1}, "class_type": "EmptyLatentImage"},
        "6": {"inputs": {"seed": seed, "steps": settings["steps"], "cfg": settings["cfg"],
                         "sampler_name": settings["sampler"], "scheduler": settings["scheduler"],
                         "denoise": 1.0, "model": ["2", 0], "positive": ["3", 0],
                         "negative": ["4", 0], "latent_image": ["5", 0]},
             "class_type": "KSampler"},
    }
    if vae_name:
        wf["9"] = {"inputs": {"vae_name": vae_name}, "class_type": "VAELoader"}
        wf["10"] = {"inputs": {"samples": ["6", 0], "vae": ["9", 0]}, "class_type": "VAEDecode"}
        wf["11"] = {"inputs": {"images": ["10", 0], "filename_prefix": prefix},
                    "class_type": "SaveImage"}
    else:
        wf["7"] = {"inputs": {"samples": ["6", 0], "vae": ["1", 2]}, "class_type": "VAEDecode"}
        wf["8"] = {"inputs": {"images": ["7", 0], "filename_prefix": prefix},
                   "class_type": "SaveImage"}
    return wf, seed


def build_workflow_vae(vae_name, prompt, negative, settings, family="photorealistic", output_dir=None, date_folder=None):
    """Build a VAE swap workflow with family-specific settings.

    Output is sorted into logical folders:
      <output_dir>[/<date_folder>/]<family>/<vae_base>/batch_vae_<vae>
    """
    width, height = settings["resolution"]
    seed = random.randint(settings["seed_range"][0], settings["seed_range"][1])

    vae_base = os.path.splitext(os.path.basename(vae_name))[0]
    out_dir = output_dir if output_dir else (cfg.vae_output_dir if cfg else "vaes")
    date_prefix = f"{date_folder}/" if date_folder else ""
    prefix = f"{out_dir}/{date_prefix}{family}/{vae_base}/batch_vae_{vae_base}"

    wf = {
        "1": {"inputs": {"ckpt_name": cfg.base_ckpt}, "class_type": "CheckpointLoaderSimple"},
        "2": {"inputs": {"vae_name": vae_name}, "class_type": "VAELoader"},
        "3": {"inputs": {"text": prompt, "clip": ["1", 1]}, "class_type": "CLIPTextEncode"},
        "4": {"inputs": {"text": negative, "clip": ["1", 1]}, "class_type": "CLIPTextEncode"},
        "5": {"inputs": {"width": width, "height": height, "batch_size": 1}, "class_type": "EmptyLatentImage"},
        "6": {"inputs": {"seed": seed, "steps": settings["steps"], "cfg": settings["cfg"],
                         "sampler_name": settings["sampler"], "scheduler": settings["scheduler"],
                         "denoise": 1.0, "model": ["1", 0], "positive": ["3", 0],
                         "negative": ["4", 0], "latent_image": ["5", 0]},
             "class_type": "KSampler"},
        "7": {"inputs": {"samples": ["6", 0], "vae": ["2", 0]}, "class_type": "VAEDecode"},
        "8": {"inputs": {"images": ["7", 0], "filename_prefix": prefix},
              "class_type": "SaveImage"},
    }
    return wf, seed


def build_workflow_controlnet(cn_name, prompt, negative, settings, control_image=None, family="photorealistic", output_dir=None, date_folder=None):
    """Build a ControlNet workflow using ControlNetLoader + ControlNetApply.

    Output is sorted into logical folders:
      <output_dir>[/<date_folder>/]<family>/<cn_base>/batch_cn_<cn>
    """
    width, height = settings["resolution"]
    seed = random.randint(settings["seed_range"][0], settings["seed_range"][1])

    if control_image is None:
        control_image = find_control_hint_image()
        if control_image is None:
            control_image = "test_controlnet_00001_.png"  # last-resort fallback

    cn_base = os.path.splitext(os.path.basename(cn_name))[0]
    out_dir = output_dir if output_dir else (cfg.cn_output_dir if cfg else "controlnet")
    date_prefix = f"{date_folder}/" if date_folder else ""
    prefix = f"{out_dir}/{date_prefix}{family}/{cn_base}/batch_cn_{cn_base}"

    wf = {
        "1": {"inputs": {"ckpt_name": cfg.base_ckpt}, "class_type": "CheckpointLoaderSimple"},
        "2": {"inputs": {"image": control_image, "image_upload": True}, "class_type": "LoadImage"},
        "3": {"inputs": {"control_net_name": cn_name}, "class_type": "ControlNetLoader"},
        "4": {"inputs": {"conditioning": ["5", 0], "control_net": ["3", 0], "image": ["2", 0], "strength": 1.0},
              "class_type": "ControlNetApply"},
        "5": {"inputs": {"text": prompt, "clip": ["1", 1]}, "class_type": "CLIPTextEncode"},
        "6": {"inputs": {"text": negative, "clip": ["1", 1]}, "class_type": "CLIPTextEncode"},
        "7": {"inputs": {"width": width, "height": height, "batch_size": 1}, "class_type": "EmptyLatentImage"},
        "8": {"inputs": {"seed": seed, "steps": settings["steps"], "cfg": settings["cfg"],
                         "sampler_name": settings["sampler"], "scheduler": settings["scheduler"],
                         "denoise": 1.0, "model": ["1", 0], "positive": ["4", 0],
                         "negative": ["6", 0], "latent_image": ["7", 0]},
             "class_type": "KSampler"},
        "9": {"inputs": {"samples": ["8", 0], "vae": ["1", 2]}, "class_type": "VAEDecode"},
        "10": {"inputs": {"images": ["9", 0], "filename_prefix": prefix},
               "class_type": "SaveImage"},
    }
    return wf, seed


def build_workflow_ckpt_combo(ckpt_name, lora_names, lora_strengths, prompt, negative, settings, vae_name=None, family="photorealistic", output_dir="combos", arch_group=None, date_folder=None):
    """Build a checkpoint + LoRA(s) combination workflow.

    Chains multiple LoraLoader nodes between the CheckpointLoader and the
    CLIP text encoders, so all LoRAs are applied to the same base model.
    Embedding tokens (Textual Inversion) should already be baked into the
    prompt/negative strings by the caller.

    Output is sorted into logical folders:
      <output_dir>[/<date_folder>/]<arch_group>/<checkpoint_base>/batch_combo_<loras>_<emb>
    """
    width, height = settings["resolution"]
    seed = random.randint(settings["seed_range"][0], settings["seed_range"][1])

    vae_suffix = f"_{vae_name}" if vae_name else ""
    lora_suffix = "_".join(f"{os.path.splitext(n)[0]}:{s}" for n, s in zip(lora_names, lora_strengths))
    ckpt_base = os.path.splitext(os.path.basename(ckpt_name))[0]

    arch = arch_group if arch_group else "uncategorized"
    date_prefix = f"{date_folder}/" if date_folder else ""
    prefix = f"{output_dir}/{date_prefix}{arch}/{ckpt_base}/batch_combo_{lora_suffix}{vae_suffix}"

    wf = {
        "1": {"inputs": {"ckpt_name": ckpt_name}, "class_type": "CheckpointLoaderSimple"},
    }

    # Chain LoRA loaders: each takes model+clip from the previous node
    model_ref = ["1", 0]
    clip_ref = ["1", 1]
    for i, (lora_name, strength) in enumerate(zip(lora_names, lora_strengths)):
        node_id = str(10 + i)
        wf[node_id] = {
            "inputs": {
                "lora_name": lora_name,
                "strength_model": strength,
                "strength_clip": strength,
                "model": model_ref,
                "clip": clip_ref,
            },
            "class_type": "LoraLoader",
        }
        model_ref = [node_id, 0]
        clip_ref = [node_id, 1]

    # CLIP text encode (using CLIP from the last LoRA in the chain)
    wf["2"] = {"inputs": {"text": prompt, "clip": clip_ref}, "class_type": "CLIPTextEncode"}
    wf["3"] = {"inputs": {"text": negative, "clip": clip_ref}, "class_type": "CLIPTextEncode"}
    wf["4"] = {"inputs": {"width": width, "height": height, "batch_size": 1}, "class_type": "EmptyLatentImage"}
    wf["5"] = {
        "inputs": {
            "seed": seed, "steps": settings["steps"], "cfg": settings["cfg"],
            "sampler_name": settings["sampler"], "scheduler": settings["scheduler"],
            "denoise": 1.0, "model": model_ref, "positive": ["2", 0],
            "negative": ["3", 0], "latent_image": ["4", 0],
        },
        "class_type": "KSampler",
    }

    node_idx = 6
    if vae_name:
        wf[str(node_idx)] = {"inputs": {"vae_name": vae_name}, "class_type": "VAELoader"}
        vae_ref = [str(node_idx), 0]
        node_idx += 1
        wf[str(node_idx)] = {"inputs": {"samples": ["5", 0], "vae": vae_ref}, "class_type": "VAEDecode"}
    else:
        wf[str(node_idx)] = {"inputs": {"samples": ["5", 0], "vae": ["1", 2]}, "class_type": "VAEDecode"}
    node_idx += 1

    wf[str(node_idx)] = {
        "inputs": {"images": [str(node_idx - 1), 0], "filename_prefix": prefix},
        "class_type": "SaveImage",
    }
    return wf, seed


def build_workflow_qwen_image_2_1(ckpt_name, prompt, negative, settings, family="qwen_image", output_dir=None, date_folder=None):
    """Build Qwen Image 2.1 workflow using native ComfyUI Qwen nodes.

    Uses: Load Diffusion Model (qwen_image_2.1_int8_convrot) + Load CLIP (qwen3vl_8b, type qwen_image)
    + VAELoader (qwen_image_2.1_vae_bf16) + TextEncodeQwenImage21 + KSampler + VAEDecode + SaveImage

    Output is sorted into logical folders:
      <output_dir>[/<date_folder>/]<family>/<ckpt_base>/batch_qwen_<ckpt>
    """
    width, height = settings["resolution"]
    seed = random.randint(settings["seed_range"][0], settings["seed_range"][1])

    ckpt_base = os.path.splitext(os.path.basename(ckpt_name))[0]
    out_dir = output_dir if output_dir else (cfg.qwen_output_dir if cfg else "qwen")
    date_prefix = f"{date_folder}/" if date_folder else ""
    prefix = f"{out_dir}/{date_prefix}{family}/{ckpt_base}/batch_qwen_{ckpt_base}"

    wf = {
        "1": {"inputs": {"model_name": ckpt_name}, "class_type": "Load Diffusion Model"},
        "2": {"inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image"},
              "class_type": "Load CLIP"},
        "3": {"inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}, "class_type": "VAELoader"},
        "4": {"inputs": {"clip": ["2", 0], "text": prompt}, "class_type": "TextEncodeQwenImage21"},
        "5": {"inputs": {"clip": ["2", 0], "text": negative}, "class_type": "TextEncodeQwenImage21"},
        "6": {"inputs": {"width": width, "height": height, "batch_size": 1}, "class_type": "EmptyLatentImage"},
        "7": {"inputs": {"seed": seed, "steps": settings["steps"], "cfg": settings["cfg"],
                         "sampler_name": settings["sampler"], "scheduler": settings["scheduler"],
                         "denoise": 1.0, "model": ["1", 0], "positive": ["4", 0],
                         "negative": ["5", 0], "latent_image": ["6", 0]},
              "class_type": "KSampler"},
        "8": {"inputs": {"samples": ["7", 0], "vae": ["3", 0]}, "class_type": "VAEDecode"},
        "9": {"inputs": {"images": ["8", 0], "filename_prefix": prefix},
              "class_type": "SaveImage"},
    }
    return wf, seed


def build_workflow_anima(ckpt_name, prompt, negative, settings, family="anima", output_dir=None, date_folder=None):
    """Build Anima workflow using UNETLoader + CLIPLoader (Qwen3-0.6B) + VAELoader.

    Uses: UNETLoader (anima-turbo-v1.0) + CLIPLoader (qwen_3_06b_base, type stable_diffusion)
    + VAELoader (qwen_image_vae) + CLIPTextEncode + KSampler + VAEDecode + SaveImage

    Output is sorted into logical folders:
      <output_dir>[/<date_folder>/]<family>/<ckpt_base>/batch_anima_<ckpt>
    """
    width, height = settings["resolution"]
    seed = random.randint(settings["seed_range"][0], settings["seed_range"][1])

    ckpt_base = os.path.splitext(os.path.basename(ckpt_name))[0]
    out_dir = output_dir if output_dir else (cfg.anima_output_dir if cfg else "anima")
    date_prefix = f"{date_folder}/" if date_folder else ""
    prefix = f"{out_dir}/{date_prefix}{family}/{ckpt_base}/batch_anima_{ckpt_base}"

    wf = {
        "1": {"inputs": {"unet_name": ckpt_name}, "class_type": "UNETLoader"},
        "2": {"inputs": {"clip_name": "qwen_3_06b_base.safetensors", "type": "stable_diffusion"},
              "class_type": "CLIPLoader"},
        "3": {"inputs": {"vae_name": "qwen_image_vae.safetensors"}, "class_type": "VAELoader"},
        "4": {"inputs": {"text": prompt, "clip": ["2", 1]}, "class_type": "CLIPTextEncode"},
        "5": {"inputs": {"text": negative, "clip": ["2", 1]}, "class_type": "CLIPTextEncode"},
        "6": {"inputs": {"width": width, "height": height, "batch_size": 1}, "class_type": "EmptyLatentImage"},
        "7": {"inputs": {"seed": seed, "steps": settings["steps"], "cfg": settings["cfg"],
                         "sampler_name": settings["sampler"], "scheduler": settings["scheduler"],
                         "denoise": 1.0, "model": ["1", 0], "positive": ["4", 0],
                         "negative": ["5", 0], "latent_image": ["6", 0]},
             "class_type": "KSampler"},
        "8": {"inputs": {"samples": ["7", 0], "vae": ["3", 0]}, "class_type": "VAEDecode"},
        "9": {"inputs": {"images": ["8", 0], "filename_prefix": prefix},
              "class_type": "SaveImage"},
    }
    return wf, seed


def build_workflow_wan22_video(model_name, prompt, negative, settings, family="wan22_video", output_dir=None, date_folder=None):
    """Build Wan2.2 video generation workflow.

    Uses Wan2.2 text-to-video nodes with dynamic offloading for RTX 3070.
    Outputs video file instead of image.

    Output is sorted into logical folders:
      <output_dir>[/<date_folder>/]<family>/<model_base>/batch_wan22_<model>
    """
    width, height = settings["resolution"]
    seed = random.randint(settings["seed_range"][0], settings["seed_range"][1])

    model_base = os.path.splitext(os.path.basename(model_name))[0]
    out_dir = output_dir if output_dir else (cfg.wan22_output_dir if cfg else "wan22")
    date_prefix = f"{date_folder}/" if date_folder else ""
    prefix = f"{out_dir}/{date_prefix}{family}/{model_base}/batch_wan22_{model_base}"

    wf = {
        "1": {"inputs": {"model_name": model_name}, "class_type": "Load Diffusion Model"},
        "2": {"inputs": {"vae_name": "wan2.2_vae.safetensors"}, "class_type": "VAELoader"},
        "3": {"inputs": {"text": prompt}, "class_type": "CLIPTextEncode"},
        "4": {"inputs": {"text": negative}, "class_type": "CLIPTextEncode"},
        "5": {"inputs": {"width": width, "height": height, "length": 81, "batch_size": 1},
              "class_type": "EmptyLatentVideo"},
        "6": {"inputs": {"seed": seed, "steps": settings["steps"], "cfg": settings["cfg"],
                         "sampler_name": settings["sampler"], "scheduler": settings["scheduler"],
                         "denoise": 1.0, "model": ["1", 0], "positive": ["3", 0],
                         "negative": ["4", 0], "latent_image": ["5", 0]},
             "class_type": "KSampler"},
        "7": {"inputs": {"samples": ["6", 0], "vae": ["2", 0]}, "class_type": "VAEDecode"},
        "8": {"inputs": {"images": ["7", 0], "filename_prefix": prefix,
                         "frame_rate": 8}, "class_type": "SaveImage"},
    }
    return wf, seed
