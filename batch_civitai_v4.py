import subprocess, json, os, time, sys, traceback

BASE_CKPT = "sdXL_base_v1.0.safetensors"
CIVITAI_SCRIPT = "/home/rizzo/.hermes/skills/civitai-api/scripts/civitai.py"
COMFYUI_BASE = "/home/rizzo/ComfyUI"
LOG_FILE = "/home/rizzo/batch_civitai_v4.log"
PROMPT = "Imagine what it would look like for an android to dream of electronic sheep, futuristic, high detail, 8k"
os.makedirs(os.path.join(COMFYUI_BASE, "output"), exist_ok=True)

def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)

def run_civitai(args):
    try:
        result = subprocess.run(["python3", CIVITAI_SCRIPT] + args, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            log(f"Error running civitai {' '.join(args)}: {result.stderr[:200]}")
            return None
        return result.stdout.strip()
    except Exception as e:
        log(f"Exception running civitai: {e}")
        return None

def get_model_info(model_id):
    out = run_civitai(["model", str(model_id)])
    if out is None:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        log(f"Failed to parse JSON for model {model_id}")
        return None

def get_download_url(version_id):
    out = run_civitai(["download-url", str(version_id)])
    if out is None:
        return None
    return out

def get_filename_from_api(model_id):
    info = get_model_info(model_id)
    if info is None:
        return None, None
    versions = info.get("modelVersions", [])
    if not versions:
        return None, None
    version = versions[0]
    version_id = version.get("id")
    files = version.get("files", [])
    if not files:
        return version_id, None
    main_file = None
    for f in files:
        if f.get("type") in ("Main", "Model"):
            main_file = f
            break
    if main_file is None:
        main_file = files[0]
    return version_id, main_file.get("name")

def download_file(url, dest_path):
    subprocess.run(["wget", "--timeout=60", "-q", "-O", dest_path, url], check=False)

def queue_prompt(workflow):
    data = json.dumps({"prompt": workflow})
    resp = subprocess.run(["curl", "-s", "-X", "POST", "http://127.0.0.1:8188/prompt",
                           "-H", "Content-Type: application/json",
                           "-d", data], capture_output=True, text=True, timeout=15)
    try:
        resp_json = json.loads(resp.stdout)
        if "prompt_id" in resp_json:
            return resp_json["prompt_id"]
        else:
            log(f"  ComfyUI error: {resp.stdout[:300]}")
            return None
    except json.JSONDecodeError:
        log(f"  Failed to parse prompt response: {resp.stdout[:200]}")
        return None

def wait_for_completion(prompt_id, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = subprocess.run(["curl", "-s", f"http://127.0.0.1:8188/history/{prompt_id}"], capture_output=True, text=True, timeout=10)
            hist = json.loads(resp.stdout)
            if prompt_id in hist:
                status = hist[prompt_id].get("status", {})
                if status.get("status_str") == "success":
                    return True
                if status.get("status_str") == "error":
                    msgs = status.get("messages", [])
                    log(f"  ComfyUI execution error: {msgs}")
                    return False
        except Exception as e:
            log(f"  History poll error: {e}")
        time.sleep(3)
    log(f"  Timeout waiting for completion ({timeout}s)")
    return False

def get_output_image(prompt_id):
    try:
        resp = subprocess.run(["curl", "-s", f"http://127.0.0.1:8188/history/{prompt_id}"], capture_output=True, text=True, timeout=10)
        hist = json.loads(resp.stdout)
        if prompt_id in hist:
            outputs = hist[prompt_id].get("outputs", {})
            for node_id, out in outputs.items():
                if "images" in out and isinstance(out["images"], list) and len(out["images"]) > 0:
                    img = out["images"][0]
                    filename = img.get("filename")
                    subfolder = img.get("subfolder", "")
                    if subfolder:
                        path = os.path.join(COMFYUI_BASE, "output", subfolder, filename)
                    else:
                        path = os.path.join(COMFYUI_BASE, "output", filename)
                    if os.path.exists(path):
                        return path
    except Exception as e:
        log(f"  Error extracting image: {e}")
    return None

def build_workflow_ckpt(ckpt_name):
    return {
        "1": {"inputs": {"ckpt_name": ckpt_name}, "class_type": "CheckpointLoaderSimple"},
        "2": {"inputs": {"text": PROMPT, "clip": ["1", 1]}, "class_type": "CLIPTextEncode"},
        "3": {"inputs": {"text": "", "clip": ["1", 1]}, "class_type": "CLIPTextEncode"},
        "4": {"inputs": {"width": 512, "height": 512, "batch_size": 1}, "class_type": "EmptyLatentImage"},
        "5": {"inputs": {"seed": 0, "steps": 20, "cfg": 7, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}, "class_type": "KSampler"},
        "6": {"inputs": {"samples": ["5", 0], "vae": ["1", 2]}, "class_type": "VAEDecode"},
        "7": {"inputs": {"images": ["6", 0], "filename_prefix": f"batch_{ckpt_name}"}, "class_type": "SaveImage"}
    }

def build_workflow_lora(lora_name):
    return {
        "1": {"inputs": {"ckpt_name": BASE_CKPT}, "class_type": "CheckpointLoaderSimple"},
        "2": {"inputs": {"lora_name": lora_name, "strength_model": 0.8, "strength_clip": 0.8, "model": ["1", 0], "clip": ["1", 1]}, "class_type": "LoraLoader"},
        "3": {"inputs": {"text": PROMPT, "clip": ["2", 1]}, "class_type": "CLIPTextEncode"},
        "4": {"inputs": {"text": "", "clip": ["2", 1]}, "class_type": "CLIPTextEncode"},
        "5": {"inputs": {"width": 512, "height": 512, "batch_size": 1}, "class_type": "EmptyLatentImage"},
        "6": {"inputs": {"seed": 0, "steps": 20, "cfg": 7, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0, "model": ["2", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["5", 0]}, "class_type": "KSampler"},
        "7": {"inputs": {"samples": ["6", 0], "vae": ["1", 2]}, "class_type": "VAEDecode"},
        "8": {"inputs": {"images": ["7", 0], "filename_prefix": f"batch_lora_{lora_name}"}, "class_type": "SaveImage"}
    }

def build_workflow_vae(vae_name):
    return {
        "1": {"inputs": {"ckpt_name": BASE_CKPT}, "class_type": "CheckpointLoaderSimple"},
        "2": {"inputs": {"text": PROMPT, "clip": ["1", 1]}, "class_type": "CLIPTextEncode"},
        "3": {"inputs": {"text": "", "clip": ["1", 1]}, "class_type": "CLIPTextEncode"},
        "4": {"inputs": {"width": 512, "height": 512, "batch_size": 1}, "class_type": "EmptyLatentImage"},
        "5": {"inputs": {"seed": 0, "steps": 20, "cfg": 7, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}, "class_type": "KSampler"},
        "6": {"inputs": {"samples": ["5", 0], "vae": ["2", 2]}, "class_type": "VAEDecode"},
        "7": {"inputs": {"images": ["6", 0], "filename_prefix": f"batch_vae_{vae_name}"}, "class_type": "SaveImage"}
    }

def process_model(model_id):
    try:
        log(f"Processing model {model_id}")
        version_id, fname = get_filename_from_api(model_id)
        if fname is None:
            log(f"  Could not determine filename")
            return
        log(f"  Filename: {fname}, Version ID: {version_id}")
        
        info = get_model_info(model_id)
        if info is None:
            log(f"  Failed to get info")
            return
        mtype = info.get("type", "").upper()
        log(f"  Type: {mtype}")
        
        if mtype == "CHECKPOINT":
            dest_dir = os.path.join(COMFYUI_BASE, "models", "checkpoints")
            dest_file = os.path.join(dest_dir, fname)
            if os.path.exists(dest_file) and os.path.getsize(dest_file) > 0:
                log(f"  Checkpoint already exists: {fname}")
            else:
                download_url = get_download_url(version_id)
                if download_url is None:
                    log(f"  Failed to get download URL")
                    return
                log(f"  Downloading checkpoint to {fname}")
                download_file(download_url, dest_file)
                time.sleep(2)
            if not os.path.exists(dest_file) or os.path.getsize(dest_file) == 0:
                log(f"  File missing or empty: {dest_file}")
                return
            workflow = build_workflow_ckpt(fname)
        elif mtype == "LORA":
            dest_dir = os.path.join(COMFYUI_BASE, "models", "loras")
            dest_file = os.path.join(dest_dir, fname)
            if os.path.exists(dest_file) and os.path.getsize(dest_file) > 0:
                log(f"  LoRA already exists: {fname}")
            else:
                download_url = get_download_url(version_id)
                if download_url is None:
                    log(f"  Failed to get download URL")
                    return
                log(f"  Downloading LoRA to {fname}")
                download_file(download_url, dest_file)
                time.sleep(2)
            if not os.path.exists(dest_file) or os.path.getsize(dest_file) == 0:
                log(f"  File missing or empty: {dest_file}")
                return
            workflow = build_workflow_lora(fname)
        elif mtype == "VAE":
            dest_dir = os.path.join(COMFYUI_BASE, "models", "vae")
            dest_file = os.path.join(dest_dir, fname)
            if os.path.exists(dest_file) and os.path.getsize(dest_file) > 0:
                log(f"  VAE already exists: {fname}")
            else:
                download_url = get_download_url(version_id)
                if download_url is None:
                    log(f"  Failed to get download URL")
                    return
                log(f"  Downloading VAE to {fname}")
                download_file(download_url, dest_file)
                time.sleep(2)
            if not os.path.exists(dest_file) or os.path.getsize(dest_file) == 0:
                log(f"  File missing or empty: {dest_file}")
                return
            workflow = build_workflow_vae(fname)
        elif mtype == "CONTROLNET":
            dest_dir = os.path.join(COMFYUI_BASE, "models", "controlnet")
            dest_file = os.path.join(dest_dir, fname)
            if os.path.exists(dest_file) and os.path.getsize(dest_file) > 0:
                log(f"  ControlNet already exists: {fname}")
            else:
                download_url = get_download_url(version_id)
                if download_url is None:
                    log(f"  Failed to get download URL")
                    return
                log(f"  Downloading ControlNet to {fname} (skipping generation)")
                download_file(download_url, dest_file)
            time.sleep(2)
            return
        else:
            log(f"  Unknown type {mtype}, skipping")
            return
        
        log(f"  Queueing prompt...")
        prompt_id = queue_prompt(workflow)
        if prompt_id is None:
            log(f"  Failed to queue prompt")
            return
        log(f"  Prompt ID: {prompt_id}")
        if wait_for_completion(prompt_id, timeout=300):
            img_path = get_output_image(prompt_id)
            if img_path and os.path.exists(img_path):
                log(f"  Generated image: {img_path}")
            else:
                log(f"  No output image found")
        else:
            log(f"  Prompt did not complete")
    except Exception as e:
        log(f"  EXCEPTION processing model {model_id}: {e}")
        traceback.print_exc()
    
    log(f"  Finished model {model_id}")

def main():
    list_path = "/home/rizzo/.hermes/cache/documents/doc_8f2905e54f60_list.txt"
    if not os.path.exists(list_path):
        log(f"List file not found: {list_path}")
        return
    ids = set()
    with open(list_path, "r") as f:
        for line in f:
            if "models/" in line:
                parts = line.strip().split("models/")
                for part in parts[1:]:
                    seg = part.split("/")[0]
                    if seg.isdigit():
                        ids.add(seg)
    log(f"Found {len(ids)} unique model IDs")
    for mid in sorted(ids, key=int):
        process_model(mid)
        time.sleep(2)
    log("Batch processing complete.")

if __name__ == "__main__":
    open(LOG_FILE, "w").close()
    main()
