"""Runs ON the Colab VM via `colab exec -f` (rendered by scripts/colab_backend.sh).

Installs ComfyUI, downloads MiniMax H3 weights, launches the ComfyUI server and
a cloudflared tunnel as detached background processes, waits for readiness, and
prints COMFYUI_URL=<public tunnel url>. Safe to re-run (idempotent).
"""
import os
import re
import subprocess
import sys
import time
import urllib.request

COMFY = "/content/ComfyUI"
PORT = 8188
MODES = "@MODES@".split(",")
DIFFUSION_FILE = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"  # or ..._fp8_scaled.safetensors
REPO = "Comfy-Org/MiniMax-H3"


def sh(*args):
    print("+", *args, flush=True)
    subprocess.run(args, check=True)


sh("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv")

if not os.path.isdir(COMFY):
    sh("git", "clone", "--depth", "1", "https://github.com/comfyanonymous/ComfyUI", COMFY)
sh(sys.executable, "-m", "pip", "install", "-q", "-r", f"{COMFY}/requirements.txt", "hf_transfer")

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download

files = [
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "vae/minimax_h3_video_vae_int8_convrot.safetensors",
    "vae/minimax_h3_audio_vae_fp32.safetensors",
]
if set(MODES) & {"t2v", "i2v"}:
    files += [f"diffusion_models/{DIFFUSION_FILE}",
              "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"]
if "r2v" in MODES:
    files += ["diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
              "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"]

for rel in files:
    dst = f"{COMFY}/models/{rel}"
    if os.path.lexists(dst):
        print("exists:", rel)
        continue
    src = hf_hub_download(REPO, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(src, dst)
    print("linked:", rel)

cf = "/content/cloudflared"
if not os.path.exists(cf):
    sh("wget", "-q",
       "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
       "-O", cf)
    sh("chmod", "+x", cf)

subprocess.run(["pkill", "-f", "main.py --listen"], check=False)
with open("/content/comfyui.log", "w") as log:
    subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(PORT)],
                     cwd=COMFY, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

subprocess.run(["pkill", "-f", "cloudflared-linux"], check=False)
with open("/content/tunnel.log", "w") as log:
    subprocess.Popen([cf, "tunnel", "--url", f"http://127.0.0.1:{PORT}", "--no-autoupdate"],
                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

url = None
for _ in range(120):
    try:
        m = re.search(r"https://[\w-]+\.trycloudflare\.com", open("/content/tunnel.log").read())
        if m:
            url = m.group(0)
            break
    except FileNotFoundError:
        pass
    time.sleep(1)

for _ in range(150):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/system_stats", timeout=2)
        break
    except Exception:
        time.sleep(2)

assert url, "tunnel URL not found; check /content/tunnel.log on the VM"
with open("/content/COMFYUI_URL", "w") as f:
    f.write(url)
print("COMFYUI_URL=" + url)
