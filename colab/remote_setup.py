"""Runs ON the Colab VM via `colab exec -f` (rendered by scripts/colab_backend.sh).

Installs ComfyUI, downloads MiniMax H3 weights, launches the ComfyUI server and
a cloudflared tunnel as detached background processes, waits for readiness, and
prints COMFYUI_URL=<public tunnel url>. Safe to re-run (idempotent).
"""
import os
import subprocess
import sys

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

files = []  # (repo, repo_path, models_subpath)
if set(MODES) & {"t2v", "i2v", "r2v"}:
    files += [(REPO, p, p) for p in [
        "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "vae/minimax_h3_video_vae_int8_convrot.safetensors",
        "vae/minimax_h3_audio_vae_fp32.safetensors",
    ]]
if set(MODES) & {"t2v", "i2v"}:
    files += [(REPO, f"diffusion_models/{DIFFUSION_FILE}", f"diffusion_models/{DIFFUSION_FILE}"),
              (REPO, "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
               "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors")]
if "r2v" in MODES:
    files += [(REPO, p, p) for p in [
        "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
    ]]
if "wan" in MODES:
    WAN_REPO = "Comfy-Org/Wan-2.2"
    files += [(WAN_REPO, f"split_files/{p}", p) for p in [
        "diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
        "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "vae/wan2.2_vae.safetensors",
    ]]

for repo, rel, sub in files:
    dst = f"{COMFY}/models/{sub}"
    if os.path.lexists(dst):
        print("exists:", sub)
        continue
    src = hf_hub_download(repo, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(src, dst)
    print("linked:", sub)

cf = "/content/cloudflared"
if not os.path.exists(cf):
    sh("wget", "-q",
       "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
       "-O", cf)
    sh("chmod", "+x", cf)

# The launcher is executed via `colab ssh` (root cgroup). Do NOT spawn the
# server from this exec'd script: jupyter-children's memcg cap (~11GB) would
# OOM-kill ComfyUI while it loads model weights on small-RAM VMs.
with open("/content/launch.sh", "w") as f:
    f.write(f"""#!/bin/bash
pkill -f 'main.py --listen' 2>/dev/null
pkill -f 'cloudflared-linux' 2>/dev/null
sleep 1
setsid bash -c 'cd {COMFY} && exec python main.py --listen 127.0.0.1 --port {PORT}' </dev/null >/content/comfyui.log 2>&1 &
setsid bash -c 'exec {cf} tunnel --url http://127.0.0.1:{PORT} --no-autoupdate' </dev/null >/content/tunnel.log 2>&1 &
for i in $(seq 1 90); do
  url=$(grep -oE 'https://[a-z0-9-]+\\.trycloudflare\\.com' /content/tunnel.log | head -1)
  [ -n "$url" ] && break
  sleep 1
done
for i in $(seq 1 150); do
  curl -sf http://127.0.0.1:{PORT}/system_stats >/dev/null && break
  sleep 2
done
[ -n "$url" ] && echo "$url" > /content/COMFYUI_URL && echo "COMFYUI_URL=$url"
""")
os.chmod("/content/launch.sh", 0o755)
print("setup done; launch with: bash /content/launch.sh")
