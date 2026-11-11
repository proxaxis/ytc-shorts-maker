#!/usr/bin/env python3
"""Submit MiniMax H3 video workflows to a ComfyUI server and download the result.

Usage:
    # text to video
    python scripts/comfy_client.py --mode t2v --prompt "A cinematic tracking shot..."

    # image to video (first frame, optional last frame)
    python scripts/comfy_client.py --mode i2v --first-frame still.png --prompt "..."

    # reference to video (1-9 reference images; use <Picture N> tags in the prompt)
    python scripts/comfy_client.py --mode r2v --ref-image char.png --ref-image style.png --prompt "..."

The server URL is the ComfyUI instance running on Colab (or anywhere reachable).
Set COMFYUI_URL in the environment or a .env file to avoid passing --url.
"""

import argparse
import json
import mimetypes
import os
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode, urlparse

import requests
import websocket

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = {
    "t2v": ("workflows/video_minimax_h3_t2v_api.json", "i2v"),
    "i2v": ("workflows/video_minimax_h3_i2v_api.json", "i2v"),
    "r2v": ("workflows/video_minimax_h3_r2v_api.json", "r2v"),
}
DEFAULT_TURBO_STEPS = {"t2v": 8, "i2v": 8, "r2v": 4}


def load_env():
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def frames_for_seconds(seconds: float) -> int:
    """Snap a duration in seconds to H3's 17k+5 frame grid at 24fps."""
    m = max(5, round(seconds * 24))
    return m + (5 - (m % 17))


def upload_image(base: str, path: Path) -> str:
    """Upload an image to the server's input dir; returns the server-side filename."""
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        r = requests.post(f"{base}/upload/image", files={"image": (path.name, f, mime)},
                          data={"overwrite": "true"}, timeout=120)
    r.raise_for_status()
    info = r.json()
    return f"{info['subfolder']}/{info['name']}" if info.get("subfolder") else info["name"]


def build_prompt(args, base: str) -> dict:
    wf_file, cond_key = WORKFLOWS[args.mode]
    wf = json.loads((ROOT / wf_file).read_text())
    cond = wf[cond_key]["inputs"]
    cond.update(prompt=args.prompt, width=args.width, height=args.height,
                length=frames_for_seconds(args.seconds))
    wf["noise"]["inputs"]["noise_seed"] = args.seed

    if args.mode == "i2v":
        for slot, path in (("first_frame", args.first_frame), ("last_frame", args.last_frame)):
            node_key = "first_img" if slot == "first_frame" else "last_img"
            wf.pop(node_key, None)
            cond.pop(slot, None)
            if path:
                name = upload_image(base, Path(path))
                wf[node_key] = {"class_type": "LoadImage", "inputs": {"image": name}}
                cond[slot] = [node_key, 0]
    elif args.mode == "r2v":
        for key in [k for k in wf if k.startswith("refimg_")]:
            del wf[key]
        for key in [k for k in cond if k.startswith("ref_image_")]:
            del cond[key]
        for i, path in enumerate(args.ref_image):
            name = upload_image(base, Path(path))
            wf[f"refimg_{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
            cond[f"ref_image_{i}"] = [f"refimg_{i}", 0]

    model_src = "lora" if args.turbo else "unet"
    for node in ("sched", "guider"):
        wf[node]["inputs"]["model"] = [model_src, 0]
    turbo_steps = args.turbo_steps or DEFAULT_TURBO_STEPS[args.mode]
    wf["sched"]["inputs"]["steps"] = turbo_steps if args.turbo else args.steps
    return wf


def main():
    load_env()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=WORKFLOWS, default="t2v")
    p.add_argument("--url", default=os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188"),
                   help="ComfyUI server URL (default: $COMFYUI_URL or http://127.0.0.1:8188)")
    p.add_argument("--prompt", required=True, help="Text prompt (scene, shots, dialogue, SFX, music)")
    p.add_argument("--first-frame", help="I2V: local image for the first frame")
    p.add_argument("--last-frame", help="I2V: local image for the last frame")
    p.add_argument("--ref-image", action="append", default=[],
                   help="R2V: local reference image (repeatable, max 9)")
    p.add_argument("--ref-image-size", choices=["match", "max"], default="match",
                   help="R2V: 'max' = best identity fidelity, several times slower")
    p.add_argument("--seconds", type=float, default=5, help="Duration in seconds (snapped to 17k+5 frames)")
    p.add_argument("--width", type=int, default=1344)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--steps", type=int, default=20, help="Sampler steps when turbo is off")
    p.add_argument("--turbo", action="store_true", help="Enable Lightning LoRA turbo mode")
    p.add_argument("--turbo-steps", type=int, default=None)
    p.add_argument("--out", default=str(ROOT / "outputs"), help="Download directory")
    p.add_argument("--timeout", type=int, default=3600, help="Max wait seconds")
    args = p.parse_args()
    if args.mode == "i2v" and not (args.first_frame or args.last_frame):
        p.error("--mode i2v requires --first-frame and/or --last-frame")
    if args.mode == "r2v" and not args.ref_image:
        p.error("--mode r2v requires at least one --ref-image")
    if len(args.ref_image) > 9:
        p.error("max 9 reference images")
    if args.seed is None:
        args.seed = int.from_bytes(os.urandom(8), "big")

    base = args.url.rstrip("/")
    parsed = urlparse(base)
    ws_url = f"{'wss' if parsed.scheme == 'https' else 'ws'}://{parsed.netloc}/ws"
    client_id = uuid.uuid4().hex

    prompt = build_prompt(args, base)
    if args.mode == "r2v":
        prompt["r2v"]["inputs"]["ref_image_size"] = args.ref_image_size
    r = requests.post(f"{base}/prompt", json={"prompt": prompt, "client_id": client_id}, timeout=30)
    r.raise_for_status()
    prompt_id = r.json()["prompt_id"]
    print(f"queued: prompt_id={prompt_id} seed={args.seed} frames={prompt[WORKFLOWS[args.mode][1]]['inputs']['length']}", flush=True)

    ws = websocket.create_connection(f"{ws_url}?clientId={client_id}", timeout=30)
    deadline = time.time() + args.timeout
    try:
        while time.time() < deadline:
            msg = json.loads(ws.recv())
            if msg["type"] == "status":
                q = msg["data"]["status"]["exec_info"]["queue_remaining"]
                print(f"queue_remaining={q}", flush=True)
            elif msg["type"] == "executing":
                if msg["data"].get("node") is None and msg["data"].get("prompt_id") == prompt_id:
                    break
            elif msg["type"] == "progress":
                d = msg["data"]
                print(f"progress {d['value']}/{d['max']}", end="\r", flush=True)
            elif msg["type"] == "execution_error" and msg["data"].get("prompt_id") == prompt_id:
                sys.exit(f"execution error: {json.dumps(msg['data'], indent=2)}")
        else:
            sys.exit("timed out waiting for execution")
    finally:
        ws.close()

    history = requests.get(f"{base}/history/{prompt_id}", timeout=30).json()
    outputs = history.get(prompt_id, {}).get("outputs", {})
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for node_out in outputs.values():
        for kind in ("videos", "gifs", "images"):
            for item in node_out.get(kind, []):
                q = urlencode({"filename": item["filename"], "subfolder": item.get("subfolder", ""),
                               "type": item.get("type", "output")})
                data = requests.get(f"{base}/view?{q}", timeout=300).content
                dest = out_dir / item["filename"]
                dest.write_bytes(data)
                saved.append(dest)
                print(f"saved: {dest} ({len(data) / 1e6:.1f} MB)")
    if not saved:
        print("no output files found in history; check the ComfyUI console on Colab", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
