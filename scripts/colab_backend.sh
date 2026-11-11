#!/usr/bin/env bash
# Manage the Colab ComfyUI backend via the official `colab` CLI.
#
#   up      provision a GPU runtime, install ComfyUI + H3 weights, start the
#           cloudflared tunnel, and write COMFYUI_URL to .env
#   url     print the current tunnel URL
#   status  show the Colab session status
#   down    stop the runtime and release the GPU
#
# Env vars:
#   COLAB_SESSION   session name        (default: comfy)
#   COLAB_GPU       T4|L4|G4|A100|H100  (default: T4)
#   H3_MODES        weight set          (default: t2v,i2v; add r2v for +23GB)
#
# One-time auth (headless-safe, ADC):
#   gcloud auth application-default login --no-browser \
#     --scopes=openid,https://www.googleapis.com/auth/cloud-platform,\
# https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
set -euo pipefail
cd "$(dirname "$0")/.."

colab() { .venv/bin/colab --auth=adc "$@"; }
SESSION="${COLAB_SESSION:-comfy}"
GPU="${COLAB_GPU:-T4}"
MODES="${H3_MODES:-t2v,i2v}"

need_auth() {
  cat >&2 <<'EOF'
[colab] not authenticated. One-time setup (headless-safe):
  gcloud auth application-default login --no-browser \
    --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
Then re-run this script.
EOF
}

cmd="${1:-}"
case "$cmd" in
  up)
    colab sessions >/dev/null 2>&1 || { need_auth; exit 1; }
    if colab status -s "$SESSION" 2>&1 | grep -q "not found"; then
      echo "[colab] provisioning '$SESSION' (gpu=$GPU)..."
      if ! colab new -s "$SESSION" --gpu "$GPU"; then
        echo "[colab] $GPU unavailable on this account, trying T4..." >&2
        colab new -s "$SESSION" --gpu T4
      fi
    else
      echo "[colab] session '$SESSION' already exists"
    fi
    sed "s/@MODES@/$MODES/" colab/remote_setup.py > /tmp/h3_remote_setup.py
    echo "[colab] remote setup: ComfyUI install + weights (~42GB, slow on first run)..."
    colab exec -s "$SESSION" --timeout 7200 -f /tmp/h3_remote_setup.py | tee /tmp/h3_setup.log
    url="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/h3_setup.log | tail -1)"
    [ -n "$url" ] || { echo "[colab] tunnel URL not found in setup output" >&2; exit 1; }
    [ -f .env ] || cp .env.example .env
    sed -i '/^COMFYUI_URL=/d' .env
    echo "COMFYUI_URL=$url" >> .env
    echo "[colab] ready: $url (written to .env)"
    ;;
  url)    colab ssh -s "$SESSION" -- cat /content/COMFYUI_URL ;;
  status) colab status -s "$SESSION" ;;
  log)    colab ssh -s "$SESSION" -- tail -n 50 /content/comfyui.log ;;
  down)   colab stop -s "$SESSION" ;;
  *)      echo "usage: $0 up|url|status|log|down" >&2; exit 2 ;;
esac
