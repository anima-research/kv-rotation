#!/usr/bin/env bash
# kvrot playground web app (FastAPI + static frontend). Colocate with the
# vLLM server (scripts/serve_playground_trinity.sh). No GPUs needed.
#
# Env knobs:
#   APP_PYTHON     python with fastapi/uvicorn/transformers      (required)
#   KVROT_VLLM_URL vLLM server base URL (default http://localhost:8013)
#   KVROT_MODEL_PATH tokenizer path (default /models/Trinity-Large-Preview)
#   APP_PORT       bind port (default 2222)
#   KVROT_TOKEN    optional shared-secret gate (unset = open; set it before
#                  any public exposure)
set -euo pipefail

KVROT_ROOT="${KVROT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$KVROT_ROOT"
APP_PYTHON="${APP_PYTHON:?set APP_PYTHON to a python with fastapi+uvicorn+transformers}"
APP_PORT="${APP_PORT:-2222}"

export PYTHONPATH=src PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export KVROT_VLLM_URL="${KVROT_VLLM_URL:-http://localhost:8013}"
export KVROT_MODEL_PATH="${KVROT_MODEL_PATH:-/models/Trinity-Large-Preview}"
export KVROT_BOT_NAME="${KVROT_BOT_NAME:-Trinity}"
export KVROT_BANK_DIR="${KVROT_BANK_DIR:-runs/playground}"

exec "$APP_PYTHON" -m uvicorn kvrot_playground.app:app \
    --host 0.0.0.0 --port "$APP_PORT" --log-level info
