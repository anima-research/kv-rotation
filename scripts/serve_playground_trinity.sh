#!/usr/bin/env bash
# Long-running trinity vLLM server for the kvrot playground (8 GPUs).
# Same validated flags as the exp12 gate job, but production-tuned:
#   * kv_load_failure_policy=recompute  — a bad load degrades to a clean
#     prefill instead of failing the request
#   * kvrot_strict=false                — preemption edge cases recompute
#     rather than raise (multi-session demo traffic)
#   * kvrot_max_sessions matches the playground app's session cap
#
# Env knobs: VLLM_ENV_BIN (required), MODEL, PORT (default 8013),
#            MAX_SESSIONS (default 6).
set -euo pipefail

KVROT_ROOT="${KVROT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$KVROT_ROOT"
VLLM_ENV_BIN="${VLLM_ENV_BIN:?set VLLM_ENV_BIN to the bin/ of a vLLM 0.16 env}"
MODEL="${MODEL:-/models/Trinity-Large-Preview}"
PORT="${PORT:-8013}"
MAX_SESSIONS="${MAX_SESSIONS:-6}"
# Preview supports up to 262144 positions (use 32768 for 20k-scale sessions);
# TrueBase caps at 8192 — NEVER serve it above that.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

export PYTHONPATH=src PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

exec "$VLLM_ENV_BIN/vllm" serve "$MODEL" --served-model-name trinity -tp 8 --port "$PORT" \
    --attention-backend FLASH_ATTN --disable-custom-all-reduce \
    --gpu-memory-utilization 0.85 --max-model-len "$MAX_MODEL_LEN" \
    --no-enable-prefix-caching --disable-hybrid-kv-cache-manager \
    --kv-transfer-config '{"kv_connector": "KvrotConnector",
        "kv_connector_module_path": "kvrot_vllm.connector",
        "kv_role": "kv_both", "kv_load_failure_policy": "recompute",
        "kv_connector_extra_config": {"kvrot_store_device": "cpu",
            "kvrot_strict": false,
            "kvrot_max_sessions": '"$MAX_SESSIONS"'}}'
