#!/usr/bin/env bash
# Quick probe: a small uniform-attention model (Llama-3.2-3B) + KvrotConnector
# under --attention-backend FLASH_ATTN. Validates (a) FlashAttention on the
# local GPU stack, (b) the kv_axis=0 (FA-layout) adapter path live, before
# committing a big hybrid-attention model to the FA backend. 1 GPU.
#
# Environment knobs:
#   VLLM_ENV_BIN  bin/ dir of a vLLM 0.16 env                      (required)
#   MODEL         checkpoint path (default: /models/llama-3.2-3b-instruct)
#   PORT          server port (default: 8014)
set -euo pipefail

KVROT_ROOT="${KVROT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$KVROT_ROOT"
VLLM_ENV_BIN="${VLLM_ENV_BIN:?set VLLM_ENV_BIN to the bin/ of a vLLM 0.16 env}"
MODEL="${MODEL:-/models/llama-3.2-3b-instruct}"
PORT="${PORT:-8014}"

export PYTHONPATH=src PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SRVLOG=runs/exp12_fa_probe_server.log
mkdir -p runs

echo "==== attempt $(date '+%F %T') ====" >> "$SRVLOG"
# vLLM 0.16 removed the VLLM_ATTENTION_BACKEND env var; the CLI flag is the
# only backend-selection mechanism.
"$VLLM_ENV_BIN/vllm" serve "$MODEL" --served-model-name l3b \
    --attention-backend FLASH_ATTN \
    --port "$PORT" --gpu-memory-utilization 0.30 --max-model-len 4096 \
    --no-enable-prefix-caching --disable-hybrid-kv-cache-manager \
    --kv-transfer-config '{"kv_connector": "KvrotConnector",
        "kv_connector_module_path": "kvrot_vllm.connector",
        "kv_role": "kv_both", "kv_load_failure_policy": "fail",
        "kv_connector_extra_config": {"kvrot_store_device": "cpu"}}' \
    >> "$SRVLOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for i in $(seq 1 60); do
    curl -s -m 3 "http://localhost:$PORT/v1/models" > /dev/null 2>&1 && break
    kill -0 "$SERVER_PID" 2>/dev/null || {
        echo "[probe] server died; log tail:"; tail -50 "$SRVLOG"; exit 1; }
    sleep 5
done

echo "[probe] connector + backend lines (this attempt):"
LAST=$(grep -n "==== attempt" "$SRVLOG" | tail -1 | cut -d: -f1)
tail -n +"${LAST:-1}" "$SRVLOG" \
    | grep -E "KvrotConnector registered|Using .* attention backend" | tail -3

"$VLLM_ENV_BIN/python" experiments/exp12_vllm_gates.py vllm \
    --base-url "http://localhost:$PORT" --model-path "$MODEL" \
    --out runs/exp12_gates_3b_fa.json --data data/eval_docs.jsonl \
    --ctx-tokens 1024 --gen-tokens 48

echo "[probe] FA PROBE COMPLETE"
