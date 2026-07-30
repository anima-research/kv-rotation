#!/usr/bin/env bash
# exp12 trinity gate run — single scheduler job (8 GPUs, run where the
# checkpoint is reachable; NFS-over-IB read is fine). Sequence: vLLM server
# with KvrotConnector -> vllm-phase cells -> teardown -> HF oracle phase.
# The compare phase runs anywhere afterwards (CPU).
#
# Environment knobs (defaults assume the repo is the script's parent dir):
#   VLLM_ENV_BIN  bin/ dir of a vLLM 0.16 env with afmoe support   (required)
#   HF_PYTHON     python with transformers/accelerate for the oracle leg
#                 (default: $VLLM_ENV_BIN/python)
#   MODEL         checkpoint path (default: /models/Trinity-Large-Preview)
#   PORT          server port (default: 8013)
#
# GPU selection comes from CUDA_VISIBLE_DEVICES, which the job scheduler
# sets — never set it here.
set -euo pipefail

KVROT_ROOT="${KVROT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$KVROT_ROOT"
VLLM_ENV_BIN="${VLLM_ENV_BIN:?set VLLM_ENV_BIN to the bin/ of a vLLM 0.16 env}"
HF_PYTHON="${HF_PYTHON:-$VLLM_ENV_BIN/python}"
MODEL="${MODEL:-/models/Trinity-Large-Preview}"
PORT="${PORT:-8013}"
# TrueBase (max_position_embeddings 8192): MAX_MODEL_LEN=8192 CTX_TOKENS=6144
# EVICT_TOKENS=1536 BANK=runs/exp12_gates_truebase.json
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
CTX_TOKENS="${CTX_TOKENS:-8192}"
EVICT_TOKENS="${EVICT_TOKENS:-2048}"
BANK="${BANK:-runs/exp12_gates_trinity.json}"

export PYTHONPATH=src PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SRVLOG="${BANK%.json}_server.log"
mkdir -p runs

echo "[job] starting trinity vLLM server (tp=8, port $PORT)"
echo "==== attempt $(date '+%F %T') pid $$ ====" >> "$SRVLOG"
# Flag notes (all learned live, 2026-07-29; details in
# notes/design-vllm-playground.md §5.1/§12):
# * --attention-backend FLASH_ATTN — with a KV connector configured, HMA is
#   off, so all 60 layers share ONE KV-cache group; FlashInfer's metadata
#   builder refuses mixed sliding windows per group ("Window left is not the
#   same for all layers"), while FlashAttention applies the window per layer
#   at kernel-call time (mixed groups merely disable AOT scheduling). vLLM
#   0.16 removed the VLLM_ATTENTION_BACKEND env var — the CLI flag is the
#   only selection mechanism.
# * --disable-custom-all-reduce — vLLM's custom AR kernel crashed at warmup
#   on an 8xB200 TP group (custom_all_reduce.cuh:455 'invalid argument');
#   the NCCL fallback is numerically identical.
# * APC off + kv_load_failure_policy=fail are gate requirements (§5.5, §7).
"$VLLM_ENV_BIN/vllm" serve "$MODEL" --served-model-name trinity -tp 8 --port "$PORT" \
    --attention-backend FLASH_ATTN --disable-custom-all-reduce \
    --gpu-memory-utilization 0.85 --max-model-len "$MAX_MODEL_LEN" \
    --no-enable-prefix-caching --disable-hybrid-kv-cache-manager \
    --kv-transfer-config '{"kv_connector": "KvrotConnector",
        "kv_connector_module_path": "kvrot_vllm.connector",
        "kv_role": "kv_both", "kv_load_failure_policy": "fail",
        "kv_connector_extra_config": {"kvrot_store_device": "cpu"}}' \
    >> "$SRVLOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for i in $(seq 1 240); do  # generous window for the ~727 GiB load
    curl -s -m 3 "http://localhost:$PORT/v1/models" > /dev/null 2>&1 && break
    kill -0 "$SERVER_PID" 2>/dev/null || {
        echo "[job] server died during startup; log tail:"; tail -60 "$SRVLOG"; exit 1; }
    sleep 5
done
curl -s -m 3 "http://localhost:$PORT/v1/models" > /dev/null \
    || { echo "[job] server not ready in time"; tail -40 "$SRVLOG"; exit 1; }

echo "[job] server up; connector init lines (this attempt):"
# slice from the LAST attempt marker; grep|tail (never grep|head: under
# pipefail an early-closing head SIGPIPEs grep and kills the job with 141)
LAST=$(grep -n "==== attempt" "$SRVLOG" | tail -1 | cut -d: -f1)
tail -n +"${LAST:-1}" "$SRVLOG" \
    | grep -E "KvrotConnector\(|registered .* paged|Using .* attention backend" \
    | tail -6

echo "[job] running vllm phase"
"$VLLM_ENV_BIN/python" experiments/exp12_vllm_gates.py vllm \
    --base-url "http://localhost:$PORT" --model-path "$MODEL" \
    --out "$BANK" --data data/eval_docs.jsonl \
    --ctx-tokens "$CTX_TOKENS" --gen-tokens 48 --evict-tokens "$EVICT_TOKENS"

echo "[job] tearing down server"
kill "$SERVER_PID"
wait "$SERVER_PID" 2>/dev/null || true
trap - EXIT
sleep 20
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo "[job] running HF oracle phase (even device map)"
"$HF_PYTHON" experiments/exp12_vllm_gates.py hf \
    --model-path "$MODEL" --inout "$BANK" --device-map even

echo "[job] EXP12 TRINITY GATES: BOTH PHASES COMPLETE"
