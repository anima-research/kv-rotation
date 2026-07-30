"""kvrot_playground — web playground for chatting with a model whose KV cache
is being live-rotated by the kvrot vLLM connector.

- ``session``  — token ledger, turn spans, eviction planning, banking
                 (vLLM-free, CPU-tested; the driver-side twin of the
                 connector's session store)
- ``client``   — stdlib HTTP client for the vLLM completions API
                 (token ids in/out, kv_transfer_params passthrough)
- ``app``      — FastAPI application serving the API + static frontend

Run (against a vLLM server started with the KvrotConnector — see
scripts/exp12_trinity_gates_job.sh for the exact serve flags)::

    KVROT_VLLM_URL=http://localhost:8013 \\
    KVROT_MODEL_PATH=/path/to/model \\
    uvicorn kvrot_playground.app:app --host 0.0.0.0 --port 8080
"""
