"""kvrot_vllm — KV-cache rotation for vLLM v1 via an out-of-tree KV connector.

Two-module split, deliberate:

- ``core``      — all the logic (session store, surgery, slot math, layout
                  adapters, claim arithmetic). Imports torch/pydantic/kvrot
                  only, so the CPU test suite runs on the no-GPU dev box
                  where vLLM is not installed.
- ``connector`` — the thin vLLM-facing wrapper (``KvrotConnector``,
                  ``KVConnectorBase_V1`` subclass). Imports vLLM; only ever
                  imported inside a vLLM process on node1.

Load into vLLM with zero installs (repo on PYTHONPATH)::

    --kv-transfer-config '{"kv_connector": "KvrotConnector",
                           "kv_connector_module_path": "kvrot_vllm.connector",
                           "kv_role": "kv_both"}'

Design + validation evidence: notes/design-vllm-playground.md.
"""
