# Running on node1 (GPU)

node1 is a **shared** machine. Other users run production services that depend on conda
environments. Follow these rules strictly:

- **Never** `pip install` / `conda install` outside an explicitly activated venv.
- **Never** write to `~/.local/`, `~/.bashrc`, or anything outside `~/luxi-files/`.
- A stray `pip install` to `~/.local/` poisons every conda env (`ENABLE_USER_SITE=True`).
- Use `uv` for everything. Our venvs intentionally have no `pip` binary.

## One-time setup (inside ~/luxi-files/)

```bash
ssh node1
cd ~/luxi-files
# sync the repo here (rsync from the dev box), e.g.:
#   rsync -av --exclude .venv --exclude runs /home/luxia/projects/kv-rotation/ node1:~/luxi-files/kv-rotation/
cd kv-rotation

uv venv .venv-kvrot
source .venv-kvrot/bin/activate
# CUDA torch (NOT the CPU pin used on the dev box) + the hf extra:
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -e ".[hf,dev]"
```

> The dev-box `pyproject.toml` pins torch to CPU wheels for a light local `uv sync`.
> On node1 we install a CUDA build explicitly (above) so it takes precedence in the venv.

## Smoke test (math only, no model)

```bash
uv run pytest      # same correctness tests as the dev box
```

## exp01 — drift vs full prompt

```bash
# hard case first: pure full-attention Llama
python experiments/exp01_rotation_drift.py \
    --model /models/llama-3.2-3b-instruct --context-len 512 --gen 32 --evict 64 128 256

# the eventual target: hybrid sliding-window MoE
python experiments/exp01_rotation_drift.py \
    --model /models/Trinity-Large-Preview --context-len 2048 --gen 32 --evict 256 1024
```

Expected shape of results: `oldest` (drops sinks) → large KL; `sink+rot` → small KL,
shrinking further as the evicted block moves outside the sliding window on trinity.

## Models on node1

| path | arch | attention | RoPE | KV heads |
|---|---|---|---|---|
| `/models/llama-3.2-3b-instruct` | Llama-3.2 3B | full (28 layers) | θ=500k, llama3 scaling | 8 |
| `/models/Trinity-Large-Preview` | afmoe MoE (~389B) | hybrid: 45 SWA(4096) + 15 full | θ=10k | 8 |
| `/models/Trinity-Large-TrueBase` | afmoe MoE | same, 8k ctx | θ=10k | 8 |
