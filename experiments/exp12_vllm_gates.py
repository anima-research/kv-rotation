"""exp12 — correctness gates for the vLLM KV-connector rotation path (design §7).

Gate 0  round-trip identity   : save P -> re-inject verbatim -> generate.
                                Must match an uninterrupted baseline generation.
Gate 1  cross-stack floor     : vLLM full-prompt generation logprobs vs the HF
                                oracle teacher-forcing the same tokens. This is
                                the numeric noise floor Gate 2 is judged against.
Gate 2  rotated parity        : evict+re-rotate in vLLM (connector plan) vs the
                                identical surgery in HF (KVSnapshot path), same
                                claim/tail-recompute semantics replicated.

All comparisons use chosen-token logprobs + greedy-token agreement (full-
distribution KL is HF-side only; see design §7 "gate metric reality check").

Three phases share one JSON bank (runs/exp12_gates_*.json):

  1) vllm phase — needs a running vLLM server WITH the connector. 3B example
     (one free GPU; $VLLM_ENV = a vLLM 0.16 env with the repo's src on
     PYTHONPATH):

     PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 \\
     $VLLM_ENV/bin/vllm serve \\
         /path/to/llama-3.2-3b-instruct --served-model-name l3b --port 8012 \\
         --gpu-memory-utilization 0.30 --max-model-len 4096 \\
         --no-enable-prefix-caching --disable-hybrid-kv-cache-manager \\
         --kv-transfer-config '{"kv_connector": "KvrotConnector",
             "kv_connector_module_path": "kvrot_vllm.connector",
             "kv_role": "kv_both", "kv_load_failure_policy": "fail",
             "kv_connector_extra_config": {"kvrot_store_device": "cpu"}}'

     PYTHONUNBUFFERED=1 PYTHONPATH=src python experiments/exp12_vllm_gates.py \\
         vllm --base-url http://localhost:8012 \\
         --model-path /path/to/llama-3.2-3b-instruct \\
         --out runs/exp12_gates_3b.json

  2) hf phase — the oracle leg (server can be down; same or another GPU):

     PYTHONUNBUFFERED=1 PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
     python experiments/exp12_vllm_gates.py hf \\
         --model-path /path/to/llama-3.2-3b-instruct --inout runs/exp12_gates_3b.json

  3) compare — CPU, prints the verdict table and writes it into the bank:

     python experiments/exp12_vllm_gates.py compare --inout runs/exp12_gates_3b.json

Trinity/afmoe: the full sequence is packaged in scripts/exp12_trinity_gates_job.sh
(-tp 8, --attention-backend FLASH_ATTN — required for hybrid-window models under
a connector — plus the HF phase's --device-map even).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# small utils
# --------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[exp12 t+{time.monotonic() - _T0:7.1f}s] {msg}", flush=True)


_T0 = time.monotonic()


def load_bank(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def save_bank(path: Path, bank: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(bank, f)
    tmp.replace(path)
    log(f"banked -> {path}")


def build_context_ids(tokenizer, data_path: str | None, n_tokens: int) -> list[int]:
    """Real long-doc context from a jsonl of {"text": ...} rows, else synthetic."""
    if data_path and Path(data_path).exists():
        parts: list[str] = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = row.get("text") or row.get("content") or ""
                if text:
                    parts.append(text)
                if sum(len(p) for p in parts) > n_tokens * 8:
                    break
        text = "\n\n".join(parts)
        if not text:
            raise ValueError(f"no usable 'text' fields in {data_path}")
    else:
        log("WARNING: no --data corpus, using synthetic filler (under-estimates drift)")
        bank_sents = [
            "The archive room smelled of cedar and old paper.",
            "A courier arrived with a sealed envelope at noon.",
            "Records from the northern district were misfiled for a decade.",
            "The clerk cross-referenced every ledger against the census.",
        ]
        text = " ".join(bank_sents[i % len(bank_sents)] for i in range(n_tokens))
    ids = tokenizer(text, add_special_tokens=True)["input_ids"][:n_tokens]
    if len(ids) < n_tokens:
        raise ValueError(f"corpus too short: got {len(ids)} tokens, need {n_tokens}")
    return ids


def sink_window_keep(seq_len: int, sinks: int, evict_count: int) -> list[int]:
    """Keep indices for a sink-protected contiguous post-sink eviction
    (mirrors kvrot.eviction 'sink_window'; inlined so the vllm phase needs no
    torch)."""
    if sinks + evict_count >= seq_len:
        raise ValueError("eviction would consume the whole context")
    return list(range(sinks)) + list(range(sinks + evict_count, seq_len))


# --------------------------------------------------------------------------
# vLLM client (raw urllib, token ids in / token ids out — the exp11 idiom)
# --------------------------------------------------------------------------


class VllmClient:
    def __init__(self, base_url: str, model: str, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _post(self, route: str, payload: dict) -> dict:
        body = json.dumps(payload).encode()
        last_err: Exception | None = None
        for attempt in range(4):
            req = urllib.request.Request(
                f"{self.base_url}{route}",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:2000]
                raise RuntimeError(f"HTTP {e.code} on {route}: {detail}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = e
                wait = 2.0 * (attempt + 1)
                log(f"transient error ({e}); retry in {wait:.0f}s")
                time.sleep(wait)
        raise RuntimeError(f"server unreachable after retries: {last_err}")

    def served_model_id(self) -> str:
        req = urllib.request.Request(f"{self.base_url}/v1/models")
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode())
        return data["data"][0]["id"]

    def complete(
        self,
        prompt_ids: list[int],
        *,
        max_tokens: int,
        logprobs: int = 5,
        seed: int = 0,
        kvrot: dict | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt_ids,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": seed,
            "logprobs": logprobs,
            "return_token_ids": True,
        }
        if kvrot is not None:
            payload["kv_transfer_params"] = {"kvrot": kvrot}
        t0 = time.monotonic()
        resp = self._post("/v1/completions", payload)
        wall = time.monotonic() - t0
        choice = resp["choices"][0]
        token_ids = choice.get("token_ids")
        if token_ids is None:
            raise RuntimeError(
                "response lacks token_ids — server missing return_token_ids support?"
            )
        lp = choice.get("logprobs") or {}
        out = {
            "token_ids": list(token_ids),
            "token_logprobs": lp.get("token_logprobs"),
            "top_logprobs": lp.get("top_logprobs"),
            "finish_reason": choice.get("finish_reason"),
            "kv_transfer_params": resp.get("kv_transfer_params")
            or choice.get("kv_transfer_params"),
            "wall_s": wall,
            "usage": resp.get("usage"),
        }
        if out["token_logprobs"] is None:
            raise RuntimeError("response lacks token_logprobs — logprobs not honored?")
        return out


# --------------------------------------------------------------------------
# phase: vllm
# --------------------------------------------------------------------------


def phase_vllm(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    client = VllmClient(args.base_url, args.served_name or "")
    client.model = client.model or client.served_model_id()
    log(f"server model: {client.model}")

    P = build_context_ids(tok, args.data, args.ctx_tokens)
    G, seed = args.gen_tokens, args.seed

    bank: dict[str, Any] = {
        "meta": {
            "model_path": args.model_path,
            "base_url": args.base_url,
            "ctx_tokens": len(P),
            "gen_tokens": G,
            "seed": seed,
            "sinks": args.sinks,
            "evict_tokens": args.evict_tokens,
            "context_ids": P,
            "created_unix": time.time(),
        },
        "vllm": {},
    }

    # --- cell A: baseline full-context generation (+ repeat for the noise floor)
    log(f"cell A: baseline, ctx={len(P)}, gen={G}")
    a = client.complete(P, max_tokens=G, seed=seed)
    log(f"cell A2: repeat (nondeterminism floor), wall_a={a['wall_s']:.2f}s")
    a2 = client.complete(P, max_tokens=G, seed=seed)
    bank["vllm"]["A_baseline"] = a
    bank["vllm"]["A2_repeat"] = a2

    # --- cell B: Gate 0 — save P, re-inject verbatim, generate
    log("cell B (gate 0): warm session store")
    b1 = client.complete(P, max_tokens=1, seed=seed, kvrot={"session_id": "gate0"})
    log("cell B (gate 0): round-trip request")
    b2 = client.complete(P, max_tokens=G, seed=seed, kvrot={"session_id": "gate0"})
    echo = (b2.get("kv_transfer_params") or {}).get("kvrot") or {}
    claimed = echo.get("claimed_tokens")
    if not claimed:
        raise RuntimeError(
            f"gate 0 request claimed no external tokens (echo={echo!r}) — "
            "injection did not happen; check connector logs"
        )
    log(f"gate 0 claimed {claimed}/{len(P)} tokens, wall={b2['wall_s']:.2f}s")
    bank["vllm"]["B1_warm"] = {"kv_transfer_params": b1.get("kv_transfer_params")}
    bank["vllm"]["B2_roundtrip"] = b2

    # --- cell C: Gate 2 vLLM side — evict + re-rotate via plan
    keep = sink_window_keep(len(P), args.sinks, args.evict_tokens)
    survivors = [P[i] for i in keep]
    log(f"cell C (gate 2): warm session store (|P|={len(P)})")
    c1 = client.complete(P, max_tokens=1, seed=seed, kvrot={"session_id": "gate2"})
    log(f"cell C (gate 2): rotated request (survivors={len(survivors)})")
    c2 = client.complete(
        survivors,
        max_tokens=G,
        seed=seed,
        kvrot={
            "session_id": "gate2",
            "plan": {"keep": keep, "src_len": len(P)},
        },
    )
    echo_c = (c2.get("kv_transfer_params") or {}).get("kvrot") or {}
    if not echo_c.get("claimed_tokens"):
        raise RuntimeError(
            f"gate 2 request claimed no external tokens (echo={echo_c!r})"
        )
    log(
        f"gate 2 claimed {echo_c['claimed_tokens']}/{len(survivors)} survivor "
        f"tokens, wall={c2['wall_s']:.2f}s"
    )
    bank["vllm"]["C1_warm"] = {"kv_transfer_params": c1.get("kv_transfer_params")}
    bank["vllm"]["C2_rotated"] = c2
    bank["meta"]["keep"] = keep

    # --- cell D: shortened-prompt clean recompute (mechanism yardstick)
    log("cell D: clean recompute of the shortened prompt")
    d = client.complete(survivors, max_tokens=G, seed=seed)
    bank["vllm"]["D_recompute"] = d

    save_bank(Path(args.out), bank)
    log("vllm phase complete")


# --------------------------------------------------------------------------
# phase: hf (the oracle leg — heavy imports deferred)
# --------------------------------------------------------------------------


def _chosen_logprobs_and_top(logits, targets) -> tuple[list[float], list[list[int]], list[bool]]:
    """Per-row chosen-token logprob, top-5 ids, greedy-match flag."""
    import torch

    logp = torch.log_softmax(logits.float(), dim=-1)
    tgt = torch.tensor(targets, device=logits.device)
    chosen = logp.gather(1, tgt.unsqueeze(1)).squeeze(1).tolist()
    top5 = logp.topk(5, dim=-1).indices.tolist()
    greedy = (logp.argmax(-1) == tgt).tolist()
    return chosen, top5, greedy


def _greedy_from_snapshot(lm, snap, feed_ids: list[int], n_steps: int):
    """Teacher-force ``feed_ids`` through ``snap`` (consumed!), then greedy-decode
    ``n_steps``. Returns (feed_logits [F,V], gen_ids, gen_logits [G,V])."""
    import torch

    from kvrot.harness import _teacher_forced_logits
    from kvrot.snapshot import to_hf_dynamic_cache

    assert feed_ids, "need at least one feed token to obtain a next-token logit"
    feed = torch.tensor([feed_ids], device=lm.device)
    feed_logits = _teacher_forced_logits(lm, snap, feed)  # [F, V]

    # continue decoding from the cache state _teacher_forced_logits left behind
    cache = to_hf_dynamic_cache(snap)
    # positions: snapshot rows 0..S-1 then feed at S..S+F-1 (recompacted regime)
    pos = snap.seq_len() + len(feed_ids)
    cur = int(feed_logits[-1].argmax().item())  # first generated token
    gen_ids: list[int] = []
    gen_logits = []
    for _ in range(n_steps):
        out = lm.model(
            input_ids=torch.tensor([[cur]], device=lm.device),
            past_key_values=cache,
            use_cache=True,
            position_ids=torch.tensor([[pos]], device=lm.device),
            cache_position=torch.tensor([pos], device=lm.device),
        )
        cache = out.past_key_values
        step = out.logits[:, -1, :].float()[0]
        gen_ids.append(cur)
        gen_logits.append(step)
        cur = int(step.argmax().item())
        pos += 1

    return feed_logits, gen_ids, torch.stack(gen_logits)


def phase_hf(args: argparse.Namespace) -> None:
    import torch

    from kvrot.harness import _teacher_forced_logits, load_model, prefill_snapshot
    from kvrot.snapshot import evict, reindex

    bank = load_bank(Path(args.inout))
    meta = bank["meta"]
    P: list[int] = meta["context_ids"]
    keep: list[int] = meta["keep"]
    block_size: int = args.block_size

    if args.device_map == "even":
        # trinity/afmoe 8-GPU oracle: accelerate's balanced/auto placement is
        # untrustworthy for this model (journal 2026-07-10) — use the explicit
        # even per-layer map and hard-fail on any disk/cpu/meta offload.
        from kvrot.natural import assert_no_offload, trinity_even_device_map

        lm = load_model(args.model_path, device_map=trinity_even_device_map())
        assert_no_offload(lm.model)
    else:
        lm = load_model(args.model_path, device=args.device)
    log(f"HF model loaded on {lm.device}")

    def tf_chosen(snap, feed_ids: list[int], targets: list[int]):
        """Chosen-token logprobs for ``targets`` teacher-forced after ``feed_ids``."""
        feed = feed_ids + targets[:-1]
        logits = _teacher_forced_logits(lm, snap, torch.tensor([feed], device=lm.device))
        rows = logits[len(feed_ids) - 1 :]
        return _chosen_logprobs_and_top(rows, targets)

    hf: dict[str, Any] = {}

    # --- Gate 1: full-context oracle teacher-forces the vLLM baseline tokens
    S = bank["vllm"]["A_baseline"]["token_ids"]
    log(f"gate 1 oracle: prefill {len(P) - 1} + teacher-force {len(S)}")
    snap_full = prefill_snapshot(lm, torch.tensor([P[:-1]], device=lm.device))
    chosen, top5, greedy = tf_chosen(snap_full.clone(), [P[-1]], S)
    hf["gate1_full_tf"] = {"chosen_logprobs": chosen, "top5": top5, "greedy_match": greedy}

    # --- Gate 2 oracle: identical surgery + claim/tail-recompute semantics
    survivors = [P[i] for i in keep]
    claim = min(
        (len(survivors) // block_size) * block_size,
        ((len(survivors) - 1) // block_size) * block_size,
    )
    log(
        f"gate 2 oracle: surgery keep={len(keep)}/{len(P)}, claim={claim}, "
        f"tail-recompute={len(survivors) - claim}"
    )
    snap_all = prefill_snapshot(lm, torch.tensor([P], device=lm.device))
    keep_t = torch.tensor(keep, device=lm.device)
    new_pos = torch.arange(len(keep), device=lm.device)
    rotated = reindex(evict(snap_all, keep_t), new_pos, lm.inv_freq)
    # replicate the connector's block-aligned claim: keep only the first
    # `claim` rotated rows; the survivor tail is teacher-forced (recomputed
    # against the rotated prefix) exactly as vLLM does
    truncated = evict(rotated, torch.arange(claim, device=lm.device))
    tail = survivors[claim:]

    SC = bank["vllm"]["C2_rotated"]["token_ids"]
    chosen, top5, greedy = tf_chosen(truncated.clone(), tail, SC)
    hf["gate2_rot_tf"] = {"chosen_logprobs": chosen, "top5": top5, "greedy_match": greedy}

    # independent greedy rollout from the rotated cache (divergence-step metric)
    _, hf_greedy_ids, _ = _greedy_from_snapshot(
        lm, truncated.clone(), tail, meta["gen_tokens"]
    )
    hf["gate2_rot_greedy"] = {"token_ids": hf_greedy_ids}

    # --- recompute yardstick: shortened prompt as a fresh context
    SD = bank["vllm"]["D_recompute"]["token_ids"]
    log("recompute oracle: prefill survivors + teacher-force D tokens")
    snap_rec = prefill_snapshot(lm, torch.tensor([survivors[:-1]], device=lm.device))
    chosen, top5, greedy = tf_chosen(snap_rec.clone(), [survivors[-1]], SD)
    hf["recompute_tf"] = {"chosen_logprobs": chosen, "top5": top5, "greedy_match": greedy}

    bank["hf"] = hf
    bank["meta"]["hf_claim"] = claim
    save_bank(Path(args.inout), bank)
    log("hf phase complete")


# --------------------------------------------------------------------------
# phase: compare
# --------------------------------------------------------------------------


def _mad(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    return sum(abs(x - y) for x, y in zip(a[:n], b[:n])) / max(n, 1)


def _first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def phase_compare(args: argparse.Namespace) -> None:
    bank = load_bank(Path(args.inout))
    v, hf = bank["vllm"], bank.get("hf")
    verdicts: dict[str, Any] = {}

    A, A2, B2, C2 = v["A_baseline"], v["A2_repeat"], v["B2_roundtrip"], v["C2_rotated"]

    # vLLM self-noise floor
    div_rep = _first_divergence(A["token_ids"], A2["token_ids"])
    floor_rep = _mad(A["token_logprobs"], A2["token_logprobs"])
    verdicts["repeat_floor"] = {"first_divergence": div_rep, "logprob_mad": floor_rep}

    # Gate 0. Floor note: the A-vs-A2 repeat floor is bitwise 0.0 (identical
    # batch -> identical reduction order), which is NOT the right yardstick for
    # a differently-batched request; 5e-3 is the kernel batch-composition noise
    # scale (cross-checked against the gate-1 cross-stack floor, ~2e-3).
    div0 = _first_divergence(A["token_ids"], B2["token_ids"])
    mad0 = _mad(A["token_logprobs"], B2["token_logprobs"])
    gate0_pass = div0 is None and mad0 <= max(3 * floor_rep, 5e-3)
    verdicts["gate0"] = {
        "first_divergence": div0,
        "logprob_mad": mad0,
        "pass": bool(gate0_pass),
    }

    if hf is not None:
        # Gate 1: cross-stack floor on the SAME tokens (A's greedy continuation)
        g1 = hf["gate1_full_tf"]
        mad1 = _mad(A["token_logprobs"], g1["chosen_logprobs"])
        agree1 = sum(g1["greedy_match"]) / max(len(g1["greedy_match"]), 1)
        verdicts["gate1_floor"] = {"logprob_mad": mad1, "hf_greedy_match": agree1}

        # Gate 2: rotated parity on the SAME tokens (C2's rotated continuation)
        g2 = hf["gate2_rot_tf"]
        mad2 = _mad(C2["token_logprobs"], g2["chosen_logprobs"])
        agree2 = sum(g2["greedy_match"]) / max(len(g2["greedy_match"]), 1)
        div2 = _first_divergence(C2["token_ids"], hf["gate2_rot_greedy"]["token_ids"])
        gate2_pass = mad2 <= max(args.gate2_margin * mad1, 5e-3)
        verdicts["gate2"] = {
            "logprob_mad": mad2,
            "hf_greedy_match": agree2,
            "greedy_first_divergence_vs_hf_rollout": div2,
            "pass": bool(gate2_pass),
            "margin_vs_gate1": (mad2 / mad1) if mad1 > 0 else None,
        }

        # context: recompute path cross-stack agreement (should look like gate1)
        rec = hf["recompute_tf"]
        verdicts["recompute_context"] = {
            "logprob_mad": _mad(v["D_recompute"]["token_logprobs"], rec["chosen_logprobs"]),
            "hf_greedy_match": sum(rec["greedy_match"]) / max(len(rec["greedy_match"]), 1),
        }

    print("\n=== exp12 gate verdicts ===")
    print(json.dumps(verdicts, indent=2))
    bank["verdicts"] = verdicts
    save_bank(Path(args.inout), bank)

    if not verdicts["gate0"]["pass"]:
        sys.exit("GATE 0 FAILED")
    if hf is not None and not verdicts["gate2"]["pass"]:
        sys.exit("GATE 2 FAILED")
    log("gates PASSED (of those measurable with current bank)")


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="phase", required=True)

    pv = sub.add_parser("vllm", help="run the vLLM-side cells against a live server")
    pv.add_argument("--base-url", required=True)
    pv.add_argument("--model-path", required=True, help="for the tokenizer")
    pv.add_argument("--served-name", default=None)
    pv.add_argument("--out", required=True)
    pv.add_argument("--data", default="data/eval_docs.jsonl")
    pv.add_argument("--ctx-tokens", type=int, default=1024)
    pv.add_argument("--gen-tokens", type=int, default=48)
    pv.add_argument("--sinks", type=int, default=4)
    pv.add_argument("--evict-tokens", type=int, default=256)
    pv.add_argument("--seed", type=int, default=1234)
    pv.set_defaults(fn=phase_vllm)

    ph = sub.add_parser("hf", help="run the HF-oracle cells")
    ph.add_argument("--model-path", required=True)
    ph.add_argument("--inout", required=True)
    ph.add_argument("--device", default="cuda")
    ph.add_argument(
        "--device-map",
        default=None,
        choices=[None, "even"],
        help="'even' = trinity 8-GPU explicit per-layer map (kvrot.natural)",
    )
    ph.add_argument("--block-size", type=int, default=16, help="vLLM block size")
    ph.set_defaults(fn=phase_hf)

    pc = sub.add_parser("compare", help="compute + print gate verdicts")
    pc.add_argument("--inout", required=True)
    pc.add_argument(
        "--gate2-margin",
        type=float,
        default=2.0,
        help="gate2 passes if its logprob MAD <= margin * gate1 MAD",
    )
    pc.set_defaults(fn=phase_compare)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
