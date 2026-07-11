"""exp10 stage 2 (generation, HTTP-only) — Trinity <-> DeepSeek-V3-Base backrooms dialogue.

Two-party naturalized dialogue with BOTH speakers served by vLLM:

  * Trinity-Large-Preview via a temporary vLLM OpenAI server on node1:8001
    (afmoe is supported by node1's kimi/glm5 vLLM builds; the in-process HF path
    measured 0.2 tok/s even with clean placement — see design note §6c throughput
    rule — so serving is the only viable route);
  * deepseek-v3-base via node2's production vLLM (raw /v1/completions transcript
    prompt, windowed to its 16384 cap) — identical to stage-2's original design.

This script runs on the LOCAL box: both model calls are HTTP. Turns are banked
per-turn (crash loses nothing, rerun resumes). Seeds are recorded for provenance;
vLLM under continuous batching is not seed-deterministic, the banked jsonl is the
canonical artifact (design note §2).

Usage:
    uv run python experiments/exp10_natural_gen2.py --smoke     # 1 conv, small target
    uv run python experiments/exp10_natural_gen2.py             # 8 convs x >=18.5k tokens
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # for exp10_natural_gen imports

from exp10_natural_gen import (  # noqa: E402  (scene + deepseek path shared with stage-2a)
    MIN_TURN_TOKENS,
    OPENERS,
    PREAMBLE,
    SHORT_TURN_RETRIES,
    SYSTEM_PROMPT,
    gen_deepseek_turn,
    log,
)
from kvrot.natural import (  # noqa: E402
    ConvRecord,
    GenParams,
    Node2Client,
    TurnRecord,
    append_turn_record,
    clean_generated_turn,
    compact_turns_to_convs,
    load_turn_records,
    messages_for_trinity,
)


import re as _re

_SENT_END_RE = _re.compile(r'[.!?…](?:["\')\]*]{0,3})(?=\s|$)')


def trim_to_sentence(text: str) -> tuple[str, bool]:
    """Trim a cap-truncated turn back to its last complete sentence.

    Only trims if a sentence end exists past 40% of the text (never guts a turn);
    returns (text, trimmed?).
    """
    ends = [m.end() for m in _SENT_END_RE.finditer(text)]
    if not ends:
        return text, False
    last = ends[-1]
    if last >= int(0.4 * len(text)) and last < len(text):
        return text[:last].rstrip(), True
    return text, False


class TrinityChatClient:
    """OpenAI-compatible /v1/chat/completions client for the temporary trinity server."""

    def __init__(self, base_url: str, model: str = "trinity", *,
                 timeout_s: float = 600.0, max_retries: int = 4, backoff_s: float = 3.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s

    def served_model_id(self) -> str:
        with urllib.request.urlopen(
            urllib.request.Request(f"{self.base_url}/v1/models"), timeout=30
        ) as resp:
            data = json.loads(resp.read())
        ids = {m["id"]: m.get("root") for m in data.get("data", [])}
        if self.model not in ids:
            raise RuntimeError(f"{self.model!r} not among served models {sorted(ids)}")
        return f"{self.model} ({ids[self.model]})"

    def chat(self, messages: list[dict[str, str]], *, max_tokens: int,
             temperature: float, top_p: float, seed: int) -> tuple[str, int, int]:
        """One chat completion. Returns (text, completion_tokens, prompt_tokens)."""
        body = json.dumps({
            "model": self.model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": top_p, "seed": seed,
        }).encode()
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    f"{self.base_url}/v1/chat/completions", data=body,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    data = json.loads(resp.read())
                choice = data["choices"][0]
                usage = data.get("usage") or {}
                return (str(choice["message"].get("content") or ""),
                        int(usage.get("completion_tokens", -1)),
                        int(usage.get("prompt_tokens", -1)))
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode(errors="replace")[:300]
                except Exception:
                    pass
                last_err = RuntimeError(f"HTTP {e.code}: {detail}")
            except (urllib.error.URLError, TimeoutError, OSError,
                    json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
                last_err = e
            if attempt < self.max_retries:
                time.sleep(self.backoff_s * (2 ** attempt))
        raise RuntimeError(f"trinity chat failed after {self.max_retries + 1} attempts: {last_err}")


def gen_trinity_turn_http(client: TrinityChatClient, turns: list[TurnRecord], *,
                          args, seed: int) -> tuple[str, int, int, list[str], int]:
    """One Trinity turn via the server. Returns (text, gen_tokens, prompt_tokens, flags, attempts)."""
    best: tuple[str, int, int, list[str]] = ("", -1, -1, ["empty"])
    attempts = 0
    for attempt in range(SHORT_TURN_RETRIES):
        attempts += 1
        msgs = messages_for_trinity(SYSTEM_PROMPT, turns)
        raw, ntok, ptok = client.chat(msgs, max_tokens=args.max_new,
                                      temperature=args.temperature, top_p=args.top_p,
                                      seed=seed + attempt * 499)
        text, flags = clean_generated_turn(raw, own_label="Trinity", other_label="DeepSeek")
        if ntok >= MIN_TURN_TOKENS and text:
            return text, ntok, ptok, flags, attempts
        if len(text) > len(best[0]):
            best = (text, ntok, ptok, flags + ["short_turn"])
        print(f"    [trinity] short turn ({ntok} toks), retrying", flush=True)
    if not best[0]:
        raise RuntimeError("trinity produced only empty turns after retries")
    return best[0], best[1], best[2], best[3], attempts


def grow_conversation(conv_id: str, conv_index: int, topic: str, opener: str, *,
                      tclient: TrinityChatClient, dclient: Node2Client,
                      bank: Path, resumed: list[TurnRecord], args) -> list[TurnRecord]:
    turns = list(resumed)
    budget_holder = {"chars": args.char_budget}
    est_render = 0  # trinity-template tokens, measured via server usage.prompt_tokens

    if not turns:
        t0 = TurnRecord(conv_id=conv_id, turn_index=0, speaker="deepseek", text=opener,
                        generated=False, model="hand-written-seed", timestamp=time.time())
        append_turn_record(bank, t0)
        turns.append(t0)
    else:
        log(f"  resumed {conv_id} at {len(turns)} banked turns")

    while True:
        if est_render >= args.target_tokens and turns[-1].speaker == "deepseek":
            log(f"  {conv_id}: target reached — est {est_render} trinity-render tokens, "
                f"{len(turns)} turns")
            break
        if len(turns) >= args.max_turns:
            print(f"  {conv_id}: WARNING hit --max-turns {args.max_turns} at est "
                  f"{est_render} tokens; stopping short", flush=True)
            break
        idx = len(turns)
        speaker = "trinity" if turns[-1].speaker == "deepseek" else "deepseek"
        seed = args.seed + conv_index * 1_000_000 + idx * 1_000
        t_start = time.perf_counter()
        if speaker == "trinity":
            text, ntok, ptok, flags, attempts = gen_trinity_turn_http(
                tclient, turns, args=args, seed=seed)
            model_id = tclient.model
            # prompt_tokens covers system + all prior turns + generation prompt;
            # + completion = the render size after this turn (headers ~ small).
            if ptok > 0:
                est_render = ptok + max(ntok, 0) + 16
        else:
            text, ntok, flags, attempts = gen_deepseek_turn(
                dclient, turns, budget_holder=budget_holder, args=args, seed=seed)
            model_id = dclient.model
            est_render += max(ntok, len(text) // 4) + 16
        if ntok >= args.max_new:  # hit the cap -> likely mid-sentence; tidy the tail
            text, trimmed = trim_to_sentence(text)
            if trimmed:
                flags = flags + ["trimmed_to_sentence"]
        wall = time.perf_counter() - t_start
        rec = TurnRecord(
            conv_id=conv_id, turn_index=idx, speaker=speaker, text=text,
            generated=True, model=model_id,
            params=GenParams(temperature=args.temperature, top_p=args.top_p,
                             max_tokens=args.max_new, seed=seed,
                             stop=["\nTrinity:", "\nDeepSeek:"] if speaker == "deepseek" else []),
            gen_tokens=ntok, attempts=attempts, flags=flags,
            wall_time_s=round(wall, 2), timestamp=time.time(),
        )
        append_turn_record(bank, rec)
        turns.append(rec)
        log(f"  {conv_id} turn {idx:>3} [{speaker:8}] {ntok:>4} toks in {wall:6.1f}s "
            f"({max(ntok, 0) / max(wall, 1e-9):5.1f} tok/s, est render {est_render})"
            f"{' ' + ','.join(flags) if flags else ''}")
    return turns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trinity-url", default="http://node1.datasci.ath:8001")
    ap.add_argument("--trinity-model", default="trinity")
    ap.add_argument("--node2-url", default="http://node2.datasci.ath:8000")
    ap.add_argument("--node2-model", default="deepseek-v3-base")
    ap.add_argument("--convs", type=int, nargs="*", default=None)
    ap.add_argument("--target-tokens", type=int, default=18_500)
    ap.add_argument("--max-turns", type=int, default=200)
    ap.add_argument("--max-new", type=int, default=450)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--char-budget", type=int, default=52_000)
    ap.add_argument("--turns-bank", type=Path, default=Path("data/natural_turns_2026-07-10.jsonl"))
    ap.add_argument("--out-convs", type=Path, default=Path("data/natural_convs_2026-07-10.jsonl"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.convs = args.convs if args.convs is not None else [0]
        args.target_tokens = min(args.target_tokens, 1_500)
        log(f"SMOKE MODE: convs={args.convs} target={args.target_tokens}")
    conv_indices = args.convs if args.convs is not None else list(range(len(OPENERS)))

    tclient = TrinityChatClient(args.trinity_url, args.trinity_model)
    dclient = Node2Client(args.node2_url, args.node2_model)
    t_served = tclient.served_model_id()
    d_served = dclient.served_model_id()
    log(f"trinity server: {t_served}")
    log(f"deepseek server: {d_served}")

    banked = load_turn_records(args.turns_bank)
    finished: list[ConvRecord] = []
    failures: list[str] = []
    for ci in conv_indices:
        topic, opener = OPENERS[ci]
        conv_id = f"conv{ci:02d}-{topic}"
        log(f"=== {conv_id} ===")
        try:
            turns = grow_conversation(conv_id, ci, topic, opener,
                                      tclient=tclient, dclient=dclient,
                                      bank=args.turns_bank,
                                      resumed=banked.get(conv_id, []), args=args)
            finished.append(ConvRecord(
                conv_id=conv_id, topic=topic, system_prompt=SYSTEM_PROMPT,
                preamble=PREAMBLE, turns=turns,
                trinity_model=t_served, deepseek_model=d_served,
                trinity_render_tokens=None,  # exact value computed at eval time
                meta={"target_tokens": args.target_tokens, "seed": args.seed,
                      "generated": "2026-07-10", "stage": 2,
                      "trinity_serving": "vllm-openai node1:8001 (temporary)"},
            ))
        except Exception as e:  # keep going across conversations; the bank has the turns
            print(f"  {conv_id} FAILED: {type(e).__name__}: {e}", flush=True)
            failures.append(f"{conv_id}: {type(e).__name__}: {e}")

    if finished:
        compact_turns_to_convs(finished, args.out_convs)
        log(f"wrote {len(finished)} conversations to {args.out_convs}")
    print("\n==== generation summary ====", flush=True)
    for c in finished:
        gen_turns = [t for t in c.turns if t.generated]
        toks = [t.gen_tokens for t in gen_turns if t.gen_tokens and t.gen_tokens > 0]
        tri = [t.gen_tokens for t in gen_turns if t.speaker == "trinity"
               and t.gen_tokens and t.gen_tokens > 0]
        print(f"  {c.conv_id}: {len(c.turns)} turns "
              f"(mean gen turn {sum(toks) / max(len(toks), 1):.0f} toks, "
              f"trinity mean {sum(tri) / max(len(tri), 1):.0f})", flush=True)
    for f in failures:
        print(f"  FAILED {f}", flush=True)


if __name__ == "__main__":
    main()
