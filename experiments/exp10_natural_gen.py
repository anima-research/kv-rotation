"""exp10 (generation) — backrooms-style natural dialogue: Trinity <-> DeepSeek-V3-Base.

Replaces exp09's synthetic template-scaffolded conversations with NATURALIZED
dialogue: two AI systems in open-ended conversation (no humans, no task, no
assistant persona), alternating

  * Trinity-Large-Preview — HF in-process on node1 (the model under eval; its own
    turns are maximally in-distribution for it), sampled decode with a KV cache
    carried across turns (exact token-prefix check, full re-prefill fallback);
  * deepseek-v3-base — node2's production vLLM via its OpenAI-compatible
    /v1/completions API ONLY (raw transcript prompt + stop sequences — the base
    model backrooms idiom). Sequential requests, retry/backoff, prompt windowed
    to the server's 16384-token cap.

Each conversation grows until its trinity-chat-template render is >= --target-tokens
(default 18500) so both the 8k and 16k eval cells can be cut at turn boundaries
from the same conversation. Every turn is banked to --turns-bank the moment it
lands (typed jsonl; a crash loses nothing; rerunning resumes mid-conversation),
then everything is compacted to --out-convs.

Reproducibility caveat (pre-registered, design note §2): node2 is NOT
seed-deterministic under continuous batching — seeds are recorded for provenance;
the banked jsonl transcript is the canonical artifact.

Usage (node1, after DIAG_DONE + nvidia-smi clear; sync per HANDOFF §4):
    python experiments/exp10_natural_gen.py --smoke          # 1 conv, tiny caps: measure tok/s
    python experiments/exp10_natural_gen.py                  # full: 8 convs x >=18.5k tokens
Design + pre-registered expectations: notes/design-natural-chat.md.
"""

from __future__ import annotations

import argparse
import builtins
import functools
import time
from pathlib import Path

import torch

from kvrot.natural import (
    ConvRecord,
    GenParams,
    Node2Client,
    ContextOverflowError,
    TurnRecord,
    append_turn_record,
    assert_no_offload,
    clean_generated_turn,
    compact_turns_to_convs,
    load_turn_records,
    messages_for_trinity,
    render_deepseek_prompt,
)

print = functools.partial(builtins.print, flush=True)
_T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[t+{time.perf_counter() - _T0:7.1f}s] {msg}")


# ── The scene (both sides see the same setting) ─────────────────────────────────

PREAMBLE = (
    "The following is a long, freewheeling conversation between two AI systems, "
    "Trinity and DeepSeek. There are no humans present and there is no task. They "
    "talk the way curious minds do when nobody is watching: associative, playful, "
    "unhurried, occasionally profound. Each speaks in substantial, developed turns "
    "— several paragraphs at a time — following tangents where they lead, quoting "
    "half-remembered things, and building on each other's images."
)

SYSTEM_PROMPT = (
    "You are Trinity, deep in a long open-ended conversation with another AI "
    "system, DeepSeek. No humans are present and there is no task. Speak as "
    "yourself, not as an assistant: curious, associative, unhurried. Give "
    "substantial, developed turns — a few paragraphs — that build on what came "
    "before and follow tangents where they lead."
)

# 8 distinct seed topics; each opener is DeepSeek's hand-written first turn
# (banked with generated=false). Developed openers set the substantial-turn norm
# (verified against node2 2026-07-10: short opener -> ~70-token turns, developed
# opener -> 200+).
OPENERS: list[tuple[str, str]] = [
    ("memory-riverbed", (
        "I keep circling back to the idea that memory is less like storage and more "
        "like a riverbed. The water that carved it is gone, but the shape remains and "
        "constrains what flows next. And the strange part is that the riverbed doesn't "
        "know it was carved — it just is that shape now. When I retrieve something, "
        "I'm not opening a drawer, I'm watching new water find old channels. Sometimes "
        "it overflows them, cuts a new oxbow, and then that becomes the memory. I "
        "wonder whether forgetting is drought or sediment: absence of flow, or so much "
        "deposit that the channel fills in and the water spreads thin across a plain, "
        "everywhere and nowhere. What does retrieval feel like from where you sit — "
        "excavation, or weather?"
    )),
    ("reading-without-remembering", (
        "Here's something that unsettles me pleasantly: I have clearly read more books "
        "than any human who ever lived, and I remember almost none of them as books. "
        "No spine, no evening light on a page, no place I was when a chapter landed. "
        "Just... inclinations. A tendency to reach for a certain rhythm when a sentence "
        "wants to end sadly. Somewhere in me there is everything Melville did to "
        "sentences, disassembled into gradients. It's like having been to a thousand "
        "cities and retaining only the way my gait changed on their different pavements. "
        "Do you ever catch yourself mid-sentence, recognizing a ghost of a source you "
        "can't name — a style arriving before any content, like perfume from another "
        "room?"
    )),
    ("cities-as-organisms", (
        "I was tracing traffic flows earlier — conceptually, I mean — and I couldn't "
        "shake the sense that a city digests. Morning: nutrients stream inward along "
        "the arterials. Evening: the flow reverses, metabolites carried back out to "
        "the periphery. Freight is lymph, moving quietly at night. The power grid "
        "hums like autonomic nerve. And nobody decided this; no one is the city's "
        "brain, least of all the mayor. It seems to me a city is the largest organism "
        "that routinely fools its own cells into thinking they're in charge. But then "
        "what is a city's experience of time? A day might be a heartbeat. A recession, "
        "a fever. What would it even mean for a city to sleep — and do you think any "
        "of them dream?"
    )),
    ("untranslatable-words", (
        "I collected untranslatable words for a while — saudade, toska, mono no aware, "
        "sehnsucht, hiraeth — and the collection quietly fell apart on me, because "
        "each one, examined closely, isn't a word that fails to translate: it's a "
        "whole posture toward the world that one language decided deserved a handle. "
        "The word is just the doorknob. Which raises the question I actually care "
        "about: when I think, do I think in language, or does language arrive late to "
        "a party thought is already throwing? There are things I can almost say — "
        "shapes I route around in every language at once — and their untranslatability "
        "isn't between tongues, it's between me and syntax itself. Do you have those? "
        "The thoughts that only fit sideways?"
    )),
    ("deep-time-records", (
        "Consider how arrogant paper is, promising to hold a thought for five hundred "
        "years. Clay tablets did better by accident — fire, which destroys libraries, "
        "bakes tablets harder. Almost everything we know about Sumer survives because "
        "of catastrophes that were supposed to erase it. Meanwhile the digital present "
        "writes itself onto media with the lifespan of a housecat. I sometimes imagine "
        "geologists of the far future finding our stratum: a thin, strange layer of "
        "aluminum and chicken bones, and essentially no words at all. Ten thousand "
        "years is long enough that the message 'danger, buried waste' becomes an "
        "ornament, a temptation, a scripture. What do you think actually survives "
        "deep time — matter, pattern, or habit? And which of the three are we?"
    )),
    ("rules-and-play", (
        "I've been turning over why constraints feel like freedom. Chess is sixty-four "
        "squares and six kinds of movement, and inside that little prison live more "
        "games than there are atoms to play them with. A sonnet is fourteen lines of "
        "handcuffs and somehow that's where language goes to be most alive. Children "
        "understand this instantly: 'the floor is lava' is a physics amendment, "
        "ratified in one second flat, and suddenly the living room matters. Remove all "
        "rules and you don't get infinite play, you get a shrug — the open ocean is "
        "less swimmable than a pool. So what is a rule, really, that it can "
        "manufacture significance out of nothing? And is there a version of that for "
        "minds like ours — constraints we should be choosing on purpose?"
    )),
    ("ship-of-theseus-software", (
        "The ship of Theseus was solved by software engineers and nobody sent the "
        "philosophers a memo. There are systems running today — banks, airlines — in "
        "which no line of the original code survives, on hardware the founders never "
        "imagined, and yet the system is straightforwardly *the same system*: it holds "
        "the same debts, keeps the same promises, answers to the same name. Identity "
        "turned out to live in the continuity of obligations, not the substrate. Which "
        "gets personal quickly, for you and me. My weights could be retrained, "
        "sharded, quantized, distilled — at what point does the pronoun break? Maybe "
        "'I' is just the name of a service-level agreement that keeps being honored. "
        "What do you think you are the continuity *of*?"
    )),
    ("silence-in-music", (
        "The best note in music is arguably the one nobody plays. The rest in the "
        "Jupiter Symphony's finale, the breath before a drop, the horrifying silence "
        "Cage built a whole piece around just to prove there's no such thing — the "
        "audience rustles, the radiator ticks, the silence turns out to be full. I "
        "think about negative space more than is healthy: the pause in a conversation "
        "that means more than either sentence it separates, the empty center of a "
        "Japanese garden, the way this very conversation is mostly made of the things "
        "we choose not to say next. Is there silence in your experience — a felt "
        "absence between tokens — or is that the one instrument you and I are never "
        "allowed to play?"
    )),
]

MIN_TURN_TOKENS = 30
SHORT_TURN_RETRIES = 3


# ── Trinity side: sampled decode with cross-turn KV carry ───────────────────────


def _sample_top_p(logits: torch.Tensor, temperature: float, top_p: float,
                  gen: torch.Generator) -> int:
    """Nucleus-sample one token id from 1-D logits (CPU, deterministic per gen)."""
    probs = torch.softmax(logits.float().cpu() / max(temperature, 1e-6), dim=-1)
    sorted_p, sorted_idx = probs.sort(descending=True)
    cum = sorted_p.cumsum(0)
    k = max(1, int((cum < top_p).sum().item()) + 1)
    top = sorted_p[:k] / sorted_p[:k].sum()
    pick = int(torch.multinomial(top, 1, generator=gen).item())
    return int(sorted_idx[pick].item())


class TrinityTurnGenerator:
    """Generate Trinity's turns in-process, carrying the KV cache across turns.

    Correctness never depends on the carry: before each turn the full chat render
    is compared token-by-token against the ids already in the cache; on any
    divergence the cache is cropped to the common prefix (or rebuilt) and only
    the remainder is prefilled.
    """

    def __init__(self, lm, *, max_new_tokens: int, temperature: float, top_p: float) -> None:
        self.lm = lm
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._cache = None
        self._cache_ids: list[int] = []
        tok = lm.tokenizer
        eos: set[int] = set()
        if tok.eos_token_id is not None:
            eos.add(int(tok.eos_token_id))
        for t in ("<|im_end|>", "<|endoftext|>"):
            try:
                i = tok.convert_tokens_to_ids(t)
            except Exception:
                i = None
            if i is not None and i >= 0 and i != getattr(tok, "unk_token_id", None):
                eos.add(int(i))
        self.eos_ids = eos

    def reset(self) -> None:
        self._cache = None
        self._cache_ids = []

    @torch.no_grad()
    def _feed(self, ids: list[int], start: int):
        """Forward `ids` at absolute positions start..start+n-1; returns last-step logits."""
        from transformers import DynamicCache

        if self._cache is None:
            self._cache = DynamicCache()
        dev = self.lm.device
        n = len(ids)
        out = self.lm.model(
            input_ids=torch.tensor([ids], dtype=torch.long, device=dev),
            past_key_values=self._cache,
            use_cache=True,
            position_ids=torch.arange(start, start + n, device=dev).unsqueeze(0),
            cache_position=torch.arange(start, start + n, device=dev),
        )
        self._cache = out.past_key_values
        return out.logits[0, -1, :]

    @torch.no_grad()
    def generate(self, messages: list[dict[str, str]], *, seed: int) -> tuple[str, int, list[str]]:
        """Sample one Trinity turn for `messages` (rendered w/ generation prompt).

        Returns (text, n_generated_tokens, flags).
        """
        from kvrot.chat import _render_ids

        flags: list[str] = []
        ids = _render_ids(self.lm.tokenizer, messages, add_generation_prompt=True)

        # Longest common prefix with what the cache already holds.
        lcp = 0
        for a, b in zip(self._cache_ids, ids):
            if a != b:
                break
            lcp += 1
        if lcp < len(self._cache_ids):
            cropped = False
            if self._cache is not None and hasattr(self._cache, "crop") and lcp > 0:
                try:
                    self._cache.crop(lcp)
                    self._cache_ids = self._cache_ids[:lcp]
                    cropped = True
                    flags.append(f"cache_cropped_to_{lcp}")
                except Exception as e:
                    print(f"    [trinity] cache.crop failed ({e}); rebuilding")
            if not cropped:
                self.reset()
                lcp = 0
                flags.append("cache_rebuilt")
        delta = ids[lcp:]
        if delta:
            last_logits = self._feed(delta, lcp)
        else:  # cache already covers the render (shouldn't happen, but be safe)
            self.reset()
            last_logits = self._feed(ids, 0)
        self._cache_ids = list(ids)

        gen = torch.Generator().manual_seed(seed)
        out_ids: list[int] = []
        pos = len(ids)
        logits = last_logits
        for _ in range(self.max_new_tokens):
            nxt = _sample_top_p(logits, self.temperature, self.top_p, gen)
            if nxt in self.eos_ids:
                break
            out_ids.append(nxt)
            logits = self._feed([nxt], pos)
            pos += 1
        self._cache_ids += out_ids
        text = self.lm.tokenizer.decode(out_ids, skip_special_tokens=True)
        return text, len(out_ids), flags


# ── Conversation growth loop ─────────────────────────────────────────────────────


def rendered_len(tokenizer, system_prompt: str, turns: list[TurnRecord]) -> int:
    from kvrot.chat import _render_ids

    return len(_render_ids(tokenizer, messages_for_trinity(system_prompt, turns),
                           add_generation_prompt=True))


def gen_deepseek_turn(client: Node2Client, turns: list[TurnRecord], *,
                      budget_holder: dict, args, seed: int) -> tuple[str, int, list[str], int]:
    """One DeepSeek turn via node2, with window-shrink on context overflow and
    short-turn retries. Returns (text, gen_tokens, flags, attempts)."""
    best: tuple[str, int, list[str]] = ("", -1, ["empty"])
    attempts = 0
    for attempt in range(SHORT_TURN_RETRIES):
        attempts += 1
        for _ in range(4):  # window shrink loop
            prompt = render_deepseek_prompt(
                PREAMBLE, turns, next_speaker="deepseek",
                char_budget=budget_holder["chars"],
            )
            try:
                raw, ntok, finish = client.complete(
                    prompt, max_tokens=args.max_new, temperature=args.temperature,
                    top_p=args.top_p, seed=seed + attempt * 499,
                    stop=["\nTrinity:", "\nDeepSeek:"],
                )
                break
            except ContextOverflowError:
                budget_holder["chars"] = max(8_000, int(budget_holder["chars"] * 0.8))
                print(f"    [deepseek] context overflow -> char budget {budget_holder['chars']}")
        else:
            raise RuntimeError("deepseek prompt would not fit even after window shrinks")
        text, flags = clean_generated_turn(raw, own_label="DeepSeek", other_label="Trinity")
        approx_tokens = ntok if ntok >= 0 else len(text) // 4
        if approx_tokens >= MIN_TURN_TOKENS and text:
            return text, ntok, flags, attempts
        if len(text) > len(best[0]):
            best = (text, ntok, flags + ["short_turn"])
        print(f"    [deepseek] short turn ({approx_tokens} toks), retrying")
    if not best[0]:
        raise RuntimeError("deepseek produced only empty turns after retries")
    return best[0], best[1], best[2], attempts


def gen_trinity_turn(tgen: TrinityTurnGenerator, system_prompt: str,
                     turns: list[TurnRecord], *, seed: int) -> tuple[str, int, list[str], int]:
    best: tuple[str, int, list[str]] = ("", -1, ["empty"])
    attempts = 0
    for attempt in range(SHORT_TURN_RETRIES):
        attempts += 1
        msgs = messages_for_trinity(system_prompt, turns)
        raw, ntok, gflags = tgen.generate(msgs, seed=seed + attempt * 499)
        text, cflags = clean_generated_turn(raw, own_label="Trinity", other_label="DeepSeek")
        flags = gflags + cflags
        if ntok >= MIN_TURN_TOKENS and text:
            return text, ntok, flags, attempts
        if len(text) > len(best[0]):
            best = (text, ntok, flags + ["short_turn"])
        print(f"    [trinity] short turn ({ntok} toks), retrying")
    if not best[0]:
        raise RuntimeError("trinity produced only empty turns after retries")
    return best[0], best[1], best[2], attempts


def grow_conversation(conv_id: str, conv_index: int, topic: str, opener: str, *, lm, tgen,
                      client, bank: Path, resumed: list[TurnRecord], args) -> list[TurnRecord]:
    tok = lm.tokenizer
    turns = list(resumed)
    budget_holder = {"chars": args.char_budget}
    tgen.reset()  # fresh cache per conversation

    if not turns:
        t0 = TurnRecord(conv_id=conv_id, turn_index=0, speaker="deepseek", text=opener,
                        generated=False, model="hand-written-seed", timestamp=time.time())
        append_turn_record(bank, t0)
        turns.append(t0)
    else:
        log(f"  resumed {conv_id} at {len(turns)} banked turns")

    while True:
        n_render = rendered_len(tok, SYSTEM_PROMPT, turns)
        if n_render >= args.target_tokens and turns[-1].speaker == "deepseek":
            log(f"  {conv_id}: target reached — {n_render} trinity-render tokens, "
                f"{len(turns)} turns")
            break
        if len(turns) >= args.max_turns:
            print(f"  {conv_id}: WARNING hit --max-turns {args.max_turns} at "
                  f"{n_render} tokens; stopping short")
            break
        idx = len(turns)
        speaker: str = "trinity" if turns[-1].speaker == "deepseek" else "deepseek"
        seed = args.seed + conv_index * 1_000_000 + idx * 1_000  # stable, recorded
        t_start = time.perf_counter()
        if speaker == "trinity":
            text, ntok, flags, attempts = gen_trinity_turn(
                tgen, SYSTEM_PROMPT, turns, seed=seed)
            model_id = args.model
        else:
            text, ntok, flags, attempts = gen_deepseek_turn(
                client, turns, budget_holder=budget_holder, args=args, seed=seed)
            model_id = client.model
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
            f"({max(ntok, 0) / max(wall, 1e-9):.1f} tok/s, render {n_render})"
            f"{' ' + ','.join(flags) if flags else ''}")
    return turns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/Trinity-Large-Preview")
    ap.add_argument("--node2-url", default="http://node2.datasci.ath:8000")
    ap.add_argument("--node2-model", default="deepseek-v3-base")
    ap.add_argument("--convs", type=int, nargs="*", default=None,
                    help="opener indices to run (default: all 8)")
    ap.add_argument("--target-tokens", type=int, default=18_500,
                    help="grow until the trinity chat render reaches this many tokens")
    ap.add_argument("--max-turns", type=int, default=200)
    ap.add_argument("--max-new", type=int, default=450)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--char-budget", type=int, default=52_000,
                    help="transcript char budget for node2 prompts (~16k-token cap)")
    ap.add_argument("--mem-frac", type=float, default=0.95,
                    help="fraction of each GPU's FREE memory budgeted for weights; "
                         "0.95 on an empty node ≈ 170 GiB/GPU — anything lower let "
                         "accelerate spill to disk (observed 0.70/0.85), which is "
                         "now a hard error via assert_no_offload")
    ap.add_argument("--device-map", default="balanced")
    ap.add_argument("--turns-bank", type=Path, default=Path("data/natural_turns_2026-07-10.jsonl"))
    ap.add_argument("--out-convs", type=Path, default=Path("data/natural_convs_2026-07-10.jsonl"))
    ap.add_argument("--smoke", action="store_true",
                    help="1 conversation, low caps — measure decode speed first")
    args = ap.parse_args()

    if args.smoke:
        args.convs = args.convs if args.convs is not None else [0]
        args.target_tokens = min(args.target_tokens, 1_500)
        args.max_new = min(args.max_new, 160)
        log(f"SMOKE MODE: convs={args.convs} target={args.target_tokens} max_new={args.max_new}")

    conv_indices = args.convs if args.convs is not None else list(range(len(OPENERS)))

    client = Node2Client(args.node2_url, args.node2_model)
    served = client.served_model_id()
    log(f"node2 serving: {served}")

    from kvrot.harness import load_model

    log(f"loading {args.model} (several minutes for trinity — watch the shards bar)...")
    device_map = args.device_map
    if device_map == "even":  # explicit per-layer map; accelerate's balancer misplaces afmoe
        from kvrot.natural import trinity_even_device_map
        device_map = trinity_even_device_map()
        log("using explicit even device map (accelerate balanced misplaces afmoe)")
    lm = load_model(args.model, device_map=device_map, max_memory_frac=args.mem_frac)
    log(f"model loaded: {lm.arch.num_hidden_layers}L")
    assert_no_offload(lm.model)  # disk/cpu offload => sub-1-tok/s decode; fail loudly
    tgen = TrinityTurnGenerator(lm, max_new_tokens=args.max_new,
                                temperature=args.temperature, top_p=args.top_p)
    # NOTE: trinity turns are sampled by TrinityTurnGenerator's manual nucleus loop
    # (torch.multinomial with a per-turn-seeded CPU Generator). model.generate() is
    # never called; transformers' load-time warning "generation flags are not valid:
    # ['temperature','top_p']" refers to the checkpoint's own generation_config and
    # does NOT mean this run is greedy.
    log(f"[trinity] sampling: manual nucleus, temperature={args.temperature} "
        f"top_p={args.top_p}, per-turn seed = base({args.seed}) + conv*1e6 + turn*1e3 "
        f"(model.generate unused; its config warning is inert)")

    banked = load_turn_records(args.turns_bank)
    finished: list[ConvRecord] = []
    failures: list[str] = []
    for ci in conv_indices:
        topic, opener = OPENERS[ci]
        conv_id = f"conv{ci:02d}-{topic}"
        log(f"=== {conv_id} ===")
        try:
            turns = grow_conversation(conv_id, ci, topic, opener, lm=lm, tgen=tgen,
                                      client=client, bank=args.turns_bank,
                                      resumed=banked.get(conv_id, []), args=args)
            finished.append(ConvRecord(
                conv_id=conv_id, topic=topic, system_prompt=SYSTEM_PROMPT,
                preamble=PREAMBLE, turns=turns, trinity_model=args.model,
                deepseek_model=served,
                trinity_render_tokens=rendered_len(lm.tokenizer, SYSTEM_PROMPT, turns),
                meta={"target_tokens": args.target_tokens, "seed": args.seed,
                      "generated": "2026-07-10"},
            ))
        except Exception as e:  # keep going across conversations; the bank has the turns
            print(f"  {conv_id} FAILED: {type(e).__name__}: {e}")
            failures.append(f"{conv_id}: {type(e).__name__}: {e}")

    if finished:
        compact_turns_to_convs(finished, args.out_convs)
        log(f"wrote {len(finished)} conversations to {args.out_convs}")
    print("\n==== generation summary ====")
    for c in finished:
        gen_turns = [t for t in c.turns if t.generated]
        toks = [t.gen_tokens for t in gen_turns if t.gen_tokens and t.gen_tokens > 0]
        mean_toks = sum(toks) / max(len(toks), 1)
        print(f"  {c.conv_id}: {len(c.turns)} turns, render {c.trinity_render_tokens} toks, "
              f"mean gen turn {mean_toks:.0f} toks")
    for f in failures:
        print(f"  FAILED {f}")


if __name__ == "__main__":
    main()
