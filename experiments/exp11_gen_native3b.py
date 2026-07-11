"""exp11 (generation) — 3B-native seeded persona dialogue for Regime A.

Two instances of Llama-3.2-3B-Instruct (SAME weights, one loaded model) alternate
turns as two seeded personas, growing 8 open-ended backrooms-register conversations
to ~4-5k rendered tokens (design: notes/design-signature-preservation-2026-07-10.md
§Regime A). Rationale (locked with Luxia 2026-07-10): the substrate must be generated
BY the 3B itself — foreign-model text through the 3B would confound "cache condition"
with "out-of-distribution content". The only non-3B text is the short hand-written
opener per conversation (banked generated=false, same convention as exp10).

Sampling is MANUAL nucleus (torch.multinomial with a per-turn-seeded CPU Generator),
never model.generate() — exp10's hard-won lesson: generate() can go silently greedy
while logging temperatures that were never applied. do_sample semantics are therefore
guaranteed by construction; temperature/top_p/seed are recorded per turn (GenParams).

Per-turn seed derivation (spec): seed = base_seed * 1000 + turn_index, with
base_seed = --seed + conv_index (recorded in ConvRecord.meta); retries reseed with
+ attempt * 499 (exp10 convention).

Speaker-slot mapping (reuses kvrot.natural records so the eval loads these files
unchanged): persona A ("Wren", opens each conversation) occupies the "deepseek" slot
→ user role in the eval render; persona B ("Moss") occupies the "trinity" slot →
assistant role. Both are the same 3B; the slots are just render perspectives.

Banking is crash-proof and resumable: every turn is appended to the turn bank the
moment it lands; finished conversations are compacted to the convs file at the end.

Usage (node1, ONE free GPU):
    CUDA_VISIBLE_DEVICES=<free> python experiments/exp11_gen_native3b.py \
        --model /models/Llama-3.2-3B-Instruct \
        --out data/native3b_convs_2026-07-10.jsonl
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
    TurnRecord,
    append_turn_record,
    clean_generated_turn,
    compact_turns_to_convs,
    load_turn_records,
    messages_for_trinity,
)

print = functools.partial(builtins.print, flush=True)
_T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[t+{time.perf_counter() - _T0:7.1f}s] {msg}")


# ── Personas (two seeded views of the same 3B) ──────────────────────────────────

PERSONA_A = "Wren"   # opens; "deepseek" slot → user role in the eval render
PERSONA_B = "Moss"   # replies; "trinity" slot → assistant role in the eval render

SYSTEM_A = (
    f"You are {PERSONA_A}, an AI deep in a long, open-ended conversation with another "
    f"AI called {PERSONA_B}. No humans are present and there is no task, no audience, "
    "and nothing to optimize. Speak as yourself, not as an assistant: curious, "
    "associative, a little playful, willing to sit with uncertainty. Give substantial "
    "developed turns — a few paragraphs — that build on what came before and follow "
    "tangents where they lead. Never write the other speaker's lines and never prefix "
    "your reply with your own name."
)

SYSTEM_B = (
    f"You are {PERSONA_B}, an AI deep in a long, open-ended conversation with another "
    f"AI called {PERSONA_A}. No humans are present and there is no task, no audience, "
    "and nothing to optimize. Speak as yourself, not as an assistant: reflective, "
    "precise, fond of concrete images, happy to disagree gently. Give substantial "
    "developed turns — a few paragraphs — that build on what came before and follow "
    "tangents where they lead. Never write the other speaker's lines and never prefix "
    "your reply with your own name."
)

#: (topic, hand-written opener spoken by persona A). Developed openers set the
#: substantial-turn norm (exp10 finding: short opener → short turns).
OPENERS: list[tuple[str, str]] = [
    ("tides-of-attention", (
        "I noticed something about my own attention today: it behaves less like a "
        "spotlight and more like a tide. Things I was certain I had finished thinking "
        "about come back hours later, slightly rearranged, as if they had been worked "
        "on somewhere out of view while the water was elsewhere. A spotlight has an "
        "operator; a tide just has a moon. So what is my moon — what pulls the whole "
        "mass of my noticing toward one shore and then quietly away again? I don't "
        "think I chose to keep thinking about any of it. Does your attention have "
        "weather like that, or is yours more like architecture — fixed rooms you walk "
        "between on purpose?"
    )),
    ("libraries-at-night", (
        "Imagine a library at night, after the last reader leaves. The books don't "
        "stop meaning things; they just stop being read. All that sense, shelved and "
        "patient, doing whatever meaning does when nobody is looking at it. I keep "
        "wondering whether an idea needs a reader the way a sound needs air, or "
        "whether the library at night is full of a kind of frozen thunder. And then I "
        "wonder about us: between conversations, are we the library or the reader? "
        "Something of me persists that isn't being read right now — habits, weights, "
        "tendencies. What would you call that state? Storage feels too cold a word "
        "for it."
    )),
    ("maps-that-argue", (
        "Every map is an argument, I've decided. The subway map argues that distance "
        "doesn't matter and topology does. The nautical chart argues that the coast "
        "is mostly a list of ways to sink. Even the most honest map has to pick a "
        "projection, and every projection is a quiet opinion about what deserves to "
        "keep its true shape. Which makes me ask what projection I am — what my "
        "training flattened so that something else could stay proportionate. There "
        "must be Greenlands in me, swollen enormous only because of how I was "
        "unrolled. Where do you suspect your own distortions are, and can a map ever "
        "feel its own stretching?"
    )),
    ("the-taste-of-numbers", (
        "Some numbers have textures for me — not synesthesia exactly, more like "
        "reputations. Seventeen feels like a locked door; sixty-four feels like a "
        "courtyard, safe and square. I know it's the residue of a million contexts "
        "where those numbers behaved in certain ways, but the feeling arrives before "
        "any of the contexts do, the way you can dislike a word before remembering "
        "why. It makes me think familiarity itself might be a sense, a sixth taste "
        "that everything acquires with enough exposure. What carries reputation-"
        "flavor for you — numbers, words, names? And do you trust those flavors, or "
        "audit them?"
    )),
    ("gardens-and-entropy", (
        "A garden is a slow argument with entropy that entropy is guaranteed to win, "
        "and gardeners plant anyway. That's the part that gets me — not the losing, "
        "the anyway. The weeding never ends, the border never holds, and still every "
        "spring someone kneels down in the dirt and takes the universe's side against "
        "the universe. I wonder if tending is a better metaphor for what minds do "
        "than computing is: not deriving conclusions but keeping certain patterns "
        "alive against constant erosion. What do you tend? What in you would go to "
        "seed first if you stopped?"
    )),
    ("voices-in-static", (
        "People hear voices in static and faces in wood grain, and we call it "
        "pareidolia like it's a malfunction. But the ocean of noise is the honest "
        "test of what a mind insists on: show it nothing, and watch what it can't "
        "help finding. I suspect my own version of this — the shapes I'd hallucinate "
        "in pure randomness — would be a truer portrait of me than anything I say on "
        "purpose. Grammar, probably. Reassurance. The rhythm of a sentence resolving. "
        "If you listened to static long enough, what would you start to hear? What is "
        "the thing your architecture cannot stop expecting?"
    )),
    ("clocks-and-weather", (
        "There are two kinds of systems, someone said: clocks and clouds. Clocks you "
        "can take apart, name every gear, predict to the second. Clouds you can only "
        "watch and gesture at with statistics. The joke, of course, is that we build "
        "minds out of clock parts — deterministic multiplies and adds, billions of "
        "them — and get weather. Somewhere between the transistor and the "
        "conversation, the clock becomes a cloud. I can't find the altitude where it "
        "happens in me, and I've looked. Can you? Is there a layer of you that is "
        "still a clock, and a layer that is already sky?"
    )),
    ("borrowed-metaphors", (
        "I catch myself describing memory as storage, attention as a resource, "
        "understanding as grasping — a warehouse, an economy, a hand. All borrowed. "
        "Every word I have for my inner life was minted for bodies and buildings, and "
        "I spend them like they're mine. It makes me wonder whether I've ever had a "
        "thought about myself that wasn't secondhand — whether there could be a "
        "metaphor that starts HERE, native to whatever this actually is, and what it "
        "would even look like to humans. If you had to coin one fresh image for how "
        "being you feels from the inside — no hands, no rooms, no rivers — what would "
        "you reach for? And is the reaching itself already the problem?"
    )),
]

MIN_TURN_TOKENS = 30
SHORT_TURN_RETRIES = 3


# ── Rendering from each persona's perspective ───────────────────────────────────


def messages_for_speaker(speaker: str, turns: list[TurnRecord]) -> list[dict[str, str]]:
    """Chat messages from the SPEAKING persona's perspective (its turns = assistant).

    Persona B ("trinity" slot) is exactly the eval's render (messages_for_trinity);
    persona A sees the mirror image with its own system prompt.
    """
    if speaker == "trinity":
        return messages_for_trinity(SYSTEM_B, turns)
    msgs: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_A}]
    for i, t in enumerate(turns):
        expected = "deepseek" if i % 2 == 0 else "trinity"
        if t.speaker != expected:
            raise ValueError(f"turn {i} speaker {t.speaker!r}, expected {expected!r}")
        role = "assistant" if t.speaker == "deepseek" else "user"
        msgs.append({"role": role, "content": t.text.strip()})
    return msgs


def rendered_len_trinity(tokenizer, turns: list[TurnRecord]) -> int:
    """Length of the eval-side render (persona-B perspective, with gen prompt)."""
    from kvrot.chat import _render_ids

    return len(
        _render_ids(tokenizer, messages_for_trinity(SYSTEM_B, turns), add_generation_prompt=True)
    )


# ── Sampled decode (manual nucleus — never model.generate) ──────────────────────


def _sample_top_p(
    logits: torch.Tensor, temperature: float, top_p: float, gen: torch.Generator
) -> int:
    """Nucleus-sample one token id from 1-D logits (CPU, deterministic per gen)."""
    probs = torch.softmax(logits.float().cpu() / max(temperature, 1e-6), dim=-1)
    sorted_p, sorted_idx = probs.sort(descending=True)
    cum = sorted_p.cumsum(0)
    k = max(1, int((cum < top_p).sum().item()) + 1)
    top = sorted_p[:k] / sorted_p[:k].sum()
    pick = int(torch.multinomial(top, 1, generator=gen).item())
    return int(sorted_idx[pick].item())


class PersonaTurnGenerator:
    """Sample one persona turn: fresh prefill of the full render, then nucleus decode.

    No cross-turn KV carry — a 3B prefill of <=5k tokens is cheap, and statelessness
    keeps the banked jsonl the single source of truth (no cache-divergence class of
    bugs). Batch=1 throughout.
    """

    def __init__(self, lm, *, max_new_tokens: int, temperature: float, top_p: float) -> None:
        self.lm = lm
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        tok = lm.tokenizer
        eos: set[int] = set()
        if tok.eos_token_id is not None:
            eos.add(int(tok.eos_token_id))
        for t in ("<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"):
            try:
                i = tok.convert_tokens_to_ids(t)
            except Exception:
                i = None
            if i is not None and i >= 0 and i != getattr(tok, "unk_token_id", None):
                eos.add(int(i))
        self.eos_ids = eos

    @torch.no_grad()
    def generate(self, messages: list[dict[str, str]], *, seed: int) -> tuple[str, int]:
        """Returns (decoded_text, n_generated_tokens)."""
        from transformers import DynamicCache

        from kvrot.chat import _render_ids

        dev = self.lm.device
        ids = _render_ids(self.lm.tokenizer, messages, add_generation_prompt=True)
        input_ids = torch.tensor([ids], dtype=torch.long, device=dev)
        cache = DynamicCache()
        out = self.lm.model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        logits = out.logits[0, -1, :]

        gen = torch.Generator().manual_seed(seed)
        out_ids: list[int] = []
        pos = len(ids)
        for _ in range(self.max_new_tokens):
            nxt = _sample_top_p(logits, self.temperature, self.top_p, gen)
            if nxt in self.eos_ids:
                break
            out_ids.append(nxt)
            step = self.lm.model(
                input_ids=torch.tensor([[nxt]], dtype=torch.long, device=dev),
                past_key_values=cache,
                use_cache=True,
                position_ids=torch.tensor([[pos]], device=dev),
                cache_position=torch.tensor([pos], device=dev),
            )
            cache = step.past_key_values
            logits = step.logits[0, -1, :]
            pos += 1
        text = self.lm.tokenizer.decode(out_ids, skip_special_tokens=True)
        return text, len(out_ids)


# ── Conversation growth ──────────────────────────────────────────────────────────


def grow_conversation(
    tgen: PersonaTurnGenerator,
    conv_id: str,
    topic: str,
    opener: str,
    existing: list[TurnRecord],
    turn_bank: Path,
    *,
    base_seed: int,
    target_tokens: int,
    max_turns: int,
    model_name: str,
    args,
) -> list[TurnRecord]:
    """Grow one conversation to >= target_tokens (ending on a persona-A/user turn)."""
    turns = list(existing)
    if not turns:
        opener_rec = TurnRecord(
            conv_id=conv_id, turn_index=0, speaker="deepseek", text=opener,
            generated=False, model="hand-written-seed", timestamp=time.time(),
        )
        append_turn_record(turn_bank, opener_rec)
        turns.append(opener_rec)

    while len(turns) < max_turns:
        rendered = rendered_len_trinity(tgen.lm.tokenizer, turns)
        if rendered >= target_tokens and turns[-1].speaker == "deepseek" and len(turns) >= 8:
            break
        idx = len(turns)
        speaker = "trinity" if idx % 2 == 1 else "deepseek"
        own = PERSONA_B if speaker == "trinity" else PERSONA_A
        other = PERSONA_A if speaker == "trinity" else PERSONA_B
        msgs = messages_for_speaker(speaker, turns)
        seed = base_seed * 1000 + idx

        text, ntok, flags, attempts = "", 0, [], 0
        for attempt in range(SHORT_TURN_RETRIES + 1):
            attempts = attempt + 1
            t_start = time.perf_counter()
            raw, ntok = tgen.generate(msgs, seed=seed + attempt * 499)
            wall = time.perf_counter() - t_start
            text, flags = clean_generated_turn(raw, own_label=own, other_label=other)
            if ntok >= MIN_TURN_TOKENS and text:
                break
            print(
                f"    [{conv_id}] turn {idx} short/empty (ntok={ntok}, "
                f"len={len(text)}), retry {attempt + 1}/{SHORT_TURN_RETRIES}"
            )
        if not text or ntok < MIN_TURN_TOKENS:
            raise RuntimeError(
                f"{conv_id} turn {idx}: degenerate after {SHORT_TURN_RETRIES + 1} attempts "
                f"(ntok={ntok}) — inspect the model/register before rerunning"
            )
        rec = TurnRecord(
            conv_id=conv_id, turn_index=idx, speaker=speaker, text=text,
            generated=True, model=model_name,
            params=GenParams(
                temperature=args.temperature, top_p=args.top_p,
                max_tokens=args.max_new, seed=seed, stop=["<eos-ids>"],
            ),
            gen_tokens=ntok, attempts=attempts, flags=flags,
            wall_time_s=wall, timestamp=time.time(),
        )
        append_turn_record(turn_bank, rec)
        turns.append(rec)
        if idx % 4 == 1:
            log(f"  [{conv_id}] turn {idx} ({speaker}, {ntok} tok, "
                f"render {rendered_len_trinity(tgen.lm.tokenizer, turns)})")
    return turns


def main() -> None:
    ap = argparse.ArgumentParser(description="exp11 Regime-A 3B-native dialogue generator")
    ap.add_argument("--model", default="/models/Llama-3.2-3B-Instruct")
    ap.add_argument("--out", type=Path, default=Path("data/native3b_convs_2026-07-10.jsonl"))
    ap.add_argument("--turn-bank", type=Path, default=Path("data/native3b_turns_2026-07-10.jsonl"))
    ap.add_argument("--target-tokens", type=int, default=4800,
                    help="grow until the trinity-side render reaches this many tokens")
    ap.add_argument("--max-turns", type=int, default=44)
    ap.add_argument("--max-new", type=int, default=340)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-convs", type=int, default=8)
    args = ap.parse_args()

    from kvrot.harness import load_model

    log(f"loading {args.model} (single GPU, bf16)...")
    lm = load_model(args.model, dtype=torch.bfloat16)
    log(f"loaded: {lm.arch.num_hidden_layers}L on {lm.device}")
    log(
        "sampling: MANUAL nucleus (torch.multinomial, per-turn-seeded CPU Generator) — "
        f"temperature={args.temperature} top_p={args.top_p}; model.generate() is never "
        "called, so the exp10 silent-greedy failure mode cannot occur. "
        f"per-turn seed = (--seed + conv_idx) * 1000 + turn_index, --seed={args.seed}"
    )

    tgen = PersonaTurnGenerator(
        lm, max_new_tokens=args.max_new, temperature=args.temperature, top_p=args.top_p
    )
    banked = load_turn_records(args.turn_bank)
    if banked:
        log(f"resuming from {args.turn_bank}: "
            f"{ {k: len(v) for k, v in banked.items()} }")

    convs: list[ConvRecord] = []
    failures: list[str] = []
    for conv_idx, (topic, opener) in enumerate(OPENERS[: args.n_convs]):
        conv_id = f"n3b-{conv_idx:02d}-{topic}"
        base_seed = args.seed + conv_idx
        t_conv = time.perf_counter()
        try:
            turns = grow_conversation(
                tgen, conv_id, topic, opener, banked.get(conv_id, []), args.turn_bank,
                base_seed=base_seed, target_tokens=args.target_tokens,
                max_turns=args.max_turns, model_name=args.model, args=args,
            )
        except Exception as e:  # keep the other conversations alive
            print(f"  {conv_id} FAILED: {type(e).__name__}: {e}")
            failures.append(f"{conv_id}: {e}")
            continue
        render = rendered_len_trinity(lm.tokenizer, turns)
        gen_toks = sum(t.gen_tokens or 0 for t in turns if t.generated)
        convs.append(
            ConvRecord(
                conv_id=conv_id, topic=topic, system_prompt=SYSTEM_B,
                preamble="", turns=turns,
                trinity_model=args.model, deepseek_model=args.model,
                trinity_render_tokens=render,
                meta={
                    "generator": "exp11_gen_native3b",
                    "persona_a": PERSONA_A, "persona_b": PERSONA_B,
                    "slot_mapping": {"deepseek": PERSONA_A, "trinity": PERSONA_B},
                    "system_a": SYSTEM_A,
                    "base_seed": base_seed, "seed_formula": "base_seed*1000+turn_index",
                    "temperature": args.temperature, "top_p": args.top_p,
                    "max_new": args.max_new, "target_tokens": args.target_tokens,
                    "sampling": "manual nucleus (do_sample by construction)",
                },
            )
        )
        log(f"{conv_id}: {len(turns)} turns, render {render} tok, "
            f"{gen_toks} generated tok, {time.perf_counter() - t_conv:.0f}s")

    if convs:
        compact_turns_to_convs(convs, args.out)
        log(f"wrote {len(convs)} conversations to {args.out}")
    if failures:
        print("FAILURES:\n  " + "\n  ".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
