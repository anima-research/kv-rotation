"""Chat-shaped contexts: turn structure, turn-aligned eviction, conversation synthesis.

The deployment shape for rotation is a rolling *chat*: when the window fills, whole
early turns are popped, never mid-sentence token ranges. This module provides the
pieces the raw-doc experiments lacked:

* :func:`turn_token_spans` — render a message list through the model's chat template
  and recover each turn's token span, with a prefix-stability assert so a template
  that rewrites history (rather than appending) fails loudly instead of silently
  misaligning spans.
* :func:`oldest_turns_to_evict` / :func:`turn_keep_indices` — the turn-aligned
  eviction policy: pick whole early turns (system prompt and the most recent turns
  protected) until a token budget is freed, then materialize the keep-index set in
  the shape :func:`kvrot.snapshot.evict` expects.
* :func:`synthesize_conversations` — deterministic multi-turn conversations built
  from *real* document prose (the eval corpus), so the KV content is realistic even
  though the turn scaffolding is templated. No LLM calls, no network.

Everything here is CPU-testable without a model: span logic is tokenizer-agnostic
(tests drive it with a fake tokenizer), and eviction returns plain LongTensors that
compose with the existing snapshot surgery.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class TurnSpan:
    """One rendered turn's half-open token span ``[start, end)`` in the full context."""

    index: int   # position in the message list
    role: str    # "system" | "user" | "assistant"
    start: int
    end: int

    @property
    def n_tokens(self) -> int:
        return self.end - self.start


class TemplateNotPrefixStableError(RuntimeError):
    """The chat template rewrote earlier tokens when a turn was appended.

    Turn spans are recovered by rendering incremental prefixes of the message list;
    that is only valid if rendering ``messages[:i]`` yields a strict token-prefix of
    rendering ``messages[:i+1]``. Standard templates (jinja append-loops) satisfy
    this; templates that inject e.g. a message count or relocate the generation
    prompt do not, and must fail loudly.
    """


def _render_ids(tokenizer, messages: list[dict], *, add_generation_prompt: bool) -> list[int]:
    """Chat-template render → flat token-id list (no tensors, device-free)."""
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt
    )
    # transformers 4.x returns list[int]; 5.x returns a BatchEncoding/dict — unwrap it.
    if hasattr(ids, "keys") and "input_ids" in ids:
        ids = ids["input_ids"]
    # Also be tolerant of tensor-returning tokenizers/mocks.
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):  # batched form
        ids = ids[0]
    return list(ids)


def turn_token_spans(
    tokenizer,
    messages: list[dict],
    *,
    add_generation_prompt: bool = False,
) -> tuple[Tensor, list[TurnSpan]]:
    """Render ``messages`` through the chat template and recover per-turn token spans.

    Returns ``(context_ids [1, T] LongTensor, spans)``. Span ``i`` covers everything
    the template emitted for message ``i`` — role headers and separators included —
    so evicting a set of spans removes those turns *entirely*, template scaffolding
    and all. When ``add_generation_prompt`` is set, the trailing generation header
    is NOT part of any span (it belongs to the reply being teased, and must never
    be evicted with the last turn).

    Raises :class:`TemplateNotPrefixStableError` if the template rewrites history.
    """
    if not messages:
        raise ValueError("messages must be non-empty")

    spans: list[TurnSpan] = []
    prev: list[int] = []
    for i in range(1, len(messages) + 1):
        # The generation prompt is only appended on the final render.
        gen = add_generation_prompt and i == len(messages)
        cur = _render_ids(tokenizer, messages[:i], add_generation_prompt=gen)
        boundary = len(prev)
        if gen:
            # Re-render without the generation prompt to find the turn's true end.
            bare = _render_ids(tokenizer, messages[:i], add_generation_prompt=False)
            if bare != cur[: len(bare)]:
                raise TemplateNotPrefixStableError(
                    "generation prompt is not a pure suffix of the rendered conversation"
                )
            end = len(bare)
        else:
            end = len(cur)
        if prev != cur[:boundary]:
            raise TemplateNotPrefixStableError(
                f"rendering messages[:{i}] rewrote tokens emitted for messages[:{i - 1}] — "
                "turn spans would be misaligned; this template needs bespoke handling"
            )
        spans.append(
            TurnSpan(index=i - 1, role=str(messages[i - 1].get("role", "?")), start=boundary, end=end)
        )
        prev = cur

    context_ids = torch.tensor([prev], dtype=torch.long)
    return context_ids, spans


def oldest_turns_to_evict(
    spans: list[TurnSpan],
    *,
    target_tokens: int,
    protect_roles: tuple[str, ...] = ("system",),
    protect_last: int = 2,
) -> list[int]:
    """Pick whole turns to evict, oldest first, until ``target_tokens`` are freed.

    ``protect_roles`` turns are never evicted (the system prompt is the booth's
    permanent fixture); the final ``protect_last`` turns are always kept (the live
    conversational state). Returns the chosen turn indices, ascending — possibly
    fewer than requested if protections leave nothing else to take.
    """
    if target_tokens <= 0:
        return []
    cutoff = max(0, len(spans) - max(protect_last, 0))
    chosen: list[int] = []
    freed = 0
    for sp in spans:
        if sp.index >= cutoff:
            break
        if sp.role in protect_roles:
            continue
        chosen.append(sp.index)
        freed += sp.n_tokens
        if freed >= target_tokens:
            break
    return chosen


def turn_keep_indices(
    spans: list[TurnSpan],
    evict_turns: list[int],
    seq_len: int,
    *,
    num_sink_tokens: int = 0,
) -> Tensor:
    """Materialize the keep-index set for turn-aligned eviction (ascending LongTensor).

    Every token outside the evicted turns' spans is kept; the first
    ``num_sink_tokens`` are kept unconditionally (attention sinks — usually inside
    the protected system turn anyway, but forced here so a system-less chat still
    keeps its sinks). ``seq_len`` may exceed the last span's end (e.g. a generation
    prompt appended after the final turn) — trailing tokens are kept.
    """
    last_end = max((sp.end for sp in spans), default=0)
    if seq_len < last_end:
        raise ValueError(f"seq_len {seq_len} shorter than final span end {last_end}")
    evict = set(evict_turns)
    unknown = evict - {sp.index for sp in spans}
    if unknown:
        raise ValueError(f"evict_turns reference unknown turn indices: {sorted(unknown)}")

    keep = torch.ones(seq_len, dtype=torch.bool)
    for sp in spans:
        if sp.index in evict:
            keep[sp.start : sp.end] = False
    if num_sink_tokens > 0:
        keep[: min(num_sink_tokens, seq_len)] = True
    return keep.nonzero(as_tuple=False).squeeze(-1)


# ── Conversation synthesis (deterministic, no LLM, real prose) ──────────────────

_USER_LEADS = (
    "Here's the next section of the document I'm working through:\n\n{chunk}\n\n"
    "Anything worth flagging before I continue?",
    "Continuing from where we left off:\n\n{chunk}\n\nNoted anything important?",
    "Next passage:\n\n{chunk}\n\nKeep tracking the details for me.",
)

_ASSISTANT_ACKS = (
    "Noted. I'm tracking the key details from that section — go ahead.",
    "Got it. That section is logged; the earlier context still stands. Continue.",
    "Understood. I've folded that into my running summary — next section when ready.",
)


def synthesize_conversations(
    docs: list[str],
    *,
    n_user_turns: int,
    chars_per_chunk: int,
    system_prompt: str = (
        "You are a careful assistant helping the user work through long documents. "
        "Track every detail they share; they will quiz you without warning."
    ),
    doc_index: int = 0,
) -> list[dict]:
    """Build one deterministic multi-turn conversation from real document prose.

    The user feeds successive ``chars_per_chunk``-sized chunks of a real document
    (wrapping to the next doc if one runs dry); the assistant replies with short
    templated acknowledgments. Realism lives in the *content* (real prose filling
    the KV cache) and the *shape* (chat template, alternating roles); only the
    scaffolding is canned. Deterministic in its arguments — no RNG, no model.
    """
    if not docs:
        raise ValueError("docs must be non-empty")
    if n_user_turns < 1:
        raise ValueError("n_user_turns must be >= 1")

    text = "\n\n".join(docs[doc_index % len(docs) :] + docs[: doc_index % len(docs)])
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    offset = 0
    for t in range(n_user_turns):
        chunk = text[offset : offset + chars_per_chunk].strip()
        if not chunk:  # corpus exhausted — recycle from the top rather than truncate
            offset = 0
            chunk = text[:chars_per_chunk].strip()
        offset += chars_per_chunk
        messages.append({"role": "user", "content": _USER_LEADS[t % len(_USER_LEADS)].format(chunk=chunk)})
        if t < n_user_turns - 1:  # the final user turn is answered by the model under test
            messages.append({"role": "assistant", "content": _ASSISTANT_ACKS[t % len(_ASSISTANT_ACKS)]})
    return messages


def plant_fact(messages: list[dict], turn_index: int, fact: str) -> None:
    """Append a probe fact to one turn's content, in place.

    The fact must be synthetic (e.g. a made-up passcode) so recall can't be
    training-data memorization — same rule as exp08's needle.
    """
    if not 0 <= turn_index < len(messages):
        raise IndexError(f"turn_index {turn_index} out of range for {len(messages)} messages")
    messages[turn_index]["content"] = f"{messages[turn_index]['content']} {fact}"
