"""Naturalized dialogue for exp10: backrooms-style Trinity <-> DeepSeek conversations.

exp09 measured turn-aligned eviction on *synthetic* chat (real prose, canned
scaffolding). exp10 replaces the data with model-generated open-ended dialogue:
Trinity (HF in-process) alternating with DeepSeek-V3-Base (node2 vLLM, raw
completions), in the backrooms register — two AIs talking with no humans, no task,
no assistant persona. This module holds everything that is CPU-testable without a
model or the network:

* typed turn/conversation records (:class:`TurnRecord`, :class:`ConvRecord`) with
  crash-proof jsonl banking (:func:`append_turn_record`, :func:`load_turn_records`,
  :func:`compact_turns_to_convs`, :func:`load_conv_records`);
* the DeepSeek-side plain-text transcript rendering with oldest-turn windowing to
  fit node2's 16384-token cap (:func:`render_deepseek_prompt`);
* the Trinity-side chat-message view (:func:`messages_for_trinity`);
* generated-turn hygiene (:func:`clean_generated_turn` — speaker-label bleed);
* eval-side helpers: :func:`choose_cut_prefix` (smallest message-prefix >= L
  tokens ending on a user turn, composing with :func:`kvrot.chat.turn_token_spans`)
  and :func:`find_passcode_leaks` (assert facts appear only where planted).

The one networked piece, :class:`Node2Client`, is stdlib-urllib with retries,
backoff, and context-overflow detection; its request/response plumbing is thin
enough to test via injected transport.

Design + pre-registered expectations: notes/design-natural-chat.md.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

Speaker = Literal["trinity", "deepseek"]

#: Chat-template role for each speaker in the trinity render (trinity is the
#: model under eval, so its turns are the assistant turns).
ROLE_FOR_SPEAKER: dict[str, str] = {"deepseek": "user", "trinity": "assistant"}

#: Display names in the DeepSeek-side plain-text transcript.
LABEL_FOR_SPEAKER: dict[str, str] = {"deepseek": "DeepSeek", "trinity": "Trinity"}

OMISSION_MARKER = "[earlier conversation omitted]"


class GenParams(BaseModel):
    """Sampling parameters actually used for one generated turn (provenance)."""

    temperature: float
    top_p: float
    max_tokens: int
    seed: int | None = None
    stop: list[str] = Field(default_factory=list)


class TurnRecord(BaseModel):
    """One conversation turn, banked to jsonl the moment it lands."""

    conv_id: str
    turn_index: int
    speaker: Speaker
    text: str
    generated: bool = True          # False for the hand-written seed opener
    model: str | None = None        # served/loaded model identity
    params: GenParams | None = None
    gen_tokens: int | None = None   # completion tokens as reported by the generator
    gen_token_ids: list[int] | None = None  # raw sampled ids (banked ground truth; exp11+)
    attempts: int = 1               # retries consumed (short/empty turns)
    flags: list[str] = Field(default_factory=list)
    wall_time_s: float | None = None
    timestamp: float | None = None


class ConvRecord(BaseModel):
    """A finished conversation (one line of natural_convs_*.jsonl)."""

    conv_id: str
    topic: str
    system_prompt: str
    preamble: str
    turns: list[TurnRecord]
    trinity_model: str | None = None
    deepseek_model: str | None = None
    trinity_render_tokens: int | None = None  # full chat-template render length
    meta: dict[str, Any] = Field(default_factory=dict)

    def messages(self) -> list[dict[str, str]]:
        """system + alternating user/assistant messages for the trinity template."""
        return messages_for_trinity(self.system_prompt, self.turns)


# ── Transcript rendering (DeepSeek side: raw base-model completion prompt) ──────


def render_deepseek_prompt(
    preamble: str,
    turns: list[TurnRecord],
    *,
    next_speaker: Speaker = "deepseek",
    char_budget: int = 52_000,
) -> str:
    """Plain-text dialogue transcript ending with the next speaker's open label.

    Base-model backrooms idiom::

        <preamble>

        DeepSeek: ...

        Trinity: ...

        DeepSeek:

    When the transcript would exceed ``char_budget`` characters (a conservative
    proxy for node2's 16384-token cap), the *oldest* turns are dropped wholesale
    behind :data:`OMISSION_MARKER` — the most recent turns always survive, and a
    turn is never split. The preamble and the trailing open label always survive.
    """
    if char_budget <= 0:
        raise ValueError("char_budget must be positive")
    label_line = f"{LABEL_FOR_SPEAKER[next_speaker]}:"
    blocks = [f"{LABEL_FOR_SPEAKER[t.speaker]}: {t.text.strip()}" for t in turns]

    fixed = len(preamble) + 2 + len(label_line) + 2  # preamble + seps + open label
    kept: list[str] = []
    used = 0
    dropped = False
    for block in reversed(blocks):
        cost = len(block) + 2  # "\n\n" separator
        marker_cost = len(OMISSION_MARKER) + 2
        if fixed + used + cost + marker_cost > char_budget and kept:
            dropped = True
            break
        if fixed + used + cost + marker_cost > char_budget and not kept:
            # A single turn larger than the whole budget: keep its tail so the
            # prompt is still well-formed (should not happen with 450-tok turns).
            block = block[-(char_budget - fixed - marker_cost - 2) :]
            dropped = True
            kept.append(block)
            used += len(block) + 2
            break
        kept.append(block)
        used += cost
    kept.reverse()
    if dropped:
        kept.insert(0, OMISSION_MARKER)
    return "\n\n".join([preamble.strip(), *kept, label_line])


def messages_for_trinity(system_prompt: str, turns: list[TurnRecord]) -> list[dict[str, str]]:
    """system + alternating user/assistant messages (deepseek→user, trinity→assistant).

    Validates strict alternation starting from a deepseek (user) turn — the shape
    both the chat template and the eval assume.
    """
    msgs: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for i, t in enumerate(turns):
        expected: Speaker = "deepseek" if i % 2 == 0 else "trinity"
        if t.speaker != expected:
            raise ValueError(
                f"turn {i} has speaker {t.speaker!r}, expected {expected!r} "
                "(conversations must alternate deepseek/trinity starting with deepseek)"
            )
        msgs.append({"role": ROLE_FOR_SPEAKER[t.speaker], "content": t.text.strip()})
    return msgs


def clean_generated_turn(text: str, *, own_label: str, other_label: str) -> tuple[str, list[str]]:
    """Strip label bleed from a generated turn. Returns (clean_text, flags).

    Handles: the model echoing its own label at the start ("DeepSeek: ..."), and
    continuing past its turn into the other speaker's ("...\\nTrinity: ..." when a
    stop-sequence variant slipped through). Flags record what was removed.
    """
    flags: list[str] = []
    out = text.strip()
    if out.startswith(f"{own_label}:"):
        out = out[len(own_label) + 1 :].lstrip()
        flags.append("stripped_own_label")
    lines = out.split("\n")
    for i, line in enumerate(lines):
        s = line.strip()
        if i > 0 and (s.startswith(f"{other_label}:") or s.startswith(f"{own_label}:")):
            lines = lines[:i]
            flags.append("truncated_label_bleed")
            break
    out = "\n".join(lines).strip()
    return out, flags


# ── jsonl banking (append-per-turn: a crash loses nothing) ──────────────────────


def append_turn_record(path: Path, rec: TurnRecord) -> None:
    """Append one turn as a jsonl line (parents created, flushed on close)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(rec.model_dump_json() + "\n")


def load_turn_records(path: Path) -> dict[str, list[TurnRecord]]:
    """Read a turn-bank jsonl → {conv_id: turns sorted by turn_index}.

    Tolerates a torn final line (crash mid-write): it is skipped with a stderr
    note rather than poisoning the resume. Duplicate (conv_id, turn_index)
    entries keep the *last* occurrence (a retried bank after a partial failure).
    """
    out: dict[str, dict[int, TurnRecord]] = {}
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = TurnRecord.model_validate_json(line)
            except Exception as e:  # torn/corrupt line — skip loudly, keep resuming
                import sys

                print(f"[natural] skipping bad line {lineno} in {path}: {e}", file=sys.stderr)
                continue
            out.setdefault(rec.conv_id, {})[rec.turn_index] = rec
    result: dict[str, list[TurnRecord]] = {}
    for conv_id, by_idx in out.items():
        idxs = sorted(by_idx)
        # Only the contiguous prefix 0..k is usable for resume.
        contiguous: list[TurnRecord] = []
        for want, idx in enumerate(idxs):
            if idx != want:
                break
            contiguous.append(by_idx[idx])
        result[conv_id] = contiguous
    return result


def compact_turns_to_convs(convs: list[ConvRecord], path: Path) -> None:
    """Write the finished conversations file (one ConvRecord per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for c in convs:
            f.write(c.model_dump_json() + "\n")
    tmp.replace(path)


def load_conv_records(path: Path) -> list[ConvRecord]:
    convs: list[ConvRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                convs.append(ConvRecord.model_validate_json(line))
    return convs


# ── Eval-side helpers ────────────────────────────────────────────────────────────


def choose_cut_prefix(
    tokenizer,
    messages: list[dict[str, str]],
    target_tokens: int,
    *,
    end_role: str = "user",
) -> tuple[int, int]:
    """Smallest message-prefix whose render is >= ``target_tokens``, ending on ``end_role``.

    Returns ``(n_messages, rendered_tokens)`` where the render includes the
    generation prompt (matching the eval's context). Mirrors exp09's
    overshoot-and-report behaviour: we take the first boundary at/after the
    target rather than chasing it. Raises if even the full conversation ends up
    short or no ``end_role`` boundary reaches the target.
    """
    from kvrot.chat import _render_ids

    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    for i in range(1, len(messages) + 1):
        if messages[i - 1].get("role") != end_role:
            continue
        n = len(_render_ids(tokenizer, messages[:i], add_generation_prompt=True))
        if n >= target_tokens:
            return i, n
    raise ValueError(
        f"conversation too short: no {end_role!r}-terminated prefix reaches "
        f"{target_tokens} tokens (full length "
        f"{len(_render_ids(tokenizer, messages, add_generation_prompt=True))})"
    )


def find_passcode_leaks(
    messages: list[dict[str, str]], passcode: str, planted_index: int
) -> list[int]:
    """Indices of messages containing ``passcode`` other than ``planted_index``."""
    return [
        i
        for i, m in enumerate(messages)
        if passcode in str(m.get("content", "")) and i != planted_index
    ]


# ── Stage 1: existing kv_perturb backrooms corpus (Trinity/Cogito group chats) ──


class Stage1Conv(BaseModel):
    """One kv_perturb backrooms transcript, re-rendered as chat-template messages.

    The mapping is byte-compatible with kv_perturb's ``to_format_b`` (its
    "standard chat format"): bot messages become raw ``assistant`` turns;
    consecutive non-bot messages are grouped into a single ``user`` turn with
    ``speaker: content`` lines. This holds exp09's *shape* (chat template,
    turn-aligned eviction) fixed while swapping in naturalized content — the
    stage-1 naturalization test. Source transcripts:
    data/backrooms_corpus/{prefill,chat}/*/transcript.json.
    """

    conv_id: str
    condition: str                    # generation probing condition: "prefill" | "chat"
    bot_name: str                     # the Trinity-persona speaker ("Cogito"/"Trinity")
    source_path: str
    approx_tokens: int                # source-side chars/4 estimate
    n_source_messages: int
    generated_pct: float | None = None
    chat_messages: list[dict[str, str]]
    meta: dict[str, Any] = Field(default_factory=dict)

    def messages(self) -> list[dict[str, str]]:
        return [dict(m) for m in self.chat_messages]


def corpus_chat_messages(
    raw_messages: list[dict[str, Any]], bot_name: str
) -> list[dict[str, str]]:
    """kv_perturb ``to_format_b``: alternating chat messages from a group transcript.

    Bot messages → ``assistant`` (content raw, name implied); consecutive non-bot
    messages → one ``user`` message of ``speaker: content`` lines. Result strictly
    alternates (never two same-role messages in a row).
    """
    out: list[dict[str, str]] = []
    pending: list[str] = []
    for m in raw_messages:
        speaker = str(m.get("speaker", "?"))
        content = str(m.get("content", ""))
        if speaker == bot_name:
            if pending:
                out.append({"role": "user", "content": "\n".join(pending)})
                pending = []
            out.append({"role": "assistant", "content": content})
        else:
            pending.append(f"{speaker}: {content}")
    if pending:
        out.append({"role": "user", "content": "\n".join(pending)})
    return out


def load_stage1_convs(path: Path) -> list[Stage1Conv]:
    convs: list[Stage1Conv] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                convs.append(Stage1Conv.model_validate_json(line))
    return convs


def load_convs_any(path: Path) -> list[ConvRecord | Stage1Conv]:
    """Load a conversations jsonl of either schema (sniffed per line).

    Stage-1 corpus lines carry ``chat_messages``; stage-2 generated lines carry
    ``turns``. Both expose ``.conv_id`` and ``.messages()``, which is all the
    eval needs.
    """
    convs: list[ConvRecord | Stage1Conv] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            head = json.loads(line)
            if "chat_messages" in head:
                convs.append(Stage1Conv.model_validate(head))
            elif "turns" in head:
                convs.append(ConvRecord.model_validate(head))
            else:
                raise ValueError(f"{path}:{lineno}: unrecognized conversation schema "
                                 f"(keys={sorted(head)[:8]})")
    return convs


def trinity_even_device_map(n_layers: int = 60, n_gpus: int = 8) -> dict[str, int]:
    """Explicit even per-layer placement for trinity/afmoe across ``n_gpus``.

    accelerate's "balanced" strategy misplaces this remote-code MoE: observed
    2026-07-10, it stacked 2x the layers (~176 GB) onto the last GPU — over any
    sane max_memory cap — then disk-offloaded the tail (layers.59, norm,
    rotary_emb, lm_head), wedging decode. A 727 GiB model over 8x183 GiB GPUs
    needs no cleverness: embeddings + the first layer block on GPU 0, an even
    split of layers across the middle, and the final norm/rotary/lm_head with
    the last block. Every GPU lands around 90-110 GB.
    """
    if n_layers < n_gpus or n_gpus < 1:
        raise ValueError(f"cannot spread {n_layers} layers over {n_gpus} gpus")
    base, rem = divmod(n_layers, n_gpus)
    quotas = [base] * n_gpus
    mid = list(range(1, n_gpus - 1)) or [0]
    i = 0
    while rem > 0:  # extras go to middle GPUs, never GPU0 (embed) or last (lm_head)
        quotas[mid[i % len(mid)]] += 1
        rem -= 1
        i += 1
    dmap: dict[str, int] = {
        "model.embed_tokens": 0,
        "model.norm": n_gpus - 1,
        "model.rotary_emb": n_gpus - 1,
        "lm_head": n_gpus - 1,
    }
    layer = 0
    for gpu, q in enumerate(quotas):
        for _ in range(q):
            dmap[f"model.layers.{layer}"] = gpu
            layer += 1
    assert layer == n_layers
    return dmap


def assert_no_offload(model) -> dict[str, float]:
    """Refuse to run against a disk/cpu-offloaded model; report params per device.

    Disk-offloaded parameters sit on the meta device and are re-read from disk on
    every forward — sub-1-tok/s decode (observed 2026-07-10 on trinity at
    mem-frac 0.70/0.85 with accelerate's unbalanced placement). Generation and
    eval must never silently run in that regime. Returns {device: billions of
    params} after printing it; raises RuntimeError listing offending modules if
    any parameter lives on disk/cpu/meta.
    """
    device_map = getattr(model, "hf_device_map", None) or {}
    bad_modules = {k: str(v) for k, v in device_map.items()
                   if str(v) in ("disk", "cpu", "meta")}
    counts: dict[str, int] = {}
    for p in model.parameters():
        d = str(p.device)
        counts[d] = counts.get(d, 0) + p.numel()
    report = {d: n / 1e9 for d, n in sorted(counts.items())}
    print("[load] params/device (B): "
          + ", ".join(f"{d}:{b:.1f}" for d, b in report.items()), flush=True)
    bad_devices = [d for d in report if d.startswith(("cpu", "meta"))]
    if bad_modules or bad_devices:
        raise RuntimeError(
            "model is partially offloaded — refusing to run "
            f"(devices={bad_devices}, modules={list(bad_modules) or 'n/a'}); "
            "raise --mem-frac / rebalance placement until everything is on GPU"
        )
    return report


# ── node2 client (the only networked piece) ─────────────────────────────────────


class ContextOverflowError(RuntimeError):
    """node2 rejected the request because the prompt exceeded max_model_len."""


class Node2Client:
    """Minimal OpenAI-compatible /v1/completions client for node2's vLLM (stdlib only).

    Sequential by design (the server is a production service others depend on);
    retries transient failures with exponential backoff; surfaces context-length
    400s as :class:`ContextOverflowError` so the caller can shrink its window.
    ``transport`` is injectable for CPU tests.
    """

    def __init__(
        self,
        base_url: str = "http://node2.datasci.ath:8000",
        model: str = "deepseek-v3-base",
        *,
        timeout_s: float = 300.0,
        max_retries: int = 4,
        backoff_s: float = 2.0,
        transport: Callable[[str, bytes, float], bytes] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self._transport = transport or self._http_post

    @staticmethod
    def _http_post(url: str, body: bytes, timeout_s: float) -> bytes:
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.read()

    def served_model_id(self) -> str:
        """Identity check against /v1/models (recorded in run metadata)."""
        req = urllib.request.Request(f"{self.base_url}/v1/models")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data = json.loads(resp.read())
        ids = [m["id"] for m in data.get("data", [])]
        if self.model not in ids:
            raise RuntimeError(f"model {self.model!r} not among served models {ids}")
        root = next(m.get("root") for m in data["data"] if m["id"] == self.model)
        return f"{self.model} ({root})"

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None,
        stop: list[str],
    ) -> tuple[str, int, str]:
        """One completion. Returns (text, completion_tokens, finish_reason).

        Raises :class:`ContextOverflowError` on prompt-too-long 400s (caller
        shrinks the transcript window and retries); retries everything transient.
        """
        body = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
                "stop": stop,
            }
        ).encode()
        url = f"{self.base_url}/v1/completions"
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = self._transport(url, body, self.timeout_s)
                data = json.loads(raw)
                choice = data["choices"][0]
                usage = data.get("usage") or {}
                return (
                    str(choice.get("text", "")),
                    int(usage.get("completion_tokens", -1)),
                    str(choice.get("finish_reason", "?")),
                )
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode(errors="replace")
                except Exception:
                    pass
                if e.code == 400 and (
                    "maximum context length" in detail
                    or "max_model_len" in detail
                    or "context length" in detail
                ):
                    raise ContextOverflowError(detail[:500]) from e
                last_err = RuntimeError(f"HTTP {e.code}: {detail[:300]}")
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError,
                    KeyError, IndexError, ValueError) as e:
                last_err = e
            if attempt < self.max_retries:
                time.sleep(self.backoff_s * (2**attempt))
        raise RuntimeError(f"node2 completion failed after {self.max_retries + 1} attempts: {last_err}")
