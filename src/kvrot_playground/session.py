"""Driver-side session state: the token ledger the connector's KV store mirrors.

Invariant that makes the whole playground honest: after every turn,
``live_ids`` (the survivor token ledger here) and the connector's session
store hold the SAME tokens in the same order — the eviction plan we send is
computed against this ledger, and the connector asserts ``src_len`` matches
its store before applying it (core.EvictionPlanMsg). Any drift fails loudly
on the server side instead of silently rotating garbage.

Rendering: base-model prefill format (the exp10/kv_perturb idiom, adapted) —
every turn is ``"{speaker}: {text}"`` followed by the tokenizer's EOS as the
turn delimiter; generation is prefilled with ``"{bot}:"`` and stops at EOS.
No chat template anywhere (Trinity-Preview's template degenerates in deep
multi-turn chat; a base-style prefill is also the natural rolling-context
shape).

Only needs a tokenizer with ``.encode(text, add_special_tokens=False)``,
``.decode(ids)``, ``.eos_token_id`` and optional ``.bos_token_id`` — the CPU
tests use a stub.
"""

from __future__ import annotations

import bisect
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from kvrot.chat import TurnSpan, oldest_turns_to_evict, turn_keep_indices
from kvrot_vllm.core import align_down

Policy = Literal["sink_rotate", "oldest", "recompute", "none"]

DEFAULT_PREAMBLE = (
    "The following is a live conversation. {bot} is a large language model "
    "speaking for itself. Messages alternate; each ends after its turn.\n"
)


class PlaygroundConfig(BaseModel):
    # "chat"    — the model's native ChatML template, compiled to exact
    #             append-only token chunks (instruct models: keeps the
    #             assistant persona; bare prefill framing makes them act
    #             like base models — observed live on Trinity-Preview)
    # "prefill" — speaker-prefixed transcript + EOS delimiters (base models)
    # "auto"    — chat if the tokenizer ships a ChatML template, else prefill
    render_mode: Literal["auto", "chat", "prefill"] = "auto"
    policy: Policy = "sink_rotate"
    budget: int = Field(8192, ge=32, description="token budget that triggers eviction")
    evict_to_frac: float = Field(0.75, gt=0.1, le=1.0, description="evict down to this fraction of budget")
    num_sink_tokens: int = Field(4, ge=0)
    protect_last_turns: int = Field(2, ge=1)
    max_reply_tokens: int = Field(320, ge=1)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    block_size: int = Field(16, ge=1, description="vLLM block size (claim alignment, display only)")


@dataclass
class Turn:
    index: int
    role: Literal["system", "user", "model"]
    speaker: str
    text: str
    start: int  # live-ledger coords; remapped on every eviction
    end: int
    evicted: bool = False
    original_tokens: int = 0

    def live_tokens(self) -> int:
        return 0 if self.evicted else self.end - self.start


@dataclass
class EvictionEvent:
    turn_indices: list[int]      # turns fully evicted
    keep: list[int]              # exact plan sent to the connector (or applied locally)
    src_len: int
    evicted_tokens: int
    policy: Policy
    recompute: bool              # True => fresh store instead of a rotation plan


@dataclass
class TurnStats:
    """Per-reply timing/accounting for the UI readout."""

    wall_s: float = 0.0
    claimed_tokens: int = 0
    prompt_tokens: int = 0
    gen_tokens: int = 0
    evicted_turns: int = 0
    evicted_tokens: int = 0
    store_tokens: int = 0


class LedgerError(RuntimeError):
    pass


class Session:
    def __init__(
        self,
        tokenizer,
        *,
        bot_name: str,
        user_name: str = "User",
        config: PlaygroundConfig | None = None,
        preamble: str | None = None,
        bank_path: Path | None = None,
        session_id: str | None = None,
    ):
        self.tok = tokenizer
        self.bot_name = bot_name
        self.user_name = user_name
        self.config = config or PlaygroundConfig()
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self._generation = 0  # bumped by recompute policy -> fresh connector store
        self.live_ids: list[int] = []
        self.turns: list[Turn] = []
        self.events: list[EvictionEvent] = []
        self.bank_path = bank_path
        # --- store-sync bookkeeping -------------------------------------
        # The connector saves PROMPT tokens only (reply suffixes are
        # recomputed, design §5.7), so its store is always a prefix of this
        # ledger. Plans must therefore be expressed in STORE coordinates:
        #   store_len   — tokens the connector holds (len of last prompt,
        #                 post any already-shipped evictions)
        #   _boundary   — how many of the CURRENT ledger's leading tokens
        #                 come from that store region (shrinks as local
        #                 evictions land before being shipped)
        #   _pending_store_keep — composed not-yet-shipped plan, indices
        #                 into the store; None = nothing pending
        self.store_len: int = 0
        self._boundary: int = 0
        self._pending_store_keep: list[int] | None = None
        if self.tok.eos_token_id is None:
            raise LedgerError("tokenizer has no eos_token_id; need a turn delimiter")
        self._delim = [int(self.tok.eos_token_id)]

        mode = self.config.render_mode
        template = getattr(self.tok, "chat_template", None) or ""
        if mode == "auto":
            mode = "chat" if "<|im_start|>" in template else "prefill"
        if mode == "chat" and "<|im_start|>" not in template:
            raise LedgerError(
                "chat render_mode requires a ChatML template on this tokenizer"
            )
        self.mode: str = mode
        if mode == "chat":
            im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
            if im_end is None or im_end < 0:
                raise LedgerError("<|im_end|> not in vocab; cannot compile ChatML")
            self._stop_ids = [int(im_end)]
        else:
            self._stop_ids = list(self._delim)

        bos = getattr(self.tok, "bos_token_id", None)
        prefix_ids = [int(bos)] if bos is not None else []
        text = (preamble if preamble is not None else DEFAULT_PREAMBLE).format(bot=bot_name)
        if mode == "chat":
            body = self._encode(f"<|im_start|>system\n{text}<|im_end|>\n")
        else:
            body = self._encode(text) + self._delim
        self._append_turn("system", "system", text, prefix_ids=prefix_ids, body_ids=body)

    # ------------------------------------------------------------------
    # rendering / ledger
    # ------------------------------------------------------------------

    @property
    def connector_session_id(self) -> str:
        return f"{self.session_id}.g{self._generation}"

    def _encode(self, text: str) -> list[int]:
        return [int(t) for t in self.tok.encode(text, add_special_tokens=False)]

    def _append_turn(
        self, role: str, speaker: str, text: str, *, prefix_ids: list[int] | None = None,
        body_ids: list[int] | None = None,
    ) -> Turn:
        """``body_ids`` must be the COMPLETE turn body including any framing/
        delimiters; when omitted, prefill-style framing is applied."""
        if body_ids is None:
            body_ids = self._encode(f"{speaker}: {text}") + self._delim
        ids = (prefix_ids or []) + body_ids
        start = len(self.live_ids)
        self.live_ids.extend(ids)
        turn = Turn(
            index=len(self.turns), role=role, speaker=speaker, text=text,
            start=start, end=len(self.live_ids), original_tokens=len(ids),
        )
        self.turns.append(turn)
        self._bank({"type": "turn", "role": role, "speaker": speaker, "text": text,
                    "n_tokens": len(ids), "ids": ids})
        return turn

    def add_user_turn(self, text: str) -> Turn:
        if self.mode == "chat":
            body = self._encode(f"<|im_start|>user\n{text}<|im_end|>\n")
            return self._append_turn("user", self.user_name, text, body_ids=body)
        return self._append_turn("user", self.user_name, text)

    def add_model_turn(self, gen_ids: list[int], text: str, prompt_tail_ids: list[int]) -> Turn:
        """Record the model's reply. ``prompt_tail_ids`` is the generation
        prefill (assistant header / ``"{bot}:"`` tag) that preceded ``gen_ids``
        in the prompt; both are part of the turn so ledger == connector store."""
        ids = list(gen_ids)
        while ids and ids[-1] in (self._stop_ids[0], self._delim[0]):
            ids.pop()  # normalize: exactly one closing delimiter, added below
        if self.mode == "chat":
            closing = self._stop_ids + self._encode("\n")
        else:
            closing = list(self._delim)
        return self._append_turn(
            "model", self.bot_name, text.strip(),
            prefix_ids=list(prompt_tail_ids), body_ids=ids + closing,
        )

    def reply_prefill_ids(self) -> list[int]:
        """The generation prefill appended after the transcript."""
        if self.mode == "chat":
            return self._encode("<|im_start|>assistant\n")
        return self._encode(f"{self.bot_name}:")

    # ------------------------------------------------------------------
    # eviction
    # ------------------------------------------------------------------

    def _live_spans(self) -> list[TurnSpan]:
        return [
            TurnSpan(index=t.index, role=t.role, start=t.start, end=t.end)
            for t in self.turns
            if not t.evicted
        ]

    def plan_eviction_if_needed(self, incoming_tokens: int) -> EvictionEvent | None:
        """Turn-aligned eviction plan when the upcoming prompt would breach the
        budget. Mutates the ledger/spans to the post-eviction state and returns
        the event (caller ships ``keep`` to the connector — or, for the
        recompute policy, bumps the store generation)."""
        cfg = self.config
        projected = len(self.live_ids) + incoming_tokens
        if cfg.policy == "none" or projected <= cfg.budget:
            return None

        target_live = max(int(cfg.budget * cfg.evict_to_frac) - incoming_tokens, 1)
        need_to_evict = len(self.live_ids) - target_live
        if need_to_evict <= 0:
            return None

        if cfg.policy == "oldest":
            protect_roles: tuple[str, ...] = ()
            sinks = 0
            protect_last = 1
        else:  # sink_rotate & recompute share the plan; they differ in transport
            protect_roles = ("system",)
            sinks = cfg.num_sink_tokens
            protect_last = cfg.protect_last_turns

        spans = self._live_spans()
        evict_turns = oldest_turns_to_evict(
            spans,
            target_tokens=need_to_evict,
            protect_roles=protect_roles,
            protect_last=protect_last,
        )
        if not evict_turns:
            return None
        keep_t = turn_keep_indices(
            spans, evict_turns, len(self.live_ids), num_sink_tokens=sinks
        )
        keep = [int(i) for i in keep_t.tolist()]
        src_len = len(self.live_ids)
        event = EvictionEvent(
            turn_indices=sorted(evict_turns),
            keep=keep,
            src_len=src_len,
            evicted_tokens=src_len - len(keep),
            policy=cfg.policy,
            recompute=(cfg.policy == "recompute"),
        )
        # compose the store-coordinate plan BEFORE mutating the ledger:
        # positions < _boundary map 1:1 onto the pending store keep (or onto
        # the store itself when nothing is pending yet)
        if event.recompute:
            self._generation += 1  # fresh connector store; plans are moot
            self._pending_store_keep = None
            self.store_len = 0
            self._boundary = 0
        else:
            in_store = [i for i in keep if i < self._boundary]
            if self._pending_store_keep is None:
                composed = in_store
            else:
                composed = [self._pending_store_keep[i] for i in in_store]
            self._pending_store_keep = composed
            self._boundary = len(in_store)
        self._apply_keep(keep, set(evict_turns))
        self.events.append(event)
        self._bank({"type": "eviction", "keep": keep, "src_len": src_len,
                    "turn_indices": event.turn_indices, "policy": cfg.policy,
                    "recompute": event.recompute})
        return event

    def _apply_keep(self, keep: list[int], evicted_turn_indices: set[int]) -> None:
        if keep and (keep[-1] >= len(self.live_ids) or keep[0] < 0):
            raise LedgerError("keep indices out of ledger range")
        self.live_ids = [self.live_ids[i] for i in keep]
        for t in self.turns:
            if t.evicted:
                continue
            new_start = bisect.bisect_left(keep, t.start)
            new_end = bisect.bisect_left(keep, t.end)
            t.start, t.end = new_start, new_end
            if new_end - new_start == 0 or t.index in evicted_turn_indices:
                # a "protected sink" remnant inside an evicted turn keeps its
                # tokens in the ledger but the turn is gone conversationally
                t.evicted = True

    # ------------------------------------------------------------------
    # prompt assembly + views
    # ------------------------------------------------------------------

    def build_request(self, user_text: str) -> dict[str, Any]:
        """Everything the vLLM client needs for one reply. Ordering matters:
        plan first (against the pre-turn ledger), then append the new user
        turn, then the generation prefill. The shipped plan is the composed
        pending plan in STORE coordinates (see __init__ bookkeeping); after a
        successful reply the caller MUST call :meth:`mark_synced`."""
        tail = self.reply_prefill_ids()
        incoming = len(self._encode(f"{self.user_name}: {user_text}")) + 1 + len(tail) \
            + self.config.max_reply_tokens
        event = self.plan_eviction_if_needed(incoming)
        self.add_user_turn(user_text)
        prompt_ids = self.live_ids + tail
        kvrot: dict[str, Any] = {"session_id": self.connector_session_id}
        if self._pending_store_keep is not None and self.store_len > 0 \
                and len(self._pending_store_keep) > 0:
            kvrot["plan"] = {
                "keep": list(self._pending_store_keep),
                "src_len": self.store_len,
            }
        return {
            "prompt_ids": prompt_ids,
            "kvrot": kvrot,
            "prefill_tail_ids": tail,
            "stop_token_ids": list(self._stop_ids),
            "event": event,
            "max_tokens": self.config.max_reply_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }

    # ------------------------------------------------------------------
    # fork / reroll / export (the A/B atomics)
    # ------------------------------------------------------------------

    def clone(self) -> "Session":
        """Fork: same transcript/config, NEW session id and a fresh connector
        store (first turn re-prefills; branches are fully independent after)."""
        import copy

        s = object.__new__(Session)
        s.__dict__.update(self.__dict__)
        s.session_id = uuid.uuid4().hex[:12]
        s.live_ids = list(self.live_ids)
        s.turns = copy.deepcopy(self.turns)
        s.events = list(self.events)
        s.config = self.config.model_copy()
        s._generation = 0
        s.store_len = 0
        s._boundary = 0
        s._pending_store_keep = None
        if self.bank_path is not None:
            s.bank_path = self.bank_path.parent / f"{s.session_id}.jsonl"
        s._bank({"type": "fork", "parent": self.session_id})
        return s

    def pop_model_turn(self) -> Turn:
        """Remove the last (model) turn for a reroll. The connector store still
        holds the exact prompt that produced it, so the reroll request reuses
        the full claimed prefix and just resamples."""
        if not self.turns or self.turns[-1].role != "model":
            raise LedgerError("last turn is not a model turn; nothing to reroll")
        last = self.turns.pop()
        self.live_ids = self.live_ids[: last.start]
        self._boundary = min(self._boundary, last.start)
        self._bank({"type": "reroll_pop", "turn_index": last.index})
        return last

    def build_reroll_request(self) -> dict[str, Any]:
        """Like build_request but with no new user turn and no fresh planning."""
        tail = self.reply_prefill_ids()
        kvrot: dict[str, Any] = {"session_id": self.connector_session_id}
        if self._pending_store_keep and self.store_len > 0:
            kvrot["plan"] = {
                "keep": list(self._pending_store_keep), "src_len": self.store_len,
            }
        return {
            "prompt_ids": self.live_ids + tail,
            "kvrot": kvrot,
            "prefill_tail_ids": tail,
            "stop_token_ids": list(self._stop_ids),
            "event": None,
            "max_tokens": self.config.max_reply_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }

    def export_dict(self) -> dict[str, Any]:
        return {
            "kvrot_playground_export": 1,
            "bot_name": self.bot_name,
            "user_name": self.user_name,
            "config": self.config.model_dump(),
            "turns": [
                {"role": t.role, "speaker": t.speaker, "text": t.text,
                 "evicted": t.evicted}
                for t in self.turns
            ],
            "events": [
                {"turn_indices": e.turn_indices, "evicted_tokens": e.evicted_tokens,
                 "policy": e.policy, "recompute": e.recompute}
                for e in self.events
            ],
        }

    def mark_synced(self, prompt_len: int) -> None:
        """Call after a successful reply: the connector saved the full prompt,
        so its store now equals the ledger prefix of that length (the shipped
        plan, if any, was consumed)."""
        self._pending_store_keep = None
        self.store_len = prompt_len
        self._boundary = min(prompt_len, len(self.live_ids))

    def expected_claim(self) -> int:
        """What the connector should claim for the *next* request (display)."""
        return align_down(len(self.live_ids), self.config.block_size)

    def view(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "connector_session_id": self.connector_session_id,
            "bot_name": self.bot_name,
            "config": self.config.model_dump(),
            "live_tokens": len(self.live_ids),
            "turns": [
                {
                    "index": t.index, "role": t.role, "speaker": t.speaker,
                    "text": t.text, "evicted": t.evicted,
                    "live_tokens": t.live_tokens(), "original_tokens": t.original_tokens,
                }
                for t in self.turns
            ],
            "events": [
                {
                    "turn_indices": e.turn_indices, "evicted_tokens": e.evicted_tokens,
                    "src_len": e.src_len, "policy": e.policy, "recompute": e.recompute,
                }
                for e in self.events
            ],
        }

    # ------------------------------------------------------------------

    def _bank(self, record: dict[str, Any]) -> None:
        if self.bank_path is None:
            return
        record = {"ts": time.time(), "session": self.session_id, **record}
        try:
            self.bank_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.bank_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            pass  # banking must never take down a live turn
