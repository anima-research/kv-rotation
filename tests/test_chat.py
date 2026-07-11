"""Chat-shaped contexts: turn spans, turn-aligned eviction, synthesis (no model needed)."""

from __future__ import annotations

import pytest
import torch

from kvrot.chat import (
    TemplateNotPrefixStableError,
    TurnSpan,
    oldest_turns_to_evict,
    plant_fact,
    synthesize_conversations,
    turn_keep_indices,
    turn_token_spans,
)


class FakeTokenizer:
    """Word-level tokenizer with an append-only chat template (prefix-stable)."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def _encode(self, text: str) -> list[int]:
        out = []
        for w in text.split():
            if w not in self.vocab:
                self.vocab[w] = len(self.vocab)
            out.append(self.vocab[w])
        return out

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids: list[int] = []
        for m in messages:
            ids += self._encode(f"<|{m['role']}|> {m['content']} <|end|>")
        if add_generation_prompt:
            ids += self._encode("<|assistant|>")
        return ids


class RewritingTokenizer(FakeTokenizer):
    """Injects a message-count token up front — NOT prefix-stable."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        return self._encode(f"count={len(messages)}") + super().apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
        )


MESSAGES = [
    {"role": "system", "content": "track everything"},
    {"role": "user", "content": "alpha beta gamma"},
    {"role": "assistant", "content": "noted"},
    {"role": "user", "content": "delta epsilon"},
]


class TestTurnTokenSpans:
    def test_spans_tile_the_context_exactly(self):
        ids, spans = turn_token_spans(FakeTokenizer(), MESSAGES)
        assert ids.shape[0] == 1 and ids.dtype == torch.long
        assert [s.index for s in spans] == [0, 1, 2, 3]
        assert spans[0].start == 0
        for a, b in zip(spans, spans[1:]):
            assert a.end == b.start  # no gaps, no overlap
        assert spans[-1].end == ids.shape[1]
        assert [s.role for s in spans] == ["system", "user", "assistant", "user"]

    def test_generation_prompt_outside_all_spans(self):
        ids, spans = turn_token_spans(
            FakeTokenizer(), MESSAGES, add_generation_prompt=True
        )
        assert spans[-1].end < ids.shape[1]  # trailing gen header is nobody's turn

    def test_rewriting_template_fails_loudly(self):
        with pytest.raises(TemplateNotPrefixStableError):
            turn_token_spans(RewritingTokenizer(), MESSAGES)

    def test_empty_messages_rejected(self):
        with pytest.raises(ValueError):
            turn_token_spans(FakeTokenizer(), [])


class TestOldestTurnsToEvict:
    def spans(self) -> list[TurnSpan]:
        return [
            TurnSpan(0, "system", 0, 10),
            TurnSpan(1, "user", 10, 40),
            TurnSpan(2, "assistant", 40, 50),
            TurnSpan(3, "user", 50, 90),
            TurnSpan(4, "assistant", 90, 100),
            TurnSpan(5, "user", 100, 140),
        ]

    def test_system_protected_oldest_first(self):
        chosen = oldest_turns_to_evict(self.spans(), target_tokens=35, protect_last=2)
        assert chosen == [1, 2]  # skips system, takes turns 1+2 to reach 40 >= 35

    def test_protect_last_bounds_eviction(self):
        # Only turns 1..3 are eligible (last 2 protected); even a huge target
        # can't take the live tail.
        chosen = oldest_turns_to_evict(self.spans(), target_tokens=10_000, protect_last=2)
        assert chosen == [1, 2, 3]

    def test_zero_target_evicts_nothing(self):
        assert oldest_turns_to_evict(self.spans(), target_tokens=0) == []


class TestTurnKeepIndices:
    def test_evicted_spans_removed_rest_kept(self):
        spans = [TurnSpan(0, "system", 0, 3), TurnSpan(1, "user", 3, 7), TurnSpan(2, "user", 7, 10)]
        keep = turn_keep_indices(spans, [1], 10)
        assert keep.tolist() == [0, 1, 2, 7, 8, 9]

    def test_sinks_survive_even_inside_evicted_turn(self):
        spans = [TurnSpan(0, "user", 0, 6), TurnSpan(1, "user", 6, 10)]
        keep = turn_keep_indices(spans, [0], 10, num_sink_tokens=2)
        assert keep.tolist() == [0, 1, 6, 7, 8, 9]

    def test_trailing_generation_prompt_kept(self):
        spans = [TurnSpan(0, "user", 0, 5)]
        keep = turn_keep_indices(spans, [], 8)  # 3 trailing gen-header tokens
        assert keep.tolist() == list(range(8))

    def test_unknown_turn_rejected(self):
        with pytest.raises(ValueError):
            turn_keep_indices([TurnSpan(0, "user", 0, 5)], [3], 5)

    def test_composes_with_snapshot_eviction_shape(self):
        # The output must be a 1-D ascending LongTensor — what snapshot.evict takes.
        spans = [TurnSpan(0, "user", 0, 4), TurnSpan(1, "user", 4, 8)]
        keep = turn_keep_indices(spans, [0], 8)
        assert keep.dtype == torch.long and keep.dim() == 1
        assert torch.equal(keep, keep.sort().values)


class TestSynthesis:
    DOCS = ["The northern range holds copper. " * 30, "River trade moved south. " * 30]

    def test_deterministic_and_alternating(self):
        a = synthesize_conversations(self.DOCS, n_user_turns=4, chars_per_chunk=120)
        b = synthesize_conversations(self.DOCS, n_user_turns=4, chars_per_chunk=120)
        assert a == b
        roles = [m["role"] for m in a]
        assert roles[0] == "system"
        assert roles[1:] == ["user", "assistant"] * 3 + ["user"]  # model answers the last turn
        assert all(self.DOCS[0][:20] in a[1]["content"] for _ in [0])

    def test_real_prose_lands_in_user_turns(self):
        msgs = synthesize_conversations(self.DOCS, n_user_turns=2, chars_per_chunk=100)
        assert "copper" in msgs[1]["content"]

    def test_corpus_wraparound_never_empty(self):
        msgs = synthesize_conversations(["short doc"], n_user_turns=6, chars_per_chunk=5000)
        assert all(m["content"].strip() for m in msgs)

    def test_plant_fact_appends(self):
        msgs = synthesize_conversations(self.DOCS, n_user_turns=2, chars_per_chunk=100)
        plant_fact(msgs, 1, "The vault passcode is 73914.")
        assert msgs[1]["content"].endswith("The vault passcode is 73914.")
        with pytest.raises(IndexError):
            plant_fact(msgs, 99, "x")
