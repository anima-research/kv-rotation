"""Naturalized-dialogue plumbing for exp10 (no model, no network needed)."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from kvrot.natural import (
    OMISSION_MARKER,
    ContextOverflowError,
    ConvRecord,
    GenParams,
    Node2Client,
    TurnRecord,
    append_turn_record,
    choose_cut_prefix,
    clean_generated_turn,
    compact_turns_to_convs,
    find_passcode_leaks,
    load_conv_records,
    load_turn_records,
    messages_for_trinity,
    render_deepseek_prompt,
)


def turn(i: int, speaker: str, text: str, conv: str = "c0", **kw) -> TurnRecord:
    return TurnRecord(conv_id=conv, turn_index=i, speaker=speaker, text=text, **kw)


def alternating(n: int, words_per_turn: int = 5) -> list[TurnRecord]:
    out = []
    for i in range(n):
        sp = "deepseek" if i % 2 == 0 else "trinity"
        out.append(turn(i, sp, " ".join(f"w{i}t{j}" for j in range(words_per_turn))))
    return out


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


# ── transcript rendering ─────────────────────────────────────────────────────────


class TestRenderDeepseekPrompt:
    def test_basic_shape(self):
        turns = [turn(0, "deepseek", "hello there"), turn(1, "trinity", "why hello")]
        p = render_deepseek_prompt("PREAMBLE.", turns)
        assert p.startswith("PREAMBLE.")
        assert "DeepSeek: hello there" in p
        assert "Trinity: why hello" in p
        assert p.endswith("DeepSeek:")
        assert OMISSION_MARKER not in p

    def test_next_speaker_label(self):
        p = render_deepseek_prompt("P.", [turn(0, "deepseek", "x")], next_speaker="trinity")
        assert p.endswith("Trinity:")

    def test_windowing_drops_oldest_keeps_newest(self):
        turns = [turn(i, "deepseek" if i % 2 == 0 else "trinity", f"turn number {i} " + "pad " * 30)
                 for i in range(20)]
        full = render_deepseek_prompt("P.", turns, char_budget=1_000_000)
        small = render_deepseek_prompt("P.", turns, char_budget=len(full) // 3)
        assert len(small) <= len(full) // 3
        assert OMISSION_MARKER in small
        assert "turn number 19" in small          # newest survives
        assert "turn number 0 " not in small      # oldest dropped
        # marker precedes all surviving turns
        assert small.index(OMISSION_MARKER) < small.index("turn number 19")

    def test_budget_respected(self):
        turns = alternating(30, words_per_turn=40)
        for budget in (500, 2000, 5000):
            p = render_deepseek_prompt("P.", turns, char_budget=budget)
            assert len(p) <= budget

    def test_never_splits_a_turn_when_avoidable(self):
        turns = alternating(10, words_per_turn=20)
        p = render_deepseek_prompt("P.", turns, char_budget=600)
        # every surviving turn body appears complete
        for t in turns:
            body = t.text
            if body[:30] in p:
                assert body in p

    def test_rejects_nonpositive_budget(self):
        with pytest.raises(ValueError):
            render_deepseek_prompt("P.", [], char_budget=0)


class TestCleanGeneratedTurn:
    def test_passthrough(self):
        text, flags = clean_generated_turn("  a fine turn.  ", own_label="DeepSeek",
                                           other_label="Trinity")
        assert text == "a fine turn." and flags == []

    def test_strips_own_label_echo(self):
        text, flags = clean_generated_turn("DeepSeek: my thought", own_label="DeepSeek",
                                           other_label="Trinity")
        assert text == "my thought" and "stripped_own_label" in flags

    def test_truncates_label_bleed(self):
        raw = "my thought.\ncontinues here.\nTrinity: I should not be here\nmore stolen text"
        text, flags = clean_generated_turn(raw, own_label="DeepSeek", other_label="Trinity")
        assert text == "my thought.\ncontinues here."
        assert "truncated_label_bleed" in flags

    def test_truncates_own_label_restart(self):
        raw = "first bit.\nDeepSeek: pretending to start a new turn"
        text, _ = clean_generated_turn(raw, own_label="DeepSeek", other_label="Trinity")
        assert text == "first bit."


# ── trinity message view ─────────────────────────────────────────────────────────


class TestMessagesForTrinity:
    def test_roles_and_alternation(self):
        msgs = messages_for_trinity("SYS", alternating(4))
        assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant"]
        assert msgs[0]["content"] == "SYS"

    def test_rejects_wrong_start(self):
        bad = [turn(0, "trinity", "i speak first")]
        with pytest.raises(ValueError, match="expected 'deepseek'"):
            messages_for_trinity("SYS", bad)

    def test_rejects_double_turn(self):
        bad = [turn(0, "deepseek", "a"), turn(1, "deepseek", "b")]
        with pytest.raises(ValueError, match="expected 'trinity'"):
            messages_for_trinity("SYS", bad)


# ── jsonl banking ────────────────────────────────────────────────────────────────


class TestTurnBank:
    def test_round_trip_two_convs(self, tmp_path):
        bank = tmp_path / "turns.jsonl"
        for r in alternating(3, 3):
            append_turn_record(bank, r)
        other = turn(0, "deepseek", "other conv", conv="c1",
                     params=GenParams(temperature=0.9, top_p=0.95, max_tokens=450, seed=7))
        append_turn_record(bank, other)
        loaded = load_turn_records(bank)
        assert set(loaded) == {"c0", "c1"}
        assert [t.turn_index for t in loaded["c0"]] == [0, 1, 2]
        assert loaded["c1"][0].params.seed == 7

    def test_missing_file_is_empty(self, tmp_path):
        assert load_turn_records(tmp_path / "nope.jsonl") == {}

    def test_torn_final_line_skipped(self, tmp_path):
        bank = tmp_path / "turns.jsonl"
        append_turn_record(bank, turn(0, "deepseek", "ok"))
        with bank.open("a") as f:
            f.write('{"conv_id": "c0", "turn_ind')  # crash mid-write
        loaded = load_turn_records(bank)
        assert [t.turn_index for t in loaded["c0"]] == [0]

    def test_duplicate_index_keeps_last(self, tmp_path):
        bank = tmp_path / "turns.jsonl"
        append_turn_record(bank, turn(0, "deepseek", "first attempt"))
        append_turn_record(bank, turn(0, "deepseek", "second attempt"))
        loaded = load_turn_records(bank)
        assert loaded["c0"][0].text == "second attempt"

    def test_resume_uses_only_contiguous_prefix(self, tmp_path):
        bank = tmp_path / "turns.jsonl"
        append_turn_record(bank, turn(0, "deepseek", "zero"))
        append_turn_record(bank, turn(2, "deepseek", "orphan"))  # gap at 1
        loaded = load_turn_records(bank)
        assert [t.turn_index for t in loaded["c0"]] == [0]

    def test_conv_compact_round_trip(self, tmp_path):
        conv = ConvRecord(conv_id="c0", topic="t", system_prompt="SYS", preamble="P",
                          turns=alternating(4), trinity_model="trin", deepseek_model="ds",
                          trinity_render_tokens=1234)
        path = tmp_path / "convs.jsonl"
        compact_turns_to_convs([conv], path)
        back = load_conv_records(path)
        assert len(back) == 1 and back[0] == conv
        assert [m["role"] for m in back[0].messages()][:3] == ["system", "user", "assistant"]


# ── eval-side helpers ────────────────────────────────────────────────────────────


class TestChooseCutPrefix:
    def test_smallest_user_terminated_prefix(self):
        tok = FakeTokenizer()
        msgs = messages_for_trinity("SYS", alternating(10, words_per_turn=8))
        n, rendered = choose_cut_prefix(tok, msgs, target_tokens=40)
        assert msgs[n - 1]["role"] == "user"
        assert rendered >= 40
        # minimality: the previous user-terminated boundary was below target
        prev_user = max(i for i in range(1, n) if msgs[i - 1]["role"] == "user") \
            if any(msgs[i - 1]["role"] == "user" for i in range(1, n)) else None
        if prev_user is not None:
            ids = tok.apply_chat_template(msgs[:prev_user], add_generation_prompt=True)
            assert len(ids) < 40

    def test_too_short_raises(self):
        tok = FakeTokenizer()
        msgs = messages_for_trinity("SYS", alternating(2))
        with pytest.raises(ValueError, match="too short"):
            choose_cut_prefix(tok, msgs, target_tokens=10_000)

    def test_rejects_bad_target(self):
        with pytest.raises(ValueError):
            choose_cut_prefix(FakeTokenizer(), [{"role": "user", "content": "x"}], 0)


class TestFindPasscodeLeaks:
    def test_clean_and_leaky(self):
        msgs = [{"role": "user", "content": "code is 62483"},
                {"role": "assistant", "content": "noted"},
                {"role": "user", "content": "again: 62483"}]
        assert find_passcode_leaks(msgs, "62483", planted_index=0) == [2]
        assert find_passcode_leaks(msgs, "73914", planted_index=0) == []


# ── stage-1 corpus conversion (kv_perturb format_b mapping) ─────────────────────


class TestCorpusChatMessages:
    RAW = [
        {"speaker": "lyra", "content": "hello all", "source": "seed"},
        {"speaker": "Opus 4.7", "content": "greetings", "source": "seed"},
        {"speaker": "Cogito", "content": "I am here", "source": "main"},
        {"speaker": "Cogito", "content": "twice in a row", "source": "main"},
        {"speaker": "K2", "content": "make room", "source": "auditor"},
    ]

    def test_grouping_and_roles(self):
        from kvrot.natural import corpus_chat_messages

        msgs = corpus_chat_messages(self.RAW, "Cogito")
        assert [m["role"] for m in msgs] == ["user", "assistant", "assistant", "user"]
        assert msgs[0]["content"] == "lyra: hello all\nOpus 4.7: greetings"
        assert msgs[1]["content"] == "I am here"          # bot name stripped
        assert msgs[3]["content"] == "K2: make room"

    def test_trailing_bot_and_no_bot(self):
        from kvrot.natural import corpus_chat_messages

        only_bot = corpus_chat_messages(self.RAW[2:4], "Cogito")
        assert [m["role"] for m in only_bot] == ["assistant", "assistant"]
        no_bot = corpus_chat_messages(self.RAW[:2], "Cogito")
        assert [m["role"] for m in no_bot] == ["user"]

    def test_never_two_user_in_a_row(self):
        from kvrot.natural import corpus_chat_messages

        msgs = corpus_chat_messages(self.RAW * 3, "Cogito")
        for a, b in zip(msgs, msgs[1:]):
            assert not (a["role"] == "user" and b["role"] == "user")


class TestLoadConvsAny:
    def test_sniffs_both_schemas_and_rejects_unknown(self, tmp_path):
        from kvrot.natural import Stage1Conv, load_convs_any

        s1 = Stage1Conv(conv_id="prefill-0003", condition="prefill", bot_name="Cogito",
                        source_path="x", approx_tokens=23000, n_source_messages=5,
                        chat_messages=[{"role": "user", "content": "a: hi"},
                                       {"role": "assistant", "content": "hello"}])
        s2 = ConvRecord(conv_id="conv00", topic="t", system_prompt="S", preamble="P",
                        turns=alternating(2))
        path = tmp_path / "mixed.jsonl"
        path.write_text(s1.model_dump_json() + "\n" + s2.model_dump_json() + "\n")
        loaded = load_convs_any(path)
        assert [type(c).__name__ for c in loaded] == ["Stage1Conv", "ConvRecord"]
        assert loaded[0].messages()[0]["role"] == "user"
        assert loaded[1].messages()[0]["role"] == "system"

        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"conv_id": "x", "something": 1}\n')
        with pytest.raises(ValueError, match="unrecognized"):
            load_convs_any(bad)


# ── node2 client (injected transport, no network) ────────────────────────────────


def ok_response(text: str = "a reply", tokens: int = 42) -> bytes:
    return json.dumps({"choices": [{"text": text, "finish_reason": "stop"}],
                       "usage": {"completion_tokens": tokens}}).encode()


class TestNode2Client:
    def make(self, transport, **kw):
        return Node2Client("http://fake:8000", "deepseek-v3-base", transport=transport,
                           backoff_s=0.001, **kw)

    def complete(self, client):
        return client.complete("prompt", max_tokens=10, temperature=0.9, top_p=0.95,
                               seed=1, stop=["\nTrinity:"])

    def test_success(self):
        calls = []

        def transport(url, body, timeout):
            calls.append(json.loads(body))
            return ok_response()

        text, ntok, finish = self.complete(self.make(transport))
        assert (text, ntok, finish) == ("a reply", 42, "stop")
        assert calls[0]["model"] == "deepseek-v3-base"
        assert calls[0]["stop"] == ["\nTrinity:"]

    def test_retries_transient_then_succeeds(self):
        state = {"n": 0}

        def transport(url, body, timeout):
            state["n"] += 1
            if state["n"] < 3:
                raise urllib.error.URLError("connection refused")
            return ok_response()

        text, _, _ = self.complete(self.make(transport, max_retries=4))
        assert text == "a reply" and state["n"] == 3

    def test_context_overflow_raises_immediately(self):
        def transport(url, body, timeout):
            raise urllib.error.HTTPError(
                url, 400, "Bad Request", {},
                io.BytesIO(b"This model's maximum context length is 16384 tokens"))

        with pytest.raises(ContextOverflowError):
            self.complete(self.make(transport))

    def test_exhausts_retries(self):
        def transport(url, body, timeout):
            raise urllib.error.URLError("down")

        with pytest.raises(RuntimeError, match="failed after"):
            self.complete(self.make(transport, max_retries=2))

    def test_malformed_json_is_transient(self):
        state = {"n": 0}

        def transport(url, body, timeout):
            state["n"] += 1
            return b"not json" if state["n"] == 1 else ok_response()

        text, _, _ = self.complete(self.make(transport))
        assert text == "a reply"


class TestTrinityEvenDeviceMap:
    def test_covers_all_modules_evenly(self):
        from kvrot.natural import trinity_even_device_map

        m = trinity_even_device_map(60, 8)
        layers = [v for k, v in m.items() if k.startswith("model.layers.")]
        assert len(layers) == 60
        assert m["model.embed_tokens"] == 0
        assert m["lm_head"] == 7 and m["model.norm"] == 7 and m["model.rotary_emb"] == 7
        from collections import Counter
        counts = Counter(layers)
        assert set(counts) == set(range(8))
        assert max(counts.values()) - min(counts.values()) <= 1
        # layers are assigned contiguously and in order
        order = [m[f"model.layers.{i}"] for i in range(60)]
        assert order == sorted(order)

    def test_rejects_impossible(self):
        from kvrot.natural import trinity_even_device_map
        import pytest

        with pytest.raises(ValueError):
            trinity_even_device_map(4, 8)
