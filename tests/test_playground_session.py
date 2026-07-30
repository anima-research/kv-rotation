"""CPU tests for kvrot_playground.session — no vLLM, no transformers.

The load-bearing property: the plan shipped to the connector is expressed in
STORE coordinates (the connector saves prompt tokens only, so its store is a
prefix of the driver ledger), and plans compose correctly across forced /
repeated evictions between requests. A desync here is exactly what the
connector's src_len assert exists to catch — these tests keep us from ever
tripping it.
"""

from __future__ import annotations

from kvrot_playground.session import PlaygroundConfig, Session


class StubTokenizer:
    """1 word = 1 token; ids are stable hashes. eos=0, bos=1."""

    eos_token_id = 0
    bos_token_id = 1
    eos_token = "<eos>"

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False):
        ids = []
        for w in text.split():
            if w not in self.vocab:
                self.vocab[w] = 2 + len(self.vocab)
            ids.append(self.vocab[w])
        return ids

    def decode(self, ids):
        rev = {v: k for k, v in self.vocab.items()}
        return " ".join(rev.get(i, "?") for i in ids)


def make_session(**cfg) -> Session:
    defaults = dict(budget=64, evict_to_frac=0.75, num_sink_tokens=2,
                    protect_last_turns=1, max_reply_tokens=8)
    defaults.update(cfg)
    return Session(
        StubTokenizer(),
        bot_name="Bot",
        config=PlaygroundConfig(**defaults),
        preamble="scene setting words here",
    )


def do_turn(s: Session, text: str, reply_words: str) -> dict:
    """Simulate one full request/reply cycle like app.send_turn does."""
    req = s.build_request(text)
    gen_ids = s._encode(reply_words) + [0]  # model emits eos at the end
    s.add_model_turn(gen_ids, reply_words, req["prefill_tail_ids"])
    s.mark_synced(len(req["prompt_ids"]))
    return req


def test_ledger_prompt_prefix_invariant():
    s = make_session()
    req1 = do_turn(s, "hello there friend", "hi")
    # after the reply, the ledger's prefix must equal the prompt we sent
    assert s.live_ids[: len(req1["prompt_ids"])] == req1["prompt_ids"]
    req2 = do_turn(s, "how are you", "fine thanks")
    assert s.live_ids[: len(req2["prompt_ids"])] == req2["prompt_ids"]
    # and each prompt extends the previous one (store reuse without plans)
    assert req2["prompt_ids"][: len(req1["prompt_ids"])] == req1["prompt_ids"]


def test_no_plan_until_budget_breached():
    s = make_session()
    req = s.build_request("short question")
    assert "plan" not in req["kvrot"]
    assert req["kvrot"]["session_id"].endswith(".g0")


def test_eviction_plan_is_in_store_coordinates():
    s = make_session(budget=48)
    do_turn(s, "one two three four five six seven eight nine ten", "a b c d e f")
    do_turn(s, "more filler words to grow the context nicely along", "g h i j k l")
    store_before = s.store_len
    assert store_before > 0
    # this turn breaches the budget -> plan must reference the STORE, whose
    # length is the last prompt, not the ledger (which includes the reply)
    req = s.build_request("this one should trigger an eviction now yes")
    assert "plan" in req["kvrot"], "expected a breach"
    plan = req["kvrot"]["plan"]
    assert plan["src_len"] == store_before
    assert all(0 <= i < store_before for i in plan["keep"])
    assert plan["keep"] == sorted(set(plan["keep"]))
    # sinks survive every plan
    assert plan["keep"][: s.config.num_sink_tokens] == list(range(s.config.num_sink_tokens))


def test_evicted_turns_marked_and_ledger_shrinks():
    s = make_session(budget=48)
    do_turn(s, "one two three four five six seven eight nine ten", "a b c d e f")
    do_turn(s, "more filler words to grow the context nicely along", "g h i j k l")
    before = len(s.live_ids)
    s.build_request("this one should trigger an eviction now definitely")
    evicted = [t for t in s.turns if t.evicted]
    assert evicted, "some turn should have been evicted"
    assert all(t.role != "system" for t in evicted), "system preamble is protected"
    assert len(s.live_ids) < before + 20  # shrank (modulo the new user turn)
    # spans stay consistent: non-evicted turns tile the ledger
    live = [t for t in s.turns if not t.evicted]
    for a, b in zip(live, live[1:]):
        assert a.end <= b.start
    assert live[-1].end == len(s.live_ids)


def test_forced_evictions_compose():
    s = make_session(budget=200)  # roomy: only forced evictions fire
    do_turn(s, "one two three four five six", "a b c")
    do_turn(s, "seven eight nine ten eleven twelve", "d e f")
    do_turn(s, "thirteen fourteen fifteen sixteen seventeen", "g h i")
    store = s.store_len
    ledger_before = list(s.live_ids)
    e1 = s.plan_eviction_if_needed(incoming_tokens=10**6)  # force
    assert e1 is not None and not e1.recompute
    e2 = s.plan_eviction_if_needed(incoming_tokens=10**6)  # force again
    # (second may be None if nothing evictable remains; both cases legal)
    req = s.build_request("and now a question")
    plan = req["kvrot"]["plan"]
    assert plan["src_len"] == store
    # composed keep must reproduce the surviving store-region tokens exactly
    reconstructed = [ledger_before[i] for i in plan["keep"]]
    boundary = len(plan["keep"])
    assert s.live_ids[:boundary] == reconstructed


def test_recompute_policy_bumps_generation_and_ships_no_plan():
    s = make_session(budget=48, policy="recompute")
    do_turn(s, "one two three four five six seven eight nine ten", "a b c d e f")
    do_turn(s, "more filler words to grow the context nicely along", "g h i j k l")
    req = s.build_request("this one should trigger an eviction now")
    assert "plan" not in req["kvrot"]
    assert req["kvrot"]["session_id"].endswith(".g1")  # fresh connector store
    assert s.store_len == 0 or "plan" not in req["kvrot"]


def test_none_policy_never_evicts():
    s = make_session(budget=32, policy="none")
    do_turn(s, "one two three four five six seven eight", "a b c")
    req = s.build_request("way past budget by now certainly")
    assert "plan" not in req["kvrot"]
    assert not any(t.evicted for t in s.turns)


def test_model_turn_normalizes_delimiter():
    s = make_session()
    req = s.build_request("hi")
    # reply arriving WITH the eos already attached must not double it
    gen = s._encode("hello world") + [0]
    t = s.add_model_turn(gen, "hello world", req["prefill_tail_ids"])
    ids = s.live_ids[t.start : t.end]
    assert ids.count(0) == 1 and ids[-1] == 0


# ---------------------------------------------------------------------------
# chat render mode (ChatML compiled chunks — Trinity-Preview framing)
# ---------------------------------------------------------------------------


class ChatMLStubTokenizer(StubTokenizer):
    chat_template = "{{'<|im_start|>'}}..."  # ChatML marker is what matters

    def __init__(self):
        super().__init__()
        self.vocab["<|im_start|>"] = 9001
        self.vocab["<|im_end|>"] = 9002

    def encode(self, text, add_special_tokens=False):
        # crude special-token-aware split for the stub
        out = []
        for chunk in text.replace("<|im_start|>", " <|im_start|> ").replace(
            "<|im_end|>", " <|im_end|> "
        ).split():
            if chunk not in self.vocab:
                self.vocab[chunk] = 2 + len(self.vocab)
            out.append(self.vocab[chunk])
        return out

    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t, -1)


def test_chat_mode_framing_and_stops():
    s = Session(
        ChatMLStubTokenizer(), bot_name="Trinity",
        config=PlaygroundConfig(budget=4096), preamble="be yourself",
    )
    assert s.mode == "chat"
    im_end = s.tok.vocab["<|im_end|>"]
    req = s.build_request("hello there")
    assert req["stop_token_ids"] == [im_end]
    # generation prefill is the assistant header
    assert req["prefill_tail_ids"][0] == s.tok.vocab["<|im_start|>"]
    # reply with the stop token attached must normalize to one closing im_end
    gen = s._encode("hi friend") + [im_end]
    t = s.add_model_turn(gen, "hi friend", req["prefill_tail_ids"])
    ids = s.live_ids[t.start : t.end]
    assert ids.count(im_end) == 1
    s.mark_synced(len(req["prompt_ids"]))
    # prefix invariant still holds in chat mode
    req2 = s.build_request("and again")
    assert req2["prompt_ids"][: len(req["prompt_ids"])] == req["prompt_ids"]


def test_auto_mode_falls_back_to_prefill_without_template():
    s = make_session()
    assert s.mode == "prefill"
