"""Context loader: source reading + needle insertion (no model/tokenizer needed)."""

from __future__ import annotations

import json

import torch

from kvrot.data import insert_needle, read_texts


def test_insert_needle_places_span_at_depth():
    ctx = torch.arange(10).unsqueeze(0)
    needle = torch.tensor([[99, 98]])
    new, start, end = insert_needle(ctx, needle, 0.5)
    assert (start, end) == (5, 7)
    assert new[0, start:end].tolist() == [99, 98]
    assert new.shape[1] == 12
    # surrounding context preserved
    assert new[0, :start].tolist() == list(range(5))
    assert new[0, end:].tolist() == list(range(5, 10))


def test_insert_needle_never_clobbers_bos():
    ctx = torch.arange(10).unsqueeze(0)
    _, start, _ = insert_needle(ctx, torch.tensor([[99]]), 0.0)
    assert start >= 1  # depth 0 still inserts after position 0 (BOS)


def test_read_plain_text(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("hello world", encoding="utf-8")
    assert read_texts(str(f)) == ["hello world"]


def test_read_directory_in_name_order(tmp_path):
    (tmp_path / "b.txt").write_text("second", encoding="utf-8")
    (tmp_path / "a.txt").write_text("first", encoding="utf-8")
    assert read_texts(str(tmp_path)) == ["first", "second"]


def test_read_jsonl_text_field(tmp_path):
    f = tmp_path / "data.jsonl"
    f.write_text(
        json.dumps({"text": "doc one"}) + "\n" + json.dumps({"content": "doc two"}) + "\n",
        encoding="utf-8",
    )
    assert read_texts(str(f)) == ["doc one", "doc two"]


def test_read_jsonl_chat_messages(tmp_path):
    f = tmp_path / "chat.jsonl"
    f.write_text(
        json.dumps({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}) + "\n",
        encoding="utf-8",
    )
    assert read_texts(str(f)) == ["hi\nyo"]
