"""Context loaders for experiments.

Real corpora are strongly preferred over synthetic filler: repetitive text under-estimates
eviction drift (dropping near-duplicate content barely changes predictions) and has no genuine
long-range dependencies to test recall against. This module loads real text and tokenizes it to
a target length, with optional needle insertion for recall probes. A synthetic fallback remains
for quick smoke tests only.

Supported sources (auto-detected):
  * a single ``.txt`` / ``.md`` file
  * a directory of ``.txt`` / ``.md`` files (concatenated in name order until long enough)
  * a ``.jsonl`` file (one object per line; first of text/content/document/body fields used)
  * a ``.json`` file (list of strings, or list of objects with a text-like field)
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor

_TEXT_KEYS = ("text", "content", "document", "body", "conversation")


def _coerce_obj_to_text(obj) -> str | None:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for k in _TEXT_KEYS:
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v
        # chat-style: list of {role, content}
        msgs = obj.get("messages") or obj.get("turns")
        if isinstance(msgs, list):
            parts = [m.get("content", "") for m in msgs if isinstance(m, dict)]
            joined = "\n".join(p for p in parts if isinstance(p, str))
            if joined.strip():
                return joined
    return None


def read_texts(source: str) -> list[str]:
    """Read raw documents from ``source``. Returns a list of strings (never empty on success)."""
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"context source not found: {source}")

    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in (".txt", ".md"))
        if not files:
            raise ValueError(f"no .txt/.md files in directory: {source}")
        return [f.read_text(encoding="utf-8", errors="ignore") for f in files]

    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        texts: list[str] = []
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                t = _coerce_obj_to_text(json.loads(line))
            except json.JSONDecodeError:
                t = None
            if t:
                texts.append(t)
        if not texts:
            raise ValueError(f"no usable text rows in {source} (expected fields {_TEXT_KEYS})")
        return texts

    if suffix == ".json":
        obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        items = obj if isinstance(obj, list) else [obj]
        texts = [t for t in (_coerce_obj_to_text(o) for o in items) if t]
        if not texts:
            raise ValueError(f"no usable text in {source}")
        return texts

    # plain text
    return [p.read_text(encoding="utf-8", errors="ignore")]


def load_text_context(
    source: str,
    tokenizer,
    n_tokens: int,
    *,
    device="cpu",
    doc_index: int = 0,
    join: str = "\n\n",
) -> Tensor:
    """Tokenize real text to exactly ``n_tokens`` -> ``[1, n_tokens]`` (long).

    Starts from document ``doc_index``; if it's shorter than ``n_tokens``, concatenates
    subsequent documents (wrapping) until long enough. Raises if the whole corpus can't fill
    ``n_tokens`` (so experiments never silently run short)."""
    texts = read_texts(source)
    ordered = texts[doc_index:] + texts[:doc_index]
    ids = tokenizer(ordered[0], return_tensors="pt").input_ids
    i = 1
    while ids.shape[1] < n_tokens and i < len(ordered):
        more = tokenizer(join + ordered[i], return_tensors="pt", add_special_tokens=False).input_ids
        ids = torch.cat([ids, more], dim=1)
        i += 1
    if ids.shape[1] < n_tokens:
        raise ValueError(
            f"corpus has only {ids.shape[1]} tokens, need {n_tokens}; provide longer/more docs"
        )
    return ids[:, :n_tokens].to(device)


def insert_needle(
    context_ids: Tensor, needle_ids: Tensor, depth: float
) -> tuple[Tensor, int, int]:
    """Insert ``needle_ids`` at fractional ``depth`` (0=start, 1=end). Returns
    (new_context, needle_start, needle_end) so recall probes know the span to evict/keep."""
    if not 0.0 <= depth <= 1.0:
        raise ValueError("depth must be in [0, 1]")
    n = context_ids.shape[1]
    pos = max(1, min(n, int(depth * n)))  # avoid clobbering BOS at 0
    needle_ids = needle_ids.to(context_ids.device)
    new = torch.cat([context_ids[:, :pos], needle_ids, context_ids[:, pos:]], dim=1)
    return new, pos, pos + needle_ids.shape[1]


def synthetic_context(tokenizer, n_tokens: int, *, device="cpu") -> Tensor:
    """Repetitive filler — smoke tests ONLY. Real workloads must use load_text_context."""
    text = "Begin a long session. " + (
        "The agent proceeds step by step, noting observations and revisiting earlier points "
        "as the conversation grows over many turns. " * 16
    )
    ids = tokenizer(text, return_tensors="pt").input_ids
    while ids.shape[1] < n_tokens:
        text += "The session continues with further detail. " * 8
        ids = tokenizer(text, return_tensors="pt").input_ids
    return ids[:, :n_tokens].to(device)
