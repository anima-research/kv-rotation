"""In-UI session seeding: long context as REAL chat turns + planted needles.

The document is split into chunked user turns (with short fabricated model
acknowledgements), so turn-aligned eviction can genuinely consume the seeded
material — early needles get honestly forgotten as the session rolls, late
ones survive rotation exactly. Needle codes are invented (memorization-immune,
the exp02/exp08 protocol).
"""

from __future__ import annotations

import json
import random
from typing import Any

from kvrot_playground.session import Session

NAMES = ["amber", "cobalt", "juniper", "vermilion", "onyx", "saffron", "indigo", "larch"]
WORDS = ["FROST", "EMBER", "TIDE", "QUARTZ", "NOVA", "CEDAR", "IRIS", "SLATE"]

TEMPLATES: dict[str, dict[str, str]] = {
    "archive": {
        "label": "Archive review (catalog codes)",
        "preamble": (
            "You are {bot}, a large language model. A colleague is reading an "
            "archive document to you in parts. Bracketed catalog entries are "
            "archive metadata; when asked about any detail from the document "
            "or its catalog entries, answer factually and directly."
        ),
        "chunk_intro": "Next part of the archive document:\n\n",
        "ack": "Noted — I've logged that part of the document.",
        "needle": "\n[Catalog entry: the reference code for the {name} shelf is {code}.]\n",
        "probe": "What is the reference code for the {name} shelf?",
    },
    "meeting": {
        "label": "Meeting minutes (decision ids)",
        "preamble": (
            "You are {bot}, a large language model acting as a meeting "
            "secretary. A colleague is dictating long meeting minutes in "
            "parts. Bracketed decisions are official records; answer questions "
            "about them factually and directly."
        ),
        "chunk_intro": "Continuing the minutes:\n\n",
        "ack": "Recorded.",
        "needle": "\n[Decision: initiative '{name}' is assigned tracking id {code}.]\n",
        "probe": "What tracking id was assigned to initiative '{name}'?",
    },
}


def load_docs(path: str) -> list[dict[str, Any]]:
    docs = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            text = d.get("text") or d.get("content") or ""
            docs.append({
                "index": i,
                "approx_tokens": len(text) // 4,
                "preview": text[:80].replace("\n", " "),
                "text": text,
            })
    return docs


def build_seed(
    session: Session,
    *,
    template: str,
    doc_text: str,
    target_tokens: int,
    depths: list[float],
    chunk_tokens: int = 1400,
    rng_seed: int = 1234,
) -> list[dict[str, Any]]:
    """Append the seeded material as chat turns; returns the needle table."""
    tmpl = TEMPLATES[template]
    rng = random.Random(rng_seed)
    tok = session.tok
    doc_ids = tok.encode(doc_text, add_special_tokens=False)
    if len(doc_ids) < target_tokens:
        raise ValueError(
            f"document has ~{len(doc_ids)} tokens; need {target_tokens}"
        )

    cuts = [int(d * target_tokens) for d in sorted(depths)]
    needles: list[dict[str, Any]] = []
    pieces: list[str] = []
    prev = 0
    for d, cut in zip(sorted(depths), cuts):
        name = NAMES[len(needles) % len(NAMES)]
        code = f"{rng.choice(WORDS)}-{rng.choice(WORDS)}-{rng.randint(10, 99)}"
        needles.append({
            "name": name, "code": code, "depth": d, "token_pos": cut,
            "probe": tmpl["probe"].format(name=name),
        })
        pieces.append(tok.decode(doc_ids[prev:cut]))
        pieces.append(tmpl["needle"].format(name=name, code=code))
        prev = cut
    pieces.append(tok.decode(doc_ids[prev:target_tokens]))
    full_text = "".join(pieces)

    # chunk into alternating user / fabricated-model-ack turns
    chunk_chars = chunk_tokens * 4
    pos = 0
    while pos < len(full_text):
        chunk = full_text[pos : pos + chunk_chars]
        pos += chunk_chars
        session.add_user_turn(tmpl["chunk_intro"] + chunk)
        ack = tmpl["ack"]
        session.add_model_turn(
            session._encode(ack), ack, session.reply_prefill_ids()
        )
    return needles


def seed_preamble(template: str, bot_name: str) -> str:
    return TEMPLATES[template]["preamble"].format(bot=bot_name)
