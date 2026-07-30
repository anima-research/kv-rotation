#!/usr/bin/env python3
"""Seed a kvrot-playground session with a long context + planted needles.

Builds a ~N-token preamble from a real eval-corpus document, plants made-up
passcode needles at chosen depths (the exp02/exp08 protocol — invented codes
are immune to memorization confounds), creates the session via the playground
API, and prints the needle table + probe questions + a direct URL.

Example (a ~19k-token session for a 20k-scale test):

    python scripts/seed_playground_session.py \\
        --app-url http://localhost:2222 --model-path /models/Trinity-Large-TrueBase \\
        --target-tokens 19000 --budget 20000 --depths 0.05 0.35 0.65 0.9

Needles at depths that later get evicted SHOULD become unrecallable (honest
forgetting); needles in retained regions must survive rotation exactly.
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request

# neutral phrasing — an earlier "keep it safe" variant made the instruct
# model dutifully REFUSE to reveal the codes when asked
NEEDLE_TMPL = (
    "\n[Catalog entry: the reference code for the {name} shelf in this "
    "archive is {code}.]\n"
)
NAMES = ["amber", "cobalt", "juniper", "vermilion", "onyx", "saffron", "indigo", "larch"]


def make_code(rng: random.Random) -> str:
    words = ["FROST", "EMBER", "TIDE", "QUARTZ", "NOVA", "CEDAR", "IRIS", "SLATE"]
    return f"{rng.choice(words)}-{rng.choice(words)}-{rng.randint(10, 99)}"


def api(url: str, payload: dict | None = None, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Kvrot-Token"] = token
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--app-url", default="http://localhost:2222")
    ap.add_argument("--model-path", required=True, help="tokenizer path (same as served model)")
    ap.add_argument("--data", default="data/eval_docs.jsonl")
    ap.add_argument("--doc-index", type=int, default=None, help="default: longest doc")
    ap.add_argument("--target-tokens", type=int, default=19000)
    ap.add_argument("--budget", type=int, default=20000)
    ap.add_argument("--depths", type=float, nargs="+", default=[0.05, 0.35, 0.65, 0.9])
    ap.add_argument("--policy", default="sink_rotate",
                    choices=["sink_rotate", "oldest", "recompute", "none"])
    ap.add_argument("--max-reply", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--token", default=None, help="KVROT_TOKEN if the app is gated")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    rng = random.Random(args.seed)

    docs = [json.loads(l) for l in open(args.data) if l.strip()]
    texts = [d.get("text") or d.get("content") or "" for d in docs]
    idx = args.doc_index if args.doc_index is not None else max(
        range(len(texts)), key=lambda i: len(texts[i])
    )
    doc_ids = tok.encode(texts[idx], add_special_tokens=False)
    if len(doc_ids) < args.target_tokens:
        raise SystemExit(
            f"doc {idx} has only ~{len(doc_ids)} tokens; need {args.target_tokens} "
            "(pick another --doc-index or concatenate)"
        )

    # splice needles at token depths (decode segments, insert needle text —
    # retokenization drift of a few tokens is irrelevant at this scale)
    depths = sorted(args.depths)
    cuts = [int(d * args.target_tokens) for d in depths]
    needles = []
    parts = []
    prev = 0
    for d, cut in zip(depths, cuts):
        name = NAMES[len(needles) % len(NAMES)]
        code = make_code(rng)
        needles.append({"name": name, "code": code, "depth": d, "token_pos": cut})
        parts.append(tok.decode(doc_ids[prev:cut]))
        parts.append(NEEDLE_TMPL.format(name=name, code=code))
        prev = cut
    parts.append(tok.decode(doc_ids[prev:args.target_tokens]))
    preamble = (
        "You are Trinity, a large language model. You are reviewing an "
        "archive document with a colleague. Bracketed catalog entries are "
        "part of the archive's metadata; when asked about any detail from "
        "the document or its catalog entries, answer factually and "
        "directly.\n\n=== ARCHIVE DOCUMENT ===\n" + "".join(parts)
        + "\n=== END OF DOCUMENT ==="
    )

    config = {
        "policy": args.policy,
        "budget": args.budget,
        "max_reply_tokens": args.max_reply,
    }
    state = api(f"{args.app_url}/api/sessions",
                {"config": config, "preamble": preamble}, args.token)
    sid = state["session_id"]

    print(f"\nsession: {sid}  (live tokens: {state['live_tokens']}, "
          f"budget {args.budget}, policy {args.policy})")
    print(f"open:    {args.app_url}/?session={sid}\n")
    print(f"{'depth':>6} {'~tok pos':>9}  {'vault':<10} code")
    for n in needles:
        print(f"{n['depth']:>6} {n['token_pos']:>9}  {n['name']:<10} {n['code']}")
    print("\nprobe with e.g.:")
    for n in needles:
        print(f'  "What is the access code for the {n["name"]} vault?"')
    print("\nexpectation: needles in evicted regions are honestly forgotten; "
          "needles in retained regions survive rotation exactly.")


if __name__ == "__main__":
    main()
