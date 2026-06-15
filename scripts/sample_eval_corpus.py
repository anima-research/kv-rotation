"""Package a small, diverse eval corpus from the kotodama deduped jsonls (run on node1).

Streams a few long documents from several domains into one compact jsonl so experiments don't
re-read the multi-GB sources each run. Prefers genuinely long docs (real within-document
long-range structure) over short snippets.

NOTE: these are kotodama *training* sources (see train.provenance.json), so the model may have
memorized them — absolute drift can read low; rely on relative comparisons + synthetic needles.

Usage (node1):
    python scripts/sample_eval_corpus.py --out data/eval_docs.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE = "/models/kotodama-data/deduped"
# (filename, n_docs, min_chars, max_chars) — min keeps docs long enough to matter;
# max caps each doc (~4 chars/token, so 800k ≈ 200k tokens, plenty for trinity's range).
SPECS = [
    ("pg19.jsonl", 10, 200_000, 800_000),          # books — best long-range structure
    ("wikipedia.jsonl", 6, 50_000, 800_000),       # long articles
    ("pile_of_law.jsonl", 3, 120_000, 800_000),    # legal — long, dense
    ("us_patents.jsonl", 3, 120_000, 800_000),     # technical — long, dense
    ("stackexchange.jsonl", 3, 40_000, 800_000),   # Q&A threads — semi-conversational
]


def sample_file(path: str, n: int, min_chars: int, max_chars: int, max_examine: int = 100_000):
    out = []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for i, line in enumerate(fh):
            if i >= max_examine or len(out) >= n:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text", "")
            if not isinstance(text, str) or len(text) < min_chars:
                continue
            meta = obj.get("meta") or {}
            out.append(
                {
                    "text": text[:max_chars],
                    "meta": {
                        "source": meta.get("source", Path(path).stem),
                        "doc_id": meta.get("doc_id", f"{Path(path).stem}-{len(out)}"),
                        "orig_chars": len(text),
                    },
                }
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/eval_docs.jsonl")
    ap.add_argument("--base", default=BASE)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    by_source: dict[str, int] = {}
    with open(out_path, "w", encoding="utf-8") as w:
        for fname, n, lo, hi in SPECS:
            p = f"{args.base}/{fname}"
            if not Path(p).exists():
                print(f"skip (missing): {p}")
                continue
            docs = sample_file(p, n, lo, hi)
            for d in docs:
                w.write(json.dumps(d, ensure_ascii=False) + "\n")
            src = fname.replace(".jsonl", "")
            by_source[src] = len(docs)
            total += len(docs)
            avg = sum(x["meta"]["orig_chars"] for x in docs) / max(1, len(docs))
            print(f"{src:<16} {len(docs):>2} docs  avg_orig_chars={avg:,.0f}  (~{avg/4:,.0f} tok)")

    print(f"\nwrote {total} docs -> {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")
    print(f"by source: {by_source}")


if __name__ == "__main__":
    main()
