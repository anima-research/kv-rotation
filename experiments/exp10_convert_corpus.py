"""exp10 stage 1 (converter) — kv_perturb backrooms corpus → eval-ready conversations.

Converts the surviving naturalized Trinity backrooms transcripts
(data/backrooms_corpus/{prefill,chat}/*/transcript.json, rsynced from
node2:/models/kv_perturb_experiment/tier2_runs/) into the chat-message jsonl that
experiments/exp10_natural_chat.py consumes. The message mapping is kv_perturb's
own ``to_format_b`` ("standard chat format"): bot (Cogito/Trinity) messages →
raw assistant turns; consecutive non-bot messages → one user turn of
``speaker: content`` lines. No system message is added (the source conversations
had none; sinks are protected token-wise by the eval's num_sink_tokens).

Pure CPU, no model, no network. Deterministic. Run locally:
    uv run python experiments/exp10_convert_corpus.py \
        --corpus data/backrooms_corpus --out data/backrooms_convs_2026-07-10.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kvrot.natural import Stage1Conv, corpus_chat_messages


def convert_run(transcript_path: Path, condition: str) -> Stage1Conv:
    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    raw = data["messages"]
    bot = str(data.get("bot_name", "Cogito"))
    chat = corpus_chat_messages(raw, bot)
    if not chat:
        raise ValueError(f"{transcript_path}: empty conversation after conversion")
    approx = data.get("approx_tokens")
    if approx is None:
        approx = data.get("metadata", {}).get("final_approx_tokens")
    if approx is None:
        approx = sum(int(m.get("approx_tokens", 0)) for m in raw)
    gen_pct = data.get("generated_pct",
                       data.get("metadata", {}).get("final_generated_pct"))
    meta = {
        "participants": data.get("participants", []),
        "seed_id": data.get("seed_id"),
        "n_user_msgs": sum(1 for m in chat if m["role"] == "user"),
        "n_assistant_msgs": sum(1 for m in chat if m["role"] == "assistant"),
    }
    return Stage1Conv(
        conv_id=f"{condition}-{data.get('seed_id', transcript_path.parent.name)}",
        condition=condition,
        bot_name=bot,
        source_path=str(transcript_path),
        approx_tokens=int(approx),
        n_source_messages=len(raw),
        generated_pct=(float(gen_pct) if gen_pct is not None else None),
        chat_messages=chat,
        meta=meta,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/backrooms_corpus"))
    ap.add_argument("--out", type=Path, default=Path("data/backrooms_convs_2026-07-10.jsonl"))
    ap.add_argument("--min-approx-tokens", type=int, default=0,
                    help="skip runs below this source-token estimate")
    args = ap.parse_args()

    convs: list[Stage1Conv] = []
    skipped: list[str] = []
    for condition in ("chat", "prefill"):
        cond_dir = args.corpus / condition
        if not cond_dir.is_dir():
            print(f"WARNING: missing condition dir {cond_dir}")
            continue
        for run_dir in sorted(cond_dir.iterdir()):
            tp = run_dir / "transcript.json"
            if not tp.is_file():
                continue
            try:
                conv = convert_run(tp, condition)
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                print(f"SKIP {tp}: {type(e).__name__}: {e}")
                skipped.append(str(tp))
                continue
            if conv.approx_tokens < args.min_approx_tokens:
                skipped.append(f"{conv.conv_id} (short: {conv.approx_tokens})")
                continue
            convs.append(conv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for c in convs:
            f.write(c.model_dump_json() + "\n")

    print(f"\nwrote {len(convs)} conversations to {args.out} "
          f"({len(skipped)} skipped)")
    print(f"{'conv_id':<28} {'src_msgs':>8} {'user':>5} {'asst':>5} "
          f"{'~toks':>7} {'gen%':>5}")
    for c in convs:
        print(f"{c.conv_id:<28} {c.n_source_messages:>8} "
              f"{c.meta['n_user_msgs']:>5} {c.meta['n_assistant_msgs']:>5} "
              f"{c.approx_tokens:>7} "
              f"{(c.generated_pct or 0) * 100:>4.0f}%")


if __name__ == "__main__":
    main()
