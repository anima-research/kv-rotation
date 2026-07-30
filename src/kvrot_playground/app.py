"""FastAPI playground app: chat with a model whose KV cache is live-rotated.

Environment:
    KVROT_VLLM_URL     vLLM server base URL (default http://localhost:8013)
    KVROT_MODEL_PATH   tokenizer path (required — same checkpoint the server runs)
    KVROT_BOT_NAME     display/prefill name (default: Trinity)
    KVROT_BANK_DIR     session jsonl bank dir (default: runs/playground)
    KVROT_TOKEN        optional shared secret; when set, every request must
                       carry it (?token= or X-Kvrot-Token) — the public-access
                       gate, off by default for local use
    KVROT_MAX_SESSIONS server-side concurrent session cap (default 6 — keep in
                       sync with the connector's kvrot_max_sessions)

Run:  uvicorn kvrot_playground.app:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from kvrot_playground.client import VllmClient, VllmClientError
from kvrot_playground.session import PlaygroundConfig, Session, TurnStats

VLLM_URL = os.environ.get("KVROT_VLLM_URL", "http://localhost:8013")
MODEL_PATH = os.environ.get("KVROT_MODEL_PATH")
BOT_NAME = os.environ.get("KVROT_BOT_NAME", "Trinity")
BANK_DIR = Path(os.environ.get("KVROT_BANK_DIR", "runs/playground"))
TOKEN = os.environ.get("KVROT_TOKEN")
MAX_SESSIONS = int(os.environ.get("KVROT_MAX_SESSIONS", "6"))
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="kvrot playground", docs_url=None, redoc_url=None)

_tokenizer = None
_client = VllmClient(VLLM_URL)
_sessions: dict[str, Session] = {}
_locks: dict[str, asyncio.Lock] = {}
_last_used: dict[str, float] = {}


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        if not MODEL_PATH:
            raise HTTPException(500, "KVROT_MODEL_PATH is not set")
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return _tokenizer


@app.middleware("http")
async def _token_gate(request: Request, call_next):
    if TOKEN and not request.url.path.startswith("/static"):
        supplied = request.query_params.get("token") or request.headers.get("X-Kvrot-Token")
        if supplied != TOKEN:
            return JSONResponse({"error": "missing or bad token"}, status_code=401)
    return await call_next(request)


class NewSessionBody(BaseModel):
    config: PlaygroundConfig = Field(default_factory=PlaygroundConfig)
    preamble: str | None = None
    user_name: str = "User"


class TurnBody(BaseModel):
    text: str = Field(min_length=1, max_length=8000)


class ConfigBody(BaseModel):
    policy: str | None = None
    budget: int | None = None
    num_sink_tokens: int | None = None
    max_reply_tokens: int | None = None
    temperature: float | None = None


def _session(sid: str) -> Session:
    s = _sessions.get(sid)
    if s is None:
        raise HTTPException(404, f"no session {sid}")
    _last_used[sid] = time.monotonic()
    return s


def _evict_lru_sessions() -> None:
    while len(_sessions) >= MAX_SESSIONS:
        victim = min(_last_used, key=_last_used.get)
        _sessions.pop(victim, None)
        _locks.pop(victim, None)
        _last_used.pop(victim, None)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health():
    ok = await asyncio.to_thread(_client.healthy)
    return {
        "vllm": "up" if ok else "down",
        "vllm_url": VLLM_URL,
        "model": _client.served_model_id() if ok else None,
        "bot_name": BOT_NAME,
        "sessions": len(_sessions),
        "max_sessions": MAX_SESSIONS,
    }


@app.post("/api/sessions")
async def new_session(body: NewSessionBody):
    _evict_lru_sessions()
    tok = _get_tokenizer()
    s = Session(
        tok,
        bot_name=BOT_NAME,
        user_name=body.user_name,
        config=body.config,
        preamble=body.preamble,
        bank_path=None,  # set after id is known
    )
    s.bank_path = BANK_DIR / f"{s.session_id}.jsonl"
    _sessions[s.session_id] = s
    _locks[s.session_id] = asyncio.Lock()
    _last_used[s.session_id] = time.monotonic()
    return s.view()


@app.get("/api/sessions/{sid}")
async def session_state(sid: str):
    return _session(sid).view()


@app.post("/api/sessions/{sid}/config")
async def update_config(sid: str, body: ConfigBody):
    s = _session(sid)
    patch = {k: v for k, v in body.model_dump().items() if v is not None}
    s.config = s.config.model_copy(update=patch)
    s._bank({"type": "config", **s.config.model_dump()})
    return s.view()


@app.post("/api/sessions/{sid}/turns")
async def send_turn(sid: str, body: TurnBody):
    s = _session(sid)
    lock = _locks[sid]
    if lock.locked():
        raise HTTPException(409, "a turn is already in flight for this session")
    async with lock:
        req = s.build_request(body.text)
        try:
            result = await asyncio.to_thread(
                _client.complete,
                req["prompt_ids"],
                max_tokens=req["max_tokens"],
                temperature=req["temperature"],
                top_p=req["top_p"],
                stop_token_ids=req["stop_token_ids"],
                kvrot=req["kvrot"],
            )
        except VllmClientError as e:
            raise HTTPException(502, f"vLLM request failed: {e}") from e

        s.add_model_turn(result.token_ids, result.text, req["prefill_tail_ids"])
        s.mark_synced(len(req["prompt_ids"]))
        ev = req["event"]
        stats = TurnStats(
            wall_s=round(result.wall_s, 3),
            claimed_tokens=int(result.kvrot_echo.get("claimed_tokens") or 0),
            prompt_tokens=result.prompt_tokens,
            gen_tokens=result.gen_tokens,
            evicted_turns=len(ev.turn_indices) if ev else 0,
            evicted_tokens=ev.evicted_tokens if ev else 0,
            store_tokens=int(result.kvrot_echo.get("stored_tokens") or 0),
        )
        return {"reply": result.text.strip(), "stats": stats.__dict__, "state": s.view()}


@app.post("/api/sessions/{sid}/fork")
async def fork_session(sid: str):
    s = _session(sid)
    if _locks[sid].locked():
        raise HTTPException(409, "a turn is in flight")
    _evict_lru_sessions()
    f = s.clone()
    _sessions[f.session_id] = f
    _locks[f.session_id] = asyncio.Lock()
    _last_used[f.session_id] = time.monotonic()
    return f.view()


@app.post("/api/sessions/{sid}/reroll")
async def reroll(sid: str):
    s = _session(sid)
    lock = _locks[sid]
    if lock.locked():
        raise HTTPException(409, "a turn is already in flight")
    async with lock:
        import random as _random

        s.pop_model_turn()
        req = s.build_reroll_request()
        try:
            result = await asyncio.to_thread(
                _client.complete, req["prompt_ids"],
                max_tokens=req["max_tokens"], temperature=req["temperature"],
                top_p=req["top_p"], stop_token_ids=req["stop_token_ids"],
                seed=_random.randint(0, 2**31), kvrot=req["kvrot"],
            )
        except VllmClientError as e:
            raise HTTPException(502, f"vLLM request failed: {e}") from e
        s.add_model_turn(result.token_ids, result.text, req["prefill_tail_ids"])
        s.mark_synced(len(req["prompt_ids"]))
        stats = TurnStats(
            wall_s=round(result.wall_s, 3),
            claimed_tokens=int(result.kvrot_echo.get("claimed_tokens") or 0),
            prompt_tokens=result.prompt_tokens, gen_tokens=result.gen_tokens,
            store_tokens=int(result.kvrot_echo.get("stored_tokens") or 0),
        )
        return {"reply": result.text.strip(), "stats": stats.__dict__, "state": s.view()}


@app.get("/api/sessions/{sid}/export")
async def export_session(sid: str):
    s = _session(sid)
    return JSONResponse(
        s.export_dict(),
        headers={"Content-Disposition": f"attachment; filename=kvrot-{sid}.json"},
    )


class ImportBody(BaseModel):
    export: dict


@app.post("/api/sessions/import")
async def import_session(body: ImportBody):
    data = body.export
    if data.get("kvrot_playground_export") != 1:
        raise HTTPException(400, "not a kvrot playground export")
    _evict_lru_sessions()
    tok = _get_tokenizer()
    from kvrot_playground.session import PlaygroundConfig as PC

    turns = [t for t in data.get("turns", []) if not t.get("evicted")]
    preamble = turns[0]["text"] if turns and turns[0]["role"] == "system" else None
    s = Session(
        tok, bot_name=BOT_NAME, user_name=data.get("user_name", "User"),
        config=PC(**data.get("config", {})), preamble=preamble,
    )
    s.bank_path = BANK_DIR / f"{s.session_id}.jsonl"
    for t in turns[1:]:
        if t["role"] == "user":
            s.add_user_turn(t["text"])
        elif t["role"] == "model":
            s.add_model_turn(s._encode(t["text"]), t["text"], s.reply_prefill_ids())
    _sessions[s.session_id] = s
    _locks[s.session_id] = asyncio.Lock()
    _last_used[s.session_id] = time.monotonic()
    return s.view()


class SeedBody(BaseModel):
    template: str = "archive"
    doc_index: int | None = None
    target_tokens: int = Field(19000, ge=500, le=28000)
    depths: list[float] = Field(default_factory=lambda: [0.05, 0.35, 0.65, 0.9])
    rng_seed: int = 1234


@app.get("/api/seed_options")
async def seed_options():
    from kvrot_playground.seeding import TEMPLATES, load_docs

    docs = load_docs(os.environ.get("KVROT_DATA", "data/eval_docs.jsonl"))
    return {
        "templates": [{"id": k, "label": v["label"]} for k, v in TEMPLATES.items()],
        "docs": [
            {k: d[k] for k in ("index", "approx_tokens", "preview")} for d in docs
        ],
    }


@app.post("/api/sessions/{sid}/seed")
async def seed_session(sid: str, body: SeedBody):
    """Seed a (fresh) session: preamble + document chunks as REAL chat turns
    with planted needles. Turns are evictable — early needles honestly
    disappear as the session rolls."""
    from kvrot_playground.seeding import build_seed, load_docs, seed_preamble

    s = _session(sid)
    if _locks[sid].locked():
        raise HTTPException(409, "a turn is in flight")
    if len(s.turns) > 1:
        raise HTTPException(409, "seed only into a fresh session (use New session)")
    docs = load_docs(os.environ.get("KVROT_DATA", "data/eval_docs.jsonl"))
    texts = [d["text"] for d in docs]
    idx = body.doc_index if body.doc_index is not None else max(
        range(len(texts)), key=lambda i: len(texts[i])
    )
    # replace the default system turn's text is not possible post-hoc; fresh
    # sessions are cheap — rebuild with the template preamble
    tok = _get_tokenizer()
    fresh = Session(
        tok, bot_name=BOT_NAME, user_name=s.user_name, config=s.config,
        preamble=seed_preamble(body.template, BOT_NAME),
        bank_path=s.bank_path, session_id=s.session_id,
    )
    try:
        needles = build_seed(
            fresh, template=body.template, doc_text=texts[idx],
            target_tokens=body.target_tokens, depths=body.depths,
            rng_seed=body.rng_seed,
        )
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e)) from e
    _sessions[sid] = fresh
    return {"needles": needles, "state": fresh.view()}


@app.post("/api/sessions/{sid}/evict")
async def force_evict(sid: str):
    """Force one eviction round now (as if the budget were breached). The
    resulting plan is composed into the session's pending store-plan and
    ships automatically with the next turn."""
    s = _session(sid)
    if _locks[sid].locked():
        raise HTTPException(409, "a turn is in flight")
    event = s.plan_eviction_if_needed(incoming_tokens=s.config.budget)  # force breach
    if event is None:
        return {"evicted": False, "state": s.view()}
    return {"evicted": True, "recompute": event.recompute,
            "evicted_tokens": event.evicted_tokens, "state": s.view()}


def _mount_static() -> None:
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


_mount_static()
