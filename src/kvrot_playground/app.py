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
