"""Stdlib HTTP client for the vLLM completions API (token ids in / out).

The exp11/exp12 idiom: prompts as token id lists, ``return_token_ids`` for a
retokenization-free ledger, per-request ``kv_transfer_params`` carrying the
kvrot session id + eviction plan. Synchronous on purpose — the FastAPI app
calls it via a worker thread, and one in-flight request per session is the
concurrency model.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class VllmClientError(RuntimeError):
    pass


@dataclass
class CompletionResult:
    text: str
    token_ids: list[int]
    finish_reason: str | None
    kvrot_echo: dict[str, Any]
    prompt_tokens: int
    gen_tokens: int
    wall_s: float


class VllmClient:
    def __init__(self, base_url: str, *, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._model: str | None = None

    def _get(self, route: str) -> dict:
        req = urllib.request.Request(f"{self.base_url}{route}")
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())

    def served_model_id(self) -> str:
        if self._model is None:
            data = self._get("/v1/models")
            self._model = data["data"][0]["id"]
        return self._model

    def healthy(self) -> bool:
        try:
            self.served_model_id()
            return True
        except Exception:
            return False

    def complete(
        self,
        prompt_ids: list[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_token_ids: list[int],
        seed: int | None = None,
        kvrot: dict[str, Any] | None = None,
    ) -> CompletionResult:
        payload: dict[str, Any] = {
            "model": self.served_model_id(),
            "prompt": prompt_ids,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop_token_ids": stop_token_ids,
            "return_token_ids": True,
        }
        if seed is not None:
            payload["seed"] = seed
        if kvrot is not None:
            payload["kv_transfer_params"] = {"kvrot": kvrot}

        body = json.dumps(payload).encode()
        t0 = time.monotonic()
        last_err: Exception | None = None
        for attempt in range(3):
            req = urllib.request.Request(
                f"{self.base_url}/v1/completions",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.loads(r.read().decode())
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:1000]
                raise VllmClientError(f"HTTP {e.code}: {detail}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        else:
            raise VllmClientError(f"vLLM server unreachable: {last_err}")

        wall = time.monotonic() - t0
        choice = resp["choices"][0]
        token_ids = choice.get("token_ids")
        if token_ids is None:
            raise VllmClientError("server did not return token_ids (return_token_ids unsupported?)")
        usage = resp.get("usage") or {}
        echo = (resp.get("kv_transfer_params") or choice.get("kv_transfer_params") or {})
        return CompletionResult(
            text=choice.get("text", ""),
            token_ids=[int(t) for t in token_ids],
            finish_reason=choice.get("finish_reason"),
            kvrot_echo=echo.get("kvrot") or {},
            prompt_tokens=int(usage.get("prompt_tokens", len(prompt_ids))),
            gen_tokens=int(usage.get("completion_tokens", len(token_ids))),
            wall_s=wall,
        )
