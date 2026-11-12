"""Stand-in model service: a Groq/OpenAI-compatible fake with configurable, seeded behaviour.

Run:  python -m relay.standin_server --config configs/standin_default.json --port 8765
Config keys are documented in configs/standin_default.json (`_doc`).

Endpoints
  POST /openai/v1/chat/completions   streaming SSE or non-streaming JSON
  GET  /admin/state                  counters, in-flight, effective config
  POST /admin/reset                  reset counters/windows/RNG; optional overrides (see `admin_reset`)

Everything about latency, limits and failures is SYNTHETIC and driven by the config file.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import random
import re
import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "standin_default.json"
_TAG_RE = re.compile(r"\[[a-z_]+\s+p{1,2}\.\s*\d+\]", re.IGNORECASE)


# ---------------------------------------------------------------- config helpers
def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text())


def default_config(**overrides: Any) -> dict:
    """Default config file with (deep-merged) overrides, e.g. default_config(limits={"rpm": 3})."""
    return deep_merge(load_config(), overrides)


def _lognormal_params(median_ms: float, p95_ms: float) -> tuple[float, float]:
    mu = math.log(max(median_ms, 1e-6))
    sigma = math.log(p95_ms / median_ms) / 1.645 if p95_ms > median_ms else 0.0
    return mu, sigma


def _trim(x: float, nd: int) -> str:
    return f"{x:.{nd}f}".rstrip("0").rstrip(".")


def _fmt_dur(s: float) -> str:
    """Go-style duration as Groq prints it: '1.507s', '2s', '47.917s', '1m26.4s', '1h2m3.5s'."""
    s = max(s, 0.0)
    if s < 60:
        return f"{_trim(s, 3)}s"
    s = round(s, 3)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return (f"{int(h)}h" if h else "") + f"{int(m)}m" + f"{_trim(sec, 3)}s"


LIMIT_MODES = ("token_bucket", "sliding_window")


def est_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


# ---------------------------------------------------------------- state
class StandIn:
    def __init__(self, config: dict):
        self.config = copy.deepcopy(config)
        self.reset()

    # -- lifecycle
    def reset(self, *, reseed: bool = True) -> None:
        cfg = self.config
        if reseed:
            self.rng = random.Random(cfg.get("seed", 0))
        self.mode  # validate limits.mode early
        self.req_times: deque[float] = deque()
        self.tok_events: deque[tuple[float, int]] = deque()
        self.req_level = float(cfg["limits"]["rpm"])    # token-bucket levels (full at start)
        self.tok_level = float(cfg["limits"]["tpm"])
        self._bucket_t: Optional[float] = None          # last refill time; set on first use
        self.rpd_used = 0
        self.tpd_used = 0
        self.in_flight = 0
        self.max_in_flight_seen = 0
        self.n_requests = 0
        self.n_429 = 0
        self.n_5xx = 0
        self.n_ok = 0
        self.day_start = time.monotonic()
        self.bank = self._load_bank(cfg.get("replay_bank"))

    def _load_bank(self, path: Optional[str]) -> dict[tuple[str, str], dict]:
        bank: dict[tuple[str, str], dict] = {}
        if not path:
            return bank
        p = Path(path)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if not p.exists():
            return bank
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (str(row["question_id"]), str(row["setting"])[:1].upper())
            bank[key] = row
        return bank

    # -- limits
    @property
    def mode(self) -> str:
        mode = self.config["limits"].get("mode", "token_bucket")
        if mode not in LIMIT_MODES:
            raise ValueError(f"limits.mode must be one of {LIMIT_MODES}, got {mode!r}")
        return mode

    @property
    def bucket(self) -> bool:
        return self.mode == "token_bucket"

    def _refill(self, now: float) -> None:
        """Token-bucket mode: top both buckets up at capacity/60 per second (continuous, not windowed)."""
        lim = self.config["limits"]
        if self._bucket_t is None:
            self._bucket_t = now
        dt = max(0.0, now - self._bucket_t)
        self._bucket_t = max(self._bucket_t, now)
        self.req_level = min(float(lim["rpm"]), self.req_level + dt * lim["rpm"] / 60.0)
        self.tok_level = min(float(lim["tpm"]), self.tok_level + dt * lim["tpm"] / 60.0)

    def _prune(self, now: float) -> None:
        if self.bucket:
            self._refill(now)
            return
        while self.req_times and now - self.req_times[0] >= 60:
            self.req_times.popleft()
        while self.tok_events and now - self.tok_events[0][0] >= 60:
            self.tok_events.popleft()

    def tokens_used_minute(self, now: Optional[float] = None) -> int:
        """Sliding window: tokens reserved in the last 60 s. Token bucket: tpm - floor(level)."""
        if self.bucket:
            if now is not None:
                self._refill(now)
            return int(self.config["limits"]["tpm"] - math.floor(self.tok_level))
        return sum(n for _, n in self.tok_events)

    def requests_used_minute(self, now: Optional[float] = None) -> int:
        if self.bucket:
            if now is not None:
                self._refill(now)
            return int(self.config["limits"]["rpm"] - math.floor(self.req_level))
        return len(self.req_times)

    def rate_headers(self, now: float) -> dict[str, str]:
        lim = self.config["limits"]
        if self.bucket:
            self._refill(now)
            remaining_tok = max(0, math.floor(self.tok_level))
            reset_tok = (lim["tpm"] - self.tok_level) * 60.0 / lim["tpm"]   # time until the bucket is full
        else:
            self._prune(now)
            remaining_tok = max(0, lim["tpm"] - self.tokens_used_minute())
            reset_tok = (self.tok_events[0][0] + 60 - now) if self.tok_events else 0.0
        return {
            "x-ratelimit-limit-requests": str(lim["rpd"]),
            "x-ratelimit-remaining-requests": str(max(0, lim["rpd"] - self.rpd_used)),
            "x-ratelimit-reset-requests": _fmt_dur(self.config.get("day_reset_s", 86400) - (now - self.day_start)),
            "x-ratelimit-limit-tokens": str(lim["tpm"]),
            "x-ratelimit-remaining-tokens": str(remaining_tok),
            "x-ratelimit-reset-tokens": _fmt_dur(reset_tok),
        }

    def complete(self, now: float, max_tokens: int, completion_tokens: int) -> None:
        """Response finished: token-bucket mode refunds the unused part of the max_tokens reservation
        (capped at capacity), so the effective cost of a request is its actual usage."""
        if not self.bucket:
            return
        self._refill(now)
        cap = float(self.config["limits"]["tpm"])
        self.tok_level = min(cap, self.tok_level + max(0, max_tokens - completion_tokens))

    def check_limits(self, now: float, prompt_tokens: int, max_tokens: int):
        """Return None if admitted (and record usage) else (kind, used, limit, requested, retry_after_s)."""
        lim = self.config["limits"]
        requested = prompt_tokens + max_tokens
        day_left = self.config.get("day_reset_s", 86400) - (now - self.day_start)
        if self.bucket:
            self._refill(now)
            if self.rpd_used >= lim["rpd"]:
                return ("rpd", self.rpd_used, lim["rpd"], 1, day_left)
            if self.tpd_used + requested > lim["tpd"]:
                return ("tpd", self.tpd_used, lim["tpd"], requested, day_left)
            eps = 1e-9
            if self.req_level + eps < 1.0:
                wait = (1.0 - self.req_level) * 60.0 / lim["rpm"]
                return ("rpm", lim["rpm"] - math.floor(self.req_level), lim["rpm"], 1, wait)
            if self.tok_level + eps < requested:
                wait = (requested - self.tok_level) * 60.0 / lim["tpm"]
                return ("tpm", lim["tpm"] - math.floor(self.tok_level), lim["tpm"], requested, wait)
            self.req_level -= 1.0
            self.tok_level -= requested
            self.rpd_used += 1
            return None
        self._prune(now)
        if self.rpd_used >= lim["rpd"]:
            return ("rpd", self.rpd_used, lim["rpd"], 1, day_left)
        if self.tpd_used + requested > lim["tpd"]:
            return ("tpd", self.tpd_used, lim["tpd"], requested, day_left)
        if len(self.req_times) >= lim["rpm"]:
            idx = len(self.req_times) - lim["rpm"]
            return ("rpm", len(self.req_times), lim["rpm"], 1, self.req_times[idx] + 60 - now)
        used = self.tokens_used_minute()
        if used + requested > lim["tpm"]:
            need = used + requested - lim["tpm"]
            freed, wait = 0, 60.0
            for t, n in self.tok_events:
                freed += n
                if freed >= need:
                    wait = t + 60 - now
                    break
            return ("tpm", used, lim["tpm"], requested, wait)
        self.req_times.append(now)
        self.tok_events.append((now, requested))
        self.rpd_used += 1
        return None

    # -- latency
    def overload_factor(self) -> float:
        o = self.config["overload"]
        extra = max(0, self.in_flight - o["knee_in_flight"])
        return o["slowdown_per_extra"] ** extra

    def draw_lognormal_ms(self, key: str) -> float:
        c = self.config[key]
        mu, sigma = _lognormal_params(c["median"], c["p95"])
        return self.rng.lognormvariate(mu, sigma)

    # -- answers
    def answer_for(self, headers, prompt: str) -> tuple[str, int]:
        qid = headers.get("x-relay-question-id")
        setting = headers.get("x-relay-setting")
        if qid and setting:
            row = self.bank.get((qid, setting[:1].upper()))
            if row:
                text = row["text"]
                return text, int(row.get("tokens_out") or est_tokens(text))
        m = _TAG_RE.search(prompt)
        if m:
            text = f"According to the retrieved passage, the answer is supported {m.group(0)}."
        else:
            text = "NOT_IN_PAPERS: the provided passages do not contain this information."
        return text, est_tokens(text)


# ---------------------------------------------------------------- app
_LIMIT_NAMES = {
    "rpm": ("requests", "requests per minute (RPM)"),
    "rpd": ("requests", "requests per day (RPD)"),
    "tpm": ("tokens", "tokens per minute (TPM)"),
    "tpd": ("tokens", "tokens per day (TPD)"),
}


def _prompt_text(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        c = m.get("content", "")
        parts.append(c if isinstance(c, str) else json.dumps(c))
    return "\n".join(parts)


def create_app(config: dict) -> FastAPI:
    si = StandIn({k: v for k, v in config.items() if not k.startswith("_")})
    app = FastAPI(title="relay stand-in")
    app.state.standin = si
    counter = {"n": 0}

    def err(status: int, message: str, etype: str, code: str, headers: Optional[dict] = None):
        return JSONResponse(
            {"error": {"message": message, "type": etype, "code": code}}, status_code=status, headers=headers or {}
        )

    @app.post("/openai/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        now = time.monotonic()
        model = body.get("model") or si.config.get("model_name", "stand-in")
        messages = body.get("messages", [])
        stream = bool(body.get("stream"))
        max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or 1024)
        prompt = _prompt_text(messages)
        prompt_tokens = est_tokens(prompt)
        si.n_requests += 1
        counter["n"] += 1
        req_id = f"req_standin_{counter['n']}"

        cap = si.config.get("max_in_flight", 0)
        if cap and si.in_flight >= cap:
            si.n_5xx += 1
            return err(503, "stand-in over capacity (max_in_flight reached)", "server_error", "service_unavailable")

        denied = si.check_limits(now, prompt_tokens, max_tokens)
        if denied:
            kind, used, limit, requested, wait = denied
            si.n_429 += 1
            etype, label = _LIMIT_NAMES[kind]
            msg = (
                f"Rate limit reached for model `{model}` in organization `org_standin` service tier "
                f"`on_demand` on {label}: Limit {limit}, Used {used}, Requested {requested}. "
                f"Please try again in {_fmt_dur(wait)}. Need more tokens? Upgrade to Dev Tier today."
            )
            hdrs = si.rate_headers(now)
            hdrs["retry-after"] = str(max(1, math.ceil(wait)))
            return err(429, msg, etype, "rate_limit_exceeded", hdrs)
        hdrs = si.rate_headers(now)

        # All random draws at admission => reproducible given arrival order.
        f = si.config["failures"]
        u = si.rng.random()
        ttft_base = si.draw_lognormal_ms("ttft_ms")
        text, tokens_out = si.answer_for(request.headers, prompt)
        if tokens_out > max_tokens:
            tokens_out = max_tokens
            text = text[: max_tokens * 4]
            finish = "length"
        else:
            finish = "stop"
        chunk_chars = max(1, int(si.config.get("tokens_per_chunk", 4)) * 4)
        pieces = [text[i : i + chunk_chars] for i in range(0, len(text), chunk_chars)] or [""]
        gaps_base = [si.draw_lognormal_ms("inter_chunk_ms") for _ in pieces]

        p_5, p_hang, p_slow = f.get("p_5xx", 0), f.get("p_hang", 0), f.get("p_slow_success", 0)
        mult = 1.0
        if u < p_5:
            si.n_5xx += 1
            return err(500, "synthetic injected failure (stand-in)", "server_error", "internal_server_error", hdrs)
        hang = u < p_5 + p_hang
        if not hang and u < p_5 + p_hang + p_slow:
            mult = float(f.get("slow_multiplier", 5.0))

        prompt_s = si.config.get("prompt_ms_per_1k_tokens", 0) * prompt_tokens / 1000 / 1000
        created = int(time.time())

        def chunk(delta: dict, finish_reason=None, extra: Optional[dict] = None) -> str:
            obj = {
                "id": req_id, "object": "chat.completion.chunk", "created": created, "model": model,
                "system_fingerprint": "fp_standin",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
            if extra:
                obj.update(extra)
            return f"data: {json.dumps(obj)}\n\n"

        si.in_flight += 1
        si.max_in_flight_seen = max(si.max_in_flight_seen, si.in_flight)
        released = False

        def release():
            nonlocal released
            if not released:
                released = True
                si.in_flight -= 1

        async def sleep_s(base_s: float) -> float:
            """Sleep base_s scaled by the slow-success multiplier and the CURRENT overload factor."""
            s = base_s * mult * si.overload_factor()
            await asyncio.sleep(s)
            return s

        ttft_s = ttft_base / 1000 + prompt_s

        def usage(completion_s: float) -> dict:
            return {
                "prompt_tokens": prompt_tokens, "completion_tokens": tokens_out,
                "total_tokens": prompt_tokens + tokens_out,
                "queue_time": 0.0, "prompt_time": round(prompt_s, 6),
                "completion_time": round(completion_s, 6), "total_time": round(prompt_s + completion_s, 6),
            }

        async def run_stream():
            try:
                yield chunk({"role": "assistant", "content": ""})
                await sleep_s(ttft_s)
                comp_s = 0.0
                for i, piece in enumerate(pieces):
                    if i > 0:
                        comp_s += await sleep_s(gaps_base[i] / 1000)
                    if piece:
                        yield chunk({"content": piece})
                u_ = usage(comp_s)
                si.tpd_used += prompt_tokens + tokens_out
                si.complete(time.monotonic(), max_tokens, tokens_out)
                si.n_ok += 1
                yield chunk({}, finish, {"x_groq": {"id": req_id, "usage": u_}, "usage": u_})
                yield "data: [DONE]\n\n"
            finally:
                release()

        try:
            if hang:
                await asyncio.sleep(3600)
            if stream:
                return StreamingResponse(run_stream(), media_type="text/event-stream", headers=hdrs)
            # non-streaming
            await sleep_s(ttft_s)
            comp_s = 0.0
            for i in range(1, len(pieces)):
                comp_s += await sleep_s(gaps_base[i] / 1000)
            u_ = usage(comp_s)
            si.tpd_used += prompt_tokens + tokens_out
            si.complete(time.monotonic(), max_tokens, tokens_out)
            si.n_ok += 1
            payload = {
                "id": req_id, "object": "chat.completion", "created": created, "model": model,
                "system_fingerprint": "fp_standin",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
                "usage": u_, "x_groq": {"id": req_id},
            }
            return JSONResponse(payload, headers=hdrs)
        finally:
            if not stream or hang:
                release()

    @app.get("/admin/state")
    async def admin_state():
        now = time.monotonic()
        si._prune(now)
        return {
            "limit_mode": si.mode,
            "in_flight": si.in_flight, "max_in_flight_seen": si.max_in_flight_seen,
            "n_requests": si.n_requests, "n_ok": si.n_ok, "n_429": si.n_429, "n_5xx": si.n_5xx,
            "rpm_used": si.requests_used_minute(), "tpm_used": si.tokens_used_minute(),
            "rpm_level": si.req_level, "tpm_level": si.tok_level,
            "rpd_used": si.rpd_used, "tpd_used": si.tpd_used,
            "bank_size": len(si.bank), "config": si.config,
        }

    @app.post("/admin/reset")
    async def admin_reset(request: Request):
        """Body (all optional): {"config": {partial overrides, deep-merged},
        "scale_limits": {"rpm": 0.5, "tpm": 0.5}, "rpd_used": int, "tpd_used": int}.
        Counters, windows and the RNG (re-seeded from config.seed) are always reset first."""
        try:
            body = await request.json()
        except Exception:
            body = {}
        if body.get("config"):
            si.config = deep_merge(si.config, body["config"])
        for k, factor in (body.get("scale_limits") or {}).items():
            si.config["limits"][k] = max(1, int(si.config["limits"][k] * factor))
        si.reset()
        if "rpd_used" in body:
            si.rpd_used = int(body["rpd_used"])
        if "tpd_used" in body:
            si.tpd_used = int(body["tpd_used"])
        return {"ok": True, "limits": si.config["limits"], "rpd_used": si.rpd_used, "tpd_used": si.tpd_used}

    return app


# ---------------------------------------------------------------- in-process server helper
class StandInThread:
    """Run the stand-in under uvicorn in a background thread (for tests and scripts)."""

    def __init__(self, config: dict, port: int = 0, host: str = "127.0.0.1"):
        import socket

        import uvicorn

        if port == 0:
            with socket.socket() as s:
                s.bind((host, 0))
                port = s.getsockname()[1]
        self.app = create_app(config)
        self.state: StandIn = self.app.state.standin
        self.url = f"http://{host}:{port}"
        self._server = uvicorn.Server(
            uvicorn.Config(self.app, host=host, port=port, log_level="warning", timeout_graceful_shutdown=1)
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def start(self) -> "StandInThread":
        self._thread.start()
        t0 = time.monotonic()
        while not self._server.started:
            if time.monotonic() - t0 > 10:
                raise RuntimeError("stand-in server failed to start")
            time.sleep(0.01)
        return self

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)


@contextmanager
def running_standin(config: dict, port: int = 0) -> Iterator[StandInThread]:
    srv = StandInThread(config, port).start()
    try:
        yield srv
    finally:
        srv.stop()


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    uvicorn.run(create_app(load_config(a.config)), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
