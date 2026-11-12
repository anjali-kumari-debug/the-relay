"""GroqProvider: one streaming chat-completion attempt against Groq (or the stand-in), no retries."""
from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Any, Optional

import groq

from relay.types import ProviderResult, Source, TokenCallback

_TIMING_KEYS = ("queue_time", "prompt_time", "completion_time", "total_time")
_KEEP_HEADER_PREFIXES = ("x-ratelimit-", "x-groq-")
_LIMIT_PATTERNS = (  # (regex, kind) -- phrases and abbreviations
    (re.compile(r"requests per minute|\bRPM\b", re.I), "rpm"),
    (re.compile(r"tokens per minute|\bTPM\b", re.I), "tpm"),
    (re.compile(r"requests per day|\bRPD\b", re.I), "rpd"),
    (re.compile(r"tokens per day|\bTPD\b", re.I), "tpd"),
)
_RETRY_IN_RE = re.compile(
    r"try again in\s+(?:(\d+(?:\.\d+)?)h)?\s*(?:(\d+(?:\.\d+)?)m(?!s))?\s*(?:(\d+(?:\.\d+)?)s)?\s*(?:(\d+(?:\.\d+)?)ms)?",
    re.I,
)


def parse_limit_kind(message: str) -> Optional[str]:
    """'rpm'|'tpm'|'rpd'|'tpd' from a Groq 429 message, else None."""
    for rx, kind in _LIMIT_PATTERNS:
        if rx.search(message or ""):
            return kind
    return None


def parse_retry_in_message(message: str) -> Optional[float]:
    """Seconds from 'Please try again in 7m12.5s' / '1.234s' / '250ms'; None if absent."""
    m = _RETRY_IN_RE.search(message or "")
    if not m or not any(m.groups()):
        return None
    h, mi, s, ms = (float(g) if g else 0.0 for g in m.groups())
    return h * 3600 + mi * 60 + s + ms / 1000


def _keep_headers(headers: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in dict(headers).items():
        lk = k.lower()
        if lk == "retry-after" or lk.startswith(_KEEP_HEADER_PREFIXES):
            out[lk] = v
    return out


def _error_message(exc: groq.APIError) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
    return str(getattr(exc, "message", None) or exc)


def _get(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _usage_of(chunk: Any) -> Any:
    """Usage from chunk.usage (OpenAI style) or chunk.x_groq.usage (Groq style)."""
    return _get(chunk, "usage") or _get(_get(chunk, "x_groq"), "usage")


class GroqProvider:
    def __init__(
        self,
        *,
        source: Source,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        http_client: Any = None,
    ):
        self.source: Source = source
        self.model = model
        kwargs: dict[str, Any] = {"max_retries": 0}  # the single retry layer lives in the pipeline
        if base_url:
            kwargs["base_url"] = base_url
        if http_client is not None:
            kwargs["http_client"] = http_client
        self._client = groq.AsyncGroq(api_key=api_key or os.environ.get("GROQ_API_KEY"), **kwargs)

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        timeout_s: float,
        on_token: Optional[TokenCallback] = None,
        relay_headers: Optional[dict[str, str]] = None,
    ) -> ProviderResult:
        res = ProviderResult(status="error", t_send=time.monotonic())
        try:
            await asyncio.wait_for(self._run(res, messages, max_tokens, timeout_s, on_token, relay_headers), timeout_s)
        except (asyncio.TimeoutError, groq.APITimeoutError):
            res.status = "timeout"
            res.error_message = f"timeout after {timeout_s}s"
        except groq.RateLimitError as e:
            self._fill_status_error(res, e, "rate_limited")
        except groq.APIStatusError as e:
            self._fill_status_error(res, e, "error")
        except groq.APIConnectionError as e:
            res.status, res.error_message = "error", f"connection error: {e}"
        except Exception as e:  # mid-stream protocol errors etc.
            res.status, res.error_message = "error", f"{type(e).__name__}: {e}"
        res.t_end = time.monotonic()
        if res.status != "ok" and res.text and not res.tokens_out:
            self._estimate_usage(res, messages)  # partial stream: tokens were still spent
        return res

    # -- internals
    async def _run(self, res, messages, max_tokens, timeout_s, on_token, relay_headers) -> None:
        extra: dict[str, Any] = {}
        if "gpt-oss" in self.model:
            extra.update(reasoning_effort="low", include_reasoning=False)
        raw = await self._client.chat.completions.with_raw_response.create(
            model=self.model,
            messages=messages,
            stream=True,
            max_completion_tokens=max_tokens,
            temperature=0.2,
            extra_headers=dict(relay_headers) if relay_headers else None,
            timeout=timeout_s,
            **extra,
        )
        res.headers = _keep_headers(raw.headers)
        stream = await raw.parse()
        parts: list[str] = []
        usage = None
        x_id = None
        stream_err = None
        try:
            async for chunk in stream:
                usage = _usage_of(chunk) or usage
                xg = _get(chunk, "x_groq")
                x_id = _get(xg, "id") or x_id
                xerr = _get(xg, "error")
                if xerr:
                    stream_err = str(xerr)
                for choice in _get(chunk, "choices") or []:
                    text = _get(_get(choice, "delta"), "content")
                    if text:  # skip role-only / reasoning-only chunks
                        if res.t_first_token is None:
                            res.t_first_token = time.monotonic()
                        parts.append(text)
                        res.text = "".join(parts)
                        if on_token:
                            on_token(text)
        finally:
            res.text = "".join(parts)
            await stream.close()
        res.status = "error" if stream_err else "ok"
        res.error_message = stream_err
        self._apply_usage(res, usage, x_id, messages)

    def _apply_usage(self, res: ProviderResult, usage: Any, x_id: Optional[str], messages) -> None:
        if x_id:
            res.provider_timing["x_groq_id"] = x_id
        if usage is None:
            self._estimate_usage(res, messages)
            return
        res.tokens_in = int(_get(usage, "prompt_tokens") or 0)
        res.tokens_out = int(_get(usage, "completion_tokens") or 0)
        res.reasoning_tokens = int(_get(_get(usage, "completion_tokens_details"), "reasoning_tokens") or 0)
        for k in _TIMING_KEYS:
            v = _get(usage, k)
            if v is not None:
                res.provider_timing[k] = v

    @staticmethod
    def _estimate_usage(res: ProviderResult, messages) -> None:
        res.tokens_in = max(1, sum(len(m.get("content", "")) for m in messages) // 4)
        res.tokens_out = max(1, len(res.text) // 4) if res.text else 0
        res.provider_timing["usage_estimated"] = True

    @staticmethod
    def _fill_status_error(res: ProviderResult, e: groq.APIStatusError, status) -> None:
        res.status = status
        res.http_status = e.status_code
        res.error_message = _error_message(e)
        res.headers = _keep_headers(e.response.headers)
        if status == "rate_limited":
            res.limit_kind = parse_limit_kind(res.error_message)
            ra = res.headers.get("retry-after")
            try:
                res.retry_after_s = float(ra) if ra is not None else None
            except ValueError:
                res.retry_after_s = None
            if res.retry_after_s is None:
                res.retry_after_s = parse_retry_in_message(res.error_message)
