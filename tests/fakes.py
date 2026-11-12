"""Test doubles: scripted provider, canned passages, fake analyze/search/verify."""
from __future__ import annotations

import asyncio
import dataclasses
import time
from typing import Optional

from relay.types import Analysis, Hit, Passage, ProviderResult, VerifyResult


def make_hit(paper: str = "attention", page: int = 3, score: float = 0.9,
             text: str = "The Transformer reaches 28.4 BLEU on WMT 2014 English-to-German.",
             section: Optional[str] = "Results") -> Hit:
    p = Passage(chunk_id=f"{paper}:B:p{page}:c0", paper_id=paper, arxiv_id="0000.00000", version="v1",
                page=page, section=section, text=text, n_words=len(text.split()))
    return Hit(p, score)


def ok_result(text: str = "It reaches 28.4 BLEU [attention p.3].", tokens_in: int = 500,
              tokens_out: int = 60) -> ProviderResult:
    return ProviderResult(status="ok", text=text, http_status=200, tokens_in=tokens_in,
                          tokens_out=tokens_out)


def limited(retry_after_s: Optional[float] = 0.05, kind: Optional[str] = "tpm") -> ProviderResult:
    return ProviderResult(status="rate_limited", http_status=429, retry_after_s=retry_after_s,
                          limit_kind=kind)


class FakeProvider:
    """Replays ``script`` (last item repeats). ``delay_s`` = total time of each call;
    ``ttft_s`` = delay before the first token for ok results."""
    source = "stand-in"

    def __init__(self, script: list[ProviderResult], delay_s: float = 0.0, ttft_s: float = 0.0):
        self.script = list(script)
        self.delay_s = delay_s
        self.ttft_s = ttft_s
        self.calls: list[dict] = []

    async def complete(self, messages, *, max_tokens, timeout_s, on_token=None, relay_headers=None):
        self.calls.append({"messages": messages, "headers": relay_headers, "timeout_s": timeout_s})
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        res = dataclasses.replace(item)
        res.t_send = time.monotonic()
        if res.status == "ok":
            await asyncio.sleep(self.ttft_s)
            res.t_first_token = time.monotonic()
            if on_token:
                on_token(res.text.split(" ")[0])
            await asyncio.sleep(max(0.0, self.delay_s - self.ttft_s))
        else:
            await asyncio.sleep(self.delay_s)
        res.t_end = time.monotonic()
        return res


def fake_analyze(question: str) -> Analysis:
    return Analysis(normalized=question.lower(), papers=("attention",), out_of_scope=False,
                    keywords=())


class FakeSearch:
    """search_fn stand-in; returns ``first`` for the first call and ``second`` for later ones."""

    def __init__(self, first: list[Hit], second: Optional[list[Hit]] = None):
        self.first, self.second = first, second if second is not None else first
        self.calls: list[dict] = []

    def __call__(self, query, setting_name, papers=None, k=None):
        self.calls.append({"query": query, "setting": setting_name, "papers": papers, "k": k})
        return self.first if len(self.calls) == 1 else self.second


def fake_verify_ok(answer, hits) -> VerifyResult:
    if answer.startswith("NOT_IN_PAPERS:"):
        return VerifyResult(True, True, (), ())
    return VerifyResult(True, False, (("attention", 3),), ())
