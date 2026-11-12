"""Per-request pipeline (SPEC "Pipeline"): admission -> analyze -> retrieve -> evidence gate ->
draft (one model call) -> verify -> optional single revise -> release.

Dependencies (provider, admission, retry policy, recorder) are injected. ``analyze_fn``,
``search_fn``, ``verify_fn`` default to ``relay.analyze.analyze``, ``relay.retrieval.search_papers``
and ``relay.verify.verify`` (resolved lazily) and can be replaced in tests.

``handle_request`` returns a ``RequestOutcome``:
    row         the ResultRow (already written via recorder.write_row)
    answer      text shown to the user (answer, fixed refusal, or fixed rejection message)
    hits        passages the answer was drawn from (final retrieval result)
    raw_text    the model's final raw text ("" if no model answer) - used for the replay bank
    model_tokens_out  completion tokens of the last successful model call (0 if none)
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from relay.admission import AdmissionController, Rejection, Ticket
from relay.metrics import Recorder
from relay.retry import RetryPolicy
from relay.types import (REFUSAL_PREFIX, SETTINGS, Analysis, Hit, ModelProvider, ProviderResult,
                         ResultRow, VerifyResult, setting_label)

DEGRADE_HEADROOM = 0.3        # below this TPM headroom, fall back to setting A
REVISE_MIN_HEADROOM = 0.2     # a revise call is a full extra call; only with this much TPM room
REVISE_MIN_PATIENCE_S = 3.0   # ... and this much patience left
TIMEOUT_BACKSTOP_S = 0.25     # asyncio.wait_for slack on top of the provider's own timeout

SYSTEM_PROMPT = (
    "You answer questions about three research papers using ONLY the passages provided. "
    "Cite every claim as [paper_id p.N] using exactly the tags shown on the passages. "
    "If the passages answer only part of the question, answer that part with citations and say "
    "plainly which part the passages do not state; never fill the gap from memory. "
    "If the passages contain none of the answer, reply starting exactly with "
    f"{REFUSAL_PREFIX} followed by one sentence saying so. Be concise: at most 150 words."
)

REFUSAL_REPLY = ("I can't answer that from the three papers I have (Attention Is All You Need, "
                 "Moshi, H-Net). I haven't guessed an answer.")
DAILY_EXHAUSTED_REPLY = (
    "I can't answer right now: today's usage allowance for the language-model service has been "
    "used up. Daily allowances reset once a day, so please try again after the reset. "
    "I have not guessed an answer.")
QUEUE_FULL_REPLY = ("I'm handling too many questions at the moment and couldn't take yours. "
                    "Please try again in a minute. I have not guessed an answer.")
GAVE_UP_REPLY = ("This is taking longer than the time you allowed, so I stopped waiting. "
                 "No answer was produced; please try again.")
PROVIDER_FAILED_REPLY = ("The language-model service didn't respond successfully. "
                         "No answer was produced; please try again.")


@dataclass
class RequestOutcome:
    row: ResultRow
    answer: str
    hits: list = field(default_factory=list)
    raw_text: str = ""
    model_tokens_out: int = 0


def format_passage(h: Hit) -> str:
    sec = f" (section: {h.passage.section})" if h.passage.section else ""
    return f"{h.passage.cite()}{sec}\n{h.passage.text}"


def build_messages(question: str, hits: list[Hit]) -> list[dict[str, str]]:
    passages = "\n\n".join(format_passage(h) for h in hits)
    user = f"Passages:\n\n{passages}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def estimate_tokens(messages: list[dict[str, str]], max_tokens: int) -> int:
    """Rough admission estimate: ~4 characters per token of prompt, plus the completion cap."""
    return sum(len(m["content"]) for m in messages) // 4 + max_tokens


def estimate_tokens_for_setting(setting_name: str, max_tokens: int) -> int:
    """Before retrieval: keep_k passages of ~piece_words words (~1.4 tokens/word) + prompt overhead."""
    s = SETTINGS[setting_name]
    return int(s.keep_k * s.piece_words * 1.4) + 300 + max_tokens


def _resolve(name: str):
    import importlib
    mod, attr = name.split(":")
    return getattr(importlib.import_module(mod), attr)


class _Run:
    """State of one request. Not public; use handle_request."""

    def __init__(self, **kw: Any):
        self.__dict__.update(kw)
        self.t0 = time.monotonic()
        self.start_time = datetime.now(timezone.utc).isoformat()
        self.deadline = self.t0 + self.patience_s
        self.request_id = uuid.uuid4().hex[:12]
        self.tokens_in = self.tokens_out = self.retries = 0
        self.wait_s = self.retry_wait_s = 0.0
        self.first_token_t: Optional[float] = None
        self.provider_ttft_ms: Optional[float] = None
        self.queue_len = self.admission.queue_len

    # ---- helpers
    def now_ms(self) -> float:
        return (time.monotonic() - self.t0) * 1000.0

    def ev(self, name: str, **f: Any) -> None:
        self.recorder.event(self.request_id, name, **f)

    def finish(self, status: str, detail: str, answer: str, *, hits=None, raw_text: str = "",
               model_tokens_out: int = 0) -> RequestOutcome:
        total = self.now_ms()
        wait, retry_wait = self.wait_s * 1000.0, self.retry_wait_s * 1000.0
        row = ResultRow(
            request_id=self.request_id, scenario=self.scenario, source=self.provider.source,
            start_time=self.start_time,
            first_token_ms=None if self.first_token_t is None else (self.first_token_t - self.t0) * 1000.0,
            total_ms=total, status=status, retries=self.retries,
            tokens_in=self.tokens_in, tokens_out=self.tokens_out,
            retrieval_setting=setting_label(SETTINGS[self.setting_used]),
            question_id=self.question_id, queue_len_at_arrival=self.queue_len,
            wait_ms=wait, work_ms=max(0.0, total - wait - retry_wait), retry_wait_ms=retry_wait,
            provider_ttft_ms=self.provider_ttft_ms,
            released_ms=total if status == "ok" else None, outcome_detail=detail)
        self.ev("done", status=status, outcome=detail, total_ms=round(total, 1))
        self.recorder.write_row(row)
        return RequestOutcome(row, answer, hits or [], raw_text, model_tokens_out)

    def reject(self, rej: Rejection, hits=None) -> RequestOutcome:
        self.ev("rejected", reason=rej.reason)
        if rej.reason == "daily_budget":
            return self.finish("rate_limited", "rejected_daily_budget", DAILY_EXHAUSTED_REPLY, hits=hits)
        if rej.reason == "queue_full":
            return self.finish("rate_limited", "rejected_queue_full", QUEUE_FULL_REPLY, hits=hits)
        return self.finish("timeout", "gave_up_in_queue", GAVE_UP_REPLY, hits=hits)

    async def acquire(self, est: int) -> Ticket | Rejection:
        self.ev("queue_enter", est_tokens=est, queue_len=self.admission.queue_len)
        t = time.monotonic()
        res = await self.admission.acquire(est, self.deadline)
        self.wait_s += time.monotonic() - t
        self.ev("queue_exit", admitted=isinstance(res, Ticket), queued_ms=round((time.monotonic() - t) * 1000, 1))
        return res

    async def call_model(self, messages: list[dict[str, str]], ticket: Optional[Ticket]
                         ) -> tuple[Optional[ProviderResult], Optional[Rejection]]:
        """One logical model call = attempts + retry waits. Returns (last result, rejection)."""
        attempt = 0
        headers = {"X-Relay-Question-Id": self.question_id, "X-Relay-Setting": self.setting_used}
        while True:
            if ticket is None:
                got = await self.acquire(estimate_tokens(messages, self.max_tokens))
                if isinstance(got, Rejection):
                    return None, got
                ticket = got
            remaining = self.deadline - time.monotonic()
            produced = {"n": 0}

            def on_token(_t: str, produced=produced) -> None:
                if produced["n"] == 0:
                    now = time.monotonic()
                    if self.first_token_t is None:
                        self.first_token_t = now
                        self.ev("first_token")
                produced["n"] += 1

            self.ev("provider_start", attempt=attempt)
            try:
                res = await asyncio.wait_for(
                    self.provider.complete(messages, max_tokens=self.max_tokens,
                                           timeout_s=max(remaining, 0.001), on_token=on_token,
                                           relay_headers=headers),
                    timeout=max(remaining, 0.001) + TIMEOUT_BACKSTOP_S)
            except asyncio.TimeoutError:
                res = ProviderResult(status="timeout", error_message="client backstop timeout",
                                     t_send=time.monotonic(), t_end=time.monotonic())
            self.tokens_in += res.tokens_in
            self.tokens_out += res.tokens_out
            self.admission.observe(res, ticket)
            ticket = None
            self.ev("provider_end", attempt=attempt, status=res.status, http_status=res.http_status,
                    limit_kind=res.limit_kind, retry_after_s=res.retry_after_s,
                    tokens_in=res.tokens_in, tokens_out=res.tokens_out)
            if res.status == "ok":
                if res.t_first_token is not None:
                    self.provider_ttft_ms = (res.t_first_token - res.t_send) * 1000.0
                return res, None
            wait = self.retry.decide(res, attempt, self.deadline - time.monotonic(), produced["n"] > 0)
            if wait is None:
                return res, None
            self.ev("retry_wait_start", wait_s=round(wait, 3))
            t = time.monotonic()
            await asyncio.sleep(wait)
            self.retry_wait_s += time.monotonic() - t
            self.ev("retry_wait_end")
            self.retries += 1
            attempt += 1

    def failure(self, res: ProviderResult, hits) -> RequestOutcome:
        if res.status == "rate_limited":
            kind = res.limit_kind or "429"
            detail = "provider_daily_limit" if kind in ("rpd", "tpd") else "provider_429"
            reply = DAILY_EXHAUSTED_REPLY if kind in ("rpd", "tpd") else QUEUE_FULL_REPLY
            return self.finish("rate_limited", detail, reply, hits=hits)
        if res.status == "timeout":
            return self.finish("timeout", "provider_timeout" if time.monotonic() < self.deadline
                               else "gave_up_waiting", GAVE_UP_REPLY, hits=hits)
        return self.finish("error", f"provider_error_{res.http_status or 'none'}",
                           PROVIDER_FAILED_REPLY, hits=hits)


async def handle_request(
    question_id: str, question: str, *, scenario: str, setting_name: str,
    provider: ModelProvider, admission: AdmissionController, retry: RetryPolicy,
    recorder: Recorder, patience_s: float = 20.0, max_tokens: int = 400,
    expected: Optional[dict] = None, allow_degrade: bool = True,
    evidence_threshold: Optional[float] = None,
    analyze_fn: Optional[Callable[[str], Analysis]] = None,
    search_fn: Optional[Callable[..., list[Hit]]] = None,
    verify_fn: Optional[Callable[[str, list[Hit]], VerifyResult]] = None,
) -> RequestOutcome:
    """``expected`` is accepted for the grading hook but unused here (grading happens later).
    ``allow_degrade`` lets admission headroom < 0.3 switch the setting to A (load scenarios)."""
    analyze_fn = analyze_fn or _resolve("relay.analyze:analyze")
    search_fn = search_fn or _resolve("relay.retrieval:search_papers")
    verify_fn = verify_fn or _resolve("relay.verify:verify")
    if evidence_threshold is None:
        evidence_threshold = _resolve("relay.retrieval:EVIDENCE_THRESHOLD")

    # degrade decision is taken at arrival, before the (possibly queued) acquire
    used = setting_name
    degraded = False
    if allow_degrade and setting_name != "A" and admission.headroom() < DEGRADE_HEADROOM:
        used, degraded = "A", True
    run = _Run(question_id=question_id, scenario=scenario, setting_used=used, provider=provider,
               admission=admission, retry=retry, recorder=recorder, patience_s=patience_s,
               max_tokens=max_tokens)
    run.ev("arrive", question_id=question_id, scenario=scenario, setting=used,
           requested_setting=setting_name, degraded=degraded, queue_len=run.queue_len)
    ticket: Optional[Ticket] = None
    hits: list[Hit] = []
    try:
        got = await run.acquire(estimate_tokens_for_setting(used, max_tokens))
        if isinstance(got, Rejection):
            return run.reject(got)
        ticket = got

        analysis = analyze_fn(question)
        if analysis.out_of_scope:   # coordinator contract: refuse without retrieval or model call
            run.ev("gate", weak=True, out_of_scope=True, expanded=False)
            admission.release(ticket)
            ticket = None
            return run.finish("ok", "refused_out_of_scope", REFUSAL_REPLY)
        run.ev("retrieve_start", papers=list(analysis.papers))
        hits = await asyncio.to_thread(search_fn, analysis.normalized, used,
                                       papers=analysis.papers or None)
        run.ev("retrieve_end", n_hits=len(hits))
        best = max((h.score for h in hits), default=0.0)
        weak = best < evidence_threshold
        expanded = False
        if weak:
            expanded = True
            k2 = SETTINGS[used].keep_k * 2
            run.ev("retrieve_start", expand=True, k=k2)
            hits = await asyncio.to_thread(search_fn, analysis.normalized, used, papers=None, k=k2)
            run.ev("retrieve_end", n_hits=len(hits), expand=True)
            best = max((h.score for h in hits), default=0.0)
            weak = best < evidence_threshold
        run.ev("gate", best_score=round(best, 4), threshold=evidence_threshold, weak=weak,
               expanded=expanded, out_of_scope=False)
        if weak:
            admission.release(ticket)
            ticket = None
            return run.finish("ok", "refused_no_evidence", REFUSAL_REPLY, hits=hits)

        messages = build_messages(question, hits)
        admission.adjust(ticket, estimate_tokens(messages, max_tokens))   # exact prompt size now known
        res, rej = await run.call_model(messages, ticket)
        ticket = None
        if rej is not None:
            return run.reject(rej, hits)
        if res.status != "ok":
            return run.failure(res, hits)
        raw = res.text.strip()
        run.ev("verify_start")
        vr = verify_fn(raw, hits)
        run.ev("verify_end", ok=vr.ok, is_refusal=vr.is_refusal, problems=list(vr.problems), raw=raw[:800])
        if vr.is_refusal:
            return run.finish("ok", "refused_model", REFUSAL_REPLY, hits=hits, raw_text=raw,
                              model_tokens_out=res.tokens_out)
        if vr.ok:
            return run.finish("ok", "answered", raw, hits=hits, raw_text=raw,
                              model_tokens_out=res.tokens_out)

        # one revise call, only if budget and patience allow
        remaining = run.deadline - time.monotonic()
        if admission.headroom() < REVISE_MIN_HEADROOM or remaining < REVISE_MIN_PATIENCE_S:
            run.ev("revise_skipped", headroom=round(admission.headroom(), 3), remaining_s=round(remaining, 2))
            return run.finish("ok", "refused_verify", REFUSAL_REPLY, hits=hits)
        run.ev("revise_start", problems=list(vr.problems))
        messages2 = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "Your answer had these problems: " + "; ".join(vr.problems)
             + ". Rewrite it using only the passages above, citing every claim with the exact "
             "[paper_id p.N] tags shown, and only numbers that appear in the cited passages. "
             f"If the passages do not contain the answer, reply starting with {REFUSAL_PREFIX}"},
        ]
        res2, rej2 = await run.call_model(messages2, None)
        if rej2 is not None or res2 is None or res2.status != "ok":
            if res2 is not None and res2.status == "timeout":
                return run.failure(res2, hits)
            return run.finish("ok", "refused_verify", REFUSAL_REPLY, hits=hits)
        raw2 = res2.text.strip()
        run.ev("verify_start", revise=True)
        vr2 = verify_fn(raw2, hits)
        run.ev("verify_end", ok=vr2.ok, is_refusal=vr2.is_refusal, problems=list(vr2.problems), revise=True, raw=raw2[:800])
        if vr2.is_refusal:
            return run.finish("ok", "refused_model", REFUSAL_REPLY, hits=hits, raw_text=raw2,
                              model_tokens_out=res2.tokens_out)
        if vr2.ok:
            return run.finish("ok", "answered_revised", raw2, hits=hits, raw_text=raw2,
                              model_tokens_out=res2.tokens_out)
        return run.finish("ok", "refused_verify", REFUSAL_REPLY, hits=hits)
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 - a crashed request must still produce a row
        if ticket is not None:
            admission.release(ticket)
        run.ev("exception", type=type(e).__name__, message=str(e)[:200])
        return run.finish("error", f"exception_{type(e).__name__}", PROVIDER_FAILED_REPLY, hits=hits)
