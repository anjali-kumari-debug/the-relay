"""Admission control: minute budgets (token bucket or sliding window), day counters, bounded FIFO queue.

Design (SPEC step 0):
  * RPM / TPM default to ``mode="token_bucket"``: continuously refilling buckets, capacity = effective
    limit, refill = capacity / ``window_s`` per second (60 s by default; injectable so tests run fast).
    This is how the Groq free tier was measured to behave. A request needs request-bucket >= 1 and
    token-bucket >= its estimate; a waiter sleeps for deficit / refill-rate. ``mode="sliding_window"``
    keeps the older sliding-window accounting. Effective limits are ``limit * safety`` in both modes
    so we stay under the provider's own accounting.
  * RPD / TPD are plain day counters with an initial ``*_used`` offset (simulate exhaustion, or
    start a real run from the budget ledger). Daily exhausted -> reject immediately, never queue.
  * Waiters are served strictly FIFO from a bounded queue; a waiter gives up at its deadline.
  * ``observe`` feeds provider truth back in: real token usage replaces the estimate, a 429 with
    ``retry_after_s`` pauses *all* admissions until then (cool-down), an rpd/tpd 429 marks the day
    as gone.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Literal, Optional

from relay.types import ProviderResult

RejectReason = Literal["daily_budget", "queue_full", "deadline"]
DAILY_KINDS = ("rpd", "tpd")
_MAX_POLL_S = 0.1   # sliding window: waiters re-check at least this often
_MIN_POLL_S = 0.005
_EPS = 1e-6         # token-bucket float tolerance (levels / deficits)
Mode = Literal["token_bucket", "sliding_window"]


@dataclass
class Ticket:
    est_tokens: int                 # tokens reserved in the minute window
    queued_s: float = 0.0
    _entry: Optional[list] = None   # [t, tokens] row in the minute window (internal)
    _tpd: int = 0                   # tokens charged to the day counter
    _tok: int = 0                   # tokens currently charged to the token bucket (bucket mode)
    _released: bool = False         # release() already refunded this ticket


@dataclass(frozen=True)
class Rejection:
    reason: RejectReason
    queued_s: float = 0.0


class _Waiter:
    """A queued acquire(); ``wake`` is set when budget may have changed (release/observe/adjust/exit)."""
    __slots__ = ("wake",)

    def __init__(self) -> None:
        self.wake = asyncio.Event()


class AdmissionController:
    def __init__(self, rpm: int, tpm: int, rpd: int, tpd: int, max_queue: int,
                 safety: float = 0.9, *, rpd_used: int = 0, tpd_used: int = 0,
                 window_s: float = 60.0, clock: Callable[[], float] = time.monotonic,
                 mode: Mode = "token_bucket"):
        if mode not in ("token_bucket", "sliding_window"):
            raise ValueError(f"mode must be 'token_bucket' or 'sliding_window', got {mode!r}")
        self.mode = mode
        self.rpm, self.tpm, self.rpd, self.tpd = rpm, tpm, rpd, tpd
        self.max_queue = max_queue
        self.safety = safety
        self.window_s = window_s
        self._clock = clock
        self.rpd_used = rpd_used
        self.tpd_used = tpd_used
        self._daily_exhausted = False
        self._cooldown_until = 0.0
        self._window: deque[list] = deque()   # [t, tokens]; one row per admitted request
        self._waiters: deque[_Waiter] = deque()
        # token-bucket state (levels may go negative after adjust/observe overshoot = debt)
        self._req_level = float(self._rpm_eff)
        self._tok_level = float(self._tpm_eff)
        self._refilled_at = clock()

    # ---- derived limits -------------------------------------------------------------------
    @property
    def _rpm_eff(self) -> int:
        return max(1, int(self.rpm * self.safety))

    @property
    def _tpm_eff(self) -> int:
        return max(1, int(self.tpm * self.safety))

    @property
    def queue_len(self) -> int:
        return len(self._waiters)

    @property
    def daily_exhausted(self) -> bool:
        return self._daily_exhausted or self.rpd_used >= int(self.rpd * self.safety)

    @property
    def _bucket(self) -> bool:
        return self.mode == "token_bucket"

    def _refill(self, now: float) -> None:
        """Token-bucket mode: add capacity/window_s per second since the last refill, capped at capacity."""
        dt = now - self._refilled_at
        if dt <= 0:
            return
        self._refilled_at = now
        self._req_level = min(float(self._rpm_eff), self._req_level + dt * self._rpm_eff / self.window_s)
        self._tok_level = min(float(self._tpm_eff), self._tok_level + dt * self._tpm_eff / self.window_s)

    def _poke(self) -> None:
        """Budget may have changed: let every waiter re-evaluate now."""
        for w in self._waiters:
            w.wake.set()

    def _prune(self, now: float) -> None:
        while self._window and self._window[0][0] <= now - self.window_s:
            self._window.popleft()

    def _tokens_in_window(self) -> int:
        return sum(e[1] for e in self._window)

    def headroom(self) -> float:
        """Fraction of the (effective) TPM budget still free in the current window, 0..1.
        0.0 while a provider cool-down is active."""
        now = self._clock()
        if now < self._cooldown_until:
            return 0.0
        if self._bucket:
            self._refill(now)
            return max(0.0, min(1.0, self._tok_level / self._tpm_eff))
        self._prune(now)
        return max(0.0, min(1.0, 1.0 - self._tokens_in_window() / self._tpm_eff))

    # ---- core -----------------------------------------------------------------------------
    def _daily_blocked(self, est: int) -> bool:
        return self.daily_exhausted or self.tpd_used + est > int(self.tpd * self.safety)

    def _wake_in(self, est: int, now: float) -> float:
        """Seconds until a request of ``est`` tokens could be admitted (0 = now)."""
        if now < self._cooldown_until:
            return self._cooldown_until - now
        if self._bucket:
            self._refill(now)
            wait = 0.0
            if self._req_level < 1.0 - _EPS:
                wait = (1.0 - self._req_level) * self.window_s / self._rpm_eff
            if self._tok_level < est - _EPS:
                wait = max(wait, (est - self._tok_level) * self.window_s / self._tpm_eff)
            return wait
        self._prune(now)
        wake = now
        if len(self._window) >= self._rpm_eff:
            wake = max(wake, self._window[len(self._window) - self._rpm_eff][0] + self.window_s)
        excess = self._tokens_in_window() + est - self._tpm_eff
        if excess > 0:
            freed = 0
            for t, tok in self._window:
                freed += tok
                if freed >= excess:
                    wake = max(wake, t + self.window_s)
                    break
        return max(0.0, wake - now)

    def _admit(self, est: int, now: float, queued_s: float) -> Ticket:
        if self._bucket:
            self._refill(now)
            self._req_level -= 1.0
            self._tok_level -= est
            self.rpd_used += 1
            self.tpd_used += est
            return Ticket(est_tokens=est, queued_s=queued_s, _tpd=est, _tok=est)
        entry = [now, est]
        self._window.append(entry)
        self.rpd_used += 1
        self.tpd_used += est
        return Ticket(est_tokens=est, queued_s=queued_s, _entry=entry, _tpd=est)

    async def acquire(self, est_tokens: int, deadline_monotonic: float) -> Ticket | Rejection:
        # A request bigger than the whole minute budget could never be admitted: clamp it.
        est = max(0, min(int(est_tokens), self._tpm_eff))
        start = self._clock()
        if self._daily_blocked(est):
            return Rejection("daily_budget")
        if start >= deadline_monotonic:
            return Rejection("deadline")
        if not self._waiters and self._wake_in(est, start) == 0.0:
            return self._admit(est, start, 0.0)
        if len(self._waiters) >= self.max_queue:
            return Rejection("queue_full")

        me = _Waiter()
        self._waiters.append(me)
        try:
            while True:
                now = self._clock()
                if self._daily_blocked(est):
                    return Rejection("daily_budget", now - start)
                if now >= deadline_monotonic:
                    return Rejection("deadline", now - start)
                wake_in = self._wake_in(est, now)
                if self._waiters[0] is me and wake_in == 0.0:
                    return self._admit(est, now, now - start)
                if self._bucket:
                    # Sleep for the bucket deficit; an event wakes us early if budget is released or we
                    # become head of the queue. A non-head waiter with budget just waits for that event.
                    pause = wake_in if wake_in > 0.0 else None
                else:
                    pause = min(max(wake_in, _MIN_POLL_S), _MAX_POLL_S)
                remaining = deadline_monotonic - now
                pause = remaining if pause is None else min(pause, remaining)
                if self._bucket:
                    me.wake.clear()
                    try:
                        await asyncio.wait_for(me.wake.wait(), timeout=max(pause, 0.0))
                    except asyncio.TimeoutError:
                        pass
                else:
                    await asyncio.sleep(max(pause, 0.0))
        finally:
            try:
                self._waiters.remove(me)
            except ValueError:
                pass
            self._poke()

    def _set_charge(self, ticket: Ticket, tokens: int) -> None:
        """Token-bucket mode: make ``tokens`` the amount this ticket has drawn from the bucket.
        A smaller charge refunds the difference (capped at capacity); a bigger one goes into debt."""
        now = self._clock()
        self._refill(now)
        delta = ticket._tok - tokens
        self._tok_level = min(float(self._tpm_eff), self._tok_level + delta) if delta > 0 else self._tok_level + delta
        ticket._tok = tokens

    def adjust(self, ticket: Ticket, est_tokens: int) -> None:
        """Replace the pre-retrieval estimate once the real prompt size is known (never waits)."""
        est = max(0, min(int(est_tokens), self._tpm_eff))
        if self._bucket:
            self._set_charge(ticket, est)
        elif ticket._entry is not None:
            ticket._entry[1] = est
        self.tpd_used += est - ticket._tpd
        ticket._tpd = est
        ticket.est_tokens = est
        self._poke()

    def release(self, ticket: Ticket) -> None:
        """Refund a ticket whose provider call was never made."""
        if self._bucket:
            if not ticket._released:
                self._set_charge(ticket, 0)     # also refills to "now"
                self._req_level = min(float(self._rpm_eff), self._req_level + 1.0)
        elif ticket._entry is not None:
            try:
                self._window.remove(ticket._entry)
            except ValueError:
                pass
            ticket._entry = None
        if not ticket._released:
            ticket._released = True
            self.rpd_used = max(0, self.rpd_used - 1)
        self.tpd_used = max(0, self.tpd_used - ticket._tpd)
        ticket._tpd = 0
        self._poke()

    def observe(self, result: ProviderResult, ticket: Optional[Ticket] = None) -> None:
        """Feed provider truth back in (call once per provider attempt).

        Day counter: charged the actual tokens. Minute budget, sliding window: providers reserve
        prompt + max_tokens at admission, so a successful call keeps max(reservation, actual) in the
        window. Minute budget, token bucket: the reservation is refunded down to the actual usage
        (capped at capacity), as the provider's remaining-tokens recovers; actual above the
        reservation is charged (debt). In both modes a failed attempt that used no tokens is refunded
        its tokens (the request itself stays counted)."""
        actual = result.tokens_in + result.tokens_out
        if ticket is not None and (ticket._entry is not None or (self._bucket and not ticket._released)):
            if result.status == "ok":
                if self._bucket:
                    self._set_charge(ticket, actual)
                else:
                    ticket._entry[1] = max(ticket._entry[1], actual)
                self.tpd_used += actual - ticket._tpd
                ticket._tpd = actual
            elif actual == 0:
                if self._bucket:
                    self._set_charge(ticket, 0)
                else:
                    ticket._entry[1] = 0
                self.tpd_used = max(0, self.tpd_used - ticket._tpd)
                ticket._tpd = 0
        if result.status == "rate_limited":
            if result.limit_kind in DAILY_KINDS:
                self._daily_exhausted = True
            elif result.retry_after_s:
                self._cooldown_until = max(self._cooldown_until,
                                           self._clock() + result.retry_after_s)
        self._poke()
