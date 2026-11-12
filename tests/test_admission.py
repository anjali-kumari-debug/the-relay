import asyncio
import time

import pytest

from relay.admission import AdmissionController, Rejection, Ticket
from relay.types import ProviderResult


MODES = ["token_bucket", "sliding_window"]


def make(**kw):
    args = dict(rpm=10, tpm=1000, rpd=100, tpd=100000, max_queue=3, safety=1.0, window_s=0.3)
    args.update(kw)
    return AdmissionController(**args)


def far():
    return time.monotonic() + 30


async def test_admits_immediately_when_budget_free():
    a = make(mode="sliding_window")
    t = await a.acquire(300, far())
    assert isinstance(t, Ticket) and t.queued_s == 0
    assert a.headroom() == pytest.approx(0.7)


async def test_daily_exhausted_rejects_immediately_without_queueing():
    a = make(rpd=5, rpd_used=5)
    t0 = time.monotonic()
    r = await a.acquire(100, far())
    assert isinstance(r, Rejection) and r.reason == "daily_budget"
    assert time.monotonic() - t0 < 0.05 and a.queue_len == 0


async def test_tpd_exhausted_rejects():
    a = make(tpd=1000, tpd_used=900)
    r = await a.acquire(200, far())
    assert isinstance(r, Rejection) and r.reason == "daily_budget"


async def test_queues_then_admits_when_window_frees():
    a = make(tpm=1000, mode="sliding_window")
    assert isinstance(await a.acquire(800, far()), Ticket)
    t0 = time.monotonic()
    t = await a.acquire(800, far())          # must wait for the 0.3 s window to expire
    assert isinstance(t, Ticket)
    assert 0.25 <= time.monotonic() - t0 < 0.6
    assert t.queued_s > 0.2


@pytest.mark.parametrize("mode", MODES)
async def test_rejects_on_deadline(mode):
    a = make(tpm=1000, window_s=5.0, mode=mode)
    await a.acquire(900, far())
    r = await a.acquire(900, time.monotonic() + 0.1)
    assert isinstance(r, Rejection) and r.reason == "deadline"
    assert a.queue_len == 0            # waiter removed


async def test_expired_deadline_rejects_without_waiting():
    a = make()
    r = await a.acquire(10, time.monotonic() - 1)
    assert isinstance(r, Rejection) and r.reason == "deadline"


@pytest.mark.parametrize("mode", MODES)
async def test_queue_full(mode):
    a = make(tpm=1000, window_s=5.0, max_queue=2, mode=mode)
    await a.acquire(900, far())
    waiters = [asyncio.create_task(a.acquire(900, time.monotonic() + 0.4)) for _ in range(2)]
    await asyncio.sleep(0.05)
    assert a.queue_len == 2
    r = await a.acquire(900, far())
    assert isinstance(r, Rejection) and r.reason == "queue_full"
    await asyncio.gather(*waiters)


@pytest.mark.parametrize("mode", MODES)
async def test_fifo_order(mode):
    a = make(rpm=1, window_s=0.2, mode=mode)
    await a.acquire(10, far())
    order = []

    async def go(name):
        r = await a.acquire(10, far())
        order.append(name)
        return r

    tasks = []
    for n in "abc":
        tasks.append(asyncio.create_task(go(n)))
        await asyncio.sleep(0.01)
    await asyncio.gather(*tasks)
    assert order == ["a", "b", "c"]


@pytest.mark.parametrize("mode", MODES)
async def test_rpm_limit_applies_with_safety(mode):
    a = make(rpm=10, safety=0.5, window_s=5.0, mode=mode)   # effective 5 requests / window
    for _ in range(5):
        assert isinstance(await a.acquire(1, far()), Ticket)
    r = await a.acquire(1, time.monotonic() + 0.05)
    assert isinstance(r, Rejection) and r.reason == "deadline"


@pytest.mark.parametrize("mode", MODES)
async def test_observe_429_with_retry_after_pauses_everyone(mode):
    a = make(mode=mode)
    t = await a.acquire(100, far())
    a.observe(ProviderResult(status="rate_limited", http_status=429, retry_after_s=0.3, limit_kind="tpm"), t)
    assert a.headroom() == 0.0
    t0 = time.monotonic()
    t2 = await a.acquire(100, far())
    assert isinstance(t2, Ticket) and time.monotonic() - t0 >= 0.25


async def test_observe_daily_kind_marks_day_gone():
    a = make()
    a.observe(ProviderResult(status="rate_limited", http_status=429, limit_kind="tpd"))
    r = await a.acquire(10, far())
    assert isinstance(r, Rejection) and r.reason == "daily_budget"


async def test_observe_keeps_minute_reservation_but_charges_actual_to_day():
    a = make(tpm=1000, mode="sliding_window")
    t = await a.acquire(800, far())
    a.observe(ProviderResult(status="ok", tokens_in=100, tokens_out=50), t)
    assert a.headroom() == pytest.approx(0.2)        # provider reserves prompt+max_tokens: keep the reservation
    assert a.tpd_used == 150


async def test_actual_above_reservation_grows_window():
    a = make(tpm=1000, mode="sliding_window")
    t = await a.acquire(300, far())
    a.observe(ProviderResult(status="ok", tokens_in=500, tokens_out=100), t)
    assert a.headroom() == pytest.approx(0.4) and a.tpd_used == 600


@pytest.mark.parametrize("mode", MODES)
async def test_failed_attempt_with_no_tokens_is_refunded(mode):
    a = make(tpm=1000, mode=mode)
    t = await a.acquire(800, far())
    a.observe(ProviderResult(status="rate_limited", http_status=429, limit_kind="tpm"), t)
    assert a.headroom() == 1.0 and a.tpd_used == 0


async def test_adjust_replaces_estimate():
    a = make(tpm=1000, mode="sliding_window")
    t = await a.acquire(800, far())
    a.adjust(t, 300)
    assert a.headroom() == pytest.approx(0.7) and a.tpd_used == 300


@pytest.mark.parametrize("mode", MODES)
async def test_release_refunds(mode):
    a = make(mode=mode)
    t = await a.acquire(500, far())
    a.release(t)
    assert a.headroom() == 1.0 and a.rpd_used == 0 and a.tpd_used == 0


@pytest.mark.parametrize("mode", MODES)
async def test_oversized_request_is_clamped_not_stuck(mode):
    a = make(tpm=1000, mode=mode)
    t = await a.acquire(5000, far())
    assert isinstance(t, Ticket)


# ---------------------------------------------------------------- token-bucket mode (default)
class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def bucket(clock=None, **kw):
    """Bucket controller on a fake clock, 60 s refill (so 1000 tpm = 16.67 tokens/s)."""
    clock = clock or FakeClock()
    args = dict(rpm=10, tpm=1000, rpd=100, tpd=100000, max_queue=3, safety=1.0, clock=clock)
    args.update(kw)
    return AdmissionController(**args), clock


def test_default_mode_is_token_bucket_and_bad_mode_rejected():
    assert make().mode == "token_bucket"
    with pytest.raises(ValueError):
        make(mode="leaky")


async def test_bucket_headroom_is_level_over_capacity_and_refills_continuously():
    a, clk = bucket()
    assert a.headroom() == 1.0
    await a.acquire(600, clk() + 30)
    assert a.headroom() == pytest.approx(0.4)
    clk.advance(12)                                   # 12 s * 1000/60 = +200 tokens
    assert a.headroom() == pytest.approx(0.6)
    clk.advance(1000)
    assert a.headroom() == 1.0                         # capped at capacity


async def test_bucket_wake_in_is_deficit_over_refill_rate():
    a, clk = bucket()
    await a.acquire(800, clk() + 30)                  # level 200
    assert a._wake_in(800, clk()) == pytest.approx(600 / (1000 / 60))      # 36 s
    clk.advance(10)
    assert a._wake_in(800, clk()) == pytest.approx(36 - 10)
    clk.advance(26)
    assert a._wake_in(800, clk()) == 0.0
    assert a._wake_in(100, clk()) == 0.0


async def test_bucket_rpm_deficit():
    a, clk = bucket(rpm=10, tpm=10**6)
    for _ in range(10):
        assert isinstance(await a.acquire(1, clk() + 30), Ticket)
    assert a._wake_in(1, clk()) == pytest.approx(6.0)  # 1 request / (10 per 60 s)
    clk.advance(3)
    assert a._wake_in(1, clk()) == pytest.approx(3.0)
    clk.advance(3)
    assert a._wake_in(1, clk()) == 0.0
    assert isinstance(await a.acquire(1, clk() + 30), Ticket)
    assert a._wake_in(1, clk()) == pytest.approx(6.0)  # one request refilled, not a burst


async def test_bucket_safety_scales_capacity_and_refill():
    a, clk = bucket(rpm=10, tpm=1000, safety=0.5)     # capacity 5 requests / 500 tokens
    await a.acquire(400, clk() + 30)
    for _ in range(4):
        assert isinstance(await a.acquire(1, clk() + 30), Ticket)
    assert a._wake_in(1, clk()) == pytest.approx(60 / 5)               # 1 request at 5 per 60 s
    assert a.headroom() == pytest.approx(96 / 500)
    clk.advance(6)                                                      # +50 tokens at 500/60 per s
    assert a.headroom() == pytest.approx(146 / 500)
    big, bclk = bucket(tpm=1000, safety=0.5)
    t = await big.acquire(5000, bclk() + 30)           # clamped to the 500-token capacity
    assert isinstance(t, Ticket) and t.est_tokens == 500


async def test_bucket_observe_refunds_down_to_actual_usage():
    a, clk = bucket()
    t = await a.acquire(800, clk() + 30)              # prompt 100 + max_tokens 700 reserved
    a.observe(ProviderResult(status="ok", tokens_in=100, tokens_out=50), t)
    assert a.headroom() == pytest.approx(0.85)         # effective cost = actual usage (150)
    assert a.tpd_used == 150


async def test_bucket_observe_actual_above_reservation_goes_into_debt():
    a, clk = bucket()
    t = await a.acquire(300, clk() + 30)
    a.observe(ProviderResult(status="ok", tokens_in=500, tokens_out=100), t)
    assert a.headroom() == pytest.approx(0.4) and a.tpd_used == 600
    assert a._wake_in(700, clk()) == pytest.approx((700 - 400) / (1000 / 60))   # debt: waits for the deficit


async def test_bucket_refund_is_capped_at_capacity():
    a, clk = bucket()
    t = await a.acquire(800, clk() + 30)
    clk.advance(60)                                    # bucket refilled to full meanwhile
    a.observe(ProviderResult(status="ok", tokens_in=100, tokens_out=50), t)
    assert a.headroom() == 1.0


async def test_bucket_failed_attempt_without_tokens_refunds_tokens_not_request():
    a, clk = bucket()
    t = await a.acquire(800, clk() + 30)
    a.observe(ProviderResult(status="rate_limited", http_status=429, limit_kind="tpm"), t)
    assert a.headroom() == 1.0 and a.tpd_used == 0
    assert a._req_level == pytest.approx(9.0)          # the request itself stays counted (as in window mode)


async def test_bucket_adjust_and_release():
    a, clk = bucket()
    t = await a.acquire(800, clk() + 30)
    a.adjust(t, 300)
    assert a.headroom() == pytest.approx(0.7) and a.tpd_used == 300
    a.adjust(t, 500)
    assert a.headroom() == pytest.approx(0.5) and a.tpd_used == 500
    a.release(t)
    assert a.headroom() == 1.0 and a.rpd_used == 0 and a.tpd_used == 0 and a._req_level == pytest.approx(10.0)
    a.release(t)                                       # idempotent
    assert a.rpd_used == 0 and a._req_level == pytest.approx(10.0)


async def test_bucket_daily_and_cooldown_unchanged():
    a, clk = bucket(rpd=5, rpd_used=5)
    assert (await a.acquire(10, clk() + 30)) == Rejection("daily_budget")
    b, clk = bucket()
    t = await b.acquire(100, clk() + 30)
    b.observe(ProviderResult(status="rate_limited", http_status=429, retry_after_s=7.0, limit_kind="tpm"), t)
    assert b.headroom() == 0.0 and b._wake_in(1, clk()) == pytest.approx(7.0)


async def test_bucket_waiter_sleeps_for_deficit_not_polling():
    a = make(tpm=1000, window_s=0.3)                   # 1000 tokens refill in 0.3 s
    await a.acquire(800, far())                        # level 200 -> need 600 more = 0.18 s
    calls = 0
    orig = a._wake_in

    def counting(est, now):
        nonlocal calls
        calls += 1
        return orig(est, now)

    a._wake_in = counting
    t0 = time.monotonic()
    t = await a.acquire(800, far())
    waited = time.monotonic() - t0
    assert isinstance(t, Ticket) and 0.15 <= waited < 0.4
    assert t.queued_s == pytest.approx(waited, abs=0.05)
    assert calls <= 5                                  # a 5 ms poll loop would make ~35 calls


async def test_bucket_fifo_small_request_does_not_jump_the_queue():
    a = make(tpm=1000, window_s=0.3)
    await a.acquire(900, far())                        # level 100
    order = []

    async def go(name, est):
        r = await a.acquire(est, far())
        order.append(name)
        return r

    big = asyncio.create_task(go("big", 900))          # needs ~0.24 s
    await asyncio.sleep(0.01)
    small = asyncio.create_task(go("small", 10))       # would fit right now, but is behind big
    await asyncio.sleep(0.05)
    assert order == [] and a.queue_len == 2
    await asyncio.gather(big, small)
    assert order == ["big", "small"]


async def test_bucket_release_wakes_waiter_early():
    a = make(tpm=1000, window_s=60.0)
    t = await a.acquire(900, far())
    waiter = asyncio.create_task(a.acquire(900, far()))   # deficit says ~48 s
    await asyncio.sleep(0.05)
    assert not waiter.done()
    a.release(t)
    r = await asyncio.wait_for(waiter, 0.5)
    assert isinstance(r, Ticket)


async def test_bucket_observe_refund_wakes_waiter_early():
    a = make(tpm=1000, window_s=60.0)
    t = await a.acquire(900, far())
    waiter = asyncio.create_task(a.acquire(900, far()))
    await asyncio.sleep(0.05)
    a.observe(ProviderResult(status="ok", tokens_in=50, tokens_out=50), t)   # refund 800
    r = await asyncio.wait_for(waiter, 0.5)
    assert isinstance(r, Ticket)


async def test_bucket_deadline_shorter_than_deficit_rejects_on_time():
    a = make(tpm=1000, window_s=60.0)
    await a.acquire(900, far())
    t0 = time.monotonic()
    r = await a.acquire(900, time.monotonic() + 0.1)
    assert isinstance(r, Rejection) and r.reason == "deadline"
    assert time.monotonic() - t0 < 0.3 and a.queue_len == 0


async def test_sliding_window_mode_keeps_window_semantics():
    a = make(rpm=1, window_s=0.2, mode="sliding_window")
    await a.acquire(10, far())
    t0 = time.monotonic()
    t = await a.acquire(10, far())
    assert isinstance(t, Ticket) and 0.15 <= time.monotonic() - t0 < 0.5
