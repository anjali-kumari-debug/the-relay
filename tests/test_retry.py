import random

from relay.retry import RetryPolicy
from relay.types import ProviderResult


def res(status="rate_limited", http=429, retry_after=None, kind="tpm"):
    return ProviderResult(status=status, http_status=http, retry_after_s=retry_after, limit_kind=kind)


def test_respects_retry_after():
    p = RetryPolicy(rng=random.Random(0))
    assert p.decide(res(retry_after=2.5), 0, 20.0, False) == 2.5


def test_full_jitter_bounded_by_base_for_first_retry():
    p = RetryPolicy(rng=random.Random(1))
    waits = [p.decide(res(), 0, 20.0, False) for _ in range(200)]
    assert all(0 <= w <= 0.5 for w in waits) and max(waits) > 0.25


def test_jitter_capped():
    p = RetryPolicy(max_retries=10, base_s=0.5, cap_s=8.0, rng=random.Random(1))
    waits = [p.decide(res(), 9, 100.0, False) for _ in range(200)]
    assert all(w <= 8.0 for w in waits) and max(waits) > 4.0


def test_never_on_daily_limits():
    p = RetryPolicy()
    assert p.decide(res(kind="rpd", retry_after=1), 0, 20, False) is None
    assert p.decide(res(kind="tpd"), 0, 20, False) is None


def test_never_when_wait_exceeds_remaining_patience():
    p = RetryPolicy()
    assert p.decide(res(retry_after=5.0), 0, 4.0, False) is None
    assert p.decide(res(retry_after=5.0), 0, 5.0, False) == 5.0


def test_never_after_tokens_streamed():
    assert RetryPolicy().decide(res(retry_after=1), 0, 20, True) is None


def test_max_retries_enforced():
    assert RetryPolicy(max_retries=1).decide(res(retry_after=1), 1, 20, False) is None


def test_retryable_statuses():
    p = RetryPolicy(rng=random.Random(0))
    assert p.decide(ProviderResult(status="timeout"), 0, 20, False) is not None
    assert p.decide(ProviderResult(status="error", http_status=503), 0, 20, False) is not None
    assert p.decide(ProviderResult(status="error", http_status=400), 0, 20, False) is None
    assert p.decide(ProviderResult(status="error", http_status=None), 0, 20, False) is None
    assert p.decide(ProviderResult(status="ok"), 0, 20, False) is None


def test_disabled_never_retries():
    assert RetryPolicy.disabled().decide(res(retry_after=0.1), 0, 20, False) is None
