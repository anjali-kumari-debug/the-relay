"""Single retry layer (SPEC "Retry policy"). Pure decision logic; the pipeline does the sleeping."""
from __future__ import annotations

import random
from typing import Optional

from relay.types import ProviderResult

DAILY_KINDS = ("rpd", "tpd")


class RetryPolicy:
    def __init__(self, max_retries: int = 1, base_s: float = 0.5, cap_s: float = 8.0,
                 rng: Optional[random.Random] = None):
        self.max_retries = max_retries
        self.base_s = base_s
        self.cap_s = cap_s
        self.rng = rng or random.Random()

    @classmethod
    def disabled(cls) -> "RetryPolicy":
        """Retry-off experiment (overload_retry_off)."""
        return cls(max_retries=0)

    def decide(self, result: ProviderResult, attempt: int, remaining_s: float,
               produced_tokens: bool) -> Optional[float]:
        """Seconds to wait before retrying, or None = do not retry.

        ``attempt`` = number of retries already made (0 after the first failed attempt).
        """
        if attempt >= self.max_retries or produced_tokens:
            return None
        if result.status == "rate_limited":
            if result.limit_kind in DAILY_KINDS:
                return None
        elif result.status == "timeout":
            pass
        elif result.status == "error":
            if result.http_status is None or not (500 <= result.http_status < 600):
                return None
        else:
            return None
        if result.retry_after_s is not None:
            wait = max(0.0, float(result.retry_after_s))
        else:   # full jitter: U(0, min(cap, base * 2^attempt))
            wait = self.rng.uniform(0.0, min(self.cap_s, self.base_s * (2 ** attempt)))
        if wait > remaining_s:
            return None
        return wait
