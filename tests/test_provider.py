"""GroqProvider tests, always against the local stand-in (never the real Groq API)."""
import time

import pytest

from relay.provider import GroqProvider, parse_limit_kind, parse_retry_in_message
from relay.standin_server import default_config, running_standin

FAST = dict(
    ttft_ms={"median": 20, "p95": 20},
    inter_chunk_ms={"median": 5, "p95": 5},
    prompt_ms_per_1k_tokens=0,
    replay_bank=None,
)
MSGS = [{"role": "system", "content": "Answer from passages."},
        {"role": "user", "content": "Passage [moshi p.4] says things. Question?"}]


@pytest.fixture
def make_provider():
    cms = []

    def make(**overrides):
        cm = running_standin(default_config(**{**FAST, **overrides}))
        srv = cm.__enter__()
        cms.append(cm)
        return GroqProvider(source="stand-in", model="openai/gpt-oss-20b", base_url=srv.url, api_key="stand-in"), srv

    yield make
    for cm in cms:
        cm.__exit__(None, None, None)


def test_parsers():
    assert parse_limit_kind("on tokens per minute (TPM): Limit 8000") == "tpm"
    assert parse_limit_kind("on requests per day (RPD)") == "rpd"
    assert parse_limit_kind("blah TPD blah") == "tpd"
    assert parse_limit_kind("nothing") is None
    assert parse_retry_in_message("Please try again in 1.234s. Need") == pytest.approx(1.234)
    assert parse_retry_in_message("try again in 7m12.5s.") == pytest.approx(432.5)
    assert parse_retry_in_message("try again in 250ms") == pytest.approx(0.25)
    assert parse_retry_in_message("nope") is None


async def test_ok_text_timing_tokens_headers(make_provider):
    prov, _ = make_provider()
    tokens: list[str] = []
    r = await prov.complete(MSGS, max_tokens=64, timeout_s=5, on_token=tokens.append,
                            relay_headers={"X-Relay-Question-Id": "q1", "X-Relay-Setting": "B"})
    assert r.status == "ok" and prov.source == "stand-in"
    assert "[moshi p.4]" in r.text and "".join(tokens) == r.text and len(tokens) >= 2
    assert r.t_send < r.t_first_token < r.t_end
    assert r.t_first_token - r.t_send >= 0.015  # role-only chunk must not count as first token
    assert r.tokens_in > 0 and r.tokens_out > 0
    assert r.provider_timing.get("prompt_time") is not None and "usage_estimated" not in r.provider_timing
    assert r.headers["x-ratelimit-limit-tokens"] == "8000"
    assert all(k == "retry-after" or k.startswith(("x-ratelimit-", "x-groq-")) for k in r.headers)


async def test_rate_limited_rpm_has_retry_after_and_kind(make_provider):
    prov, _ = make_provider(limits={"rpm": 1})
    assert (await prov.complete(MSGS, max_tokens=32, timeout_s=5)).status == "ok"
    r = await prov.complete(MSGS, max_tokens=32, timeout_s=5)
    assert r.status == "rate_limited" and r.http_status == 429
    assert r.limit_kind == "rpm" and r.retry_after_s is not None and 1 <= r.retry_after_s <= 60
    assert r.headers["retry-after"] and r.text == ""


async def test_rate_limited_tpm_and_daily(make_provider):
    prov, _ = make_provider(limits={"tpm": 100})
    r = await prov.complete(MSGS, max_tokens=500, timeout_s=5)
    assert r.status == "rate_limited" and r.limit_kind == "tpm"
    prov2, srv2 = make_provider(limits={"rpd": 3})
    srv2.state.rpd_used = 3
    r2 = await prov2.complete(MSGS, max_tokens=32, timeout_s=5)
    assert r2.status == "rate_limited" and r2.limit_kind == "rpd" and r2.retry_after_s > 60


async def test_5xx_is_error_with_http_status(make_provider):
    prov, _ = make_provider(failures={"p_5xx": 1.0})
    r = await prov.complete(MSGS, max_tokens=32, timeout_s=5)
    assert r.status == "error" and r.http_status == 500 and r.error_message


async def test_hang_times_out(make_provider):
    prov, _ = make_provider(failures={"p_hang": 1.0})
    t0 = time.monotonic()
    r = await prov.complete(MSGS, max_tokens=32, timeout_s=0.5)
    assert r.status == "timeout"
    assert 0.4 < time.monotonic() - t0 < 2


async def test_slow_stream_times_out_with_partial_text(make_provider):
    prov, _ = make_provider(ttft_ms={"median": 10, "p95": 10}, inter_chunk_ms={"median": 150, "p95": 150})
    r = await prov.complete(MSGS, max_tokens=64, timeout_s=0.6)
    assert r.status == "timeout" and r.t_first_token is not None
    assert r.text and r.tokens_out > 0 and r.provider_timing.get("usage_estimated") is True


async def test_connection_refused_is_error():
    prov = GroqProvider(source="stand-in", model="openai/gpt-oss-20b", base_url="http://127.0.0.1:9", api_key="x")
    r = await prov.complete(MSGS, max_tokens=8, timeout_s=3)
    assert r.status == "error" and r.http_status is None
