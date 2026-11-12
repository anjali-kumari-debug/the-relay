"""Stand-in server tests (real HTTP over a local uvicorn thread; never touches Groq)."""
import asyncio
import json
import math
import statistics
import time

import httpx
import pytest

from relay.standin_server import StandIn, _fmt_dur, default_config, running_standin

FAST = dict(
    ttft_ms={"median": 20, "p95": 20},
    inter_chunk_ms={"median": 5, "p95": 5},
    prompt_ms_per_1k_tokens=0,
    replay_bank=None,
)
PATH = "/openai/v1/chat/completions"


def body(content="hello [attention p.3] question", stream=True, max_tokens=50):
    return {
        "model": "openai/gpt-oss-20b", "messages": [{"role": "user", "content": content}],
        "stream": stream, "max_tokens": max_tokens,
    }


async def stream_events(client, payload, headers=None):
    events = []
    async with client.stream("POST", PATH, json=payload, headers=headers or {}) as r:
        assert r.status_code == 200, await r.aread()
        async for line in r.aiter_lines():
            if line.startswith("data: "):
                events.append(line[6:])
    return events


@pytest.fixture
def make_server():
    servers = []

    def make(**overrides):
        cm = running_standin(default_config(**{**FAST, **overrides}))
        srv = cm.__enter__()
        servers.append(cm)
        return srv

    yield make
    for cm in servers:
        cm.__exit__(None, None, None)


async def test_streams_multiple_chunks_and_final_usage(make_server):
    srv = make_server()
    async with httpx.AsyncClient(base_url=srv.url) as c:
        ev = await stream_events(c, body())
    assert ev[-1] == "[DONE]"
    chunks = [json.loads(e) for e in ev[:-1]]
    content = [ch["choices"][0]["delta"].get("content") for ch in chunks if ch["choices"][0]["delta"].get("content")]
    assert len(content) >= 2
    assert "".join(content).count("[attention p.3]") == 1  # fallback cites first tag in prompt
    final = chunks[-1]
    assert final["choices"][0]["finish_reason"] == "stop"
    for u in (final["usage"], final["x_groq"]["usage"]):
        assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"] > 0
        assert {"queue_time", "prompt_time", "completion_time"} <= set(u)


async def test_fallback_refuses_without_tag_and_non_streaming(make_server):
    srv = make_server()
    async with httpx.AsyncClient(base_url=srv.url) as c:
        r = await c.post(PATH, json=body("no citations here", stream=False))
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"].startswith("NOT_IN_PAPERS:")
    assert r.headers["x-ratelimit-limit-tokens"] == "8000"


@pytest.mark.parametrize("mode", ["token_bucket", "sliding_window"])
async def test_rpm_exceeded_returns_429_with_retry_after(make_server, mode):
    srv = make_server(limits={"rpm": 2, "mode": mode})
    async with httpx.AsyncClient(base_url=srv.url) as c:
        codes = [(await c.post(PATH, json=body(stream=False))).status_code for _ in range(2)]
        r = await c.post(PATH, json=body(stream=False))
    assert codes == [200, 200]
    assert r.status_code == 429
    assert 1 <= int(r.headers["retry-after"]) <= 60
    err = r.json()["error"]
    assert err["code"] == "rate_limit_exceeded" and err["type"] == "requests"
    assert "requests per minute (RPM)" in err["message"] and "Please try again in" in err["message"]
    assert "x-ratelimit-remaining-tokens" in r.headers


async def test_tpm_counts_prompt_plus_max_tokens(make_server):
    # sliding window: the max_tokens reservation stays in the window for 60 s (no refund)
    srv = make_server(limits={"tpm": 1000, "mode": "sliding_window"})
    async with httpx.AsyncClient(base_url=srv.url) as c:
        ok = await c.post(PATH, json=body(max_tokens=600, stream=False))  # ~600+prompt
        bad = await c.post(PATH, json=body(max_tokens=600, stream=False))  # would exceed 1000
        small = await c.post(PATH, json=body(max_tokens=10, stream=False))  # fits in the rest
    assert ok.status_code == 200
    assert bad.status_code == 429
    assert "tokens per minute (TPM)" in bad.json()["error"]["message"]
    assert bad.json()["error"]["type"] == "tokens"
    assert small.status_code == 200
    assert srv.state.tokens_used_minute() > 600


async def test_rpd_exhausted_via_admin_reset(make_server):
    srv = make_server(limits={"rpd": 5})
    async with httpx.AsyncClient(base_url=srv.url) as c:
        assert (await c.post("/admin/reset", json={"rpd_used": 5})).status_code == 200
        r = await c.post(PATH, json=body(stream=False))
        state = (await c.get("/admin/state")).json()
        await c.post("/admin/reset", json={})
        again = await c.post(PATH, json=body(stream=False))
    assert r.status_code == 429
    assert "requests per day (RPD)" in r.json()["error"]["message"]
    assert r.headers["x-ratelimit-remaining-requests"] == "0"
    assert state["rpd_used"] == 5
    assert again.status_code == 200


async def test_admin_scale_limits(make_server):
    srv = make_server(limits={"rpm": 10, "tpm": 8000})
    async with httpx.AsyncClient(base_url=srv.url) as c:
        r = await c.post("/admin/reset", json={"scale_limits": {"rpm": 0.5, "tpm": 0.5}})
    assert r.json()["limits"]["rpm"] == 5 and r.json()["limits"]["tpm"] == 4000


async def test_replay_bank_keyed_by_headers(make_server, tmp_path):
    bank = tmp_path / "bank.jsonl"
    bank.write_text(json.dumps({"question_id": "q1", "setting": "B", "text": "Recorded answer [moshi p.2]", "tokens_out": 7}) + "\n")
    srv = make_server(replay_bank=str(bank))
    hdr = {"X-Relay-Question-Id": "q1", "X-Relay-Setting": "B:450w/k4/rerank=no"}
    async with httpx.AsyncClient(base_url=srv.url) as c:
        hit = await c.post(PATH, json=body(stream=False), headers=hdr)
        miss = await c.post(PATH, json=body(stream=False), headers={**hdr, "X-Relay-Setting": "A"})
    assert hit.json()["choices"][0]["message"]["content"] == "Recorded answer [moshi p.2]"
    assert hit.json()["usage"]["completion_tokens"] == 7
    assert miss.json()["choices"][0]["message"]["content"] != "Recorded answer [moshi p.2]"


async def test_slowdown_grows_with_concurrency(make_server):
    srv = make_server(
        overload={"knee_in_flight": 1, "slowdown_per_extra": 2.0},
        ttft_ms={"median": 30, "p95": 30}, inter_chunk_ms={"median": 1, "p95": 1},
        limits={"rpm": 1000, "tpm": 10**7},
    )

    async def timed(c):
        t0 = time.monotonic()
        await c.post(PATH, json=body(stream=False))
        return time.monotonic() - t0

    async with httpx.AsyncClient(base_url=srv.url) as c:
        solo = statistics.median([await timed(c) for _ in range(3)])
        loaded = await asyncio.gather(*[timed(c) for _ in range(5)])
    # 5 in flight with knee 1 -> factor up to 3**4 on the slowest; compare the max conservatively
    assert max(loaded) > 2 * solo
    assert srv.state.max_in_flight_seen >= 4


async def test_max_in_flight_returns_503(make_server):
    srv = make_server(max_in_flight=1, ttft_ms={"median": 200, "p95": 200}, limits={"rpm": 1000})
    async with httpx.AsyncClient(base_url=srv.url) as c:
        a = asyncio.create_task(c.post(PATH, json=body(stream=False)))
        await asyncio.sleep(0.05)
        b = await c.post(PATH, json=body(stream=False))
        first = await a
    assert first.status_code == 200 and b.status_code == 503


async def test_5xx_injection(make_server):
    srv = make_server(failures={"p_5xx": 1.0})
    async with httpx.AsyncClient(base_url=srv.url) as c:
        r = await c.post(PATH, json=body())
    assert r.status_code == 500


async def test_seeded_latency_is_reproducible():
    from relay.standin_server import StandIn

    cfg = default_config(seed=7, replay_bank=None)
    a, b = StandIn(cfg), StandIn(cfg)
    assert [a.draw_lognormal_ms("ttft_ms") for _ in range(5)] == [b.draw_lognormal_ms("ttft_ms") for _ in range(5)]


# ---------------------------------------------------------------- token-bucket mode
def bucket_standin(**limits) -> StandIn:
    return StandIn(default_config(replay_bank=None, limits={"mode": "token_bucket", **limits}))


def test_default_config_mode_is_token_bucket():
    si = StandIn(default_config(replay_bank=None))
    assert si.config["limits"]["mode"] == "token_bucket" and si.mode == "token_bucket"


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        StandIn(default_config(replay_bank=None, limits={"mode": "leaky"}))


def test_fmt_dur_is_go_style():
    assert _fmt_dur(1.507) == "1.507s"
    assert _fmt_dur(47.917) == "47.917s"
    assert _fmt_dur(2.0) == "2s"
    assert _fmt_dur(0) == "0s" and _fmt_dur(-3) == "0s"
    assert _fmt_dur(86.4) == "1m26.4s"
    assert _fmt_dur(120) == "2m0s"
    assert _fmt_dur(3723.5) == "1h2m3.5s"


def test_tpm_bucket_deficit_and_refill_match_measured_groq_429():
    # measured: Limit 8000, Used 6389, Requested 2177 -> "try again in 4.245s", recovered in 3.8 s
    si = bucket_standin(rpm=1000, tpm=8000)
    t0 = 1000.0
    assert si.check_limits(t0, 389, 6000) is None                 # reserves 6389
    kind, used, limit, requested, wait = si.check_limits(t0, 177, 2000)
    assert (kind, used, limit, requested) == ("tpm", 6389, 8000, 2177)
    assert wait == pytest.approx((6389 + 2177 - 8000) / (8000 / 60))
    assert round(wait, 3) == 4.245
    assert si.check_limits(t0 + wait - 0.05, 177, 2000) is not None   # not quite refilled yet
    assert si.check_limits(t0 + wait + 1e-6, 177, 2000) is None       # admitted as soon as the deficit refilled


def test_rpm_bucket_refills_half_request_per_second_at_30_rpm():
    si = bucket_standin(rpm=30, tpm=10**7)
    t0 = 50.0
    for _ in range(30):
        assert si.check_limits(t0, 1, 1) is None
    kind, used, limit, requested, wait = si.check_limits(t0, 1, 1)
    assert (kind, used, limit, requested) == ("rpm", 30, 30, 1)
    assert wait == pytest.approx(2.0)                              # measured: "try again in 2s"
    assert si.check_limits(t0 + 1.0, 1, 1) is not None             # only half a request so far
    assert si.check_limits(t0 + 2.0, 1, 1) is None
    # one refilled request, not a whole burst
    assert si.check_limits(t0 + 2.0, 1, 1) is not None


def test_bucket_refill_is_capped_at_capacity():
    si = bucket_standin(rpm=30, tpm=8000)
    assert si.check_limits(0.0, 100, 900) is None
    si.rate_headers(10_000.0)
    assert si.tok_level == 8000 and si.req_level == 30


def test_headers_remaining_floor_and_reset_is_time_to_full():
    si = bucket_standin(rpm=30, tpm=8000)
    t0 = 10.0
    assert si.check_limits(t0, 1, 200) is None                    # 201 of 8000 reserved
    h = si.rate_headers(t0)
    assert h["x-ratelimit-limit-tokens"] == "8000"
    assert h["x-ratelimit-remaining-tokens"] == "7799"
    assert h["x-ratelimit-reset-tokens"] in ("1.507s", "1.508s")  # 201 / (8000/60) = 1.5075 s
    h2 = si.rate_headers(t0 + 0.75)                               # 100 more tokens back
    assert h2["x-ratelimit-remaining-tokens"] == "7899"           # floor(7899.0..) 
    assert h2["x-ratelimit-reset-tokens"] in ("0.757s", "0.758s")
    assert si.rate_headers(t0 + 5)["x-ratelimit-reset-tokens"] == "0s"
    assert si.rate_headers(t0 + 5)["x-ratelimit-remaining-tokens"] == "8000"


def test_headers_reset_tokens_minutes_format():
    si = bucket_standin(rpm=1000, tpm=8000)
    assert si.check_limits(0.0, 0, 8000) is None                   # empty bucket -> full in 60 s
    assert si.rate_headers(0.0)["x-ratelimit-reset-tokens"] == "1m0s"
    assert si.rate_headers(0.0)["x-ratelimit-remaining-tokens"] == "0"
    assert si.rate_headers(13.6)["x-ratelimit-reset-tokens"] == "46.4s"


def test_complete_refunds_unused_max_tokens_capped_at_capacity():
    si = bucket_standin(rpm=1000, tpm=1000)
    assert si.check_limits(0.0, 10, 600) is None                  # level 390
    si.complete(0.0, max_tokens=600, completion_tokens=20)        # refund 580 -> 970 (cost = 10 + 20)
    assert si.tok_level == pytest.approx(970)
    si.complete(0.0, max_tokens=600, completion_tokens=0)         # refund would overshoot -> cap
    assert si.tok_level == 1000


def test_complete_is_noop_in_sliding_window_mode():
    si = StandIn(default_config(replay_bank=None, limits={"mode": "sliding_window", "tpm": 1000}))
    assert si.check_limits(0.0, 10, 600) is None
    si.complete(0.0, 600, 20)
    assert si.tokens_used_minute() == 610


def test_rpd_and_tpd_unchanged_in_bucket_mode():
    si = bucket_standin(rpd=2, tpd=500)
    assert si.check_limits(0.0, 10, 100) is None
    si.tpd_used = 450
    assert si.check_limits(0.0, 10, 100)[0] == "tpd"
    si.tpd_used = 0
    si.rpd_used = 2
    assert si.check_limits(0.0, 1, 1)[0] == "rpd"


def test_oversized_request_never_admitted_in_bucket_mode():
    si = bucket_standin(rpm=1000, tpm=1000)
    assert si.check_limits(0.0, 10, 5000)[0] == "tpm"
    assert si.check_limits(1000.0, 10, 5000)[0] == "tpm"


async def test_bucket_429_body_and_retry_after_over_http(make_server):
    # a slow first response keeps its max_tokens reservation outstanding while the second arrives
    srv = make_server(limits={"tpm": 1000}, ttft_ms={"median": 400, "p95": 400})
    async with httpx.AsyncClient(base_url=srv.url) as c:
        first = asyncio.create_task(c.post(PATH, json=body(max_tokens=600, stream=False)))
        await asyncio.sleep(0.1)
        r = await c.post(PATH, json=body(max_tokens=600, stream=False))
        await first
    assert r.status_code == 429
    err = r.json()["error"]
    msg = err["message"]
    assert err["type"] == "tokens" and "tokens per minute (TPM)" in msg
    used = int(msg.split("Used ")[1].split(",")[0])
    requested = int(msg.split("Requested ")[1].split(".")[0])
    limit = int(msg.split("Limit ")[1].split(",")[0])
    assert limit == 1000 and used >= 600 and requested >= 600
    z = (used + requested - limit) / (limit / 60)                   # deficit / refill rate
    shown = float(msg.split("try again in ")[1].split("s.")[0])
    assert shown == pytest.approx(z, abs=0.5)
    assert int(r.headers["retry-after"]) == math.ceil(shown) or int(r.headers["retry-after"]) == math.ceil(shown) + 1
    assert int(r.headers["x-ratelimit-remaining-tokens"]) <= limit - used + 1


async def test_bucket_refunds_unused_max_tokens_after_response(make_server):
    srv = make_server(limits={"tpm": 1000})
    async with httpx.AsyncClient(base_url=srv.url) as c:
        ok = await c.post(PATH, json=body(max_tokens=600, stream=False))   # reserves ~607, uses ~25
        second = await c.post(PATH, json=body(max_tokens=600, stream=False))  # fits only thanks to the refund
        state = (await c.get("/admin/state")).json()
    assert ok.status_code == 200 and second.status_code == 200
    assert state["limit_mode"] == "token_bucket"
    assert state["tpm_used"] < 100                                      # ~ actual usage, not 2 x 607


async def test_bucket_streaming_also_refunds(make_server):
    srv = make_server(limits={"tpm": 1000})
    async with httpx.AsyncClient(base_url=srv.url) as c:
        await stream_events(c, body(max_tokens=600))
        await stream_events(c, body(max_tokens=600))
        state = (await c.get("/admin/state")).json()
    assert state["tpm_used"] < 100


async def test_admin_reset_refills_buckets(make_server):
    srv = make_server(limits={"rpm": 2})
    async with httpx.AsyncClient(base_url=srv.url) as c:
        for _ in range(2):
            await c.post(PATH, json=body(stream=False))
        assert (await c.post(PATH, json=body(stream=False))).status_code == 429
        await c.post("/admin/reset", json={})
        assert (await c.post(PATH, json=body(stream=False))).status_code == 200


async def test_sliding_window_mode_still_available_over_http(make_server):
    srv = make_server(limits={"mode": "sliding_window", "rpm": 1})
    async with httpx.AsyncClient(base_url=srv.url) as c:
        a = await c.post(PATH, json=body(stream=False))
        b = await c.post(PATH, json=body(stream=False))
        state = (await c.get("/admin/state")).json()
    assert a.status_code == 200 and b.status_code == 429
    assert 1 <= int(b.headers["retry-after"]) <= 60
    assert state["limit_mode"] == "sliding_window" and state["rpm_used"] == 1
