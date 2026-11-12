"""Measure the real Groq service: latency spread, concurrency, and what happens at a limit.

SPENDS REAL BUDGET (~30K tokens, ~90 requests). Phases (run individually or all):
  seq      : 20 sequential streamed requests (12 small, 8 passage-sized) paced under TPM
  inflight : 12 concurrent tiny requests (max 5 tokens) -> concurrency behaviour
  rpm      : 36 tiny requests fired back-to-back -> find the RPM 429 and its body
  tpm      : 8 concurrent passage-sized requests -> TPM 429, then poll until recovered

Every attempt -> outputs/calibration.jsonl ; running token total -> outputs/budget_ledger.jsonl
Usage: .venv/bin/python scripts/calibrate_groq.py seq inflight rpm tpm --i-understand-this-spends-real-budget
"""
import asyncio, json, os, pathlib, sys, time

import httpx
import pymupdf
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
URL = "https://api.groq.com/openai/v1/chat/completions"
KEEP = ("x-ratelimit", "retry-after", "x-groq")
CAL = ROOT / "outputs/calibration.jsonl"
LEDGER = ROOT / "outputs/budget_ledger.jsonl"


def passage(words: int, offset: int) -> str:
    d = pymupdf.open(ROOT / "data/papers/1706.03762v7.pdf")
    toks = " ".join(p.get_text() for p in d).split()
    return " ".join(toks[offset: offset + words])


def append(path: pathlib.Path, rec: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


async def call(client: httpx.AsyncClient, phase: str, prompt: str, max_tokens: int) -> dict:
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "stream": True,
            "stream_options": {"include_usage": True}, "max_completion_tokens": max_tokens,
            "temperature": 0.2, "reasoning_effort": "low", "include_reasoning": False}
    rec = {"phase": phase, "wall_start": time.time(), "prompt_chars": len(prompt), "max_tokens": max_tokens}
    t0 = time.monotonic()
    gaps, last, first_content, usage, xgroq = [], None, None, None, None
    try:
        async with client.stream("POST", URL, json=body, timeout=60) as r:
            rec["status"] = r.status_code
            rec["headers"] = {k: v for k, v in r.headers.items() if k.lower().startswith(KEEP)}
            rec["headers_ms"] = (time.monotonic() - t0) * 1000
            if r.status_code != 200:
                rec["body"] = (await r.aread()).decode()[:1000]
            else:
                async for line in r.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    obj = json.loads(line[6:])
                    now = (time.monotonic() - t0) * 1000
                    ch = obj.get("choices") or []
                    if ch and (ch[0].get("delta") or {}).get("content"):
                        if first_content is None:
                            first_content = now
                        elif last is not None:
                            gaps.append(round(now - last, 2))
                        last = now
                    usage = obj.get("usage") or usage
                    xgroq = obj.get("x_groq") or xgroq
    except httpx.TimeoutException as e:
        rec["status"], rec["error"] = "timeout", repr(e)
    except httpx.HTTPError as e:
        rec["status"], rec["error"] = "error", repr(e)
    rec.update(total_ms=(time.monotonic() - t0) * 1000, first_content_ms=first_content,
               gaps_ms=gaps, usage=usage or (xgroq or {}).get("usage"), x_groq=xgroq)
    append(CAL, rec)
    u = rec["usage"] or {}
    append(LEDGER, {"t": time.time(), "phase": f"calibrate:{phase}", "status": rec["status"],
                    "tokens": u.get("total_tokens", 0),
                    "remaining_tokens_min": rec.get("headers", {}).get("x-ratelimit-remaining-tokens"),
                    "remaining_requests_day": rec.get("headers", {}).get("x-ratelimit-remaining-requests")})
    print(phase, rec["status"], f"ttft={first_content and round(first_content)}ms total={round(rec['total_ms'])}ms",
          f"tok={u.get('prompt_tokens')}/{u.get('completion_tokens')}", rec.get("body", "")[:160])
    return rec


async def seq(c):
    small = "Explain in two sentences what {} means in the Transformer paper."
    topics = ["scaled dot-product attention", "multi-head attention", "positional encoding", "label smoothing",
              "residual dropout", "the encoder stack", "the decoder stack", "beam search", "byte-pair encoding",
              "the Adam optimizer schedule", "self-attention complexity", "layer normalization"]
    for t in topics:
        await call(c, "seq_small", small.format(t), 120)
        await asyncio.sleep(2.5)
    for i in range(8):
        p = passage(1200, 800 * i) + "\n\nSummarize the passage above in 3 sentences."
        await call(c, "seq_passage", p, 200)
        await asyncio.sleep(16)  # ~1.8K tokens each -> stay under 8K TPM


async def inflight(c):
    await asyncio.gather(*[call(c, "inflight", f"Reply with the number {i}.", 5) for i in range(12)])


async def rpm(c):
    await asyncio.gather(*[call(c, "rpm", "Reply OK.", 1) for _ in range(36)])
    # recovery: poll every 2 s until a 200
    t0 = time.monotonic()
    while time.monotonic() - t0 < 90:
        r = await call(c, "rpm_recovery", "Reply OK.", 1)
        if r["status"] == 200:
            append(CAL, {"phase": "rpm_recovered_after_s", "value": time.monotonic() - t0})
            break
        await asyncio.sleep(2)


async def tpm(c):
    await asyncio.sleep(61)  # let the minute window clear after previous phases
    p = passage(1400, 2000) + "\n\nGive a one-line summary."
    await asyncio.gather(*[call(c, "tpm", p, 60) for _ in range(8)])
    t0 = time.monotonic()
    while time.monotonic() - t0 < 90:
        r = await call(c, "tpm_recovery", p, 30)
        if r["status"] == 200:
            append(CAL, {"phase": "tpm_recovered_after_s", "value": time.monotonic() - t0})
            break
        await asyncio.sleep(3)


async def main(phases):
    headers = {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}
    async with httpx.AsyncClient(headers=headers, limits=httpx.Limits(max_connections=50)) as c:
        for ph in phases:
            await {"seq": seq, "inflight": inflight, "rpm": rpm, "tpm": tpm}[ph](c)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--i-understand-this-spends-real-budget" not in sys.argv:
        sys.exit("refusing: pass --i-understand-this-spends-real-budget")
    asyncio.run(main(args or ["seq", "inflight", "rpm", "tpm"]))
