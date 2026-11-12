"""One-off probe of the real Groq API: dumps raw stream chunks + rate-limit headers.

Spends real budget (2 tiny requests). Run once:  .venv/bin/python scripts/probe_groq.py
Output: outputs/probe.json  (API key is never written).
"""
import json, os, pathlib, time

import httpx
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
KEY = os.environ["GROQ_API_KEY"]
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
URL = "https://api.groq.com/openai/v1/chat/completions"
KEEP = ("x-ratelimit", "retry-after", "x-groq", "x-request-id", "date", "server")


def probe(include_reasoning: bool) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "In one sentence: what is multi-head attention?"}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_completion_tokens": 120,
        "temperature": 0.2,
        "reasoning_effort": "low",
        "include_reasoning": include_reasoning,
    }
    t0 = time.monotonic()
    chunks, first_any, first_content = [], None, None
    with httpx.stream("POST", URL, json=body, headers={"Authorization": f"Bearer {KEY}"}, timeout=60) as r:
        headers = {k: v for k, v in r.headers.items() if k.lower().startswith(KEEP)}
        status = r.status_code
        if status != 200:
            return {"status": status, "headers": headers, "body": r.read().decode()}
        for line in r.iter_lines():
            now = (time.monotonic() - t0) * 1000
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if first_any is None:
                first_any = now
            if data == "[DONE]":
                chunks.append({"t_ms": now, "raw": "[DONE]"})
                break
            obj = json.loads(data)
            delta = (obj.get("choices") or [{}])[0].get("delta", {}) if obj.get("choices") else {}
            if first_content is None and delta.get("content"):
                first_content = now
            chunks.append({"t_ms": round(now, 1), "obj": obj})
    return {
        "include_reasoning": include_reasoning, "status": status, "headers": headers,
        "first_chunk_ms": first_any, "first_content_ms": first_content,
        "total_ms": (time.monotonic() - t0) * 1000, "n_chunks": len(chunks),
        "first_chunks": chunks[:3], "last_chunks": chunks[-3:],
        "text": "".join(((c.get("obj", {}).get("choices") or [{}])[0].get("delta", {}) or {}).get("content") or ""
                        for c in chunks if c.get("obj", {}).get("choices")),
    }


if __name__ == "__main__":
    out = {"model": MODEL, "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "runs": [probe(False), probe(True)]}
    p = ROOT / "outputs/probe.json"
    p.write_text(json.dumps(out, indent=2))
    for run in out["runs"]:
        print({k: run.get(k) for k in ("include_reasoning", "status", "first_content_ms", "total_ms", "n_chunks")})
        print("headers:", run.get("headers"))
    print("wrote", p)
