"""Open-loop load generator. One asyncio task per arrival, scheduled at its time regardless of others.

Examples:
  .venv/bin/python scripts/run_load.py --scenario steady --source stand-in
  .venv/bin/python scripts/run_load.py --scenario sustained_overload --source stand-in --duration 120
  .venv/bin/python scripts/run_load.py --scenario real_steady --source real --i-understand-this-spends-real-budget

Rates are derived from capacity = min(rpm, tpm*safety / est_tokens_per_request) of the --limits file
unless --rate-per-min is given: steady/rpm_halved/daily_exhausted 0.6x, sustained_overload and
overload_retry_{on,off} 2.5x, burst trickle 0.3x; real_steady is a fixed 2/min.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relay.admission import AdmissionController  # noqa: E402
from relay.metrics import Recorder  # noqa: E402
from relay.pipeline import estimate_tokens_for_setting, handle_request  # noqa: E402
from relay.retry import RetryPolicy  # noqa: E402

SCENARIOS = ["steady", "burst", "sustained_overload", "rpm_halved", "daily_exhausted",
             "real_steady", "real_daily_exhausted", "overload_retry_on", "overload_retry_off"]
# (rate multiplier of capacity or fixed rate_per_min, default duration seconds)
PRESETS = {
    "steady": (0.6, 300), "burst": (0.3, 120), "sustained_overload": (2.5, 180),
    "rpm_halved": (0.6, 300), "daily_exhausted": (0.6, 60), "real_steady": (2.0, 600),
    "real_daily_exhausted": (2.0, 60),   # real Groq after its TPD bucket is empty: 429s cost no tokens
    "overload_retry_on": (2.5, 180), "overload_retry_off": (2.5, 180),
}
REAL_GUARD = "--i-understand-this-spends-real-budget"
DEFAULT_STANDIN_URL = None


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--source", choices=["stand-in", "real"], default="stand-in")
    ap.add_argument("--setting", default="B", choices=["A", "B", "C"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--questions", default=str(ROOT / "data" / "questions.json"))
    ap.add_argument("--standin-url", default=None, help="use an already running stand-in instead of in-process")
    ap.add_argument("--standin-config", default=str(ROOT / "configs" / "standin_default.json"))
    ap.add_argument("--limits", default=str(ROOT / "configs" / "limits_measured.json"))
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--rpd-used", type=int, default=0, help="ledger: requests already used today")
    ap.add_argument("--tpd-used", type=int, default=0, help="ledger: tokens already used today")
    ap.add_argument("--outputs-dir", default=str(ROOT / "outputs"))
    ap.add_argument(REAL_GUARD, dest="real_ok", action="store_true")


def check_real_guard(args) -> None:
    if args.source == "real" and not args.real_ok:
        sys.exit(f"refusing to run with --source real without {REAL_GUARD} (spends the real daily budget)")


def load_limits(path) -> dict:
    return json.loads(Path(path).read_text())


def load_questions(path, answerable_only: bool = False) -> list[dict]:
    qs = json.loads(Path(path).read_text())
    if isinstance(qs, dict):
        qs = qs.get("questions", [])
    if answerable_only:
        qs = [q for q in qs if q.get("answerable", True) is not False and not q.get("should_refuse", False)]
    return qs


def make_admission(limits: dict, *, rpd_used=0, tpd_used=0, max_queue=20, scale=1.0) -> AdmissionController:
    return AdmissionController(
        rpm=max(1, int(limits["rpm"] * scale)), tpm=max(1, int(limits["tpm"] * scale)),
        rpd=limits["rpd"], tpd=limits["tpd"], max_queue=max_queue,
        rpd_used=rpd_used, tpd_used=tpd_used)


def build_provider(source: str, model: str, base_url=None):
    from relay.provider import GroqProvider
    if source == "real":
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
        except ImportError:
            pass
        if not os.environ.get("GROQ_API_KEY"):
            sys.exit("GROQ_API_KEY is not set (put it in .env)")
        return GroqProvider(source="real", model=model)
    return GroqProvider(source="stand-in", model=model, base_url=base_url, api_key="stand-in")


def capacity_per_min(limits: dict, setting: str, max_tokens: int, safety: float = 0.9) -> float:
    est = estimate_tokens_for_setting(setting, max_tokens)
    return min(limits["rpm"] * safety, limits["tpm"] * safety / est)


def build_schedule(scenario: str, duration_s: float, rate_per_min: float, burst_size: int) -> list[float]:
    """Arrival offsets in seconds. Constant spacing (deterministic)."""
    import math
    gap = 60.0 / rate_per_min
    if scenario == "burst":
        burst = [i * (1.0 / max(burst_size, 1)) for i in range(burst_size)]   # all within 1 s
        trickle, t = [], 1.0 + gap
        while t < duration_s:
            trickle.append(t)
            t += gap
        return burst + trickle
    n = math.ceil(duration_s / gap)
    return [i * gap for i in range(max(n, 1))]


def reset_standin(url: str, payload: dict) -> None:
    import httpx
    r = httpx.post(url.rstrip("/") + "/admin/reset", json=payload, timeout=10)
    r.raise_for_status()


async def run_open_loop(schedule, questions, *, scenario, setting, provider, admission, retry,
                        recorder, patience_s, max_tokens):
    start = time.monotonic()
    outcomes = []

    async def fire(i: int, offset: float):
        await asyncio.sleep(max(0.0, start + offset - time.monotonic()))
        q = questions[i % len(questions)]
        outcomes.append(await handle_request(
            str(q["id"]), q["question"], scenario=scenario, setting_name=setting, provider=provider,
            admission=admission, retry=retry, recorder=recorder, patience_s=patience_s,
            max_tokens=max_tokens, expected=q))

    await asyncio.gather(*(fire(i, off) for i, off in enumerate(schedule)))
    return outcomes


def summary_line(scenario: str, outcomes) -> str:
    n = len(outcomes)
    by = {}
    for o in outcomes:
        by[o.row.status] = by.get(o.row.status, 0) + 1
    return (f"{scenario}: {n} requests " + " ".join(f"{k}={v}" for k, v in sorted(by.items())) +
            f" retries={sum(o.row.retries for o in outcomes)} "
            f"peak_queue={max((o.row.queue_len_at_arrival for o in outcomes), default=0)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=SCENARIOS, required=True)
    add_common_args(ap)
    ap.add_argument("--duration", type=float, default=None, help="arrival window, seconds")
    ap.add_argument("--rate-per-min", type=float, default=None)
    ap.add_argument("--burst-size", type=int, default=10)
    ap.add_argument("--patience", type=float, default=20.0)
    ap.add_argument("--admission-scale", type=float, default=1.0,
                    help="multiply the CLIENT-side admission RPM/TPM (provider limits unchanged). >1 lets "
                         "traffic through to the provider so its 429s (and our retries) are exercised")
    args = ap.parse_args(argv)
    check_real_guard(args)
    if args.scenario.startswith("real_") and args.source != "real":
        sys.exit(f"{args.scenario} requires --source real")
    if args.scenario != "real_steady" and args.source == "real" and args.scenario in (
            "sustained_overload", "overload_retry_on", "overload_retry_off", "rpm_halved", "daily_exhausted"):
        sys.exit(f"{args.scenario} is a stand-in scenario; use --source stand-in")

    random.seed(args.seed)
    limits = load_limits(args.limits)
    questions = load_questions(args.questions, answerable_only=True)
    if not questions:
        sys.exit(f"no questions in {args.questions}")
    mult, default_dur = PRESETS[args.scenario]
    cap = capacity_per_min(limits, args.setting, args.max_tokens)
    rate = args.rate_per_min or (mult if args.scenario.startswith("real_") else mult * cap)
    duration = args.duration or default_dur
    schedule = build_schedule(args.scenario, duration, rate, args.burst_size)

    scale = 0.5 if args.scenario == "rpm_halved" else 1.0
    rpd_used = limits["rpd"] if args.scenario == "daily_exhausted" else args.rpd_used
    admission = make_admission(limits, rpd_used=rpd_used, tpd_used=args.tpd_used,
                               scale=scale * args.admission_scale)
    retry = (RetryPolicy.disabled() if args.scenario == "overload_retry_off"
             else RetryPolicy(rng=random.Random(args.seed)))
    out_dir = Path(args.outputs_dir)
    recorder = Recorder(out_dir / "events.jsonl", out_dir / "results.csv")

    if True:
        try:   # warm the BM25 indexes so index build time is not counted as request latency
            from relay import retrieval
            for s in {args.setting, "A"}:
                retrieval.get_index(s)
        except Exception as e:  # noqa: BLE001
            print(f"warning: could not pre-build indexes: {e}", file=sys.stderr)

    print(f"{args.scenario}: {len(schedule)} arrivals over ~{duration:.0f}s "
          f"(rate {rate:.2f}/min, capacity ~{cap:.2f}/min, limits source: {limits.get('source')})")

    async def go(base_url):
        provider = build_provider(args.source, args.model, base_url)
        return await run_open_loop(schedule, questions, scenario=args.scenario, setting=args.setting,
                                   provider=provider, admission=admission, retry=retry,
                                   recorder=recorder, patience_s=args.patience, max_tokens=args.max_tokens)

    if args.source == "real":
        outcomes = asyncio.run(go(None))
    else:
        from relay.standin_server import load_config, running_standin
        eff = {k: limits[k] for k in ("rpm", "tpm", "rpd", "tpd")}
        eff["rpm"], eff["tpm"] = int(eff["rpm"] * scale), int(eff["tpm"] * scale)
        payload = {"config": {"seed": args.seed, "limits": eff}, "rpd_used": rpd_used,
                   "tpd_used": args.tpd_used}
        if args.standin_url:
            reset_standin(args.standin_url, payload)
            outcomes = asyncio.run(go(args.standin_url))
        else:
            with running_standin(load_config(args.standin_config)) as s:
                reset_standin(s.url, payload)
                outcomes = asyncio.run(go(s.url))
    print(summary_line(args.scenario, outcomes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
