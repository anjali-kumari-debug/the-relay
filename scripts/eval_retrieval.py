"""Retrieval-setting evaluation: every question under settings A, B, C, sequentially.

Resumable: (question, setting) pairs already in outputs/answers_{setting}.jsonl are skipped, so
budget is not spent twice. Only `ok` outcomes are recorded there; failed attempts are retried on re-run
(their rows stay in results.csv).

  .venv/bin/python scripts/eval_retrieval.py --source stand-in
  .venv/bin/python scripts/eval_retrieval.py --source real --i-understand-this-spends-real-budget
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_load import (ROOT, add_common_args, build_provider, check_real_guard, load_limits,  # noqa: E402
                      load_questions, make_admission)

from relay.metrics import Recorder  # noqa: E402
from relay.pipeline import handle_request  # noqa: E402
from relay.retry import RetryPolicy  # noqa: E402
from relay.types import parse_citations  # noqa: E402

BANKABLE = ("answered", "answered_revised", "refused_model")


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(json.loads(l)["question_id"]) for l in path.read_text().splitlines() if l.strip()}


def append_jsonl(path: Path, rec: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


async def evaluate(questions, settings, *, provider, admission, retry, recorder, out_dir: Path,
                   min_interval_s: float, patience_s: float, max_tokens: int) -> int:
    n_run = 0
    # question-outer, setting-inner: if the daily budget runs out mid-run, every setting has been
    # asked the same questions, so Table 3 stays a like-for-like comparison.
    bank_path = out_dir / "answer_bank.jsonl"
    for q in questions:
        for setting in settings:
            answers_path = out_dir / f"answers_{setting}.jsonl"
            qid = str(q["id"])
            if qid in done_ids(answers_path):
                continue
            t0 = time.monotonic()
            o = await handle_request(
                qid, q["question"], scenario="retrieval_eval", setting_name=setting, provider=provider,
                admission=admission, retry=retry, recorder=recorder, patience_s=patience_s,
                max_tokens=max_tokens, expected=q, allow_degrade=False)
            n_run += 1
            if o.row.status == "ok":
                cited = sorted(set(parse_citations(o.answer)))
                append_jsonl(answers_path, {
                    "request_id": o.row.request_id, "question_id": qid, "setting": setting, "question": q["question"],
                    "answer": o.answer, "cited_pages": [list(c) for c in cited],
                    "outcome": o.row.outcome_detail})
                if o.row.outcome_detail in BANKABLE:
                    append_jsonl(bank_path, {"question_id": qid, "setting": setting, "text": o.raw_text,
                                             "tokens_out": o.model_tokens_out})
            spare = min_interval_s - (time.monotonic() - t0)
            if spare > 0:
                await asyncio.sleep(spare)
    return n_run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--settings", default="A,B,C")
    ap.add_argument("--min-interval-s", type=float, default=None,
                    help="min seconds between request starts (default 25 real, 0 stand-in)")
    ap.add_argument("--patience", type=float, default=60.0)
    args = ap.parse_args(argv)
    check_real_guard(args)
    interval = args.min_interval_s if args.min_interval_s is not None else (25.0 if args.source == "real" else 0.0)
    questions = load_questions(args.questions)
    out_dir = Path(args.outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    limits = load_limits(args.limits)
    admission = make_admission(limits, rpd_used=args.rpd_used, tpd_used=args.tpd_used)
    recorder = Recorder(out_dir / "events.jsonl", out_dir / "results.csv")
    retry = RetryPolicy(rng=random.Random(args.seed))
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]

    async def go(base_url):
        provider = build_provider(args.source, args.model, base_url)
        return await evaluate(questions, settings, provider=provider, admission=admission, retry=retry,
                              recorder=recorder, out_dir=out_dir, min_interval_s=interval,
                              patience_s=args.patience, max_tokens=args.max_tokens)

    if args.source == "real":
        n = asyncio.run(go(None))
    else:
        import run_load
        from relay.standin_server import load_config, running_standin
        payload = {"config": {"seed": args.seed, "limits": {k: limits[k] for k in ("rpm", "tpm", "rpd", "tpd")}}}
        if args.standin_url:
            run_load.reset_standin(args.standin_url, payload)
            n = asyncio.run(go(args.standin_url))
        else:
            with running_standin(load_config(args.standin_config)) as s:
                run_load.reset_standin(s.url, payload)
                n = asyncio.run(go(s.url))
    print(f"eval_retrieval: ran {n} (question, setting) pairs; settings={settings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
