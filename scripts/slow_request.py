"""Heading 4: split one slow request's latency into waiting, working and retrying, from events.jsonl.

Picks the slowest `ok` request in a scenario (or --request-id) and prints its timeline:
  waiting  = queue (queue_enter->queue_exit) + any admission waits before a provider attempt
  retrying = retry_wait_start->retry_wait_end sleeps + time spent in failed provider attempts
  working  = retrieval + successful provider attempt(s) + verify (+ revise call)
and whether the delay landed before or after the first token.
Usage: .venv/bin/python scripts/slow_request.py --scenario sustained_overload [--events outputs/events.jsonl]
"""
import argparse, csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario")
    ap.add_argument("--request-id")
    ap.add_argument("--results", default=str(ROOT / "outputs/results.csv"))
    ap.add_argument("--events", default=str(ROOT / "outputs/events.jsonl"))
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.results)))
    if a.request_id:
        row = next(r for r in rows if r["request_id"] == a.request_id)
    else:
        cand = [r for r in rows if r["scenario"] == a.scenario and r["status"] == "ok" and r["first_token_ms"]]
        row = max(cand, key=lambda r: float(r["total_ms"]))
    rid = row["request_id"]
    ev = [json.loads(l) for l in open(a.events) if f'"{rid}"' in l]
    ev = [e for e in ev if e["request_id"] == rid]
    print(f"request {rid} scenario={row['scenario']} setting={row['retrieval_setting']} q={row['question_id']}")
    print(f"total {row['total_ms']} ms; first token {row['first_token_ms']} ms; retries {row['retries']}; "
          f"csv split wait/work/retry_wait = {row['wait_ms']}/{row['work_ms']}/{row['retry_wait_ms']} ms\n")
    prev = 0.0
    for e in ev:
        extra = {k: v for k, v in e.items() if k not in ("request_id", "t_ms", "event", "raw")}
        print(f"{e['t_ms']:>9.1f} ms  (+{e['t_ms'] - prev:>8.1f})  {e['event']:<17} {json.dumps(extra)[:140]}")
        prev = e["t_ms"]
    t = {e["event"]: e["t_ms"] for e in ev}
    ft = float(row["first_token_ms"])
    print(f"\nbefore first token: {ft:.0f} ms of {float(row['total_ms']):.0f} ms "
          f"({100 * ft / float(row['total_ms']):.1f}%); after first token: {float(row['total_ms']) - ft:.0f} ms")
    if "queue_exit" in t:
        print(f"queue wait (queue_enter->queue_exit): {t['queue_exit'] - t.get('queue_enter', 0):.1f} ms")


if __name__ == "__main__":
    main()
