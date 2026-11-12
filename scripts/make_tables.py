"""Build outputs/tables.md from outputs/results.csv ONLY.

Usage: .venv/bin/python scripts/make_tables.py [--results outputs/results.csv] [--out outputs/tables.md]
                                               [--price-in USD_PER_M] [--price-out USD_PER_M]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from relay.metrics import percentile  # noqa: E402

NM = "not measured"
SCENARIO_ORDER = ["steady", "burst", "sustained_overload", "rpm_halved", "daily_exhausted",
                  "real_steady", "overload_retry_on", "overload_retry_off"]
GRADES = ["correct", "wrong", "refused_correctly", "refused_wrongly"]
DEFAULT_COST = "$0.00 (free tier); paid-equivalent not measured"


def _f(x: str) -> Optional[float]:
    return float(x) if x not in ("", None) else None


def read_rows(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt_ms(v) -> str:
    return NM if v is None else f"{v:,.0f}"


def fmt_pct(num: int, den: int) -> str:
    return NM if den == 0 else f"{100.0 * num / den:.1f}%"


def fmt_num(v, nd=1) -> str:
    return NM if v is None else f"{v:,.{nd}f}"


def pcts(values: list[float], ps) -> str:
    if not values:
        return NM
    return " / ".join(fmt_ms(percentile(values, p)) for p in ps) + f" (n={len(values)})"


def _start_epoch(r: dict) -> float:
    return datetime.fromisoformat(r["start_time"]).timestamp()


def _group_key(r):
    return (r["scenario"], r["source"])


def _ordered(keys):
    def k(key):
        sc = key[0]
        return (SCENARIO_ORDER.index(sc) if sc in SCENARIO_ORDER else len(SCENARIO_ORDER), sc, key[1])
    return sorted(keys, key=k)


def _md(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _tokens(r) -> int:
    return int(_f(r["tokens_in"]) or 0) + int(_f(r["tokens_out"]) or 0)


def table_2a(groups: dict) -> str:
    rows = []
    for key in _ordered(groups):
        g = groups[key]
        starts = [_start_epoch(r) for r in g]
        ends = [_start_epoch(r) + float(r["total_ms"]) / 1000.0 for r in g]
        duration = max(ends) - min(starts)
        arrival_span = max(starts) - min(starts)
        rate = (len(g) / arrival_span * 60.0) if arrival_span > 0 else None
        ft = [float(r["first_token_ms"]) for r in g if r["status"] == "ok" and r["first_token_ms"] != ""]
        tot = [float(r["total_ms"]) for r in g]
        rows.append([key[0], "Real" if key[1] == "real" else "Stand-in", str(len(g)),
                     f"{duration:.1f}", fmt_num(rate), pcts(ft, (50, 95, 99)), pcts(tot, (50, 95, 99))])
    return _md(["Scenario", "Real or stand-in", "Requests sent", "Duration (s)",
                "Offered rate per min", "First token ms p50 / p95 / p99",
                "Total time ms p50 / p95 / p99"], rows)


def table_2b(groups: dict, price_in, price_out) -> str:
    rows = []
    for key in _ordered(groups):
        g = groups[key]
        n = len(g)
        ok = [r for r in g if r["status"] == "ok"]
        cnt = lambda s: sum(1 for r in g if r["status"] == s)  # noqa: E731
        tokens_ok = [_tokens(r) for r in ok]
        if price_in is None or price_out is None:
            cost = DEFAULT_COST
        elif not ok:
            cost = NM
        else:   # all tokens spent (failed rows included) / answers delivered
            spent = sum(float(r["tokens_in"] or 0) * price_in + float(r["tokens_out"] or 0) * price_out
                        for r in g) / 1e6
            cost = f"${spent / len(ok) * 1000:,.2f}"
        rows.append([
            key[0], "Real" if key[1] == "real" else "Stand-in",
            fmt_pct(len(ok), n), fmt_pct(cnt("rate_limited"), n), fmt_pct(cnt("timeout"), n),
            fmt_pct(cnt("error"), n),
            fmt_num(sum(int(r["retries"] or 0) for r in g) / n, 2),
            str(max(int(r["queue_len_at_arrival"] or 0) for r in g)),
            fmt_num(sum(tokens_ok) / len(tokens_ok) if tokens_ok else None, 0),
            cost])
    return _md(["Scenario", "Real or stand-in", "Succeeded %", "Rate-limited %",
                "Timed out or given up %", "Other errors %", "Retries per request (mean)",
                "Peak queue length", "Tokens per answer (mean)", "Cost per 1,000 answers"], rows)


def table_3(rows_all: list[dict]) -> str:
    groups = defaultdict(list)
    for r in rows_all:
        if r["scenario"] == "retrieval_eval":
            groups[(r["retrieval_setting"], r["source"])].append(r)
    if not groups:
        return NM
    out = []
    for key in sorted(groups):
        setting, source = key
        g = groups[key]
        graded = [r for r in g if r["answer_grade"] != ""]
        share = lambda grade: fmt_pct(sum(1 for r in graded if r["answer_grade"] == grade), len(graded))  # noqa: E731
        ok = [r for r in g if r["status"] == "ok"]
        tok = [_tokens(r) for r in ok]
        ft = [float(r["first_token_ms"]) for r in ok if r["first_token_ms"] != ""]
        tot = [float(r["total_ms"]) for r in g]
        n_q = len({r["question_id"] or r["request_id"] for r in g})
        asked = str(n_q) if n_q == len(g) else f"{n_q} ({len(g)} rows)"
        out.append([setting, "Real" if source == "real" else "Stand-in", asked, str(len(graded)), share("correct"), share("wrong"),
                    share("refused_correctly"), share("refused_wrongly"),
                    fmt_num(sum(tok) / len(tok) if tok else None, 0),
                    pcts(ft, (50, 95)), pcts(tot, (50, 95))])
    return _md(["Retrieval setting", "Real or stand-in", "Questions asked", "Graded (n)", "Correct and cited %", "Wrong %",
                "Correctly refused %", "Wrongly refused %", "Tokens per answer (mean)",
                "First token ms p50 / p95", "Total time ms p50 / p95"], out)


def build_tables(rows: list[dict], price_in=None, price_out=None) -> str:
    load = [r for r in rows if r["scenario"] != "retrieval_eval"]
    groups = defaultdict(list)
    for r in load:
        groups[_group_key(r)].append(r)
    parts = ["# Tables (generated from outputs/results.csv)\n"]
    parts.append("## Table 2a - Latency and offered load\n")
    parts.append(table_2a(groups) if groups else NM)
    parts.append("""
Notes: percentiles use linear interpolation between order statistics (numpy default, "type 7"); n is the
number of rows each percentile is computed over. First token = arrival to first answer content token
(includes queueing and retrieval), over `ok` rows that produced a token. Total time = arrival to final
outcome (answer, refusal or rejection), over all rows. Duration = max(start + total) - min(start).
Offered rate per min = requests / (max(start) - min(start)) * 60, where the denominator is the span of
arrival times (a burst concentrated in one second therefore inflates it); "not measured" if all arrivals
are simultaneous. results.csv is append-only: use one run per scenario per results file (or --outputs-dir),
otherwise Duration and Offered rate span the gap between repeated runs.
""")
    parts.append("## Table 2b - Outcomes, queueing and cost\n")
    parts.append(table_2b(groups, price_in, price_out) if groups else NM)
    parts.append("""
Notes: percentages are shares of all requests sent in that scenario (they sum to 100% with "Other errors").
Peak queue length = max `queue_len_at_arrival` (queue length seen by a request on arrival). Tokens per answer =
mean tokens_in + tokens_out over `ok` rows (refusals made without a model call count as 0 tokens; tokens of
all attempts of a request are included). Cost per 1,000 answers = all tokens spent in the scenario, including
failed requests, priced per million tokens, divided by `ok` answers, times 1,000.
""")
    parts.append("## Table 3 - Retrieval settings (scenario = retrieval_eval)\n")
    parts.append(table_3(rows))
    parts.append("""
Notes: the four grades (correct, wrong, refused correctly, refused wrongly) are shares of the graded rows
(non-blank `answer_grade`, n shown), so they sum to 100% per setting; "Questions asked" counts distinct questions, with the number of
rows in brackets when a question was attempted more than once (a budget-rejected attempt was re-run).
Non-ok rows are never graded. Percentiles: linear interpolation (numpy "type 7"),
n in parentheses. Tokens per answer as in Table 2b. First token over `ok` rows with a token, total over all rows.
""")
    return "\n".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "outputs" / "results.csv"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "tables.md"))
    ap.add_argument("--price-in", type=float, default=None, help="USD per million input tokens")
    ap.add_argument("--price-out", type=float, default=None, help="USD per million output tokens")
    a = ap.parse_args(argv)
    text = build_tables(read_rows(a.results), a.price_in, a.price_out)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text, encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
