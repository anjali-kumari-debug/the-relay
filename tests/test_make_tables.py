import csv

from relay.types import RESULTS_COLUMNS
from scripts.make_tables import build_tables, main, read_rows


def mk(**kw):
    base = {c: "" for c in RESULTS_COLUMNS}
    base.update(request_id="r", scenario="steady", source="stand-in", status="ok", retries="0",
                tokens_in="0", tokens_out="0", queue_len_at_arrival="0", total_ms="1000",
                start_time="2026-01-01T00:00:00+00:00", retrieval_setting="B:450w/k4/rerank=no")
    base.update({k: str(v) for k, v in kw.items()})
    return base


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def synthetic():
    rows = []
    # steady: 4 requests at t=0,10,20,30 s; totals 1000..4000 ms; one rate_limited, one timeout
    for i, (tot, st, ft) in enumerate([(1000, "ok", 100), (2000, "ok", 200), (3000, "rate_limited", ""),
                                       (4000, "timeout", "")]):
        rows.append(mk(request_id=f"s{i}", start_time=f"2026-01-01T00:00:{10 * i:02d}+00:00",
                       total_ms=tot, status=st, first_token_ms=ft, retries=i % 2,
                       queue_len_at_arrival=i, tokens_in=100 if st == "ok" else 0,
                       tokens_out=50 if st == "ok" else 0))
    # retrieval_eval, setting A: 2 correct, 1 wrong, 1 refused_correctly
    for i, g in enumerate(["correct", "correct", "wrong", "refused_correctly"]):
        rows.append(mk(request_id=f"e{i}", scenario="retrieval_eval", answer_grade=g,
                       retrieval_setting="A:250w/k3/rerank=no", first_token_ms=100 * (i + 1),
                       total_ms=1000 * (i + 1), tokens_in=200, tokens_out=100))
    return rows


def test_tables_exact_numbers(tmp_path):
    write(tmp_path / "r.csv", synthetic())
    md = build_tables(read_rows(tmp_path / "r.csv"))
    # Table 2a: duration = max(start+total) - min(start) = 34 s (30 s + 4 s)
    # offered rate = 4 / 30 * 60 = 8.0 ; first token over ok rows [100, 200] -> 150 / 195 / 199
    # total over [1000..4000] -> p50 2,500, p95 3,850, p99 3,970
    assert ("| steady | Stand-in | 4 | 34.0 | 8.0 | 150 / 195 / 199 (n=2) | "
            "2,500 / 3,850 / 3,970 (n=4) |") in md
    # Table 2b: 50.0% ok, 25.0% rate-limited, 25.0% timeout, 0.0% other, retries mean 0.50,
    # peak queue 3, tokens per answer 150, default cost text
    assert ("| steady | Stand-in | 50.0% | 25.0% | 25.0% | 0.0% | 0.50 | 3 | 150 | "
            "$0.00 (free tier); paid-equivalent not measured |") in md
    # Table 3: 4 graded -> 50.0 / 25.0 / 25.0 / 0.0; tokens 300; first token p50/p95 250 / 385; total 2,500 / 3,850
    assert ("| A:250w/k3/rerank=no | Stand-in | 4 | 4 | 50.0% | 25.0% | 25.0% | 0.0% | 300 | "
            "250 / 385 (n=4) | 2,500 / 3,850 (n=4) |") in md
    assert "linear interpolation" in md and "sum to 100%" in md


def test_cost_from_prices(tmp_path):
    write(tmp_path / "r.csv", synthetic())
    # spent = 2 ok rows * (100*1 + 50*2)/1e6 = 0.0004 ; / 2 answers * 1000 = $0.20
    md = build_tables(read_rows(tmp_path / "r.csv"), 1.0, 2.0)
    assert "| 0.50 | 3 | 150 | $0.20 |" in md


def test_no_data_says_not_measured(tmp_path):
    write(tmp_path / "r.csv", [mk(scenario="steady", status="ok", first_token_ms="")])
    md = build_tables(read_rows(tmp_path / "r.csv"))
    assert "not measured" in md and "## Table 3" in md


def test_cli_writes_file(tmp_path):
    write(tmp_path / "r.csv", synthetic())
    assert main(["--results", str(tmp_path / "r.csv"), "--out", str(tmp_path / "t.md")]) == 0
    assert (tmp_path / "t.md").read_text().startswith("# Tables")


def test_table3_separates_real_from_stand_in(tmp_path):
    rows = synthetic() + [mk(request_id="x", scenario="retrieval_eval", source="real", answer_grade="wrong",
                             retrieval_setting="A:250w/k3/rerank=no", total_ms=500)]
    write(tmp_path / "r.csv", rows)
    md = build_tables(read_rows(tmp_path / "r.csv"))
    assert "| A:250w/k3/rerank=no | Real | 1 | 1 | 0.0% | 100.0% |" in md
    assert "| A:250w/k3/rerank=no | Stand-in | 4 | 4 |" in md
