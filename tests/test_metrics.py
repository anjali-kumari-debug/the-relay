import csv
import json

from relay.metrics import Recorder, percentile
from relay.types import RESULTS_COLUMNS, ResultRow


def row(**kw):
    base = dict(request_id="r1", scenario="steady", source="stand-in", start_time="2026-01-01T00:00:00+00:00",
                first_token_ms=123.456, total_ms=2000.04, status="ok", retries=0, tokens_in=10,
                tokens_out=5, retrieval_setting="B:450w/k4/rerank=no")
    base.update(kw)
    return ResultRow(**base)


def test_header_exact_and_written_once(tmp_path):
    rec = Recorder(tmp_path / "e.jsonl", tmp_path / "r.csv")
    rec.write_row(row())
    rec.write_row(row(request_id="r2"))
    lines = (tmp_path / "r.csv").read_text().splitlines()
    assert lines[0] == ",".join(RESULTS_COLUMNS)
    assert len(lines) == 3
    # a new Recorder appends without a second header
    Recorder(tmp_path / "e.jsonl", tmp_path / "r.csv").write_row(row(request_id="r3"))
    assert (tmp_path / "r.csv").read_text().count("request_id,scenario") == 1


def test_none_blank_and_ms_rounded(tmp_path):
    rec = Recorder(tmp_path / "e.jsonl", tmp_path / "r.csv")
    rec.write_row(row(first_token_ms=None))
    rec.write_row(row(first_token_ms=123.456))
    rows = list(csv.DictReader((tmp_path / "r.csv").open()))
    assert rows[0]["first_token_ms"] == "" and rows[0]["provider_ttft_ms"] == ""
    assert rows[1]["first_token_ms"] == "123.5" and rows[1]["total_ms"] == "2000.0"
    assert rows[1]["answer_grade"] == ""


def test_events_have_t_ms_since_arrival(tmp_path):
    import time
    rec = Recorder(tmp_path / "e.jsonl", tmp_path / "r.csv")
    rec.event("a", "arrive", scenario="x")
    time.sleep(0.05)
    rec.event("a", "done", status="ok")
    rec.event("b", "arrive")
    ev = [json.loads(l) for l in (tmp_path / "e.jsonl").read_text().splitlines()]
    assert ev[0]["t_ms"] == 0 and ev[0]["event"] == "arrive" and ev[0]["scenario"] == "x"
    assert ev[1]["t_ms"] >= 45 and ev[1]["request_id"] == "a"
    assert ev[2]["t_ms"] == 0


def test_percentile_linear_interpolation():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([10, 20, 30, 40, 50], 95) == 48.0
    assert percentile([7], 99) == 7.0
    assert percentile([], 50) is None
