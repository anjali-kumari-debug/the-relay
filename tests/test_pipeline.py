import csv
import json
import random

import pytest

from relay.admission import AdmissionController
from relay.metrics import Recorder
from relay.pipeline import (DAILY_EXHAUSTED_REPLY, REFUSAL_REPLY, SYSTEM_PROMPT, build_messages,
                            handle_request)
from relay.retry import RetryPolicy
from relay.types import Analysis, RESULTS_COLUMNS, VerifyResult
from tests.fakes import (FakeProvider, FakeSearch, fake_analyze, fake_verify_ok, limited, make_hit,
                         ok_result)


def adm(**kw):
    a = dict(rpm=30, tpm=8000, rpd=1000, tpd=200000, max_queue=10, safety=1.0)
    a.update(kw)
    return AdmissionController(**a)


@pytest.fixture
def rec(tmp_path):
    return Recorder(tmp_path / "events.jsonl", tmp_path / "results.csv")


async def run(rec, provider, search=None, admission=None, retry=None, verify=None, analyze=None,
              **kw):
    return await handle_request(
        "q1", "How many BLEU?", scenario=kw.pop("scenario", "steady"), setting_name=kw.pop("setting_name", "B"),
        provider=provider, admission=admission or adm(), retry=retry or RetryPolicy(rng=random.Random(0)),
        recorder=rec, analyze_fn=analyze or fake_analyze, search_fn=search or FakeSearch([make_hit()]),
        verify_fn=verify or fake_verify_ok, evidence_threshold=0.5, **kw)


def events(tmp_path):
    return [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]


async def test_happy_path_row_fields(rec, tmp_path):
    prov = FakeProvider([ok_result()], delay_s=0.1, ttft_s=0.05)
    out = await run(rec, prov)
    r = out.row
    assert r.status == "ok" and r.outcome_detail == "answered" and r.retries == 0
    assert r.source == "stand-in" and r.scenario == "steady" and r.question_id == "q1"
    assert r.tokens_in == 500 and r.tokens_out == 60 and r.answer_grade == ""
    assert r.retrieval_setting == "B:450w/k4/rerank=no"
    assert 50 <= r.provider_ttft_ms < 150
    assert r.first_token_ms >= r.provider_ttft_ms and r.first_token_ms < r.total_ms
    assert r.released_ms == r.total_ms and r.retry_wait_ms == 0
    assert r.wait_ms + r.work_ms + r.retry_wait_ms == pytest.approx(r.total_ms, abs=1)
    assert out.answer == "It reaches 28.4 BLEU [attention p.3]." and out.raw_text == out.answer
    assert prov.calls[0]["headers"] == {"X-Relay-Question-Id": "q1", "X-Relay-Setting": "B"}
    names = [e["event"] for e in events(tmp_path)]
    for n in ("arrive", "queue_enter", "queue_exit", "retrieve_start", "retrieve_end", "gate",
              "provider_start", "first_token", "provider_end", "verify_start", "verify_end", "done"):
        assert n in names
    # one row in the CSV with the exact header
    lines = (tmp_path / "results.csv").read_text().splitlines()
    assert lines[0] == ",".join(RESULTS_COLUMNS) and len(lines) == 2


async def test_prompt_contains_cite_tags_and_instructions():
    msgs = build_messages("Q?", [make_hit("moshi", 12, text="Moshi has 7B params.", section="Model")])
    assert msgs[0]["content"] == SYSTEM_PROMPT and "NOT_IN_PAPERS:" in SYSTEM_PROMPT
    assert "[moshi p.12] (section: Model)\nMoshi has 7B params." in msgs[1]["content"]


async def test_weak_evidence_expands_then_refuses_without_model_call(rec):
    weak = [make_hit(score=0.1)]
    search = FakeSearch(weak, weak)
    prov = FakeProvider([ok_result()])
    adm_ = adm()
    out = await run(rec, prov, search=search, admission=adm_)
    assert prov.calls == []
    assert out.row.status == "ok" and out.row.outcome_detail == "refused_no_evidence"
    assert out.answer == REFUSAL_REPLY and out.row.tokens_in == 0 and out.row.first_token_ms is None
    assert len(search.calls) == 2 and search.calls[1]["k"] == 8 and search.calls[1]["papers"] is None
    assert adm_.rpd_used == 0 and adm_.tpd_used == 0          # reservation refunded


async def test_weak_then_expansion_finds_evidence(rec):
    search = FakeSearch([make_hit(score=0.1)], [make_hit(score=0.9)])
    prov = FakeProvider([ok_result()])
    out = await run(rec, prov, search=search)
    assert len(prov.calls) == 1 and out.row.outcome_detail == "answered"


async def test_out_of_scope_refuses_without_model_call(rec):
    prov = FakeProvider([ok_result()])
    out = await run(rec, prov, analyze=lambda q: Analysis(q, (), True, ()))
    assert prov.calls == [] and out.row.outcome_detail == "refused_out_of_scope"
    assert out.row.status == "ok"


async def test_429_then_success_counts_retry_and_wait(rec):
    prov = FakeProvider([limited(retry_after_s=0.2, kind="tpm"), ok_result()])
    a = adm()
    out = await run(rec, prov, admission=a)
    r = out.row
    assert r.status == "ok" and r.retries == 1 and r.retry_wait_ms >= 190
    assert len(prov.calls) == 2 and r.tokens_in == 500     # failed attempt reported 0 tokens
    assert r.wait_ms + r.work_ms + r.retry_wait_ms == pytest.approx(r.total_ms, abs=1)


async def test_retry_off_gives_rate_limited(rec):
    prov = FakeProvider([limited(retry_after_s=0.2), ok_result()])
    out = await run(rec, prov, retry=RetryPolicy.disabled())
    assert out.row.status == "rate_limited" and out.row.retries == 0 and out.row.outcome_detail == "provider_429"
    assert len(prov.calls) == 1


async def test_daily_exhausted_rejects_with_helpful_message(rec):
    prov = FakeProvider([ok_result()])
    out = await run(rec, prov, admission=adm(rpd=10, rpd_used=10))
    assert prov.calls == []
    assert out.row.status == "rate_limited" and out.row.outcome_detail == "rejected_daily_budget"
    assert out.answer == DAILY_EXHAUSTED_REPLY and out.row.released_ms is None
    assert "reset" in DAILY_EXHAUSTED_REPLY and "not guessed" in DAILY_EXHAUSTED_REPLY


async def test_provider_daily_429_is_not_retried(rec):
    prov = FakeProvider([limited(retry_after_s=1, kind="tpd"), ok_result()])
    out = await run(rec, prov)
    assert len(prov.calls) == 1 and out.row.status == "rate_limited"
    assert out.row.outcome_detail == "provider_daily_limit" and out.answer == DAILY_EXHAUSTED_REPLY


async def test_patience_exceeded_is_timeout(rec):
    prov = FakeProvider([ok_result()], delay_s=2.0, ttft_s=1.5)
    out = await run(rec, prov, patience_s=0.3)
    r = out.row
    assert r.status == "timeout" and r.total_ms < 1000 and r.released_ms is None


async def test_gave_up_in_queue_is_timeout(rec):
    a = adm(tpm=1000, window_s=30.0)
    await a.acquire(900, 1e12)       # window now full
    out = await run(rec, FakeProvider([ok_result()]), admission=a, patience_s=0.2)
    assert out.row.status == "timeout" and out.row.outcome_detail == "gave_up_in_queue"
    assert out.row.wait_ms >= 150 and out.row.first_token_ms is None


async def test_queue_full_is_rate_limited(rec):
    a = adm(tpm=1000, window_s=30.0, max_queue=0)
    await a.acquire(900, 1e12)
    out = await run(rec, FakeProvider([ok_result()]), admission=a)
    assert out.row.status == "rate_limited" and out.row.outcome_detail == "rejected_queue_full"


async def test_model_refusal_is_ok(rec):
    prov = FakeProvider([ok_result("NOT_IN_PAPERS: not covered.")])
    out = await run(rec, prov)
    assert out.row.status == "ok" and out.row.outcome_detail == "refused_model"
    assert out.answer == REFUSAL_REPLY and out.raw_text.startswith("NOT_IN_PAPERS:")


async def test_verify_fail_then_revise_succeeds_sums_tokens(rec):
    calls = {"n": 0}

    def verify(answer, hits):
        calls["n"] += 1
        if calls["n"] == 1:
            return VerifyResult(False, False, (), ("no citation found",))
        return VerifyResult(True, False, (("attention", 3),), ())

    prov = FakeProvider([ok_result("bad"), ok_result("good [attention p.3]")])
    out = await run(rec, prov, verify=verify)
    assert out.row.outcome_detail == "answered_revised" and out.answer == "good [attention p.3]"
    assert out.row.tokens_in == 1000 and out.row.tokens_out == 120
    assert "no citation found" in prov.calls[1]["messages"][-1]["content"]
    assert prov.calls[1]["messages"][2] == {"role": "assistant", "content": "bad"}


async def test_verify_fail_without_headroom_refuses(rec):
    def verify(answer, hits):
        return VerifyResult(False, False, (), ("x",))

    a = adm(tpm=1000)
    prov = FakeProvider([ok_result("bad", tokens_in=900, tokens_out=50)])
    out = await run(rec, prov, verify=verify, admission=a)
    assert out.row.outcome_detail == "refused_verify" and out.row.status == "ok"
    assert len(prov.calls) == 1


async def test_low_headroom_degrades_to_setting_a(rec):
    a = adm(tpm=1000, window_s=30.0)
    await a.acquire(800, 1e12)        # headroom 0.2 < 0.3
    search = FakeSearch([make_hit()])
    out = await run(rec, FakeProvider([ok_result()]), admission=a, search=search, patience_s=0.3)
    assert search.calls == [] or search.calls[0]["setting"] == "A"
    assert out.row.retrieval_setting.startswith("A:")


async def test_exception_becomes_error_row(rec):
    def boom(q):
        raise RuntimeError("x")

    a = adm()
    out = await run(rec, FakeProvider([ok_result()]), analyze=boom, admission=a)
    assert out.row.status == "error" and a.rpd_used == 0


async def test_integration_with_real_retrieval_and_verify(rec):
    """Real analyze/retrieval/verify (local, no network) with a fake provider."""
    pytest.importorskip("rank_bm25")
    prov = FakeProvider([ok_result("NOT_IN_PAPERS: nothing relevant.")])
    out = await handle_request(
        "q9", "How many attention heads does the base Transformer use?", scenario="steady",
        setting_name="B", provider=prov, admission=adm(), retry=RetryPolicy(), recorder=rec)
    assert out.row.status == "ok" and out.hits and len(prov.calls) == 1
    user = prov.calls[0]["messages"][1]["content"]
    assert "[attention p." in user


def test_run_load_schedules_and_real_guard():
    import pytest as _p
    from scripts import run_load
    burst = run_load.build_schedule("burst", 60, 6.0, 10)
    assert len([t for t in burst if t <= 1.0]) == 10 and len(burst) > 10
    steady = run_load.build_schedule("steady", 60, 12.0, 0)
    assert steady == [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
    with _p.raises(SystemExit):
        run_load.main(["--scenario", "real_steady", "--source", "real"])


async def test_eval_resume_skips_done_pairs(tmp_path):
    import json
    from scripts import eval_retrieval
    (tmp_path / "answers_B.jsonl").write_text(json.dumps({"question_id": "q1"}) + "\n")
    prov = FakeProvider([ok_result("NOT_IN_PAPERS: x")])
    rec = Recorder(tmp_path / "e.jsonl", tmp_path / "r.csv")
    # patch the pipeline hooks through handle_request defaults is heavy; use real local modules
    n = await eval_retrieval.evaluate(
        [{"id": "q1", "question": "How many attention heads?"}, {"id": "q2", "question": "How many layers does the encoder have?"}],
        ["B"], provider=prov, admission=adm(), retry=RetryPolicy(), recorder=rec, out_dir=tmp_path,
        min_interval_s=0, patience_s=5, max_tokens=100)
    assert n == 1 and len(prov.calls) == 1
    assert "q2" in (tmp_path / "answers_B.jsonl").read_text()
    assert (tmp_path / "answer_bank.jsonl").exists()
