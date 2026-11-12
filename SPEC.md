# SPEC — The Relay

## Objective
A small cited-answer assistant over three exact-version arXiv papers, plus the measurement
apparatus to show how it behaves under load against Groq's free tier and a stand-in model service.

## Hard constraint that shapes the design
One Groq free-tier daily budget is available (deadline is today). For `openai/gpt-oss-20b` the
binding limit is tokens per day (~200K, to be confirmed). **So: exactly one model call per answer
in the normal path.** Query analysis and citation verification are local and deterministic.

## Pipeline (per request)
| # | Step | Where | Notes |
|---|------|-------|-------|
| 0 | Admission | `admission.py` | RPM/TPM token buckets, daily ledger (RPD/TPD), bounded queue, per-request patience deadline. Reject honestly instead of queueing past the deadline. |
| 1 | Analyze | `analyze.py` | Normalize; map keywords to paper_ids (may be several or none); flag obviously out-of-library questions. |
| 2 | Retrieve (tool call) | `retrieval.py` via tool `search_papers(query, k, papers)` | BM25 over page-aware chunks for the chosen retrieval setting. |
| 3 | Evidence gate (branch/loop) | `pipeline.py` | If best score < threshold: expand once (k×2, drop paper filter). Still weak → refuse without a model call. |
| 4 | Draft | `provider.py` (Groq or stand-in), streamed | Answer only from passages; cite `[paper_id p.N]`; refuse with `NOT_IN_PAPERS:` prefix. |
| 5 | Verify | `verify.py` | Citations must point to retrieved passages; numbers in the answer must appear in a cited passage. Fail → one revise call if budget allows, else refuse. |

Degrade under pressure: when TPM headroom is low, admission tells the pipeline to use the smaller
retrieval setting (fewer tokens in) — "weaker answer" instead of "no answer".

## Retrieval settings
| name | piece_words | overlap | fetch_k | keep_k | rerank |
|------|-------------|---------|---------|--------|--------|
| A | 250 | 50 | 3 | 3 | no |
| B | 450 | 75 | 4 | 4 | no |
| C | 450 | 75 | 8 | 4 | yes (lexical overlap + section/paper prior) |

## Citation format
`[attention p.3]`, `[moshi p.12]`, `[hnet p.7]` — paper_id + 1-based PDF page. Regex in `types.py`.
paper_ids: `attention` (1706.03762v7), `moshi` (2410.00037v1), `hnet` (2507.07955v2).
Refusal: model output beginning `NOT_IN_PAPERS:`; pipeline returns a fixed user-facing refusal.

## results.csv (exact)
Brief columns first, in this order:
`request_id, scenario, source, start_time, first_token_ms, total_ms, status, retries, tokens_in, tokens_out, retrieval_setting, answer_grade`
then extras, appended:
`question_id, queue_len_at_arrival, wait_ms, work_ms, retry_wait_ms, provider_ttft_ms, released_ms, outcome_detail`

Rules:
- `source` ∈ {`real`, `stand-in`}. `start_time` = ISO-8601 UTC wall time of arrival.
- `first_token_ms` = arrival → first answer content token from the model (includes queueing and
  retrieval). Empty if no token was produced. `released_ms` = arrival → verified reply released to user.
- `total_ms` = arrival → final outcome (success, refusal, or rejection).
- `status`: `ok` (an answer or a correct-by-design refusal was delivered), `rate_limited` (rejected
  or failed because a provider/admission limit was reached, including daily exhausted),
  `timeout` (provider/tool timeout **or the user gave up** — patience deadline passed), `error` (other).
- `retries` = number of extra provider attempts.
- `tokens_in`/`tokens_out` from provider usage (stand-in reports its own estimate). 0 if no call made.
- `answer_grade` ∈ {`correct`, `wrong`, `refused_correctly`, `refused_wrongly`} for `ok` rows;
  **blank for non-ok rows**; tables exclude blanks from quality percentages.
- Peak queue length = max `queue_len_at_arrival` within a scenario.
- `wait_ms` (queue + rate-limit waiting), `work_ms` (retrieval + provider + verify),
  `retry_wait_ms` (backoff sleeps). wait+work+retry_wait ≈ total.
- `outcome_detail`: short machine tag, e.g. `answered`, `refused_no_evidence`, `refused_model`,
  `refused_verify`, `rejected_daily_budget`, `rejected_queue_full`, `gave_up_in_queue`, `provider_429`.

Supplementary per-event timeline in `outputs/events.jsonl`:
`{"request_id", "t_ms" (since arrival), "event", ...fields}` with events
`arrive, queue_enter, queue_exit, retrieve_start, retrieve_end, gate, provider_start, first_token,
provider_end, retry_wait_start, retry_wait_end, verify_start, verify_end, done`.

## Retry policy
Single retry layer (the SDK's own retries are disabled: `max_retries=0`).
Max 1 retry. Retry on 429 (TPM/RPM), 5xx, timeout. Wait = `retry-after` if present else full-jitter
exponential (base 0.5 s, cap 8 s). Never retry when: daily budget exhausted; wait would exceed the
request's remaining patience; the stream already produced tokens (no duplicate partial answers).

## Stand-in model service
FastAPI, OpenAI/Groq-compatible `POST /openai/v1/chat/completions` (streaming SSE + non-streaming),
so the same `AsyncGroq` client is pointed at it with `base_url`. Behaviours, all seeded and configured
from a JSON file: calibrated lognormal TTFT and per-chunk gaps; RPM/TPM/RPD/TPD counters returning 429
with `retry-after` and `x-ratelimit-*` headers; slowdown that grows with in-flight requests;
injectable 5xx and hangs (marked synthetic); replays recorded real answers keyed by
`X-Relay-Question-Id` + `X-Relay-Setting` headers so stand-in rows carry real grades.

## Load scenarios (Table 2)
steady, burst, sustained_overload, rpm_halved (halve TPM and RPM — TPM binds), daily_exhausted;
plus real_steady (real Groq, ~2/min) and overload_retry_on vs overload_retry_off (heading 6).
Open-loop arrivals (scheduled times, not waiting for previous to finish). Each request has a
patience deadline (default 20 s) after which the user gives up.

## Commands
- `.venv/bin/pytest -q`
- `.venv/bin/python scripts/run_load.py --scenario steady --source stand-in`
- `.venv/bin/python scripts/make_tables.py` → `outputs/tables.md`

## Boundaries
- Always: tests for new behaviour; raw outputs kept; key never logged.
- Ask first: changing shared types/columns; spending real Groq budget.
- Never: real Groq calls from tests/builders; invented numbers; committing `.env`.
