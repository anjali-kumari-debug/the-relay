# The Relay

A cited question-answering pipeline over three arXiv papers (exact versions in `data/papers/manifest.json`).
It runs against Groq's free tier and against a calibrated stand-in model service, with load experiments.
Write-up: `report/submission.md`. Spec and results-file rules: `SPEC.md`.

## Setup
```
/opt/homebrew/bin/python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env    # add GROQ_API_KEY (never committed)
.venv/bin/python scripts/fetch_papers.py      # downloads and checks sha256
.venv/bin/pytest -q                           # 167 tests, no network
```

## Reproduce
| Step | Command | Spends real budget |
|---|---|---|
| Probe the stream and header shapes | `scripts/probe_groq.py` | 2 requests |
| Limits and latency | `scripts/calibrate_groq.py seq inflight rpm tpm --i-understand-this-spends-real-budget` | ~80 requests / 30K tokens |
| Retrieval evaluation (Table 3) | `scripts/eval_retrieval.py --source real --i-understand-this-spends-real-budget --min-interval-s 25` | ~140K tokens |
| Real load rows | `scripts/run_load.py --scenario real_steady --source real ...` / `real_daily_exhausted` | ~31K tokens |
| Stand-in load (Table 2) | `DUR=900 scripts/run_all_standin.sh` | none |
| Grade, then tables | `scripts/grade.py && scripts/make_tables.py` → `outputs/tables.md` | none |
| Slow-request breakdown | `scripts/slow_request.py --request-id 6e2f7c76314f` | none |
| Transcript export | `scripts/export_transcript.py` | none |

Run every script with `.venv/bin/python`.

## Raw data
- `outputs/results.csv` has one row per request. The brief's columns come first, then the extra columns documented in `SPEC.md`.
- `outputs/events.jsonl` holds the per-request timeline, including the raw model text at verification.
- `outputs/calibration.jsonl` holds every real calibration attempt with its headers and 429 bodies. `outputs/probe.json` holds the raw stream chunks.
- `outputs/answers_{A,B,C}.jsonl`, `outputs/grades_review.csv` and `data/grades_manual.json` (optional overrides) hold the grading.
- `outputs/runs/` keeps each load run's own files. `outputs/archive/eval_attempt1/` is the aborted first evaluation (3 rows), kept rather than deleted.
- Stand-in calibration provenance is in `configs/standin_calibrated.json` (`_doc`). Measured limits are in `configs/limits_measured.json`.

## Known issues, found in the measured runs and deliberately left unfixed so code matches results
1. The system prompt's citation example `[paper_id p.N]` was sometimes copied literally (4 drafts). Fix: show a concrete example tag, e.g. `[hnet p.7]`.
2. `verify.py` splits numbers on U+202F (the thin space gpt-oss uses in "300 000"), so it read "000". This wrongly refused one correct answer. Fix: normalise U+202F/U+00A0 before number extraction.
3. `DAILY_EXHAUSTED_REPLY` says allowances "reset once a day". Groq's TPD bucket refills continuously, so the reply should quote `retry-after`.
