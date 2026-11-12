# The Relay — Engineering Rules

Source of truth: `The_Relay_Assignment_Brief.pdf` (copy in `docs/brief.pdf`) and `SPEC.md`.
Installed skills: addyosmani/agent-skills in `.claude/skills/` (use spec-driven-development,
test-driven-development, incremental-implementation, observability-and-instrumentation,
code-review-and-quality as relevant).

1. `SPEC.md` and `src/relay/types.py` are the contract. Do not change a shared interface
   without the orchestrator's approval; if you must, say so in your report.
2. Test first. Every module ships with pytest tests in `tests/`. `pytest -q` must pass.
3. **Never call the real Groq API** from tests or while building. Only the orchestrator runs
   real calls, through `scripts/` with the budget ledger. Use the stand-in or fakes.
4. Never invent measurements, API limits, prices, citations or results. Unknown = "not measured".
5. Keep raw request-level results reproducible: seeds, configs and raw rows go to `outputs/`.
6. Do not optimize before measuring. No frontend, GPU, LangChain, vector DB or Docker.
7. Never print, log or commit `GROQ_API_KEY` or `.env`.
8. Python 3.13, venv at `.venv/` (`.venv/bin/python`, `.venv/bin/pytest`). Async code uses asyncio.
9. Time: use `time.monotonic()` for durations, `time.time()` only for wall-clock `start_time`.
10. End each task with: files changed, tests run + result, open questions.
