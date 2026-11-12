#!/usr/bin/env bash
# Run every stand-in load scenario in parallel (each with its own in-process stand-in and output dir),
# then merge into outputs/results.csv and outputs/events.jsonl. Needs outputs/answer_bank.jsonl
# (written by the real retrieval eval) so stand-in answers replay real, graded answers.
#
# Durations are long because the stand-in enforces the REAL measured limits (8000 TPM => ~2.5
# pipeline requests/min): short runs would give too few requests for p95/p99.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
CFG=configs/standin_calibrated.json
LIM=configs/limits_measured.json
DUR=${DUR:-900}
RUN=outputs/runs/$(date +%Y%m%dT%H%M%S)
mkdir -p "$RUN"
common=(--source stand-in --setting B --standin-config "$CFG" --limits "$LIM" --questions data/questions.json --seed 7)

run() {  # name, extra args...
  local name=$1; shift
  mkdir -p "$RUN/$name"
  $PY scripts/run_load.py --scenario "$name" "${common[@]}" --outputs-dir "$RUN/$name" "$@" \
      > "$RUN/$name/log.txt" 2>&1 && echo "done $name: $(tail -1 "$RUN/$name/log.txt")" \
      || echo "FAILED $name (see $RUN/$name/log.txt)"
}

run steady              --duration "$DUR" &
run burst               --duration "$DUR" --burst-size 20 &
run sustained_overload  --duration "$DUR" &
run rpm_halved          --duration "$DUR" &
run daily_exhausted     --duration 300 --rate-per-min 6 &
run overload_retry_on   --duration "$DUR" --admission-scale 10 &
run overload_retry_off  --duration "$DUR" --admission-scale 10 &
wait

# merge (header once)
first=1
for d in "$RUN"/*/; do
  [ -f "$d/results.csv" ] || continue
  if [ $first = 1 ] && [ ! -f outputs/results.csv ]; then head -1 "$d/results.csv" > outputs/results.csv; fi
  first=0
  tail -n +2 "$d/results.csv" >> outputs/results.csv
  cat "$d/events.jsonl" >> outputs/events.jsonl
done
echo "merged into outputs/results.csv from $RUN"
