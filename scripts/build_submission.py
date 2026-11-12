"""Assemble report/submission.md from report/submission_template.md + the tables in outputs/tables.md.
Run after scripts/grade.py and scripts/make_tables.py so the write-up always matches results.csv."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tables = (ROOT / "outputs/tables.md").read_text()


def block(header_start: str) -> str:
    lines, on = [], False
    for ln in tables.splitlines():
        if ln.startswith(header_start):
            on = True
        if on:
            if not ln.startswith("|"):
                break
            lines.append(ln)
    return "\n".join(lines)


t2b = block("| Scenario | Real or stand-in | Succeeded").replace(
    "$0.00 (free tier); paid-equivalent not measured", "$0 (free tier); paid: not measured")
out = (ROOT / "report/submission_template.md").read_text()
for key, val in (("{{T2A}}", block("| Scenario | Real or stand-in | Requests sent")), ("{{T2B}}", t2b),
                 ("{{T3}}", block("| Retrieval setting"))):
    out = out.replace(key, "\n" + val + "\n")
assert "{{" not in out
(ROOT / "report/submission.md").write_text(out)
print("wrote report/submission.md;", len(re.sub(r"^\|.*$", "", out, flags=re.M).split()), "prose words")
