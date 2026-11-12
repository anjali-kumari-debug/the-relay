"""Grade answers and write answer_grade into outputs/results.csv.

1. Auto-suggest a grade for every (question_id, setting) in outputs/answers_{A,B,C}.jsonl:
   - should_refuse question: refused -> refused_correctly ; answered -> wrong
   - answerable question:    refused -> refused_wrongly
                             answered -> correct if every key fact appears in the answer AND at least
                                         one citation lands on a gold page; otherwise wrong
   Key facts = q["key_facts"] if present, else the numbers in the first clause of expected_answer.
2. Human overrides in data/grades_manual.json: {"q05:B": "wrong", ...} always win.
   Review sheet with both columns: outputs/grades_review.csv
3. Apply to results.csv rows with status ok (blank for non-ok, per SPEC):
   - retrieval_eval rows: grade of their (question_id, setting)
   - load-scenario rows: refusals made without/after the model (outcome refused_*) are graded by
     should_refuse; answered rows inherit the (question_id, setting) grade, because the stand-in
     replays the recorded real answer for that pair. Rows with no graded pair stay blank.

Usage: .venv/bin/python scripts/grade.py [--results outputs/results.csv] [--questions data/questions.json]
"""
import argparse, csv, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUM = re.compile(r"\d+(?:[.,]\d+)?")


def key_facts(q: dict) -> list[str]:
    if q.get("key_facts"):
        return [str(k) for k in q["key_facts"]]
    first = re.split(r"(?<=[.;])\s|\(", q.get("expected_answer", ""), maxsplit=1)[0]
    return sorted(set(NUM.findall(first)))


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace(",", "").replace("−", "-")).lower()


def has_fact(ans: str, fact: str) -> bool:
    """Numbers must match as whole numbers (so '6' does not match '16'); words match as substrings."""
    f = norm(fact)
    if NUM.fullmatch(f):
        return re.search(rf"(?<![\d.]){re.escape(f)}(?![\d]|\.\d)", ans) is not None
    return f in ans


def gold_set(q: dict) -> set[tuple[str, int]]:
    out = set()
    for g in q.get("gold_pages", []):
        if isinstance(g, str) and ":" in g:
            pid, pg = g.split(":")
            out.add((pid, int(pg)))
        else:
            for pid in q.get("paper_ids", []):
                out.add((pid, int(g)))
    return out


def auto_grade(q: dict, rec: dict) -> tuple[str, str]:
    refused = rec["outcome"].startswith("refused")
    if q.get("should_refuse"):
        return ("refused_correctly", "") if refused else ("wrong", "answered an unanswerable question")
    if refused:
        return "refused_wrongly", rec["outcome"]
    ans = norm(rec["answer"])
    missing = [k for k in key_facts(q) if not has_fact(ans, k)]
    cited = {(p, int(n)) for p, n in rec.get("cited_pages", [])}
    gold = gold_set(q)
    on_gold = any((p, n) in gold or (p, n - 1) in gold or (p, n + 1) in gold for p, n in cited)
    if missing:
        return "wrong", f"missing {missing}"
    if not on_gold:
        return "wrong", f"no citation on gold pages {sorted(gold)}; cited {sorted(cited)}"
    return "correct", ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "outputs/results.csv"))
    ap.add_argument("--questions", default=str(ROOT / "data/questions.json"))
    ap.add_argument("--manual", default=str(ROOT / "data/grades_manual.json"))
    a = ap.parse_args()
    qs = {str(q["id"]): q for q in json.loads(Path(a.questions).read_text())}
    manual = json.loads(Path(a.manual).read_text()) if Path(a.manual).exists() else {}

    grades, review = {}, []
    for setting in "ABC":
        p = ROOT / f"outputs/answers_{setting}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            rec = json.loads(line)
            q = qs.get(rec["question_id"])
            if q is None:
                continue
            g, why = auto_grade(q, rec)
            key = f"{rec['question_id']}:{setting}"
            final = manual.get(key, g)
            grades[key] = final
            review.append({"key": key, "auto": g, "final": final, "why": why,
                           "answer": rec["answer"][:400], "expected": q.get("expected_answer", "")})
    with (ROOT / "outputs/grades_review.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "auto", "final", "why", "answer", "expected"])
        w.writeheader()
        w.writerows(review)

    rows = list(csv.DictReader(open(a.results)))
    fields = list(rows[0].keys()) if rows else []
    n = 0
    for r in rows:
        r["answer_grade"] = ""
        if r["status"] != "ok":
            continue
        setting = r["retrieval_setting"][:1]
        q = qs.get(r["question_id"])
        key = f"{r['question_id']}:{setting}"
        if r["scenario"] != "retrieval_eval" and r["outcome_detail"].startswith("refused") and q:
            r["answer_grade"] = "refused_correctly" if q.get("should_refuse") else "refused_wrongly"
        elif key in grades:
            r["answer_grade"] = grades[key]
        n += bool(r["answer_grade"])
    with open(a.results, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    counts = {}
    for g in grades.values():
        counts[g] = counts.get(g, 0) + 1
    print(f"graded pairs: {counts}; results rows graded: {n}; review sheet: outputs/grades_review.csv")


if __name__ == "__main__":
    main()
