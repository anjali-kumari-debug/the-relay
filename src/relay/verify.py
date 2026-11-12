"""Local citation + number verification of a drafted answer against retrieved passages."""
from __future__ import annotations

import re

from relay.types import CITATION_RE, REFUSAL_PREFIX, Hit, VerifyResult, parse_citations

_NUM_RE = re.compile(r"(?<![A-Za-z0-9.])\d+(?:[.,]\d+)*")
_LIST_MARKER_RE = re.compile(r"(?m)^\s*(?:[-*•]\s*)?\d+[.)]\s+")


def _norm_num(tok: str) -> str:
    return tok.replace(",", "").rstrip(".")


def _passage_numbers(text: str) -> set[str]:
    # looser than the answer side: digits glued to letters still count ("160ms")
    return {_norm_num(m) for m in re.findall(r"\d+(?:[.,]\d+)*", text)}


def _answer_numbers(answer: str) -> list[str]:
    """Numbers worth checking: >=2 digits or a decimal; citations and list markers excluded."""
    body = CITATION_RE.sub(" ", answer)
    body = _LIST_MARKER_RE.sub(" ", body)
    out: list[str] = []
    for tok in _NUM_RE.findall(body):
        n = _norm_num(tok)
        if "." in n or sum(c.isdigit() for c in n) >= 2:
            out.append(n)
    return list(dict.fromkeys(out))


def verify(answer: str, hits: list[Hit]) -> VerifyResult:
    text = answer.strip()
    if text.startswith(REFUSAL_PREFIX):
        return VerifyResult(ok=True, is_refusal=True, citations=(), problems=())

    cites = tuple(parse_citations(text))
    problems: list[str] = []
    if not cites:
        problems.append("no citation found")

    cited_text: list[str] = []
    for paper, page in cites:
        # a chunk starting on page P may run onto P+1, so cited page may be P or P+1
        matched = [h.passage for h in hits
                   if h.passage.paper_id == paper and h.passage.page in (page, page - 1)]
        if not matched:
            problems.append(f"citation [{paper} p.{page}] does not match any retrieved passage")
        cited_text.extend(p.text for p in matched)

    if cited_text:
        supported = set().union(*(_passage_numbers(t) for t in cited_text))
        for n in _answer_numbers(text):
            if n not in supported:
                problems.append(f"number {n} not found in any cited passage")

    return VerifyResult(ok=not problems, is_refusal=False, citations=cites, problems=tuple(problems))
