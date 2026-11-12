"""PDF -> page-aware word-window chunks (Passages). Local only, no network."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import pymupdf

from relay.types import Passage, RetrievalSetting, SETTINGS

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "papers" / "manifest.json"
CACHE_DIR = ROOT / "outputs" / "cache"
CACHE_VERSION = 2  # bump when extraction/chunking logic changes

_HEADING_NUM = re.compile(r"^\d+(\.\d+)*\.?\s{1,3}[A-Z]")
_HEADING_APPX = re.compile(r"^(Appendix\s+[A-Z]\b|[A-F]\s{2}[A-Z][a-z]|[A-F]\.\d+\s+[A-Z])")
_REFS = re.compile(r"^(\d+\s+)?(References|REFERENCES|Bibliography)\s*$")
_PAGE_NUM = re.compile(r"^\s*\d{1,3}\s*$")


def load_manifest() -> list[dict]:
    return json.loads(MANIFEST.read_text())


def _paper(paper_id: str) -> dict:
    for m in load_manifest():
        if m["paper_id"] == paper_id:
            return m
    raise KeyError(paper_id)


@lru_cache(maxsize=None)
def _raw_pages(paper_id: str) -> tuple[str, ...]:
    """Raw per-page text (index 0 = page 1), reading order by position."""
    doc = pymupdf.open(ROOT / _paper(paper_id)["file"])
    try:
        return tuple(p.get_text("text", sort=True) for p in doc)
    finally:
        doc.close()


def page_count(paper_id: str) -> int:
    return len(_raw_pages(paper_id))


def page_text(paper_id: str, page: int) -> str:
    """Raw text of a 1-based PDF page (for graders)."""
    pages = _raw_pages(paper_id)
    if not 1 <= page <= len(pages):
        raise ValueError(f"{paper_id} has pages 1..{len(pages)}, got {page}")
    return pages[page - 1]


def _is_heading(line: str) -> bool:
    if len(line) > 80 or len(line.split()) > 12 or line.endswith("."):
        return False
    if _HEADING_NUM.match(line) or _HEADING_APPX.match(line):
        return True
    letters = [c for c in line if c.isalpha()]
    return len(letters) >= 4 and len(line.split()) <= 6 and line.upper() == line and not any(c.isdigit() for c in line)


def _repeated_lines(pages: tuple[str, ...]) -> set[str]:
    """Running headers/footers: short lines present on >=30% of pages."""
    if len(pages) < 4:
        return set()
    counts: dict[str, int] = {}
    for t in pages:
        for l in {l.strip() for l in t.splitlines() if l.strip()}:
            if len(l) < 100:
                counts[l] = counts.get(l, 0) + 1
    return {l for l, c in counts.items() if c >= 0.3 * len(pages)}


def paper_words(paper_id: str) -> list[tuple[str, int, str | None]]:
    """(word, page, section) for every kept word, references dropped."""
    pages = _raw_pages(paper_id)
    skip_lines = _repeated_lines(pages)
    out: list[tuple[str, int, str | None]] = []
    section: str | None = None
    in_refs = False
    for pno, text in enumerate(pages, start=1):
        text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)  # de-hyphenate line breaks
        for raw in text.splitlines():
            line = raw.strip()
            if not line or _PAGE_NUM.match(line) or line in skip_lines:
                continue
            if _REFS.match(line):
                in_refs = True
                section = "References"
                continue
            if in_refs:
                if _HEADING_APPX.match(line):
                    in_refs = False
                else:
                    continue
            if _is_heading(line):
                section = line
            out.extend((w, pno, section) for w in line.split())
    return out


def _chunk_paper(paper_id: str, setting: RetrievalSetting) -> list[Passage]:
    m = _paper(paper_id)
    words = paper_words(paper_id)
    step = max(1, setting.piece_words - setting.overlap_words)
    chunks: list[Passage] = []
    start = 0
    while start < len(words):
        win = words[start:start + setting.piece_words]
        page = win[0][1]
        chunks.append(Passage(
            chunk_id=f"{paper_id}:{setting.name}:p{page}:c{len(chunks)}",
            paper_id=paper_id, arxiv_id=m["arxiv_id"], version=m["version"],
            page=page, section=win[0][2],
            text=" ".join(w for w, _, _ in win), n_words=len(win),
        ))
        if start + setting.piece_words >= len(words):
            break
        start += step
    return chunks


def _fingerprint(setting: RetrievalSetting) -> str:
    shas = "".join(m["sha256"] for m in load_manifest())
    key = f"{CACHE_VERSION}|{setting}|{shas}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def build_chunks(setting: RetrievalSetting) -> list[Passage]:
    """All chunks for a setting, per paper (never spanning papers). Cached as JSONL."""
    cache = CACHE_DIR / f"chunks_{setting.name}.jsonl"
    fp = _fingerprint(setting)
    if cache.exists():
        with cache.open() as f:
            header = json.loads(f.readline())
            if header.get("fingerprint") == fp:
                return [Passage(**json.loads(l)) for l in f if l.strip()]
    chunks: list[Passage] = []
    for m in load_manifest():
        chunks.extend(_chunk_paper(m["paper_id"], setting))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    with tmp.open("w") as f:
        f.write(json.dumps({"fingerprint": fp}) + "\n")
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")
    tmp.replace(cache)
    return chunks


__all__ = ["build_chunks", "page_text", "page_count", "paper_words", "SETTINGS"]
