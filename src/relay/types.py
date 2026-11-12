"""Shared contract between modules. Change only with orchestrator approval (see AGENTS.md)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Optional, Protocol

PAPER_IDS = ("attention", "moshi", "hnet")

# One bracketed citation group. Canonical: [attention p.3]. Also tolerated, because the real model
# produced them: "p. 3", "pp. 3-4", several pages "[attention p.1, p.8]", several papers
# "[attention p.3; hnet p.8]", and gpt-oss's lenticular brackets 【attention p.3】.
CITATION_RE = re.compile(r"[\[【]\s*(?:attention|moshi|hnet)\b[^\]】\n]{0,80}[\]】]", re.IGNORECASE)
_CITE_INNER = re.compile(r"(attention|moshi|hnet)?[\s,;:]*\bp{1,2}\.?\s*(\d+)(?:\s*[-–]\s*(\d+))?", re.IGNORECASE)


def parse_citations(text: str) -> list[tuple[str, int]]:
    """All (paper_id, page) pairs cited in text, in order, de-duplicated. Page ranges expand."""
    out: list[tuple[str, int]] = []
    for m in CITATION_RE.finditer(text):
        paper = None
        for inner in _CITE_INNER.finditer(m.group(0)):
            paper = (inner.group(1) or paper or "").lower()
            if not paper:
                continue
            lo = int(inner.group(2))
            hi = int(inner.group(3)) if inner.group(3) else lo
            for pg in range(lo, min(hi, lo + 3) + 1):
                if (paper, pg) not in out:
                    out.append((paper, pg))
    return out
REFUSAL_PREFIX = "NOT_IN_PAPERS:"

Source = Literal["real", "stand-in"]
Status = Literal["ok", "rate_limited", "timeout", "error"]
Grade = Literal["correct", "wrong", "refused_correctly", "refused_wrongly", ""]

RESULTS_COLUMNS = [
    # brief columns, exact order
    "request_id", "scenario", "source", "start_time", "first_token_ms", "total_ms", "status",
    "retries", "tokens_in", "tokens_out", "retrieval_setting", "answer_grade",
    # extras
    "question_id", "queue_len_at_arrival", "wait_ms", "work_ms", "retry_wait_ms",
    "provider_ttft_ms", "released_ms", "outcome_detail",
]


@dataclass(frozen=True)
class Passage:
    chunk_id: str          # e.g. "attention:B:p5:c2"  (paper:setting:page:index)
    paper_id: str          # one of PAPER_IDS
    arxiv_id: str          # e.g. "1706.03762"
    version: str           # e.g. "v7"
    page: int              # 1-based PDF page where the chunk starts
    section: Optional[str] # best-effort section heading, may be None
    text: str
    n_words: int

    def cite(self) -> str:
        return f"[{self.paper_id} p.{self.page}]"


@dataclass(frozen=True)
class RetrievalSetting:
    name: str
    piece_words: int
    overlap_words: int
    fetch_k: int
    keep_k: int
    rerank: bool


SETTINGS: dict[str, RetrievalSetting] = {
    "A": RetrievalSetting("A", 250, 50, 3, 3, False),
    "B": RetrievalSetting("B", 450, 75, 4, 4, False),
    "C": RetrievalSetting("C", 450, 75, 8, 4, True),
}


def setting_label(s: RetrievalSetting) -> str:
    """Value written to results.csv retrieval_setting column."""
    return f"{s.name}:{s.piece_words}w/k{s.keep_k}/rerank={'yes' if s.rerank else 'no'}"


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float


@dataclass(frozen=True)
class Analysis:
    normalized: str
    papers: tuple[str, ...]     # likely target papers; empty = unknown, search all
    out_of_scope: bool          # clearly not about the library (e.g. "weather in Paris")
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    is_refusal: bool
    citations: tuple[tuple[str, int], ...]     # (paper_id, page) found in the answer
    problems: tuple[str, ...]                  # human-readable reasons when not ok


@dataclass
class ProviderResult:
    """One provider attempt (no retries inside)."""
    status: Status
    text: str = ""
    http_status: Optional[int] = None
    t_send: float = 0.0                    # time.monotonic() when request sent
    t_first_token: Optional[float] = None  # monotonic time of first *content* token
    t_end: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    reasoning_tokens: int = 0
    retry_after_s: Optional[float] = None
    limit_kind: Optional[str] = None       # "rpm"|"tpm"|"rpd"|"tpd"|None, parsed from 429 when possible
    headers: dict[str, str] = field(default_factory=dict)   # x-ratelimit-*, retry-after only
    provider_timing: dict[str, Any] = field(default_factory=dict)  # Groq x_groq/usage timings if present
    error_message: Optional[str] = None


TokenCallback = Callable[[str], None]


class ModelProvider(Protocol):
    source: Source

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        timeout_s: float,
        on_token: Optional[TokenCallback] = None,
        relay_headers: Optional[dict[str, str]] = None,  # X-Relay-Question-Id / X-Relay-Setting
    ) -> ProviderResult: ...


@dataclass
class ResultRow:
    request_id: str
    scenario: str
    source: Source
    start_time: str
    first_token_ms: Optional[float]
    total_ms: float
    status: Status
    retries: int
    tokens_in: int
    tokens_out: int
    retrieval_setting: str
    answer_grade: Grade = ""
    question_id: str = ""
    queue_len_at_arrival: int = 0
    wait_ms: float = 0.0
    work_ms: float = 0.0
    retry_wait_ms: float = 0.0
    provider_ttft_ms: Optional[float] = None
    released_ms: Optional[float] = None
    outcome_detail: str = ""


EventSink = Callable[[str, dict[str, Any]], Awaitable[None] | None]
