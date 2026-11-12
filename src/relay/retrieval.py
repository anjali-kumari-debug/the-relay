"""BM25 retrieval over page-aware chunks; the pipeline's `search_papers` tool.

Hit.score semantics (important for the evidence gate in pipeline.py)
-------------------------------------------------------------------
`Hit.score` is a NORMALIZED relevance in [0, 1], comparable across queries and settings:
the idf-weighted fraction of the query's distinct content terms that occur in the passage
(terms absent from the whole corpus carry maximum idf, so nonsense / off-topic queries score
low). Setting C blends in bigram matches, section-heading overlap and a paper prior.
Ranking order is BM25 (A, B) or the rerank score (C); the raw BM25 value is available via
`Index.raw_search`. Gate on `best_score(hits)`; recommended threshold: see EVIDENCE_THRESHOLD.
"""
from __future__ import annotations

import re
from collections import Counter
from math import log

from rank_bm25 import BM25Okapi

from relay.ingest import build_chunks
from relay.types import Hit, Passage, SETTINGS

STOPWORDS = frozenset("""a an and are as at be been by do does did for from has have how i in is it its of on or
that the their this to was were what when where which who whom why with about can could would should into than then
there these those they we you your our also but not so such if any some more most many much used use using paper
papers why authors author choose chose get got give tell explain describe""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-.'][a-z0-9]+)*")

# Recommended gate on best_score(hits). Observed on the real papers over 19 answerable questions:
# best score 0.57-1.00 on A, B and C (A min 0.57, B/C min 0.57). Clearly unrelated/nonsense queries:
# 0.00-0.45 ('xyzzy foo bar qux' 0.00, 'tall is Mount Everest' 0.00, 'train a dog' 0.13,
# 'stock price of Nvidia' 0.29, 'recipe for lasagna' 0.42, 'capital of France' 0.45; these last two
# match words that genuinely occur in Moshi's appendix transcripts). Caveat: ML-adjacent
# out-of-library questions (GPT-4 params 0.51-0.67, Llama 3 context 0.71, BERT MLM 0.57-0.58) and
# 'weather in Paris' (0.60, Moshi appendix) overlap with answerable scores; a lexical gate cannot
# catch those, that is the model's NOT_IN_PAPERS job (and analyze.out_of_scope for weather/recipes).
# Sample is small (19 + 13 queries): treat 0.50 as a starting point to tune against the real question set.
#
# DECISION 2026-11-12 (orchestrator, from the real eval set data/questions.json): lowered 0.50 -> 0.30.
# At 0.50 the gate wrongly refused 3 of 15 answerable eval questions before any model call
# (q01 Adam betas 0.36-0.37: the paper writes beta as a Greek symbol; q09 ratio-loss alpha 0.33;
# q15 cross-paper normalization 0.43-0.44), and blocked a real-run question (outputs/archive/eval_attempt1).
# Off-topic questions are already caught by analyze.out_of_scope, and ML-adjacent unanswerables score
# high anyway (GPT-4 params 0.78). Cost of the change: junk that passes the gate spends ~1.5K tokens on a
# model refusal. The gate now only stops near-empty matches (xyzzy 0.00, dog 0.13, Nvidia stock 0.29); margin is thin (lowest answerable 0.33).
EVIDENCE_THRESHOLD = 0.30


def tokenize(text: str) -> list[str]:
    """Lowercase; keep numbers (28.4) and hyphenated terms (en-de), also emit hyphen parts."""
    out: list[str] = []
    for m in _TOKEN_RE.findall(text.lower().replace("’", "'")):
        m = m.strip(".-'")
        if not m:
            continue
        parts = re.split(r"[-']", m)
        if len(parts) > 1:
            out.append(m)
            out.extend(p for p in parts if p and p not in STOPWORDS and len(p) > 1)
        elif m not in STOPWORDS:
            out.append(m)
    return out


def _bigrams(toks: list[str]) -> set[tuple[str, str]]:
    return set(zip(toks, toks[1:]))


class Index:
    def __init__(self, chunks: list[Passage]):
        self.passages = chunks
        self._toks = [tokenize(c.text + " " + (c.section or "")) for c in chunks]
        self._sets = [set(t) for t in self._toks]
        self._bigrams = [_bigrams(t) for t in self._toks]
        self.bm25 = BM25Okapi(self._toks)
        n = len(chunks)
        df: Counter[str] = Counter()
        for s in self._sets:
            df.update(s)
        self._df = df
        self._n = n

    # ---- scoring helpers -------------------------------------------------
    def _idf(self, term: str) -> float:
        return log(1 + self._n / (self._df.get(term, 0) + 0.5))

    def _query_terms(self, query: str) -> list[str]:
        seen: dict[str, None] = {}
        for t in tokenize(query):
            seen.setdefault(t)
        return list(seen)

    def coverage(self, terms: list[str], i: int) -> float:
        if not terms:
            return 0.0
        tot = sum(self._idf(t) for t in terms)
        got = sum(self._idf(t) for t in terms if t in self._sets[i])
        return got / tot if tot else 0.0

    def raw_search(self, query: str, k: int, papers: tuple[str, ...] | None = None) -> list[tuple[int, float]]:
        """(passage index, raw BM25) best-first, optionally restricted to papers."""
        qt = tokenize(query)
        if not qt:
            return []
        scores = self.bm25.get_scores(qt)
        order = sorted(range(self._n), key=lambda i: -scores[i])
        if papers:
            order = [i for i in order if self.passages[i].paper_id in papers]
        return [(i, float(scores[i])) for i in order[:k]]

    # ---- public API ------------------------------------------------------
    def search(self, query: str, k: int, papers: tuple[str, ...] | None = None) -> list[Hit]:
        """Top-k by BM25; Hit.score is the normalized coverage score (see module docstring)."""
        terms = self._query_terms(query)
        return [Hit(self.passages[i], round(self.coverage(terms, i), 4))
                for i, _ in self.raw_search(query, k, papers)]

    def rerank(self, query: str, cands: list[tuple[int, float]], keep_k: int,
               prefer: tuple[str, ...] | None = None) -> list[Hit]:
        """Cheap lexical reranker: coverage + bigram matches + section overlap + small paper prior."""
        qtoks = tokenize(query)
        terms = self._query_terms(query)
        qbi = _bigrams(qtoks)
        best_raw = max((s for _, s in cands), default=0.0) or 1.0
        scored = []
        for i, raw in cands:
            p = self.passages[i]
            cov = self.coverage(terms, i)
            bi = len(qbi & self._bigrams[i]) / len(qbi) if qbi else 0.0
            sec = set(tokenize(p.section or ""))
            sec_ov = len(sec & set(terms)) / len(sec) if sec else 0.0
            prior = 1.0 if prefer and p.paper_id in prefer else 0.0
            score = 0.70 * cov + 0.15 * bi + 0.05 * sec_ov + 0.05 * prior + 0.05 * max(raw, 0) / best_raw
            scored.append((score, cov, i))
        scored.sort(key=lambda x: -x[0])
        # Hit.score stays on the comparable coverage scale (priors must not inflate the gate).
        return [Hit(self.passages[i], round(cov, 4)) for _, cov, i in scored[:keep_k]]


_INDEX_CACHE: dict[str, Index] = {}


def get_index(setting_name: str) -> Index:
    if setting_name not in _INDEX_CACHE:
        _INDEX_CACHE[setting_name] = Index(build_chunks(SETTINGS[setting_name]))
    return _INDEX_CACHE[setting_name]


def search_papers(query: str, setting_name: str, papers: tuple[str, ...] | None = None,
                  k: int | None = None) -> list[Hit]:
    """The pipeline's retrieval tool call.

    A/B: BM25 top-k (k defaults to keep_k), `papers` is a hard filter.
    C: BM25 top max(fetch_k, 2k) over ALL papers, lexical rerank down to k; `papers` acts as a
       small prior (not a filter), so a wrong analyzer guess does not hide the right paper.
    """
    s = SETTINGS[setting_name]
    idx = get_index(setting_name)
    k = k or s.keep_k
    if s.rerank:
        cands = idx.raw_search(query, max(s.fetch_k, 2 * k), None)
        return idx.rerank(query, cands, k, prefer=papers)
    return idx.search(query, max(k, 1), papers)


def best_score(hits: list[Hit]) -> float:
    """Evidence-gate input: best normalized score (0.0 when no hits)."""
    return max((h.score for h in hits), default=0.0)
