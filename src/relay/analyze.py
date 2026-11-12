"""Local, deterministic query analysis: normalize, map keywords to papers, flag out-of-scope."""
from __future__ import annotations

import re

from relay.types import Analysis

# paper_id -> keyword phrases (lowercase, matched on word boundaries after normalization)
PAPER_KEYWORDS: dict[str, tuple[str, ...]] = {
    "moshi": ("moshi", "mimi", "helium", "inner monologue", "full-duplex", "full duplex", "speech",
              "spoken", "audio", "codec", "dialogue", "voice", "rvq", "depth transformer",
              "temporal transformer", "kyutai", "wavlm", "latency"),
    "hnet": ("h-net", "hnet", "dynamic chunking", "byte-level", "byte level", "chunking",
             "routing module", "dechunking", "tokenizer-free", "tokenization", "bytes",
             "smoothing module", "hierarchical"),
    "attention": ("attention is all you need", "multi-head", "multihead", "positional encoding",
                  "positional encodings", "wmt", "bleu", "scaled dot-product", "self-attention",
                  "encoder-decoder", "sinusoidal", "label smoothing", "newstest", "en-de", "en-fr",
                  "vaswani", "attention heads", "attention head"),
}
# Weak: "Transformer" appears in all three papers; only used when nothing else matched.
WEAK_KEYWORDS: dict[str, tuple[str, ...]] = {
    "attention": ("transformer", "transformers", "attention", "encoder", "decoder"),
}
LIBRARY_HINTS = ("paper", "papers", "model", "models", "neural", "network", "training", "trained",
                 "token", "tokens", "language model", "architecture", "layer", "layers", "benchmark",
                 "dataset", "ablation", "author", "authors", "table", "figure", "section", "encoder",
                 "decoder", "embedding", "loss", "parameters", "perplexity", "latency")
OFF_TOPIC = ("weather", "recipe", "recipes", "cook", "cooking", "stock price", "stocks", "bitcoin",
             "football", "soccer", "basketball", "movie", "movies", "restaurant", "restaurants",
             "horoscope", "lyrics", "flight", "flights", "hotel", "capital of", "mount everest",
             "who won", "election", "president of", "celebrity", "pizza", "lasagna", "joke",
             "jokes", "dog", "cat", "vacation", "travel", "temperature in", "news")


def normalize(q: str) -> str:
    q = q.replace("’", "'").replace("‐", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", q).strip()


def _has(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def analyze(question: str) -> Analysis:
    norm = normalize(question)
    low = norm.lower()
    keywords: list[str] = []
    papers: list[str] = []
    for pid, kws in PAPER_KEYWORDS.items():
        hit = [k for k in kws if _has(low, k)]
        if hit:
            papers.append(pid)
            keywords.extend(hit)
    if not papers:
        for pid, kws in WEAK_KEYWORDS.items():
            hit = [k for k in kws if _has(low, k)]
            if hit:
                papers.append(pid)
                keywords.extend(hit)
    # all three matched = no real narrowing information
    if len(papers) == 3:
        papers = []
    out_of_scope = (
        not papers
        and any(_has(low, k) for k in OFF_TOPIC)
        and not any(_has(low, k) for k in LIBRARY_HINTS)
    )
    return Analysis(normalized=norm, papers=tuple(papers), out_of_scope=out_of_scope,
                    keywords=tuple(dict.fromkeys(keywords)))
