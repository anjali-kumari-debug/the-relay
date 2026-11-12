from relay.retrieval import (EVIDENCE_THRESHOLD, Index, best_score, get_index, search_papers,
                             tokenize)
from relay.types import SETTINGS


def test_tokenize_keeps_numbers_and_hyphens():
    t = tokenize("BLEU 28.4 on EN-DE, the Moshi's full-duplex model?")
    assert "28.4" in t and "en-de" in t and "full-duplex" in t and "duplex" in t
    assert "the" not in t


def test_get_index_cached():
    assert get_index("B") is get_index("B")
    assert isinstance(get_index("B"), Index)


def test_bleu_query_retrieves_attention_on_B():
    hits = search_papers("BLEU 28.4 EN-DE", "B")
    assert len(hits) == SETTINGS["B"].keep_k
    assert hits[0].passage.paper_id == "attention"
    assert any("28.4" in h.passage.text for h in hits)


def test_k_and_paper_filter():
    hits = search_papers("dynamic chunking routing module", "A", papers=("hnet",), k=2)
    assert len(hits) == 2 and all(h.passage.paper_id == "hnet" for h in hits)
    assert len(search_papers("transformer", "A", k=6)) == 6


def test_default_k_per_setting():
    for name, s in SETTINGS.items():
        assert len(search_papers("inner monologue Moshi text tokens", name)) == s.keep_k


def test_setting_C_reranks_and_returns_keep_k():
    hits = search_papers("What is the inner monologue in Moshi?", "C", papers=("moshi",))
    assert len(hits) == 4
    assert hits[0].passage.paper_id == "moshi"
    assert any("monologue" in h.passage.text.lower() for h in hits)


def test_scores_normalized_0_1():
    for q in ("BLEU 28.4 EN-DE", "xyzzy foo", "mimi frame rate"):
        for name in "ABC":
            assert all(0.0 <= h.score <= 1.0 for h in search_papers(q, name))


def test_real_questions_clear_threshold_nonsense_does_not():
    real = ["How many layers does the Transformer encoder stack have?",
            "What is the frame rate of the Mimi codec?",
            "What is inner monologue in Moshi?",
            "How does the H-Net dynamic chunking routing module work?",
            "BLEU 28.4 EN-DE",
            "Why did the authors choose sinusoidal positional encodings?"]
    junk = ["xyzzy foo bar qux", "How tall is Mount Everest?", "how do I train a dog to sit",
            "what is the stock price of Nvidia?"]
    for name in "ABC":
        assert min(best_score(search_papers(q, name)) for q in real) >= EVIDENCE_THRESHOLD
        assert max(best_score(search_papers(q, name)) for q in junk) < EVIDENCE_THRESHOLD


def test_empty_query():
    assert search_papers("", "B") == [] or best_score(search_papers("", "B")) == 0.0
    assert best_score([]) == 0.0
