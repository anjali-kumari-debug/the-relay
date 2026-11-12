from relay.types import Hit, Passage, REFUSAL_PREFIX
from relay.verify import verify


def P(paper, page, text):
    return Hit(Passage(f"{paper}:B:p{page}:c0", paper, "x", "v1", page, None, text, len(text.split())), 1.0)


HITS = [
    P("attention", 8, "The big model achieves 28.4 BLEU on WMT 2014 English-to-German, trained for 3.5 days."),
    P("moshi", 12, "Mimi runs at a frame rate of 12.5 Hz with 160 ms latency, 24 kHz audio."),
]


def test_refusal_ok():
    r = verify("  NOT_IN_PAPERS: the library does not cover this.", HITS)
    assert r.ok and r.is_refusal and r.citations == () and r.problems == ()


def test_valid_answer():
    r = verify("The big Transformer reaches 28.4 BLEU [attention p.8].", HITS)
    assert r.ok and not r.is_refusal
    assert r.citations == (("attention", 8),)


def test_no_citation():
    r = verify("It reaches good BLEU.", HITS)
    assert not r.ok and any("no citation" in p for p in r.problems)


def test_fabricated_citation_page():
    r = verify("It reaches 28.4 BLEU [attention p.5].", HITS)
    assert not r.ok and any("p.5" in p for p in r.problems)


def test_wrong_paper_citation():
    r = verify("It reaches 28.4 BLEU [moshi p.8].", HITS)
    assert not r.ok


def test_next_page_allowed_for_cross_page_chunk():
    assert verify("Mimi uses 12.5 Hz [moshi p.13].", HITS).ok
    assert not verify("Mimi uses 12.5 Hz [moshi p.14].", HITS).ok


def test_unsupported_number():
    r = verify("It reaches 41.8 BLEU [attention p.8].", HITS)
    assert not r.ok and any("41.8" in p for p in r.problems)


def test_number_must_be_in_cited_passage_not_just_any_hit():
    r = verify("Mimi has 28.4 Hz [moshi p.12].", HITS)
    assert not r.ok and any("28.4" in p for p in r.problems)


def test_small_integers_list_markers_and_citation_pages_ignored():
    ans = "1. It has 8 heads [attention p.8].\n2. Another point [attention p.8]."
    assert verify(ans, HITS).ok


def test_number_formats():
    assert verify("Latency is 160ms [moshi p.12] at 24 kHz [moshi p.12].", HITS).ok
    assert verify("Trained 3.5 days [attention p.8].", HITS).ok


def test_case_insensitive_and_pp_citation():
    assert verify("28.4 BLEU [Attention pp. 8].", HITS).ok


def test_multiple_citations_numbers_from_any_cited():
    ans = "BLEU 28.4 [attention p.8]; Mimi 160 ms [moshi p.12]."
    r = verify(ans, HITS)
    assert r.ok and len(r.citations) == 2


def test_refusal_prefix_constant():
    assert verify(REFUSAL_PREFIX + " x", []).is_refusal


def test_parse_citations_tolerates_real_model_formats():
    from relay.types import parse_citations
    assert parse_citations("x [attention p.3] y") == [("attention", 3)]
    assert parse_citations("x [attention p.1, p.8]") == [("attention", 1), ("attention", 8)]
    assert parse_citations("x [attention p.3; hnet p.8]") == [("attention", 3), ("hnet", 8)]
    assert parse_citations("x 【moshi p.26】") == [("moshi", 26)]
    assert parse_citations("x [hnet pp. 3-4]") == [("hnet", 3), ("hnet", 4)]
    assert parse_citations("see [1] and [Table 2]") == []
