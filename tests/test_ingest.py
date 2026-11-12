import json

from relay.ingest import build_chunks, load_manifest, page_count, page_text
from relay.types import PAPER_IDS, SETTINGS


def test_manifest_papers_present():
    assert {m["paper_id"] for m in load_manifest()} == set(PAPER_IDS)


def test_chunks_valid_pages_and_ids():
    for s in SETTINGS.values():
        chunks = build_chunks(s)
        assert chunks
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))
        for c in chunks:
            assert 1 <= c.page <= page_count(c.paper_id)
            assert c.chunk_id.startswith(f"{c.paper_id}:{s.name}:p{c.page}:c")
            assert 0 < c.n_words <= s.piece_words
            assert c.n_words == len(c.text.split())


def test_every_paper_chunked_and_never_spans_papers():
    chunks = build_chunks(SETTINGS["B"])
    assert {c.paper_id for c in chunks} == set(PAPER_IDS)
    # papers are chunked one after another: paper ids appear as contiguous runs, once each
    runs = [c.paper_id for i, c in enumerate(chunks) if i == 0 or chunks[i - 1].paper_id != c.paper_id]
    assert sorted(runs) == sorted(PAPER_IDS)
    # distinctive text of one paper never shows up in another paper's chunks
    assert not any("Moshi" in c.text and c.paper_id == "attention" for c in chunks)


def test_smaller_setting_gives_more_chunks_and_respects_window():
    a, b = build_chunks(SETTINGS["A"]), build_chunks(SETTINGS["B"])
    assert len(a) > len(b)
    assert max(c.n_words for c in a) <= 250


def test_overlap_between_consecutive_chunks():
    chunks = [c for c in build_chunks(SETTINGS["B"]) if c.paper_id == "attention"]
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-75:] == second[:75]


def test_attention_bleu_text_is_extracted():
    text = " ".join(c.text for c in build_chunks(SETTINGS["B"]) if c.paper_id == "attention")
    assert "28.4" in text and "BLEU" in text


def test_references_dropped():
    chunks = build_chunks(SETTINGS["B"])
    assert not any("arXiv preprint arXiv:1601.06733" in c.text for c in chunks)


def test_sections_detected_best_effort():
    chunks = build_chunks(SETTINGS["B"])
    secs = {c.section for c in chunks if c.section}
    assert len(secs) > 10
    assert any("Inner Monologue" in s for s in secs)


def test_page_text_helper():
    assert "Introduction" in page_text("attention", 2)
    assert "Moshi" in page_text("moshi", 1)


def test_page_text_out_of_range():
    import pytest
    with pytest.raises(ValueError):
        page_text("attention", 0)
    with pytest.raises(ValueError):
        page_text("attention", 999)


def test_cache_written_and_reused(tmp_path, monkeypatch):
    import relay.ingest as ing
    monkeypatch.setattr(ing, "CACHE_DIR", tmp_path)
    first = build_chunks(SETTINGS["A"])
    f = tmp_path / "chunks_A.jsonl"
    assert f.exists()
    lines = f.read_text().splitlines()
    assert len(lines) == len(first) + 1 and "fingerprint" in json.loads(lines[0])
    assert build_chunks(SETTINGS["A"]) == first
