import pytest

from relay.analyze import analyze


@pytest.mark.parametrize("q,expected", [
    ("What is the inner monologue in Moshi?", ("moshi",)),
    ("How does Mimi encode audio?", ("moshi",)),
    ("What is dynamic chunking in H-Net?", ("hnet",)),
    ("Explain byte-level modeling", ("hnet",)),
    ("What BLEU did the model get on WMT 2014?", ("attention",)),
    ("Why multi-head attention?", ("attention",)),
    ("How many layers does the Transformer have?", ("attention",)),
])
def test_paper_mapping(q, expected):
    a = analyze(q)
    assert a.papers == expected and not a.out_of_scope


def test_multiple_papers():
    a = analyze("Compare Moshi's Mimi codec with H-Net dynamic chunking")
    assert set(a.papers) == {"moshi", "hnet"}


def test_transformer_is_weak_signal():
    assert analyze("H-Net's Transformer layers").papers == ("hnet",)


def test_unsure_returns_empty():
    a = analyze("What is the main contribution?")
    assert a.papers == () and not a.out_of_scope


def test_all_three_means_unknown():
    assert analyze("Moshi, H-Net and attention is all you need").papers == ()


@pytest.mark.parametrize("q", ["What's the weather in Paris?", "Give me a recipe for lasagna",
                               "Who won the football game?"])
def test_out_of_scope(q):
    assert analyze(q).out_of_scope


def test_conservative_out_of_scope():
    assert not analyze("What is the temperature parameter used in training the model?").out_of_scope
    assert not analyze("What's the weather in the Moshi paper dialogues?").out_of_scope
    assert not analyze("Something totally vague").out_of_scope


def test_normalization():
    a = analyze("  What  is\n  Moshi’s   latency? ")
    assert a.normalized == "What is Moshi's latency?"
    assert "moshi" in a.keywords
