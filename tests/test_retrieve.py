"""Ranking against hand-checkable vectors, with no model or HTTP calls."""
import pytest

from rag_portfolio.models import Section, Source
from rag_portfolio.retrieve import rank_sections


@pytest.fixture
def sections():
    source = Source("test", "v1", "test edition", "test.pdf", "0" * 64)
    return [
        Section("test-text", "article", str(number), f"Article {number}.", [number], None, source)
        for number in (3, 2, 4, 1)
    ]


@pytest.fixture
def passage_vectors():
    return [[0.0, 1.0], [1.0, 1.0], [-1.0, 0.0], [1.0, 0.0]]


def test_ranks_sections_by_cosine_similarity(sections, passage_vectors):
    hits = rank_sections([1.0, 0.0], sections, passage_vectors)

    assert [section.number for section, _ in hits] == ["1", "2", "3", "4"]
    assert [score for _, score in hits] == pytest.approx([1.0, 0.7071067811865476, 0.0, -1.0])
    assert all(
        section is sections[index]
        for (section, _), index in zip(hits, (3, 1, 0, 2), strict=True)
    )


@pytest.mark.parametrize(
    "k, expected",
    [(1, ["1"]), (2, ["1", "2"]), (10, ["1", "2", "3", "4"])],
)
def test_returns_up_to_k_sections(sections, passage_vectors, k, expected):
    hits = rank_sections([1.0, 0.0], sections, passage_vectors, k=k)

    assert [section.number for section, _ in hits] == expected


def test_ties_preserve_original_section_order(sections):
    hits = rank_sections([1.0, 0.0], sections[:2], [[1.0, 0.0], [2.0, 0.0]], k=1)

    assert len(hits) == 1
    assert hits[0][0] is sections[0]
    assert hits[0][1] == pytest.approx(1.0)


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_is_rejected(sections, passage_vectors, k):
    with pytest.raises(ValueError, match="k"):
        rank_sections([1.0, 0.0], sections, passage_vectors, k=k)


def test_section_vector_count_mismatch_is_rejected(sections, passage_vectors):
    with pytest.raises(ValueError):
        rank_sections([1.0, 0.0], sections, passage_vectors[:-1])


def test_vector_dimension_mismatch_is_rejected(sections):
    with pytest.raises(ValueError):
        rank_sections([1.0, 0.0], sections[:1], [[1.0, 0.0, 0.0]])


@pytest.mark.parametrize(
    "query, passage",
    [
        ([0.0, 0.0], [1.0, 0.0]),
        ([1.0, 0.0], [0.0, 0.0]),
        ([], [1.0, 0.0]),
        ([1.0, 0.0], []),
    ],
)
def test_zero_or_empty_vectors_are_rejected(sections, query, passage):
    with pytest.raises(ValueError):
        rank_sections(query, sections[:1], [passage])


def test_empty_corpus_returns_no_hits():
    assert rank_sections([1.0, 0.0], [], []) == []


def test_none_vectors_is_a_caller_error_not_an_empty_corpus(sections):
    # Returning [] here would hide a wiring mistake: two real sections went in
    # and nothing came back, with no error to explain it.
    with pytest.raises(TypeError):
        rank_sections([1.0, 0.0], sections, None)


def test_the_dimension_mismatch_message_names_the_sizes(sections):
    with pytest.raises(ValueError, match=r"\[2, 3\]"):
        rank_sections([1.0, 0.0], sections[:1], [[1.0, 0.0, 0.0]])
