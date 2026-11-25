"""Scoring and baseline writing, with no model and no corpus."""
import json

import pytest

from rag_portfolio.answer import (
    GenerationFormatError,
    UngroundedCitationError,
    UnknownCitationError,
)
from rag_portfolio.evaluate import (
    OUTCOMES,
    classify,
    gold_ranks,
    normalise_article,
    score_case,
    write_baseline,
)
from rag_portfolio.generate import GenerationError, GenerationTransportError
from rag_portfolio.models import Citation, Section, Source

SOURCE = Source("constitution", "edition-2015-01-01", "1 janvier 2015",
                "constitution.pdf", "0" * 64)


def section(text_id, number, text="Le texte."):
    return Section(text_id, "article", number, text, [2], None, SOURCE)


def case(answerable=True, elements=("le francais",), gold=(("constitution-1958", "2"),)):
    return {
        "id": "SM-XX",
        "question": "Une question ?",
        "answerable": answerable,
        "expected_elements": list(elements),
        "gold_evidence": [{"text_id": t, "article": a, "pages": [2]} for t, a in gold],
    }


def answered(text, citations=()):
    return {"status": "answered", "answer": text, "attempts": 1,
            "citations": list(citations)}


def abstained():
    return {"status": "insufficient_evidence", "answer": "Impossible.",
            "attempts": 1, "citations": []}


def test_article_aliases_are_normalised():
    # The PDF prints "ARTICLE PREMIER"; the gold rows use digits.
    assert normalise_article("PREMIER") == "1"
    assert normalise_article("1er") == "1"
    assert normalise_article("I") == "1"
    assert normalise_article("66-1") == "66-1"
    assert normalise_article(None) == ""


def test_gold_rank_is_found_or_reported_missing():
    hits = [(section("declaration-1789", None), 0.7),
            (section("declaration-1789", "1"), 0.66)]

    found = gold_ranks(case(gold=(("declaration-1789", "1"),)), hits)
    assert found == [{"text_id": "declaration-1789", "article": "1", "rank": 2}]

    missing = gold_ranks(case(gold=(("charte-environnement-2004", "1"),)), hits)
    assert missing[0]["rank"] is None


def test_a_grounded_answer_is_distinguished_from_an_incomplete_one():
    assert classify(case(), answered("La langue est le francais."), None) == "grounded"
    assert classify(
        case(elements=("cinq ans", "majorite absolue")),
        answered("Elu pour cinq ans."), None
    ) == "answered_incomplete"


def test_abstention_is_scored_against_the_case_not_the_model():
    assert classify(case(answerable=False), abstained(), None) == "abstained_correctly"
    assert classify(case(answerable=True), abstained(), None) == "abstained_wrongly"


@pytest.mark.parametrize(
    "error, expected",
    [
        (UngroundedCitationError("S1", "cite inventee", "le vrai passage"),
         "ungrounded_citation"),
        (UnknownCitationError("S9", ["S1", "S2"]), "unknown_citation"),
        (GenerationFormatError("output is not JSON"), "format_failure"),
        (GenerationError("empty content from gemma4:12b (0 chars)"),
         "generation_failure"),
        (GenerationTransportError("could not reach"), "transport_failure"),
    ],
)
def test_each_failure_keeps_its_own_outcome(error, expected):
    assert classify(case(), None, error) == expected


def test_a_rejected_quote_is_not_an_abstention():
    # These score the same and mean opposite things: one never saw the answer,
    # the other saw it and would not vouch for the quote. SM-04 is the second.
    rejected = classify(case(), None,
                        UngroundedCitationError("S1", "cite", "passage"))
    declined = classify(case(answerable=True), abstained(), None)

    assert rejected != declined
    assert rejected == "ungrounded_citation"


def test_an_unclassified_failure_is_refused():
    # Silently bucketing an unknown failure would corrupt every count above it.
    with pytest.raises(ValueError, match="unclassified"):
        classify(case(), None, RuntimeError("something else"))


def test_a_scored_row_separates_retrieval_from_generation():
    gold_section = section("constitution-1958", "2", "La langue est le francais.")
    hits = [(gold_section, 0.6612), (section("constitution-1958", "87"), 0.55)]
    citation = Citation("S1", "La langue est le francais.", gold_section)

    row = score_case(case(), hits,
                     result=answered("La langue est le francais.", [citation]))

    assert row["outcome"] == "grounded"
    assert row["gold"][0]["rank"] == 1
    assert row["retrieved"][0]["score"] == 0.6612
    assert row["citations"][0]["article"] == "2"
    assert row["cited_matches_gold"] is True
    assert row["missing_elements"] == []


def test_a_failed_row_keeps_the_message_verbatim():
    hits = [(section("declaration-1789", "1"), 0.66)]
    error = UngroundedCitationError(
        "S1",
        "Les hommes naissent et demeurent libres et egaux en droits. "
        "Les distinctions sociaux ne peuvent etre fondees.",
        "ARTICLE  PREMIER \nLes hommes naissent et demeurent libres et egaux en "
        "droits. Les distinctions \nsociales ne peuvent etre fondees.",
    )

    row = score_case(case(gold=(("declaration-1789", "1"),)), hits, error=error)

    assert row["outcome"] == "ungrounded_citation"
    assert row["gold"][0]["rank"] == 1          # retrieval worked
    assert "status" not in row
    # The message truncates at 80 chars, which cut off before the disputed word
    # and left a row that recorded the rejection without the evidence.
    assert "sociaux" not in row["failure"]["message"]
    assert "sociaux" in row["failure"]["excerpt"]
    assert "sociales" in row["failure"]["passage"]


def test_an_unknown_label_records_what_was_supplied():
    hits = [(section("constitution-1958", "2"), 0.6)]

    row = score_case(case(), hits, error=UnknownCitationError("S9", ["S1", "S2"]))

    assert row["failure"]["source_id"] == "S9"
    assert row["failure"]["supplied"] == ["S1", "S2"]


def test_the_baseline_counts_only_outcomes_that_occurred(tmp_path):
    rows = [
        score_case(case(), [(section("constitution-1958", "2"), 0.6)],
                   result=answered("le francais")),
        score_case(case(answerable=False, elements=(), gold=()), [], result=abstained()),
    ]

    path = tmp_path / "baseline.json"
    digest = write_baseline(path, {"retrieval": {"k": 5}}, rows)
    payload = json.loads(path.read_text())

    assert payload["outcome_counts"] == {"grounded": 1, "abstained_correctly": 1}
    assert payload["case_count"] == 2
    assert payload["provenance"] == {"retrieval": {"k": 5}}
    assert len(digest) == 64


def test_the_baseline_is_byte_stable(tmp_path):
    rows = [score_case(case(), [(section("constitution-1958", "2"), 0.6)],
                       result=answered("le francais"))]
    first = write_baseline(tmp_path / "a.json", {"k": 5}, rows)
    second = write_baseline(tmp_path / "b.json", {"k": 5}, rows)

    # No timestamp inside: two identical runs must produce identical files, so a
    # diff between baselines means something changed in the system.
    assert first == second
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


def test_answering_an_unanswerable_question_is_its_own_outcome():
    # expected_elements is empty for an unanswerable case, so scoring by
    # elements alone marked a fabricated answer "grounded".
    fabricated = answered("Le SMIC est de 1800 euros.")

    assert classify(case(answerable=False, elements=()), fabricated, None) == (
        "answered_unanswerable"
    )


def test_every_outcome_name_is_reachable():
    # A name in OUTCOMES that classify can never return is dead vocabulary.
    produced = {
        classify(case(), answered("le francais"), None),
        classify(case(elements=("absent",)), answered("autre"), None),
        classify(case(answerable=False), abstained(), None),
        classify(case(answerable=True), abstained(), None),
        classify(case(answerable=False, elements=()), answered("invente"), None),
        classify(case(), None, UngroundedCitationError("S1", "c", "p")),
        classify(case(), None, UnknownCitationError("S9", ["S1"])),
        classify(case(), None, GenerationFormatError("x")),
        classify(case(), None, GenerationError("empty content")),
        classify(case(), None, GenerationTransportError("x")),
    }

    assert produced == set(OUTCOMES)
