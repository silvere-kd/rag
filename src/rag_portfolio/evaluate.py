"""Scoring a question set, and saving the result as a record.

A saved baseline is evidence, not a test gate. Generation is not guaranteed
reproducible, so asserting on these outcomes inside the suite would flake;
comparing two baselines is a deliberate act, not a CI check.

The outcome vocabulary is the point of this module. Four of the six Constitution
cases have already produced three different failures, and a pass/fail column
would have flattened all of them into the same thing.
"""
import hashlib
import json
from pathlib import Path

from rag_portfolio.answer import (
    GenerationFormatError,
    UngroundedCitationError,
    UnknownCitationError,
)
from rag_portfolio.generate import GenerationError, GenerationTransportError

# Ordered worst-understood last, so a listing reads as a diagnosis.
OUTCOMES = (
    "grounded",              # answered, expected elements present, quotes verified
    "answered_incomplete",   # answered and verified, but an expected element is missing
    "abstained_correctly",   # unanswerable, and the model declined
    "abstained_wrongly",     # answerable, and the model declined anyway
    "answered_unanswerable", # answered a question the corpus cannot support
    "ungrounded_citation",   # retrieval right, answer right, quote not verbatim
    "unknown_citation",      # cited a label that was never supplied
    "format_failure",        # unparseable or contract-breaking after the repair
    "generation_failure",    # the model returned nothing usable
    "transport_failure",     # the host could not be reached
)

# The PDF prints "ARTICLE PREMIER"; the gold rows use digits.
_ARTICLE_ALIASES = {"premier": "1", "1er": "1", "i": "1"}


def normalise_article(value: str | None) -> str:
    return _ARTICLE_ALIASES.get((value or "").strip().lower(), (value or "").strip().lower())


def gold_ranks(case: dict, hits: list) -> list[dict]:
    """Where each gold passage landed, or that it was never retrieved.

    Scored separately from generation on purpose: "the system never saw the
    answer" and "the system saw it and would not vouch for its quote" are the
    same score and completely different diagnoses.
    """
    retrieved = [(s.text_id, normalise_article(s.number)) for s, _ in hits]
    rows = []
    for gold in case.get("gold_evidence", []):
        key = (gold["text_id"], normalise_article(gold["article"]))
        rows.append({
            "text_id": key[0],
            "article": key[1],
            "rank": retrieved.index(key) + 1 if key in retrieved else None,
        })
    return rows


def classify(case: dict, result: dict | None, error: Exception | None) -> str:
    """One outcome per case. Failures are told apart by their own exceptions."""
    if error is not None:
        if isinstance(error, GenerationTransportError):
            return "transport_failure"
        if isinstance(error, UngroundedCitationError):
            return "ungrounded_citation"
        if isinstance(error, UnknownCitationError):
            return "unknown_citation"
        if isinstance(error, GenerationFormatError):
            return "format_failure"
        # Checked last: GenerationTransportError is a subclass, and an empty
        # reply is neither a transport fault nor a malformed one. Without this,
        # one empty reply aborted the entire baseline run.
        if isinstance(error, GenerationError):
            return "generation_failure"
        raise ValueError(f"unclassified failure: {type(error).__name__}: {error}")

    if result["status"] == "insufficient_evidence":
        return "abstained_correctly" if not case["answerable"] else "abstained_wrongly"

    # Checked before the elements, which are empty for an unanswerable case and
    # would otherwise score a fabricated answer as grounded.
    if not case["answerable"]:
        return "answered_unanswerable"

    missing = [e for e in case.get("expected_elements", [])
               if e.casefold() not in result["answer"].casefold()]
    return "answered_incomplete" if missing else "grounded"


def score_case(case: dict, hits: list, result=None, error=None) -> dict:
    """One row of a baseline: what was retrieved, what came back, and the verdict."""
    row = {
        "id": case["id"],
        "question": case["question"],
        "answerable": case["answerable"],
        "outcome": classify(case, result, error),
        "gold": gold_ranks(case, hits),
        "retrieved": [
            {"text_id": s.text_id, "kind": s.kind, "number": s.number,
             "pages": s.page_numbers, "score": round(float(score), 4)}
            for s, score in hits
        ],
    }
    if error is not None:
        # The message truncates; these fields do not. A row holding only the
        # message recorded that a quote was rejected without the evidence for
        # it -- the cut landed before the disputed word.
        row["failure"] = {"type": type(error).__name__, "message": str(error)}
        for field in ("source_id", "excerpt", "passage", "supplied"):
            if hasattr(error, field):
                row["failure"][field] = getattr(error, field)
        return row

    row["status"] = result["status"]
    row["attempts"] = result["attempts"]
    row["answer"] = result["answer"]
    row["missing_elements"] = [
        e for e in case.get("expected_elements", [])
        if e.casefold() not in result["answer"].casefold()
    ]
    row["citations"] = [
        {"source_id": c.source_id, "text_id": c.section.text_id,
         "article": normalise_article(c.section.number),
         "pages": c.section.page_numbers, "excerpt": c.excerpt}
        for c in result["citations"]
    ]
    gold_keys = {(g["text_id"], g["article"]) for g in row["gold"]}
    cited_keys = {(c["text_id"], c["article"]) for c in row["citations"]}
    row["cited_matches_gold"] = bool(cited_keys) and cited_keys <= gold_keys
    return row


def write_baseline(path: Path, provenance: dict, rows: list[dict]) -> str:
    """Write the baseline and return its sha256.

    No timestamp inside: the filename carries the date, and leaving it out keeps
    two identical runs byte-identical, which is itself a signal worth having.
    """
    counts = {outcome: 0 for outcome in OUTCOMES}
    for row in rows:
        counts[row["outcome"]] += 1
    payload = {
        "provenance": provenance,
        "outcome_counts": {k: v for k, v in counts.items() if v},
        "case_count": len(rows),
        "cases": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()
