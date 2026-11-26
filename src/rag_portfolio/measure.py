"""Does the corpus fit an embedding model's budget?

Separate from ingest.py because this answers a question about a model, not
about the PDF. ingest.py turns pages into passages; this one counts tokens and
reports where they land against a limit, and the two change for unrelated
reasons.

It measures embedding_text(section), not section.text: the document title rides
along in every request, so a count of the bare passage understates it.
"""
import math

from rag_portfolio.ingest import embedding_text
from rag_portfolio.models import Section


def length_distribution(lengths: list[int]) -> dict | None:
    """Nearest-rank percentiles, or None when there is nothing to describe."""
    if not lengths:
        return None

    def pick(percent: int) -> int:
        # Nearest rank: the 1-based position is ceil(n * p / 100), so subtract one
        # for the index. int() instead of ceil() lands one too high whenever
        # n * p / 100 is a whole number.
        return lengths[math.ceil(len(lengths) * percent / 100) - 1]

    return {
        "min": lengths[0],
        "p50": pick(50),
        "p90": pick(90),
        "p99": pick(99),
        "max": lengths[-1],
    }


def estimate_embedding_fit(sections: list[Section],
                           max_tokens: int,
                           chars_per_token: float,
                           ) -> dict:
    """Which sections might exceed an embedding input budget, by character ratio.

    An estimate, not a measurement. The chosen model's own tokenizer still has to
    verify the complete input, including any required query/passage prefix and
    special tokens, and a different tokenizer can produce more tokens rather than
    fewer. Use this to shortlist models and plan splitting, not to claim a limit.
    """
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    if chars_per_token <= 0:
        raise ValueError(f"chars_per_token must be positive, got {chars_per_token}")

    # Rendered like measure_fit, so the script's "N measured, M estimated"
    # comparison is between the same inputs.
    rendered = {id(section): embedding_text(section) for section in sections}
    estimates = [
        (section, math.ceil(len(rendered[id(section)]) / chars_per_token))
        for section in sections
    ]
    over = [
        {
            "text_id": section.text_id,
            "kind": section.kind,
            "number": section.number,
            "page_numbers": section.page_numbers,
            "chars": len(rendered[id(section)]),
            "estimated_tokens": tokens,
        }
        for section, tokens in estimates
        if tokens > max_tokens
    ]
    return {
        "method": "character-ratio estimate; verify with the model's own tokenizer",
        "section_count": len(sections),
        "max_tokens": max_tokens,
        "assumed_chars_per_token": chars_per_token,
        "chars": length_distribution(sorted(len(rendered[id(s)]) for s in sections)),
        "over_limit_count": len(over),
        "over_limit": sorted(over, key=lambda row: -row["chars"]),
    }


def measure_fit(
    sections: list[Section],
    count_tokens,
    prefix: str,
    max_tokens: int,
    model: str = "",
    revision: str = "",
) -> dict:
    """Which sections exceed an embedding input budget, by actual token count.

    `count_tokens` takes the finished passage string and returns its token count.
    It is injected rather than being a tokenizer object so this function stays
    testable without a download: the script wires a real tokenizer into it.

    The report mirrors estimate_embedding_fit so the two can be compared directly.
    """
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")

    # Budgeted against embedding_text, not section.text: the title rides along
    # in every request, so a measurement of the bare passage understates it.
    rendered = {id(section): prefix + embedding_text(section) for section in sections}
    measured = [(section, count_tokens(rendered[id(section)])) for section in sections]
    over = [
        {
            "text_id": section.text_id,
            "kind": section.kind,
            "number": section.number,
            "page_numbers": section.page_numbers,
            "chars": len(rendered[id(section)]),
            "tokens": tokens,
        }
        for section, tokens in measured
        if tokens > max_tokens
    ]
    total_chars = sum(len(rendered[id(s)]) for s in sections)
    total_tokens = sum(tokens for _, tokens in measured)
    return {
        "method": "measured with the model's own tokenizer",
        "model": model,
        "revision": revision,
        "prefix": prefix,
        "section_count": len(sections),
        "max_tokens": max_tokens,
        # Feed this back into estimate_embedding_fit to calibrate the cheap screen.
        "implied_chars_per_token": (
            total_chars / total_tokens if total_tokens else None
        ),
        "chars": length_distribution(sorted(len(rendered[id(s)]) for s in sections)),
        "tokens": length_distribution(sorted(t for _, t in measured)),
        "over_limit_count": len(over),
        "over_limit": sorted(over, key=lambda row: -row["tokens"]),
    }
