"""Measure how the corpus sections fit each candidate embedding model's budget.

Needs the tokenizer-check extra, which is deliberately not a runtime dependency:

    uv pip install -e ".[tokenizer-check]"
    PYTHONPATH=src python scripts/measure_tokenizer_fit.py

Tokenizer files only. No model weights are downloaded and no inference runs.

measure_fit itself lives in rag_portfolio.measure, not here, so the tests can
exercise it with a stand-in counter and no download.
"""
import json
import sys
from pathlib import Path

from rag_portfolio.config import CONSTITUTION_DIR
from rag_portfolio.ingest import ingest_pdf
from rag_portfolio.measure import estimate_embedding_fit, measure_fit

CORPUS = CONSTITUTION_DIR

# Revisions are pinned because from_pretrained tracks main otherwise, and a
# tokenizer update upstream would silently change these counts. The PDF is
# pinned by sha256 for the same reason.
CANDIDATES = (
    {
        "model": "BAAI/bge-m3",
        "revision": "5617a9f61b028005a4858fdac845db406aefb181",
        # Deliberately empty: bge-m3's card specifies no query/passage prefix.
        "prefix": "",
        # The card's limit. The Ollama host truncates at 2048 regardless, and
        # options.num_ctx does not raise it, so that is the number that decides
        # whether a passage is actually safe to send.
        "max_tokens": 8192,
        "serving_max_tokens": 2048,
    },
    {
        "model": "intfloat/multilingual-e5-small",
        "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        # E5 requires this prefix on passages; omitting it degrades retrieval.
        "prefix": "passage: ",
        "max_tokens": 512,
        "serving_max_tokens": None,
    },
)

# The ratio the cheap screen assumed, so the two can be compared.
ASSUMED_CHARS_PER_TOKEN = 3.5


def load_counter(model: str, revision: str):
    """A count_tokens function bound to one pinned tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)

    def count_tokens(passage: str) -> int:
        encoded = tokenizer(passage, add_special_tokens=True, truncation=False)
        return len(encoded["input_ids"])

    return count_tokens


def report(candidate: dict, sections: list) -> None:
    try:
        counter = load_counter(candidate["model"], candidate["revision"])
    except Exception as exc:
        print(f"{candidate['model']}: could not load its tokenizer")
        print(f"  {type(exc).__name__}: {exc}")
        print(f"  cache: {Path.home() / '.cache' / 'huggingface'}")
        return

    measured = measure_fit(
        sections,
        counter,
        candidate["prefix"],
        candidate["max_tokens"],
        candidate["model"],
        candidate["revision"],
    )
    estimated = estimate_embedding_fit(
        sections, candidate["max_tokens"], ASSUMED_CHARS_PER_TOKEN
    )

    print(f"{candidate['model']}")
    print(f"  revision {candidate['revision']}")
    print(f"  limit {candidate['max_tokens']} tokens, prefix {candidate['prefix']!r}")
    serving = candidate.get("serving_max_tokens")
    if serving:
        print(f"  serving ceiling {serving} tokens (Ollama truncates above this)")
        # Model-safe passages can still exceed the serving limit.
        serving_fit = measure_fit(
            sections,
            counter,
            candidate["prefix"],
            serving,
            candidate["model"],
            candidate["revision"],
        )
        print(f"  over the serving ceiling: {serving_fit['over_limit_count']}")
    print(f"  tokens  {json.dumps(measured['tokens'])}")
    print(f"  chars   {json.dumps(measured['chars'])}")
    print(
        f"  implied chars/token {measured['implied_chars_per_token']:.2f}"
        f"  (the screen assumed {ASSUMED_CHARS_PER_TOKEN})"
    )
    print(
        f"  over the limit: {measured['over_limit_count']} measured,"
        f" {estimated['over_limit_count']} estimated"
    )
    for row in measured["over_limit"]:
        print(
            f"    {row['text_id']:<26} {str(row['kind']):<9}"
            f" n={str(row['number']):<6} {row['tokens']:>5} tok"
            f" {row['chars']:>5} chars  pages={row['page_numbers']}"
        )
    print()


def main() -> None:
    try:
        import transformers  # noqa: F401
    except ImportError:
        sys.exit(
            "transformers is missing. Install it with:\n"
            '  uv pip install -e ".[tokenizer-check]"'
        )

    sections = ingest_pdf(CORPUS)
    print(f"{len(sections)} sections from {CORPUS.name}")
    print("title_context is metadata here and is NOT part of the embedded input;")
    print("including it later would add to every count below.\n")

    for candidate in CANDIDATES:
        report(candidate, sections)


if __name__ == "__main__":
    main()
