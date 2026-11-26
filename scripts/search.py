"""Ask the corpus a question, end to end.

    PYTHONPATH=src python scripts/search.py "Quelle est la langue de la R\u00e9publique ?"

Needs a running Ollama host with both models pulled. Nothing is persisted: this
is the demonstration, not the measured baseline.
"""
import sys
from pathlib import Path

from rag_portfolio.answer import AnswerError, answer_question
from rag_portfolio.config import CONSTITUTION_DIR, EMBED_MODEL, INDEX_FILENAME, TOP_K
from rag_portfolio.embed import embed_passages, model_digest
from rag_portfolio.generate import DEFAULT_MODEL, ollama_generator
from rag_portfolio.index import IndexMismatchError, load_index, save_index
from rag_portfolio.ingest import embedding_text, ingest_pdf
from rag_portfolio.ollama import OllamaError
from rag_portfolio.retrieve import rank_sections

CORPUS = CONSTITUTION_DIR
INDEX = CORPUS / INDEX_FILENAME
K = TOP_K
DEFAULT_QUESTION = "Quelle est la langue de la R\u00e9publique ?"


def embed_corpus(inputs: list[str], digest: str) -> list[list[float]]:
    """Embed every passage and save the index for next time."""
    vectors = embed_passages(inputs, model=EMBED_MODEL)["vectors"]
    fingerprint = save_index(INDEX, inputs, vectors, EMBED_MODEL, digest)
    print(f"index: wrote {INDEX.name} ({len(vectors)} vectors, {fingerprint[:12]})")
    return vectors


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or DEFAULT_QUESTION

    sections = ingest_pdf(CORPUS)
    digest = model_digest(EMBED_MODEL)
    print(f"{len(sections)} sections from {CORPUS.name}")
    print(f"embedding: {EMBED_MODEL} ({digest[:12]})")
    print(f"generation: {DEFAULT_MODEL}")

    # embedding_text, not section.text: the document title goes to the embedder
    # so a question naming a text can find its articles. The model still sees
    # only section.text, and citations are still checked against section.text.
    inputs = [embedding_text(section) for section in sections]

    try:
        passage_vectors = load_index(INDEX, inputs, EMBED_MODEL, digest)
        print(f"index: reused {INDEX.name}")
    except FileNotFoundError:
        # Absent is normal on a first run.
        passage_vectors = embed_corpus(inputs, digest)
    except IndexMismatchError as exc:
        # Stale is news. Rebuilding silently would hide a corpus or renderer
        # change; refusing outright would make the script useless.
        print(f"index: stale, rebuilding -- {exc}")
        passage_vectors = embed_corpus(inputs, digest)

    query_vector = embed_passages([question], model=EMBED_MODEL)["vectors"][0]
    hits = rank_sections(query_vector, sections, passage_vectors, k=K)

    print(f"\nQuestion : {question}\n")
    print(f"Passages retrouv\u00e9s (les {K} plus proches) :")
    for index, (section, score) in enumerate(hits, start=1):
        print(
            f"  [S{index}] {score:.3f}  {section.text_id}"
            f"  {section.kind} {section.number}  page(s) {section.page_numbers}"
        )

    try:
        result = answer_question(question, hits, ollama_generator())
    except (AnswerError, OllamaError) as exc:
        # Shown rather than swallowed: a fabricated citation or an unreachable
        # host must not be mistaken for the model declining to answer.
        print(f"\n\u00c9CHEC {type(exc).__name__}: {exc}")
        raise SystemExit(1)

    print(f"\nStatut : {result['status']}  (tentatives : {result['attempts']})")
    print(f"\nR\u00e9ponse : {result['answer']}")

    if not result["citations"]:
        print("\nAucune citation : rien n'est affirm\u00e9 sur la base de ces passages.")
        return

    print("\nCitations v\u00e9rifi\u00e9es dans le texte source :")
    for citation in result["citations"]:
        section = citation.section
        print(
            f"  {citation.source_id}  {section.text_id},"
            f" {section.kind} {section.number}, page(s) {section.page_numbers}"
        )
        print(
            f"        {section.source.filename}"
            f"  \u2014  \u00e9dition {section.source.edition_label}"
        )
        print(f"        \u00ab {citation.excerpt} \u00bb")


if __name__ == "__main__":
    main()
