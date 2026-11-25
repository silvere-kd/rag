from rag_portfolio.models import Section
from numpy import dot
from numpy.linalg import norm


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if v1 is None or v2 is None:
        raise ValueError("Empty vector")
    if norm(v1) == 0 or norm(v2) == 0:
        raise ValueError("all zero vector detected")
    return dot(v1, v2) / (norm(v1) * norm(v2))

def rank_sections(
        query_vector: list[float],
        sections: list[Section],
        passage_vectors: list[list[float]],
        k: int = 5,
) -> list[tuple[Section, float]]:
    if k <= 0:
        raise ValueError("k must be positive")
    # No None guard: an empty corpus is [], and returning [] for None would hide
    # a wiring mistake behind a result that looks like "nothing matched".
    if len(sections) != len(passage_vectors):
        raise ValueError(f"Sections count {len(sections)} do not match the embeddings vectors {len(passage_vectors)}")

    # Checked here rather than left to numpy, which reports it as
    # "shapes (2,) and (3,) not aligned" and says nothing about the corpus.
    dimensions = {len(vector) for vector in passage_vectors} | {len(query_vector)}
    if len(dimensions) > 1:
        raise ValueError(f"vector dimensions disagree: {sorted(dimensions)}")

    similarities : list = []
    top_k = min(k, len(sections))
    for passage, section in zip(passage_vectors, sections):
        similarities.append(
            (section, cosine_similarity(query_vector, passage))
        )


    return sorted(similarities, key=lambda sim: -sim[1])[:top_k]
