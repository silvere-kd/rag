"""Embedding through Ollama's HTTP API, with truncation refused.

Nothing here imports torch or transformers. The model is served by Ollama, so a
clean clone needs no Python ML stack, only a running Ollama host.
"""
import math

from rag_portfolio.ollama import EMBED_URL, TAGS_URL, OllamaError, request

# Measured against bge-m3 on 2026-11-17: /api/embed truncates at 2048 tokens,
# not the model card's 8192, and options.num_ctx does not raise it. Record this
# in any baseline; it is a property of the serving setup, not of the model.
SERVING_MAX_TOKENS = 2048



class EmbeddingError(OllamaError):
    """Provider-neutral embedding failure."""


def validate_vectors(vectors, error, count=None, width=None) -> int:
    """Check a batch of vectors and return their single width.

    Shared with the saved index, deliberately: a cached vector reaches cosine
    similarity by the same path as a freshly embedded one, so it has to clear
    the same bar. `error` is the class to raise, so each caller reports in its
    own vocabulary.

    A NaN or an infinity here poisons every score it touches, and len() of a
    string passes a width check, so neither is caught by shape alone.
    """
    if not isinstance(vectors, list) or not vectors:
        raise error("expected a non-empty list of vectors")
    if count is not None and len(vectors) != count:
        raise error(f"asked for {count} vectors, received {len(vectors)}")
    for vector in vectors:
        if not isinstance(vector, list) or not vector:
            raise error("each vector must be a non-empty list")
        if any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in vector
        ):
            raise error("vector values must be finite numbers")
    widths = {len(vector) for vector in vectors}
    if len(widths) != 1:
        raise error(f"ragged vectors, widths {sorted(widths)}")
    found = widths.pop()
    if width is not None and found != width:
        raise error(f"vectors are {found} wide, expected {width}")
    return found


def embed_passages(
    texts: list[str],
    model: str = "bge-m3",
    post=None,
    timeout: float = 120.0,
) -> dict:
    """Embed a batch, refusing truncation.

    `truncate` is deliberately not a parameter. A truncated vector indexes only
    the opening of its passage and is indistinguishable from a complete one, so
    the only safe behaviour is to fail. Splitting an oversized passage is a
    decision made upstream, never a runtime fallback.

    `post` is injected so this is testable with no Ollama host: it takes the
    request payload and returns the decoded response.
    """
    if not texts:
        return {"model": model, "vectors": [], "dimensions": None, "prompt_tokens": 0}

    send = post or (lambda payload: request(EMBED_URL, payload, timeout, EmbeddingError))
    # bge-m3 takes raw text: its model card specifies no query/passage prefix,
    # unlike E5. Any prefix belongs to the caller, not here.
    response = send({"model": model, "input": list(texts), "truncate": False})

    if not isinstance(response, dict):
        raise EmbeddingError("expected a JSON object in the embedding response")
    vectors = response.get("embeddings")
    if not isinstance(vectors, list):
        raise EmbeddingError(f"no embeddings in the response, keys {sorted(response)}")
    # A short batch would silently pair every passage with the wrong vector.
    dimensions = validate_vectors(vectors, EmbeddingError, count=len(texts))

    return {
        "model": response.get("model", model),
        "vectors": vectors,
        "dimensions": dimensions,
        "prompt_tokens": response.get("prompt_eval_count"),
    }


def model_digest(model: str = "bge-m3", get=None, timeout: float = 30.0) -> str:
    """The pulled model's digest, for pinning a baseline.

    The embed response does not carry it, so it comes from /api/tags. Pin it the
    way the corpus PDF is pinned by sha256: a re-pull can change the vectors.
    """
    fetch = get or (lambda: request(TAGS_URL, None, timeout, EmbeddingError))
    wanted = model if ":" in model.rsplit("/", 1)[-1] else f"{model}:latest"
    for entry in fetch().get("models", []):
        if entry.get("name") == wanted:
            return entry.get("digest", "")
    raise EmbeddingError(f"{model} is not pulled on this host")
