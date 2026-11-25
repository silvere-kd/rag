"""The embedding adapter, exercised with no Ollama host and no model."""
from io import BytesIO
from urllib.error import HTTPError

import pytest

from rag_portfolio.embed import (
    SERVING_MAX_TOKENS,
    EmbeddingError,
    embed_passages,
    model_digest,
)


def fake_post(response, seen=None):
    """A stand-in transport that records the payload it was handed."""

    def post(payload):
        if seen is not None:
            seen.append(payload)
        return response

    return post


def test_truncation_is_always_refused():
    seen = []
    embed_passages(
        ["un", "deux"],
        post=fake_post({"embeddings": [[0.1, 0.2], [0.3, 0.4]]}, seen),
    )

    assert seen[0]["truncate"] is False
    assert seen[0]["input"] == ["un", "deux"]
    assert seen[0]["model"] == "bge-m3"


def test_a_successful_batch_reports_its_shape():
    result = embed_passages(
        ["un"],
        post=fake_post({"embeddings": [[0.0] * 1024], "prompt_eval_count": 11}),
    )

    assert result["dimensions"] == 1024
    assert result["prompt_tokens"] == 11
    assert len(result["vectors"]) == 1


def test_a_short_batch_is_rejected():
    # Two passages, one vector: pairing them would misalign every section.
    with pytest.raises(EmbeddingError, match="received 1"):
        embed_passages(["un", "deux"], post=fake_post({"embeddings": [[0.1]]}))


def test_ragged_vectors_are_rejected():
    with pytest.raises(EmbeddingError, match="ragged"):
        embed_passages(
            ["un", "deux"], post=fake_post({"embeddings": [[0.1, 0.2], [0.3]]})
        )


def test_a_response_without_embeddings_is_rejected():
    with pytest.raises(EmbeddingError, match="no embeddings"):
        embed_passages(["un"], post=fake_post({"error": "something went wrong"}))


def test_no_texts_needs_no_request():
    def explode(payload):
        raise AssertionError("should not have called the provider")

    result = embed_passages([], post=explode)

    assert result["vectors"] == []
    assert result["dimensions"] is None


def test_the_digest_comes_from_the_tag_listing():
    listing = {
        "models": [
            {"name": "gemma4:12b", "digest": "aaa"},
            {"name": "bge-m3:latest", "digest": "7907646426070047"},
        ]
    }

    assert model_digest("bge-m3", get=lambda: listing) == "7907646426070047"


def test_an_unpulled_model_is_reported():
    with pytest.raises(EmbeddingError, match="not pulled"):
        model_digest("bge-m3", get=lambda: {"models": [{"name": "llama3", "digest": "x"}]})


def test_the_serving_ceiling_is_recorded_not_assumed():
    # The model card says 8192; this host truncates at 2048 and num_ctx does not
    # raise it. The constant exists so a baseline records the serving fact.
    assert SERVING_MAX_TOKENS == 2048


@pytest.mark.parametrize("model", ["bge-m3", "bge-m3:latest"])
def test_digest_matches_the_requested_tag_instead_of_the_first_base_name(model):
    listing = {
        "models": [
            {"name": "bge-m3:older", "digest": "older-digest"},
            {"name": "bge-m3:latest", "digest": "latest-digest"},
        ]
    }

    assert model_digest(model, get=lambda: listing) == "latest-digest"


def test_an_unpulled_tag_does_not_match_another_tag():
    listing = {"models": [{"name": "bge-m3:latest", "digest": "latest-digest"}]}

    with pytest.raises(EmbeddingError, match="not pulled"):
        model_digest("bge-m3:missing", get=lambda: listing)


@pytest.mark.parametrize(
    "vector",
    [
        pytest.param([], id="empty"),
        pytest.param("abc", id="string-vector"),
        pytest.param(None, id="null-vector"),
        pytest.param(["0.1"], id="string-value"),
        pytest.param([None], id="null-value"),
        pytest.param([True], id="boolean-value"),
        pytest.param([float("nan")], id="nan"),
        pytest.param([float("inf")], id="positive-infinity"),
        pytest.param([float("-inf")], id="negative-infinity"),
    ],
)
def test_malformed_vectors_are_rejected(vector):
    response = {"model": "bge-m3", "embeddings": [vector], "prompt_eval_count": 1}

    with pytest.raises(EmbeddingError):
        embed_passages(["texte"], post=fake_post(response))


def test_a_non_object_response_is_rejected():
    with pytest.raises(EmbeddingError):
        embed_passages(["texte"], post=fake_post([]))


def test_http_errors_preserve_the_provider_message(monkeypatch):
    body = b'{"error":"the input length exceeds the context length"}'
    error = HTTPError("http://test.invalid/api/embed", 400, "Bad Request", {}, BytesIO(body))

    def refuse_request(*args, **kwargs):
        raise error

    # The transport moved to rag_portfolio.ollama; embed still has to surface
    # the provider's message through it.
    monkeypatch.setattr("rag_portfolio.ollama.urllib.request.urlopen", refuse_request)

    with pytest.raises(EmbeddingError) as caught:
        embed_passages(["passage trop long"])

    assert "HTTP 400" in str(caught.value)
    assert "the input length exceeds the context length" in str(caught.value)
