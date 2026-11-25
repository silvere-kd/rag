"""The chat adapter, exercised with no Ollama host and no model."""
import pytest

from rag_portfolio.answer import REQUESTED_SHAPE
from rag_portfolio.generate import (
    DEFAULT_MODEL,
    GenerationError,
    GenerationTransportError,
    ollama_generator,
)
from rag_portfolio.ollama import OllamaError


def fake_post(response, seen=None):
    def post(payload):
        if seen is not None:
            seen.append(payload)
        return response

    return post


def reply(content, thinking=None):
    message = {"role": "assistant", "content": content}
    if thinking is not None:
        message["thinking"] = thinking
    return {"model": DEFAULT_MODEL, "message": message}


def test_the_payload_carries_the_schema_and_pins_the_temperature():
    seen = []
    generate = ollama_generator(post=fake_post(reply('{"status": "answered"}'), seen))

    generate([{"role": "user", "content": "Question ?"}])

    assert seen[0]["format"] == REQUESTED_SHAPE
    assert seen[0]["stream"] is False
    assert seen[0]["options"]["temperature"] == 0
    assert seen[0]["model"] == DEFAULT_MODEL
    # Measured: thinking on with this schema returned empty content after 56.5s
    # and 3891 tokens. Off, the same question answered in 2.4s on 144.
    assert seen[0]["think"] is False
    assert seen[0]["messages"] == [{"role": "user", "content": "Question ?"}]


def test_the_content_is_returned_and_the_reasoning_dropped():
    generate = ollama_generator(
        post=fake_post(reply('{"status": "answered"}', thinking="x" * 700))
    )

    assert generate([]) == '{"status": "answered"}'


def test_the_schema_can_be_omitted():
    seen = []
    generate = ollama_generator(schema=None, post=fake_post(reply("texte"), seen))

    generate([])

    assert "format" not in seen[0]


def test_empty_content_reports_how_much_reasoning_happened():
    # A thinking model can spend its budget reasoning and return nothing.
    generate = ollama_generator(post=fake_post(reply("", thinking="y" * 512)))

    with pytest.raises(GenerationError, match="512 chars of thinking"):
        generate([])


@pytest.mark.parametrize(
    "response",
    [
        pytest.param([], id="not-an-object"),
        pytest.param({"error": "model not found"}, id="no-message"),
        pytest.param({"message": "oops"}, id="message-not-an-object"),
        pytest.param({"message": {"content": None}}, id="content-not-text"),
    ],
)
def test_unusable_replies_are_rejected(response):
    generate = ollama_generator(post=fake_post(response))

    with pytest.raises(GenerationError):
        generate([])


def test_a_generation_failure_is_an_ollama_failure():
    # So a caller can catch one provider error for embedding and generation.
    assert issubclass(GenerationError, OllamaError)


def test_thinking_can_be_turned_back_on():
    seen = []
    generate = ollama_generator(think=True, post=fake_post(reply("texte"), seen))

    generate([])

    assert seen[0]["think"] is True


def test_a_transport_failure_is_distinguishable_from_an_empty_reply(monkeypatch):
    # These were indistinguishable by type, and I misread a measurement because
    # of it: seven transport failures counted as the model returning nothing.
    def refuse(*args, **kwargs):
        raise OSError("connection reset")

    monkeypatch.setattr("rag_portfolio.ollama.urllib.request.urlopen", refuse)

    with pytest.raises(GenerationTransportError):
        ollama_generator()([])


def test_an_empty_reply_is_not_a_transport_failure():
    generate = ollama_generator(post=fake_post(reply("")))

    with pytest.raises(GenerationError) as caught:
        generate([])

    # Still a GenerationError, so existing handlers keep working...
    assert not isinstance(caught.value, GenerationTransportError)
    # ...but a retry aimed at the transport will not swallow it.
