"""The shared transport, with urlopen replaced and no host involved."""
import io
import json
import socket
import urllib.error

import pytest

from rag_portfolio.ollama import _BODY_LIMIT, CHAT_URL, OllamaError, request


class Fake(io.BytesIO):
    """A response object json.load can read, usable as a context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_urlopen(body=b'{"ok": true}', seen=None):
    def urlopen(req, timeout=None):
        if seen is not None:
            seen.append((req, timeout))
        return Fake(body)

    return urlopen


def raising_urlopen(exc):
    def urlopen(req, timeout=None):
        raise exc

    return urlopen


def test_a_payload_is_posted_as_json(monkeypatch):
    seen = []
    monkeypatch.setattr("rag_portfolio.ollama.urllib.request.urlopen",
                        fake_urlopen(seen=seen))

    assert request(CHAT_URL, {"model": "x"}, 5.0) == {"ok": True}

    req, timeout = seen[0]
    assert req.data == json.dumps({"model": "x"}).encode("utf-8")
    assert req.get_header("Content-type") == "application/json"
    assert req.get_method() == "POST"
    assert timeout == 5.0


def test_no_payload_means_a_get(monkeypatch):
    seen = []
    monkeypatch.setattr("rag_portfolio.ollama.urllib.request.urlopen",
                        fake_urlopen(seen=seen))

    request(CHAT_URL, None, 1.0)

    # /api/tags takes no body, and sending an empty one would make it a POST.
    assert seen[0][0].data is None
    assert seen[0][0].get_method() == "GET"


def test_an_http_error_keeps_the_body(monkeypatch):
    # The reason this module exists: repr(HTTPError) drops the body, and Ollama
    # puts the real cause there.
    body = b'{"error":"the input length exceeds the context length"}'
    monkeypatch.setattr(
        "rag_portfolio.ollama.urllib.request.urlopen",
        raising_urlopen(urllib.error.HTTPError(CHAT_URL, 400, "Bad Request", {},
                                               io.BytesIO(body))),
    )

    with pytest.raises(OllamaError) as caught:
        request(CHAT_URL, {}, 1.0)

    message = str(caught.value)
    assert "HTTP 400" in message
    assert "the input length exceeds the context length" in message
    assert CHAT_URL in message


def test_a_huge_error_body_is_truncated(monkeypatch):
    monkeypatch.setattr(
        "rag_portfolio.ollama.urllib.request.urlopen",
        raising_urlopen(urllib.error.HTTPError(CHAT_URL, 500, "Server Error", {},
                                               io.BytesIO(b"x" * 20000))),
    )

    with pytest.raises(OllamaError) as caught:
        request(CHAT_URL, {}, 1.0)

    # An HTML error page from a proxy would otherwise flood the log.
    assert str(caught.value).count("x") <= _BODY_LIMIT


def test_an_unreachable_host_is_reported(monkeypatch):
    monkeypatch.setattr("rag_portfolio.ollama.urllib.request.urlopen",
                        raising_urlopen(urllib.error.URLError(socket.timeout())))

    with pytest.raises(OllamaError, match="could not reach"):
        request(CHAT_URL, {}, 1.0)


def test_a_non_json_body_does_not_leak_a_decode_error(monkeypatch):
    # A proxy can answer 200 with an HTML page. That is still a provider
    # failure, and callers catch OllamaError, not json.JSONDecodeError.
    monkeypatch.setattr("rag_portfolio.ollama.urllib.request.urlopen",
                        fake_urlopen(body=b"<html>Bad Gateway</html>"))

    with pytest.raises(OllamaError):
        request(CHAT_URL, {}, 1.0)


def test_the_caller_chooses_the_exception_class(monkeypatch):
    class MyError(OllamaError):
        pass

    monkeypatch.setattr("rag_portfolio.ollama.urllib.request.urlopen",
                        raising_urlopen(OSError("boom")))

    # This argument is why one transport can serve embedding and generation
    # without either of them reporting failures in the transport's vocabulary.
    with pytest.raises(MyError):
        request(CHAT_URL, {}, 1.0, MyError)
