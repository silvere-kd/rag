"""The shared HTTP transport for this project's Ollama host.

One copy on purpose. The HTTPError handling below is the only hard-won part of
it: repr() on an HTTPError drops the response body, and Ollama puts the real
cause there. A second copy is a second place for that lesson to rot.
"""
import json
import urllib.error
import urllib.request

HOST = "http://host.docker.internal:11434"
EMBED_URL = f"{HOST}/api/embed"
TAGS_URL = f"{HOST}/api/tags"
CHAT_URL = f"{HOST}/api/chat"

_BODY_LIMIT = 500


class OllamaError(Exception):
    """A call to the Ollama host failed."""


def request(
    url: str,
    payload: dict | None,
    timeout: float,
    error: type[Exception] = OllamaError,
) -> dict:
    """POST the payload as JSON, or GET when payload is None.

    `error` is the class to raise, so each caller reports failures in its own
    vocabulary instead of leaking this module's.
    """
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        # repr(HTTPError) omits the response body, and Ollama puts the real
        # cause there: "the input length exceeds the context length".
        body = exc.read().decode("utf-8", "replace")[:_BODY_LIMIT]
        raise error(f"HTTP {exc.code} from {url}: {body}") from exc
    except json.JSONDecodeError as exc:
        # A proxy can answer 200 with an HTML page. That is a provider failure
        # like any other, and callers catch this module's error, not a decode
        # error from the standard library.
        raise error(f"{url} returned a body that is not JSON: {exc}") from exc
    except OSError as exc:
        raise error(f"could not reach {url}: {exc!r}") from exc
