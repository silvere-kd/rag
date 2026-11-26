"""Generation through Ollama's chat API, shaped by a JSON schema.

This is the whole provider boundary. Everything above it takes a
generate(messages) -> str callable, which is why answer_question can be tested
without a model and this module can be tested without a host.
"""
from rag_portfolio.answer import REQUESTED_SHAPE
from rag_portfolio.config import GENERATION_MODEL
from rag_portfolio.ollama import CHAT_URL, OllamaError, request

# Kept as a name so callers and tests have one symbol to import.
DEFAULT_MODEL = GENERATION_MODEL


class GenerationError(OllamaError):
    """The chat call failed, or returned nothing usable."""


class GenerationTransportError(GenerationError):
    """The host could not be reached, or answered with an HTTP error.

    Separate from its parent because the two are not the same failure and must
    not be counted together. Tallying bare GenerationError once led me to
    attribute seven transport failures to the model returning empty content,
    and to conclude a retry was needed on evidence that said no such thing.
    A retry belongs here; nowhere else in this module.
    """


def ollama_generator(
    model: str = DEFAULT_MODEL,
    schema: dict | None = REQUESTED_SHAPE,
    timeout: float = 300.0,
    think: bool = False,
    post=None,
):
    """Build the generate(messages) -> str callable that answer_question takes.

    The schema goes to Ollama as `format`, which constrains decoding to that
    shape. It is load-bearing. Measurement says it does two things that ONE
    tested prompt wording did not achieve (gemma4:12b, three questions x three
    trials). A different wording might; that was not tested, and "the prompt
    cannot" would overstate nine runs:

      1. It suppresses the markdown fence. Asked in prose for JSON, the model
         produces JSON wrapped in ```json, and 0 of 9 replies parsed. Stripping
         the fence afterwards recovers 6 of 9.
      2. It guarantees `answer` is a string. The other 3 of those 9 return
         `answer: null` on an abstention, which no prompt wording prevented.

    Ollama's own structured-outputs page recommends passing the schema in the
    prompt as well, to ground the response. Measured here it did not help, and
    on one question it hurt. Three easy questions already parse 9 of 9 with no
    repairs, which is a ceiling that cannot show an improvement. On five harder
    questions x three trials at k=8 both arms scored identically: 12 answered,
    3 abstained, 0 repairs, 0 failures in 30 calls. The only difference was
    citation completeness, and it favoured leaving the prompt alone: with the
    schema described in the prompt, the model asserted the same four facts about
    the Conseil constitutionnel while quoting evidence for one of them instead
    of three, consistently across trials.

    An earlier version of this paragraph blamed "empty content from the
    provider" for failures in a first, badly instrumented run of that
    comparison. That was wrong. The tally recorded exception types without their
    messages, and GenerationError covered transport failures too, which is why
    GenerationTransportError now exists. Those seven failures have no
    established cause and did not recur in 45 later calls.

    So schema=None is NOT a working fallback for this model, only a hook for a
    provider with structured output of its own. It cannot constrain meaning
    either way, so answer_question checks every citation regardless.

    Thinking is off by default. One observation, not a law: on gemma4:12b with
    this schema and a two-part question, thinking on spent 56.5s and 3891 tokens
    and returned EMPTY content, while think=False answered in 2.4s on 144
    tokens. Whether that generalises to other models, prompts or question shapes
    is untested, so the flag stays a parameter. Harmless on models that cannot
    think at all: llama3 and mistral both accept it.

    temperature 0 narrows variation without making generation reproducible: only
    the prompt side of this pipeline is deterministic.

    `post` is injected so this is testable with no host, like embed_passages.
    """
    send = post or (
        lambda payload: request(CHAT_URL, payload, timeout, GenerationTransportError)
    )

    def generate(messages: list[dict[str, str]]) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": think,
            "options": {"temperature": 0},
        }
        if schema is not None:
            payload["format"] = schema
        response = send(payload)

        if not isinstance(response, dict):
            raise GenerationError("expected a JSON object from the chat API")
        message = response.get("message")
        if not isinstance(message, dict):
            raise GenerationError(f"no message in the reply, keys {sorted(response)}")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            # A thinking model can spend its whole budget reasoning and return
            # empty content, so the reasoning length is the useful diagnostic.
            thinking = len(message.get("thinking") or "")
            raise GenerationError(
                f"empty content from {model} ({thinking} chars of thinking)"
            )
        # message["thinking"] is deliberately dropped: the contract is a string
        # of JSON, and the reasoning is not part of the answer.
        return content

    # shortcut: no transport retry and no latency/token capture here. Add them
    # when the persisted baseline needs the numbers, not before.
    # shortcut: no markdown-fence stripping. The schema path never produces a
    # fence (9/9 clean), so it would be three lines guarding a case the
    # supported configuration does not reach. Add it if schema=None is ever
    # genuinely used.
    return generate
