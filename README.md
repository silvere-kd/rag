# rag-portfolio

## Purpose

Ask questions in French about the French Constitution and get answers whose citations
code has checked against the source text. The model is told to abstain when the
passages do not support an answer.

The corpus is 40 pages holding four distinct texts, split into 159 passages.

## How it works

```
PDF -> passages -> embed (title + body) -> rank by cosine -> prompt -> verify quotes
```

Each passage is embedded with its document title attached, which matters because
three of these texts each contain an article 1 and the fourth a paragraph 1. The model receives the passage alone, labelled
`S1`, `S2`, fenced inside `<Passage>` elements, and answers in JSON constrained by a
schema. Code then checks that every cited label was supplied and every quoted excerpt
appears in the passage it was attributed to. A failed check raises; a refusal does not.

Needs [Ollama](https://ollama.com) with `bge-m3` and `gemma4:12b` pulled.

```bash
uv sync --extra dev
uv run pytest                                  # the whole suite, no network needed
uv run python scripts/search.py "Quelle est la langue de la République ?"
```

Measurements and findings live in the comments beside the code that produced them.

## Layout

| File | Job |
| --- | --- |
| `config.py` | models, k, paths, and the Ollama host |
| `ingest.py` | PDF to passages; builds the embedding input |
| `measure.py` | token counts against an embedding model's budget |
| `embed.py` | Ollama `/api/embed`, truncation refused |
| `generate.py` | Ollama `/api/chat` with the schema applied |
| `index.py` | saves vectors with a fingerprint, refuses a stale file |
| `retrieve.py` | cosine ranking |
| `answer.py` | the JSON contract and the citation check |
| `evaluate.py` | scores a run and writes the baseline |
| `models.py` | `Source`, `Section`, `Citation` |

`scripts/` holds the entry points and `data/` the corpus with its manifest.
