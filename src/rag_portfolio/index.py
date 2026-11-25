"""A saved passage-vector index, with integrity checks.

What this establishes, exactly: these vectors were saved beside inputs with this
fingerprint and this model digest, they are well-formed, and there are as many
of them as there are inputs.

What it does NOT establish: that the vectors were produced from those inputs, or
that re-embedding would reproduce them. The fingerprint covers the inputs, not
the vectors, so a match after a rebuild says the inputs are unchanged and
nothing more. Reproducibility would mean regenerating the vectors independently
and comparing them; a digest over the vectors would make that comparison
possible but would not by itself demonstrate it.

Deliberately NOT the baseline either. The baseline needs the questions, the
ranked results, the corpus sha256 and the observed serving ceiling, and
conflating the two would leave a file that looks like evidence and is not.

Nothing here touches the network, so it tests without even a fake transport.
"""
import hashlib
import json
from pathlib import Path

from rag_portfolio.embed import validate_vectors


class IndexMismatchError(Exception):
    """The saved index does not belong to these inputs or this model."""


def input_fingerprint(inputs: list[str]) -> str:
    """sha256 over each input's own digest, in order.

    NOT sha256("\\n".join(inputs)). The rendered inputs are multiline, so a
    join loses the boundaries between them and different corpora collide:
    measured, ["t\\nun", "deux"] and ["t", "un\\ndeux"] share a joined hash and
    differ under this one. Chaining per-input digests has no separator to
    escape and stays order-sensitive.
    """
    chained = b"".join(hashlib.sha256(text.encode("utf-8")).digest() for text in inputs)
    return hashlib.sha256(chained).hexdigest()


def save_index(
    path: Path,
    inputs: list[str],
    vectors: list[list[float]],
    model: str,
    digest: str,
) -> str:
    """Write the index and return its input fingerprint.

    Byte-stable by construction: sorted keys, no timestamp, explicit newline.
    Floats round-trip exactly through json, so the vectors survive unchanged.
    """
    if len(inputs) != len(vectors):
        raise ValueError(f"{len(inputs)} inputs but {len(vectors)} vectors")
    widths = {len(vector) for vector in vectors}
    if len(widths) > 1:
        raise ValueError(f"ragged vectors, widths {sorted(widths)}")

    fingerprint = input_fingerprint(inputs)
    payload = {
        "model": model,
        "digest": digest,
        "input_fingerprint": fingerprint,
        # Not redundant with the fingerprint: these catch a half-written file
        # before 159 passages get paired with 140 vectors.
        "dimensions": widths.pop() if widths else None,
        "count": len(vectors),
        "vectors": vectors,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    return fingerprint


def load_index(
    path: Path,
    inputs: list[str],
    model: str,
    digest: str,
) -> list[list[float]]:
    """Return the saved vectors, or refuse if the file contradicts these inputs.

    A missing file raises FileNotFoundError, which is NOT a mismatch: absent is
    normal on a first run, stale is news, and a caller has to tell them apart.

    `model` is recorded for diagnostics and deliberately not compared. The
    digest is the identity, so a retagged copy of the same weights must not
    invalidate a cache that is provably correct.
    """
    raw = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        # A half-written file is a stale cache, not a crash: the caller handles
        # FileNotFoundError and IndexMismatchError, and nothing else.
        raise IndexMismatchError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise IndexMismatchError(
            f"{path} holds a {type(payload).__name__}, not an object"
        )

    expected = input_fingerprint(inputs)
    for field, want, found in (
        ("digest", digest, payload.get("digest")),
        ("input_fingerprint", expected, payload.get("input_fingerprint")),
    ):
        if want != found:
            raise IndexMismatchError(
                f"{field} differs: the index holds {str(found)[:16]!r}, "
                f"these inputs and this model give {str(want)[:16]!r}"
            )

    vectors = payload.get("vectors")
    if not isinstance(vectors, list):
        raise IndexMismatchError(f"no vectors in {path}, keys {sorted(payload)}")

    recorded_dimensions, recorded_count = payload.get("dimensions"), payload.get("count")
    for field, value in (("dimensions", recorded_dimensions), ("count", recorded_count)):
        # Absent is not benign: these exist to catch a half-written file, and a
        # missing one would let that file through unchecked.
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise IndexMismatchError(
                f"{path} records {field}={value!r}, which cannot be checked"
            )
    if recorded_count != len(vectors):
        raise IndexMismatchError(
            f"{path} records count={recorded_count} but holds {len(vectors)} vectors"
        )

    # The same bar a freshly embedded batch clears: a cached NaN reaches cosine
    # similarity by exactly the same path.
    validate_vectors(
        vectors, IndexMismatchError, count=len(inputs), width=recorded_dimensions
    )
    return vectors
