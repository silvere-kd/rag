"""The saved vector index: integrity checks, no network, no model."""
import json

import pytest

from rag_portfolio.index import (  # noqa: I001
    IndexMismatchError,
    input_fingerprint,
    load_index,
    save_index,
)

INPUTS = ["Constitution du 4 octobre 1958, article 2\nLa langue...",
          "Constitution du 4 octobre 1958, article 6\nLe President..."]
VECTORS = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
DIGEST = "7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab"


def saved(tmp_path, inputs=INPUTS, vectors=VECTORS, digest=DIGEST):
    path = tmp_path / "vectors.json"
    save_index(path, inputs, vectors, model="bge-m3:latest", digest=digest)
    return path


def test_saving_and_loading_preserves_the_vectors_and_their_order(tmp_path):
    path = saved(tmp_path)

    assert load_index(path, INPUTS, model="bge-m3:latest", digest=DIGEST) == VECTORS


def test_changed_inputs_are_rejected(tmp_path):
    path = saved(tmp_path)
    altered = [INPUTS[0], INPUTS[1] + " modifie"]

    with pytest.raises(IndexMismatchError, match="input_fingerprint"):
        load_index(path, altered, model="bge-m3:latest", digest=DIGEST)


def test_a_different_model_digest_is_rejected(tmp_path):
    path = saved(tmp_path)

    with pytest.raises(IndexMismatchError, match="digest"):
        load_index(path, INPUTS, model="bge-m3:latest", digest="0" * 64)


def test_a_retagged_identical_model_is_accepted(tmp_path):
    # The digest is the identity. A tag is a label and must not invalidate a
    # cache whose weights are provably the same.
    path = saved(tmp_path)

    assert load_index(path, INPUTS, model="bge-m3:v1.5", digest=DIGEST) == VECTORS


def test_reordered_inputs_are_rejected(tmp_path):
    path = saved(tmp_path)

    with pytest.raises(IndexMismatchError, match="input_fingerprint"):
        load_index(path, list(reversed(INPUTS)), model="bge-m3:latest", digest=DIGEST)


def test_the_fingerprint_is_order_sensitive():
    assert input_fingerprint(["a", "b"]) != input_fingerprint(["b", "a"])


def test_multiline_inputs_do_not_collide():
    # The regression test for sha256("\n".join(inputs)), which collides here:
    # the rendered inputs are multiline, so a join loses their boundaries.
    assert input_fingerprint(["t\nun", "deux"]) != input_fingerprint(["t", "un\ndeux"])


def test_a_truncated_index_is_rejected(tmp_path):
    path = saved(tmp_path)
    payload = json.loads(path.read_text())
    payload["vectors"] = payload["vectors"][:1]
    path.write_text(json.dumps(payload))

    # Would otherwise pair 2 passages with 1 vector downstream.
    with pytest.raises(IndexMismatchError):
        load_index(path, INPUTS, model="bge-m3:latest", digest=DIGEST)


def test_a_missing_index_is_not_a_mismatch(tmp_path):
    # Absent is normal on a first run; stale is news. The caller must be able to
    # tell them apart.
    with pytest.raises(FileNotFoundError):
        load_index(tmp_path / "absent.json", INPUTS, model="x", digest=DIGEST)


def test_the_file_is_byte_stable(tmp_path):
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    save_index(first, INPUTS, VECTORS, model="bge-m3:latest", digest=DIGEST)
    save_index(second, INPUTS, VECTORS, model="bge-m3:latest", digest=DIGEST)

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes().endswith(b"\n")


def test_mismatched_counts_are_refused_at_save(tmp_path):
    with pytest.raises(ValueError, match="2 inputs"):
        save_index(tmp_path / "x.json", INPUTS, VECTORS[:1],
                   model="bge-m3:latest", digest=DIGEST)


def test_ragged_vectors_are_refused_at_save(tmp_path):
    with pytest.raises(ValueError, match="ragged"):
        save_index(tmp_path / "x.json", INPUTS, [[0.1, 0.2], [0.3]],
                   model="bge-m3:latest", digest=DIGEST)


def test_the_index_records_its_shape(tmp_path):
    path = saved(tmp_path)
    payload = json.loads(path.read_text())

    assert payload["dimensions"] == 3
    assert payload["count"] == 2
    assert payload["model"] == "bge-m3:latest"
    # No timestamp: it would break byte-stability and prove nothing.
    assert not any("time" in key or "date" in key for key in payload)


# --- a corrupt cache must not crash the caller or poison ranking -------------
# Reproduced before these were written: truncated JSON leaked JSONDecodeError,
# a JSON array leaked AttributeError, and NaN components, a missing
# "dimensions" and a contradictory "count" were all accepted.


def valid_body(**overrides):
    payload = {
        "model": "bge-m3:latest",
        "digest": DIGEST,
        "input_fingerprint": input_fingerprint(INPUTS),
        "dimensions": 3,
        "count": 2,
        "vectors": VECTORS,
    }
    payload.update(overrides)
    return json.dumps(payload)


@pytest.mark.parametrize(
    "raw, why",
    [
        pytest.param('{"digest": "x", "vec', "truncated json", id="truncated"),
        pytest.param("[]", "a json array, not an object", id="array"),
        pytest.param("null", "json null", id="null"),
        pytest.param("", "an empty file", id="empty"),
    ],
)
def test_malformed_files_are_a_mismatch_not_a_crash(tmp_path, raw, why):
    path = tmp_path / "vectors.json"
    path.write_text(raw)

    # search.py handles FileNotFoundError and IndexMismatchError. Anything else
    # reaches the user as a traceback.
    with pytest.raises(IndexMismatchError):
        load_index(path, INPUTS, model="bge-m3:latest", digest=DIGEST)


@pytest.mark.parametrize(
    "overrides, why",
    [
        pytest.param({"vectors": [[float("nan"), 0.2, 0.3], [0.4, 0.5, 0.6]]},
                     "NaN poisons every cosine it touches", id="nan"),
        pytest.param({"vectors": [[float("inf"), 0.2, 0.3], [0.4, 0.5, 0.6]]},
                     "infinity likewise", id="inf"),
        pytest.param({"vectors": ["abc", "def"]},
                     "len() of a string passes a width check", id="strings"),
        pytest.param({"vectors": [["0.1", "0.2", "0.3"], [0.4, 0.5, 0.6]]},
                     "string components", id="string-values"),
        pytest.param({"vectors": [[0.1, 0.2], [0.3, 0.4]]},
                     "width contradicts the recorded dimensions", id="width-vs-dimensions"),
        pytest.param({"count": 99}, "count contradicts the vectors", id="bad-count"),
        pytest.param({"dimensions": None}, "dimensions absent", id="no-dimensions"),
        pytest.param({"vectors": [[], []]}, "empty vectors", id="empty-vectors"),
    ],
)
def test_invalid_vectors_are_rejected(tmp_path, overrides, why):
    path = tmp_path / "vectors.json"
    path.write_text(valid_body(**overrides))

    with pytest.raises(IndexMismatchError):
        load_index(path, INPUTS, model="bge-m3:latest", digest=DIGEST)


def test_a_valid_body_still_loads(tmp_path):
    # So the parametrised cases above are rejected for their own reason and not
    # because the fixture was broken.
    path = tmp_path / "vectors.json"
    path.write_text(valid_body())

    assert load_index(path, INPUTS, model="bge-m3:latest", digest=DIGEST) == VECTORS
