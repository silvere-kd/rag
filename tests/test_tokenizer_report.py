"""The tokenizer report runs offline with a known character-counting stand-in."""
import importlib.util
from pathlib import Path

from rag_portfolio.ingest import embedding_text
from rag_portfolio.models import Section, Source


def test_serving_limit_checks_passages_below_the_model_limit(monkeypatch, capsys):
    script = Path(__file__).resolve().parents[1] / "scripts" / "measure_tokenizer_fit.py"
    spec = importlib.util.spec_from_file_location("measure_tokenizer_fit", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_counter", lambda model, revision: len)

    prefix = "passage: "
    source = Source("test", "v1", "fixture", "sections.txt", "0" * 64)

    def sized(total):
        """A section whose rendered input is exactly `total` characters."""
        bare = Section("constitution-1958", "article", str(total), "", [], None, source)
        head = len(prefix) + len(embedding_text(bare))
        return Section("constitution-1958", "article", str(total),
                       "x" * (total - head), [], None, source)

    sections = [sized(tokens) for tokens in (2048, 2049, 3000, 9000)]
    candidate = {
        "model": "test-model",
        "revision": "test-revision",
        "prefix": prefix,
        "max_tokens": 8192,
        "serving_max_tokens": 2048,
    }

    module.report(candidate, sections)

    output = capsys.readouterr().out
    assert "over the serving ceiling: 3" in output
    assert "over the limit: 1 measured" in output
