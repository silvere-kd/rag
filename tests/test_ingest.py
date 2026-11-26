import copy
import json
from dataclasses import asdict
import pytest
from pathlib import Path

from rag_portfolio.models import Section, Source
from rag_portfolio.ingest import extract_pages, clean_page_text, detect_text_heading, assign_text_ids, detect_article_start, normalize_article_id, detect_title_heading, detect_preamble_heading, assemble_sections, file_sha256, load_sources, ingest_pdf, write_sections, export_sections, embedding_text, DOCUMENT_TITLES, TEXT_IDS_BY_TITLE
from rag_portfolio.measure import estimate_embedding_fit, measure_fit

PATH = Path(__file__).resolve().parents[1]
CONSTITUTION = PATH / "data" / "constitution" / "constitution.pdf"
FAKE_SOURCE = Source("constitution", "test", "test edition", "fake.pdf", "0" * 64)


@pytest.fixture(scope="module")
def raw_records():
    return extract_pages(CONSTITUTION)


def test_extraction(raw_records):        
    assert len(raw_records) == 40
    assert [record["page_number"] for record in raw_records] == list(range(1, 41))
    assert all(record["raw_text"].strip() for record in raw_records)

def test_header_and_footer_are_removed_exactly():
    assert clean_page_text("1 janvier 2015 \n  Texte conservé.  \n2", 2) == "  Texte conservé.  "

def test_surrounding_blank_lines_do_not_defeat_cleanup():
    assert clean_page_text("\n1 janvier 2015 \n  Texte conservé.  \n2\n", 2) == "  Texte conservé.  "

def test_body_numbers_and_roman_numerals_survive():
    raw = "1 janvier 2015\n1. Au lendemain\nXVI\n13. La Nation\n36"
    assert clean_page_text(raw, 36) == "1. Au lendemain\nXVI\n13. La Nation"

def test_the_date_inside_the_body_is_preserved():
    raw = "1 janvier 2015\nEn vigueur au 1 janvier 2015 selon le texte.\n7"
    assert clean_page_text(raw, 7) == "En vigueur au 1 janvier 2015 selon le texte."


def test_a_footer_that_is_not_the_page_number_is_kept():
    assert clean_page_text("1 janvier 2015\nTexte\n99", 7) == "Texte\n99"


def test_a_page_without_furniture_is_unchanged():
    assert clean_page_text("Texte seul", 5) == "Texte seul"


def test_every_real_page_loses_its_furniture(raw_records):
    for record in raw_records:
        cleaned = clean_page_text(record["raw_text"], record["page_number"])
        lines = [line for line in cleaned.splitlines() if line.strip()]
        assert lines, record["page_number"]
        assert lines[0].strip() != "1 janvier 2015"
        assert lines[-1].strip() != str(record["page_number"])
        assert not cleaned.startswith("\n") and not cleaned.endswith("\n")


def test_heading():
    assert detect_text_heading("PRÉAMBULE DE LA CONSTITUTION\nDU 27 OCTOBRE 1946\n\n1. Au lendemain de la victoire remportée") == "preambule-1946"

def test_each_real_heading_page_is_identified(raw_records):
    cleaned = {r["page_number"]: clean_page_text(r["raw_text"], r["page_number"]) for r in raw_records}
    assert detect_text_heading(cleaned[1]) == "constitution-1958"
    assert detect_text_heading(cleaned[34]) == "charte-environnement-2004"
    assert detect_text_heading(cleaned[36]) == "preambule-1946"
    assert detect_text_heading(cleaned[38]) == "declaration-1789"


def test_continuation_pages_have_no_heading(raw_records):
    cleaned = {r["page_number"]: clean_page_text(r["raw_text"], r["page_number"]) for r in raw_records}
    assert [detect_text_heading(cleaned[p]) for p in (2, 35, 37, 39)] == [None] * 4

def test_assigned_text_ids(raw_records):
    records = assign_text_ids(raw_records)
    assert [record["text_id"] for record in records[:33]] == ["constitution-1958"] * 33
    assert [record["text_id"] for record in records[33:35]] == ["charte-environnement-2004"] * 2
    assert [record["text_id"] for record in records[35:37]] == ["preambule-1946"] * 2
    assert [record["text_id"] for record in records[37:40]] == ["declaration-1789"] * 3

def test_a_page_before_any_heading_is_rejected():
    with pytest.raises(ValueError, match="7"):
        assign_text_ids([{"page_number": 7, "raw_text": "Texte sans titre reconnu."}])


def test_assign_returns_one_record_per_input_page(raw_records):
    assert len(assign_text_ids(raw_records)) == len(raw_records) == 40


def test_assign_preserves_page_order(raw_records):
    records = assign_text_ids(raw_records)
    assert [record["page_number"] for record in records] == list(range(1, 41))


def test_assign_preserves_raw_text_unchanged(raw_records):
    records = assign_text_ids(raw_records)
    assert [record["raw_text"] for record in records] == [
        source["raw_text"] for source in raw_records
    ]


def test_assign_does_not_mutate_its_input(raw_records):
    before = copy.deepcopy(raw_records)
    assign_text_ids(raw_records)
    assert raw_records == before


def test_detect_article_identifier():
    assert detect_article_start("ARTICLE PREMIER. La France…", "constitution-1958") == "1"
    assert detect_article_start("ARTICLE 1er. Chacun…", "charte-environnement-2004") == "1"
    assert detect_article_start("ARTICLE 34-1. Les assemblées…", "constitution-1958") == "34-1"
    assert detect_article_start("ARTICLE PREMIER", "constitution-1958") == "1"
    assert detect_article_start("La procédure prévue à l’article 34…", "constitution-1958") is None
    assert detect_article_start("2004.", "constitution-1958") is None


def test_malformed_article_markers_are_rejected():
    assert detect_article_start("ARTICLES 34 et 35. Texte.", "constitution-1958") is None
    assert detect_article_start("ARTICLE", "constitution-1958") is None


def test_whitespace_variants_are_tolerated():
    assert detect_article_start("ARTICLE\t2. Texte.", "constitution-1958") == "2"
    assert detect_article_start("ARTICLE 2 . Texte.", "constitution-1958") == "2"
    assert detect_article_start("Article 88-4. Le Gouvernement…", "constitution-1958") == "88-4"


def test_each_text_uses_its_own_numbering_convention():
    assert detect_article_start("13. La Nation garantit…", "preambule-1946") == "13"
    assert detect_article_start("XVI", "declaration-1789") == "16"
    assert detect_article_start("ARTICLE PREMIER", "declaration-1789") == "1"
    # A bare Roman numeral is not a boundary in a text that does not use them.
    assert detect_article_start("XVI", "constitution-1958") is None


def test_normalize_article_id():
    assert normalize_article_id("PREMIER") == "1"
    assert normalize_article_id("1er") == "1"
    assert normalize_article_id("XVII") == "17"
    assert normalize_article_id("IV") == "4"
    assert normalize_article_id("88-4") == "88-4"

def test_detect_title_heading():
    assert detect_title_heading("Titre premier ") == "Titre premier"
    assert detect_title_heading("Titre XI BIS ") == "Titre XI BIS"
    assert detect_title_heading("Titre VIII  ") == "Titre VIII"
    assert detect_title_heading("DE LA SOUVERAINETÉ ") is None
    assert detect_title_heading("ARTICLE  2.    La langue de la République. ") is None


def test_the_constitution_has_seventeen_title_headings(raw_records):
    records = assign_text_ids(raw_records)
    pages = [
        record["page_number"]
        for record in records
        for line in record["clean_text"].splitlines()
        if detect_title_heading(line)
    ]
    assert pages == [2, 2, 7, 8, 10, 18, 19, 21, 22, 23, 24, 25, 25, 30, 31, 31, 33]


def test_detect_preamble_heading():
    assert detect_preamble_heading("PRÉAMBULE") == "preamble"
    assert detect_preamble_heading("   PRÉAMBULE   ") == "preamble"
    assert detect_preamble_heading("PRÉAMBULE DE LA CONSTITUTION") is None
    assert detect_preamble_heading("complétée par le préambule de la Constitution de 1946") is None


def test_preamble_headings_occur_only_on_pages_1_and_38(raw_records):
    records = assign_text_ids(raw_records)
    pages = [
        record["page_number"]
        for record in records
        for line in record["clean_text"].splitlines()
        if detect_preamble_heading(line)
    ]
    assert pages == [1, 38]

@pytest.fixture(scope="module")
def source():
    return load_sources(CONSTITUTION.parent)[0]


@pytest.fixture(scope="module")
def sections(raw_records, source):
    return assemble_sections(assign_text_ids(raw_records), source)


def unit(sections, text_id, number):
    return next(s for s in sections if s.text_id == text_id and s.number == number)


def test_a_title_between_two_articles_closes_the_first():
    records = [
        {"page_number": 1, "raw_text": "", "clean_text": "ARTICLE PREMIER. Texte A.",
         "text_id": "constitution-1958"},
        {"page_number": 2, "raw_text": "",
         "clean_text": "Titre premier\nDE LA SOUVERAINET\u00c9\nARTICLE 2. Texte B.",
         "text_id": "constitution-1958"},
    ]

    first, second = assemble_sections(records, FAKE_SOURCE)

    assert (first.number, first.page_numbers, first.title_context) == ("1", [1], None)
    assert "Texte A." in first.text and "Titre premier" not in first.text
    assert (second.number, second.page_numbers) == ("2", [2])
    assert second.title_context == "Titre premier\nDE LA SOUVERAINET\u00c9"
    assert "Texte B." in second.text and "SOUVERAINET\u00c9" not in second.text


def test_constitution_article_1_stops_at_the_end_of_page_1(sections):
    article = unit(sections, "constitution-1958", "1")

    assert article.page_numbers == [1]
    assert article.kind == "article"
    assert "Titre premier" not in article.text


def test_charte_article_5_keeps_its_real_continuation(sections):
    article = unit(sections, "charte-environnement-2004", "5")

    assert article.page_numbers == [34, 35]


def test_a_two_line_subtitle_lands_entirely_in_title_context(sections):
    article = unit(sections, "constitution-1958", "34")

    assert article.title_context == "Titre V\nDES RAPPORTS ENTRE LE PARLEMENT\nET LE GOUVERNEMENT"
    assert "ET LE GOUVERNEMENT" not in article.text


def test_the_1946_numbered_units_are_paragraphs(sections):
    units = [s for s in sections if s.text_id == "preambule-1946" and s.number]

    assert {s.kind for s in units} == {"paragraph"}
    assert [s.number for s in units] == [str(n) for n in range(1, 19)]


def test_no_section_is_empty(sections):
    assert all(s.text.strip() for s in sections)


def test_file_sha256_is_stable_across_reads(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"bonjour")

    assert file_sha256(path) == file_sha256(path)


def test_file_sha256_depends_on_bytes_not_path(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"bonjour")
    (tmp_path / "b.bin").write_bytes(b"bonjour")

    assert file_sha256(tmp_path / "a.bin") == file_sha256(tmp_path / "b.bin")


def test_file_sha256_changes_with_content(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"bonjour")
    before = file_sha256(path)

    path.write_bytes(b"bonjour!")

    assert file_sha256(path) != before


def test_file_sha256_matches_the_known_digest(tmp_path):
    # Pins SHA-256 itself. MD5 satisfies all three tests above; a path hash or a
    # constant does not, so this is the only one that names the algorithm.
    path = tmp_path / "a.bin"
    path.write_bytes(b"bonjour")

    assert file_sha256(path) == (
        "2cb4b1431b84ec15d35ed83bb927e27e8967d75f4bcd9cc4b25c8d879ae23e18"
    )


def test_load_sources_for_the_constitution():
    sources = load_sources(CONSTITUTION.parent)

    assert len(sources) == 1
    source = sources[0]
    assert source.corpus_id == "constitution"
    assert source.version == "edition-2015-01-01"
    assert source.edition_label == "1 janvier 2015"
    assert source.filename == "constitution.pdf"
    assert source.sha256 == (
        "5fcfc33d6ae3023712060b2707b17c48289d7ae8e254f6c3784ce22859367984"
    )


def test_load_sources_keeps_manifest_order_and_hashes_each_file():
    sources = load_sources(PATH / "data" / "fictional" / "v1")

    assert [s.filename for s in sources] == ["reglement-antf.md", "bareme-antf.md"]
    assert {s.version for s in sources} == {"v1"}
    assert {s.corpus_id for s in sources} == {"fictional"}
    assert len({s.sha256 for s in sources}) == 2


def test_every_section_carries_the_supplied_source(sections, source):
    # The annotation does not enforce the type, so a Path passed here would be
    # invisible to every other test: they only read text_id, number, pages and text.
    assert all(section.source == source for section in sections)
    assert all(isinstance(section.source, Source) for section in sections)


@pytest.fixture(scope="module")
def pipeline_sections():
    return ingest_pdf(CONSTITUTION.parent)


def test_ingest_pdf_returns_a_flat_list_of_sections(pipeline_sections, source):
    # The only test that would catch a wiring mistake in ingest_pdf: every other
    # test assembles the chain by hand, so a broken composition stays green.
    assert len(pipeline_sections) == 159
    assert all(isinstance(s, Section) for s in pipeline_sections)
    assert all(s.source == source for s in pipeline_sections)


def test_ingest_pdf_covers_the_four_texts(pipeline_sections):
    assert {s.text_id for s in pipeline_sections} == {
        "constitution-1958",
        "charte-environnement-2004",
        "preambule-1946",
        "declaration-1789",
    }


def test_ingest_pdf_keeps_page_provenance(pipeline_sections):
    assert unit(pipeline_sections, "constitution-1958", "1").page_numbers == [1]
    assert unit(pipeline_sections, "charte-environnement-2004", "5").page_numbers == [34, 35]


def test_ingest_pdf_is_repeatable():
    assert ingest_pdf(CONSTITUTION.parent) == ingest_pdf(CONSTITUTION.parent)


def test_written_sections_round_trip(tmp_path, pipeline_sections):
    out = tmp_path / "sections.jsonl"

    write_sections(pipeline_sections, out)

    raw = out.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in raw.splitlines()]

    # One comparison covers the count, the order, every Source field, the page
    # references and the text. It pins the round trip, not the schema: a field
    # renamed in Section would still pass, since asdict is on both sides.
    assert rows == [asdict(section) for section in pipeline_sections]
    # Separate, because json.loads would undo \uXXXX escapes and hide ensure_ascii=True.
    assert "République" in raw


def test_two_independent_runs_write_identical_bytes(tmp_path):
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"

    write_sections(ingest_pdf(CONSTITUTION.parent), first)
    write_sections(ingest_pdf(CONSTITUTION.parent), second)

    assert first.read_bytes() == second.read_bytes()


def test_writing_twice_replaces_rather_than_appends(tmp_path, pipeline_sections):
    out = tmp_path / "sections.jsonl"

    write_sections(pipeline_sections, out)
    write_sections(pipeline_sections, out)

    assert len(out.read_text(encoding="utf-8").splitlines()) == 159


def test_export_writes_both_files_with_the_real_artifact_hash(
    tmp_path, pipeline_sections, source
):
    export_sections(pipeline_sections, tmp_path)

    manifest = json.loads((tmp_path / "export.json").read_text(encoding="utf-8"))

    assert manifest["section_count"] == 159
    assert manifest["sources"] == [asdict(source)]
    assert manifest["artifact"]["filename"] == "sections.jsonl"
    # Compared against the bytes on disk, not against a value the writer kept.
    assert manifest["artifact"]["sha256"] == file_sha256(tmp_path / "sections.jsonl")


def test_repeated_exports_produce_identical_bytes(tmp_path, pipeline_sections):
    first, second = tmp_path / "first", tmp_path / "second"

    export_sections(pipeline_sections, first)
    export_sections(pipeline_sections, second)

    for name in ("sections.jsonl", "export.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_manifest_keeps_one_record_per_distinct_source(tmp_path):
    # Same filename, different version: deduplicating by filename would merge them.
    v1 = Source("fictional", "v1", "ANTF v1", "reglement-antf.md", "a" * 64)
    v2 = Source("fictional", "v2", "ANTF v2", "reglement-antf.md", "b" * 64)
    sections = [
        Section("antf-reglement", "article", "1", "Texte v1.", [], None, v1),
        Section("antf-reglement", "article", "2", "Autre texte v1.", [], None, v1),
        Section("antf-reglement", "article", "1", "Texte v2.", [], None, v2),
    ]

    export_sections(sections, tmp_path)

    manifest = json.loads((tmp_path / "export.json").read_text(encoding="utf-8"))
    assert manifest["sources"] == [asdict(v1), asdict(v2)]
    assert manifest["section_count"] == 3


def test_export_leaves_an_existing_input_manifest_untouched(tmp_path, pipeline_sections):
    # The collision is prevented by a filename, not by code. Renaming export.json
    # back to manifest.json for tidiness would destroy a corpus manifest silently.
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    manifest = corpus / "manifest.json"
    manifest.write_text('{"corpus_id": "x", "files": []}', encoding="utf-8")
    before = manifest.read_bytes()

    export_sections(pipeline_sections, corpus)

    assert manifest.read_bytes() == before
    assert (corpus / "export.json").exists()


def fit_section(text, number="1"):
    return Section("constitution-1958", "article", number, text, [1], None, FAKE_SOURCE)


def rendered_to(total, number="1"):
    """A section whose embedding input is exactly `total` characters.

    The budget applies to embedding_text, not to the bare passage, so a test
    about budgets has to size the thing actually measured.
    """
    head = len(embedding_text(fit_section("", number=number)))
    return fit_section("x" * (total - head), number=number)


def test_a_section_exactly_at_the_budget_is_not_flagged():
    # 350 rendered chars / 3.5 = exactly 100 tokens, and the limit is 100.
    report = estimate_embedding_fit([rendered_to(350)], max_tokens=100, chars_per_token=3.5)

    assert report["over_limit_count"] == 0
    assert report["over_limit"] == []


def test_one_character_over_the_budget_is_flagged():
    # 351 rendered chars / 3.5 = 100.29, rounded up to 101, over a limit of 100.
    report = estimate_embedding_fit(
        [rendered_to(351, number="65")], max_tokens=100, chars_per_token=3.5
    )

    assert report["over_limit_count"] == 1
    flagged = report["over_limit"][0]
    assert flagged["text_id"] == "constitution-1958"
    assert flagged["number"] == "65"
    assert flagged["page_numbers"] == [1]
    assert flagged["chars"] == 351
    assert flagged["estimated_tokens"] == 101


def test_the_report_records_its_own_assumptions():
    report = estimate_embedding_fit([rendered_to(350)], max_tokens=100, chars_per_token=3.5)

    assert report["max_tokens"] == 100
    assert report["assumed_chars_per_token"] == 3.5
    assert report["section_count"] == 1


def test_no_sections_is_handled():
    report = estimate_embedding_fit([], max_tokens=512, chars_per_token=3.5)

    assert report["section_count"] == 0
    assert report["over_limit_count"] == 0
    assert report["chars"] is None


def test_a_non_positive_budget_or_ratio_is_rejected():
    section = [fit_section("x" * 10)]
    for bad in (0, -1):
        with pytest.raises(ValueError):
            estimate_embedding_fit(section, max_tokens=bad, chars_per_token=3.5)
    for bad in (0, -0.5):
        with pytest.raises(ValueError):
            estimate_embedding_fit(section, max_tokens=512, chars_per_token=bad)


def test_estimate_on_the_real_corpus(pipeline_sections):
    generous = estimate_embedding_fit(pipeline_sections, max_tokens=8192, chars_per_token=3.5)
    tight = estimate_embedding_fit(pipeline_sections, max_tokens=512, chars_per_token=3.5)

    assert generous["section_count"] == 159
    assert generous["over_limit_count"] == 0
    assert tight["over_limit_count"] == 6
    # 3701 when the bare passage was measured; the title now rides along, so
    # the figure is derived rather than restated.
    assert generous["chars"]["max"] == max(
        len(embedding_text(s)) for s in pipeline_sections
    )
    assert generous["chars"]["max"] > 3701


def test_the_distribution_uses_nearest_rank_percentiles():
    # Lengths 1..10. int(n*p/100) lands one index too high whenever n*p/100 is
    # whole, which 159 sections never make it. This case does.
    sections = [rendered_to(n, number="1") for n in range(100, 110)]

    report = estimate_embedding_fit(sections, max_tokens=8192, chars_per_token=3.5)

    # Rendered lengths 100..109: nearest rank gives index ceil(n*p/100)-1.
    assert report["chars"] == {"min": 100, "p50": 104, "p90": 108, "p99": 109,
                               "max": 109}


def one_token_per_character(passage):
    """A stand-in tokenizer, so this runs with no download and no network."""
    return len(passage)


def test_measure_fit_counts_the_prefix_as_part_of_the_input():
    section = fit_section("x" * 10)
    # The budget applies to prefix + embedding_text, so the expectation is
    # computed rather than hardcoded: the title rides along in every request.
    whole = len("passage: ") + len(embedding_text(section))
    report = measure_fit(
        [section],
        count_tokens=one_token_per_character,
        prefix="passage: ",
        max_tokens=whole - 1,
    )

    assert report["over_limit_count"] == 1
    assert report["over_limit"][0]["tokens"] == whole
    # Dropping the prefix brings it back under, which is the point.
    assert measure_fit(
        [section],
        count_tokens=one_token_per_character,
        prefix="",
        max_tokens=whole - 1,
    )["over_limit_count"] == 0


def test_measure_fit_reports_the_implied_ratio_for_calibration():
    report = measure_fit(
        [fit_section("x" * 100)],
        count_tokens=lambda passage: len(passage) // 4,
        prefix="",
        max_tokens=8192,
    )

    quarter = len(embedding_text(fit_section("x" * 100))) // 4
    assert report["tokens"] == dict.fromkeys(("min", "p50", "p90", "p99", "max"), quarter)
    # The stand-in counter floors, so the ratio lands just above 4.
    assert report["implied_chars_per_token"] == pytest.approx(4.0, abs=0.1)


def test_measure_fit_records_the_model_and_revision():
    report = measure_fit(
        [fit_section("x" * 10)],
        count_tokens=one_token_per_character,
        prefix="",
        max_tokens=8192,
        model="BAAI/bge-m3",
        revision="5617a9f6",
    )

    assert report["model"] == "BAAI/bge-m3"
    assert report["revision"] == "5617a9f6"
    assert report["method"] == "measured with the model's own tokenizer"


def test_measure_fit_handles_no_sections():
    report = measure_fit(
        [], count_tokens=one_token_per_character, prefix="", max_tokens=512
    )

    assert report["section_count"] == 0
    assert report["implied_chars_per_token"] is None
    assert report["tokens"] is None


# --- embedding_text ----------------------------------------------------------
# What actually goes to the embedder. Measured: with the document title in the
# input, SM-03's gold article moves from rank 5 to 1 and SM-04's from 13 to 1,
# because a question naming "la Declaration ... de 1789" shares no vocabulary
# with an article body that only says "Les hommes naissent et demeurent libres".


def test_the_body_is_the_passage_verbatim():
    # The invariant everything else depends on: exports hash Section.text and
    # citation checking matches against Section.text, so neither may change.
    #
    # The original is captured BEFORE the call. Comparing the result against
    # section.text afterwards would pass even if the renderer had mutated the
    # section, since both sides would move together -- and Section is a mutable
    # dataclass, so that is reachable.
    original = "Les hommes naissent  et demeurent libres."
    section = fit_section(original)

    rendered = embedding_text(section)

    assert section.text == original
    assert rendered.endswith(original)


def test_the_title_and_unit_lead_the_input():
    section = fit_section("Le corps de l'article.", number="66-1")

    assert embedding_text(section) == (
        "Constitution du 4 octobre 1958, article 66-1\nLe corps de l'article."
    )


def test_a_numberless_section_carries_the_title_alone():
    intro = Section("declaration-1789", "intro", None, "DECLARATION\n", [38], None,
                    FAKE_SOURCE)

    first_line = embedding_text(intro).splitlines()[0]
    assert first_line == DOCUMENT_TITLES["declaration-1789"]
    assert "," not in first_line


def test_every_ingested_text_id_has_a_title():
    # A new text in the corpus must declare one rather than inherit a worse
    # configuration silently.
    assert set(DOCUMENT_TITLES) == set(TEXT_IDS_BY_TITLE.values())


def test_a_text_without_a_declared_title_is_refused():
    orphan = Section("antf-reglement", "article", "2", "Trente jours.", [], None,
                     FAKE_SOURCE)

    # Falling back to the slug is NOT neutral: measured, it puts SM-04's gold at
    # rank 4 instead of 1. A silent fallback degrades retrieval invisibly.
    with pytest.raises(ValueError, match="antf-reglement"):
        embedding_text(orphan)


def test_rendering_is_stable():
    section = fit_section("Le meme texte.")

    assert embedding_text(section) == embedding_text(section)


def test_measure_fit_measures_what_is_embedded_not_the_bare_text():
    # The whole point of the function: the budget applies to the real input.
    section = fit_section("x" * 10)
    report = measure_fit([section], one_token_per_character, "", max_tokens=1000)

    assert report["tokens"]["max"] == len(embedding_text(section))
    assert report["tokens"]["max"] > len(section.text)
