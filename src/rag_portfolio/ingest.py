import math
import re
from dataclasses import asdict
from pathlib import Path
from pypdf import PdfReader
from .models import Section, Source
import hashlib
import json


HEADER = "1 janvier 2015"
TEXT_IDS_BY_TITLE = {
    "CONSTITUTION": "constitution-1958",
    "CHARTE DE L'ENVIRONNEMENT DE 2004": "charte-environnement-2004",
    "PRÉAMBULE DE LA CONSTITUTION DU 27 OCTOBRE 1946": "preambule-1946",
    "DÉCLARATION DES DROITS DE L’HOMME ET DU CITOYEN DU 26 AOÛT 1789": "declaration-1789",
}
# The human title of each text, used only as embedding input. Deliberately not
# derived by inverting TEXT_IDS_BY_TITLE: that map classifies pages by the
# heading printed on them, and its Constitution key is the bare word
# "CONSTITUTION", which is a heading rather than a title.
DOCUMENT_TITLES = {
    "constitution-1958": "Constitution du 4 octobre 1958",
    "charte-environnement-2004": "Charte de l'environnement de 2004",
    "preambule-1946": "Préambule de la Constitution du 27 octobre 1946",
    "declaration-1789": "Déclaration des droits de l'homme et du citoyen du 26 août 1789",
}
_KEYWORD = re.compile(r"^\s*article\s+(premier|1er|\d+(?:-\d+)?)\b", re.IGNORECASE)
_NUMBERED_PARAGRAPH = re.compile(r"^\s*(\d+)\s*\.")
_ROMAN = re.compile(r"^\s*([IVXL]+)\s*\.?\s*$")

# Each text numbers its articles its own way. The Constitution and the Charte share
# the ARTICLE keyword; the 1946 preamble uses bare numbered paragraphs; the 1789
# Declaration keyword-marks its first article and uses Roman numerals after it.
ARTICLE_PATTERNS = {
    "constitution-1958": (_KEYWORD,),
    "charte-environnement-2004": (_KEYWORD,),
    "preambule-1946": (_NUMBERED_PARAGRAPH,),
    "declaration-1789": (_KEYWORD, _ROMAN),
}

# Only the Constitution carries these; nothing else in the document opens a line
# with "Titre". The numeral is deliberately not parsed: the heading is context text,
# not an identifier, which is why "Titre XI BIS" needs no special case.
_TITLE_HEADING = re.compile(r"^\s*titre\b", re.IGNORECASE)

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50}

def extract_pages(source_path: Path):
    doc = PdfReader(source_path)
    records = [{"page_number": page_number, "raw_text": page.extract_text()} for 
               page_number, page in enumerate(doc.pages, start=1)]
    return records

def clean_page_text(raw_text: str, page_number: int) -> str:
    """Drop this PDF's running header and page-number footer, nothing else."""
    lines = raw_text.splitlines()
    filled = [i for i, line in enumerate(lines) if line.strip()]

    if not filled:
         return ""
    
    if lines[filled[0]].strip() == HEADER:
        filled = filled[1:]

    if filled and lines[filled[-1]].strip() == str(page_number):
         filled = filled[:-1]

    if not filled:
         return ""

    return "\n".join(lines[filled[0]:filled[-1] + 1])

def detect_text_heading(cleaned_text: str) -> str | None:
    """The text id this page opens, or None when it continues the previous one."""
    opening = " ".join(cleaned_text.split()).upper()
    for title, text_id in TEXT_IDS_BY_TITLE.items():
        if opening.startswith(title):
            return text_id
    return None

def assign_text_ids(raw_records: list[dict]) -> list[dict]:
    extended_records = []
    current_text_id = None
    for record in raw_records:
        clean_text = clean_page_text(record["raw_text"], record["page_number"])
        text_id = detect_text_heading(clean_text)
        if text_id is not None:
            current_text_id = text_id
        if current_text_id is None:
            raise ValueError(
                f"page {record['page_number']} opens no known text and has none to inherit"
            )

        extended_records.append(
            {
                "page_number": record["page_number"],
                "raw_text": record["raw_text"],
                "clean_text": clean_text,
                "text_id": current_text_id
            }
        )
    return extended_records

def normalize_article_id(raw: str) -> str:
    """One id per article, whatever its text calls it: PREMIER, 1er and I all mean 1."""
    token = raw.strip().upper()
    if token in {"PREMIER", "1ER"}:
        return "1"
    if re.fullmatch(r"[IVXL]+", token):
        total = previous = 0
        for char in reversed(token):
            value = _ROMAN_VALUES[char]
            total += -value if value < previous else value
            previous = max(previous, value)
        return str(total)
    return token.lower()

def detect_article_start(line: str, text_id: str) -> str | None:
    """The canonical article id this line opens, or None."""
    for pattern in ARTICLE_PATTERNS[text_id]:
        match = pattern.match(line)
        if match:
            return normalize_article_id(match.group(1))
    return None

def detect_title_heading(line: str) -> str | None:
    """The Constitution title heading on this line, such as "Titre XI BIS", or None."""
    return line.strip() if _TITLE_HEADING.match(line) else None

def detect_preamble_heading(line: str) -> str | None:
    """The section kind when this line is a standalone preamble label, or None."""
    if line.strip() == "PRÉAMBULE":
        return "preamble"
    return None

def assemble_sections(records: list[dict], source: Source) -> list[Section]:
    """Cut each text into its logical units, keeping the pages each one covers.

    A page joins a section only when it contributes body text to it, which is why a
    title heading has to close the previous section: otherwise an article running to
    the next article marker would swallow the following page's heading block.
    """
    sections: list[Section] = []
    text_id: str | None = None
    title_context: str | None = None
    title_lines: list[str] = []
    collecting_title = False
    kind: str | None = None
    number: str | None = None
    body: list[str] = []
    pages: list[int] = []

    def close() -> None:
        nonlocal kind, number, body, pages
        if kind is not None and any(line.strip() for line in body):
            sections.append(
                Section(text_id, kind, number, "\n".join(body), pages, title_context, source)
            )
        kind, number, body, pages = None, None, [], []

    def open_unit(new_kind: str, new_number: str | None) -> None:
        nonlocal kind, number, body, pages
        kind, number, body, pages = new_kind, new_number, [], []

    def add(line: str, page: int) -> None:
        body.append(line)
        if line.strip() and page not in pages:
            pages.append(page)

    for record in records:
        page = record["page_number"]
        if record["text_id"] != text_id:
            close()
            text_id = record["text_id"]
            title_context = None
            title_lines, collecting_title = [], False

        for line in record["clean_text"].splitlines():
            preamble = detect_preamble_heading(line)
            title = detect_title_heading(line)
            unit_number = detect_article_start(line, text_id)

            if preamble:
                close()
                title_lines, collecting_title = [], False
                open_unit(preamble, None)
                add(line, page)
            elif title:
                close()
                title_lines, collecting_title = [title], True
            elif unit_number:
                close()
                if collecting_title:
                    title_context = "\n".join(title_lines)
                    title_lines, collecting_title = [], False
                open_unit(
                    "paragraph" if text_id == "preambule-1946" else "article", unit_number
                )
                add(line, page)
            elif collecting_title:
                if line.strip():
                    title_lines.append(line.strip())
            elif kind is not None:
                add(line, page)
            elif line.strip():
                open_unit("intro", None)
                add(line, page)

    close()
    return sections

def file_sha256(path: Path) -> str:
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def load_sources(source_path: Path) -> list[Source]:
    sources: list[Source] = []
    manifest_path = source_path / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    # data describes the corpus once; entry describes one file within it.
    for entry in data["files"]:
        sources.append(
            Source(
                data["corpus_id"],
                data["version"],
                data["edition_label"],
                entry["filename"],
                file_sha256(source_path / entry["filename"]),
            )
        )
    return sources

def ingest_pdf(source_dir: Path) -> list[Section]:
    list_sections: list[Section] = []
    # load sources
    sources = load_sources(source_dir)

    for source in sources:
        file_path =  source_dir / source.filename
        pages = extract_pages(file_path)
        records = assign_text_ids(pages)
        list_sections.extend(assemble_sections(records, source))

    return list_sections


def write_sections(sections: list[Section], output_path: Path) -> None:
    """Write one JSON object per line, replacing whatever was there before."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for section in sections:
            handle.write(
                json.dumps(asdict(section), 
                           ensure_ascii=False,
                           sort_keys=True) + "\n"
            )


def export_sections(sections: list[Section], output_dir: Path) -> None:
    """Write the artifact, then a manifest describing it.

    The order is load-bearing: the manifest records the hash of the finished
    sections file, so it cannot be inside its own hash.

    Named export.json rather than manifest.json on purpose: a corpus directory
    already holds a hand-written manifest.json declaring its inputs, and writing
    an output here under that name would destroy it.
    """
    sections_path = output_dir / "sections.jsonl"
    write_sections(sections, sections_path)

    manifest = {
        # dict.fromkeys dedupes while keeping first appearance; it needs Source
        # frozen. Deduplicating by filename would merge v1 and v2 of one document.
        "sources": [
            asdict(source)
            for source in dict.fromkeys(section.source for section in sections)
        ],
        "section_count": len(sections),
        "artifact": {
            "filename": sections_path.name,
            "sha256": file_sha256(sections_path),
        },
    }
    (output_dir / "export.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

def length_distribution(lengths: list[int]) -> dict | None:
    """Nearest-rank percentiles, or None when there is nothing to describe."""
    if not lengths:
        return None

    def pick(percent: int) -> int:
        # Nearest rank: the 1-based position is ceil(n * p / 100), so subtract one
        # for the index. int() instead of ceil() lands one too high whenever
        # n * p / 100 is a whole number.
        return lengths[math.ceil(len(lengths) * percent / 100) - 1]

    return {
        "min": lengths[0],
        "p50": pick(50),
        "p90": pick(90),
        "p99": pick(99),
        "max": lengths[-1],
    }


def estimate_embedding_fit(sections: list[Section],
                           max_tokens: int,
                           chars_per_token: float,
                           ) -> dict:
    """Which sections might exceed an embedding input budget, by character ratio.

    An estimate, not a measurement. The chosen model's own tokenizer still has to
    verify the complete input, including any required query/passage prefix and
    special tokens, and a different tokenizer can produce more tokens rather than
    fewer. Use this to shortlist models and plan splitting, not to claim a limit.
    """
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    if chars_per_token <= 0:
        raise ValueError(f"chars_per_token must be positive, got {chars_per_token}")

    # Rendered like measure_fit, so the script's "N measured, M estimated"
    # comparison is between the same inputs.
    rendered = {id(section): embedding_text(section) for section in sections}
    estimates = [
        (section, math.ceil(len(rendered[id(section)]) / chars_per_token))
        for section in sections
    ]
    over = [
        {
            "text_id": section.text_id,
            "kind": section.kind,
            "number": section.number,
            "page_numbers": section.page_numbers,
            "chars": len(rendered[id(section)]),
            "estimated_tokens": tokens,
        }
        for section, tokens in estimates
        if tokens > max_tokens
    ]
    return {
        "method": "character-ratio estimate; verify with the model's own tokenizer",
        "section_count": len(sections),
        "max_tokens": max_tokens,
        "assumed_chars_per_token": chars_per_token,
        "chars": length_distribution(sorted(len(rendered[id(s)]) for s in sections)),
        "over_limit_count": len(over),
        "over_limit": sorted(over, key=lambda row: -row["chars"]),
    }


def embedding_text(section: Section) -> str:
    """What is actually sent to the embedder: provenance, then the passage.

    A question can identify a passage by facts its body never states. "Que dit
    l'article premier de la Déclaration des droits de l'homme et du citoyen de
    1789 ?" shares no vocabulary with "Les hommes naissent et demeurent libres
    et égaux en droits", because the document's name lives in its title. Four
    texts in this corpus each have an article 1, so without the title their
    vectors are nearly indistinguishable.

    Measured on the smoke cases with the titles below: SM-03's gold article moves
    from rank 5 to 2 and SM-04's from 13 (outside k=5) to 1, with no movement on
    the three that already worked and no score inflation on the unanswerable
    question. An exploratory run using title-cased headings put SM-03 at rank 1;
    the exact title string is worth one rank here, so these numbers belong to
    DOCUMENT_TITLES as written and not to the idea of adding a title.

    Section.text is returned verbatim, artefacts included. The title is input to
    the embedder only: exports hash Section.text and citation checking matches
    against Section.text, and neither may see this.
    """
    title = DOCUMENT_TITLES.get(section.text_id)
    if title is None:
        # Not a neutral default: falling back to the slug was measured at rank 4
        # for SM-04 against rank 1 with the title. A silent fallback would
        # degrade retrieval invisibly, which is the bug this function fixes.
        raise ValueError(
            f"{section.text_id} has no entry in DOCUMENT_TITLES; declare one "
            f"rather than embedding the passage without its provenance"
        )
    head = f"{title}, {section.kind} {section.number}" if section.number else title
    return f"{head}\n{section.text}"


def measure_fit(
    sections: list[Section],
    count_tokens,
    prefix: str,
    max_tokens: int,
    model: str = "",
    revision: str = "",
) -> dict:
    """Which sections exceed an embedding input budget, by actual token count.

    `count_tokens` takes the finished passage string and returns its token count.
    It is injected rather than being a tokenizer object so this function stays
    testable without a download: the script wires a real tokenizer into it.

    The report mirrors estimate_embedding_fit so the two can be compared directly.
    """
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")

    # Budgeted against embedding_text, not section.text: the title rides along
    # in every request, so a measurement of the bare passage understates it.
    rendered = {id(section): prefix + embedding_text(section) for section in sections}
    measured = [(section, count_tokens(rendered[id(section)])) for section in sections]
    over = [
        {
            "text_id": section.text_id,
            "kind": section.kind,
            "number": section.number,
            "page_numbers": section.page_numbers,
            "chars": len(rendered[id(section)]),
            "tokens": tokens,
        }
        for section, tokens in measured
        if tokens > max_tokens
    ]
    total_chars = sum(len(rendered[id(s)]) for s in sections)
    total_tokens = sum(tokens for _, tokens in measured)
    return {
        "method": "measured with the model's own tokenizer",
        "model": model,
        "revision": revision,
        "prefix": prefix,
        "section_count": len(sections),
        "max_tokens": max_tokens,
        # Feed this back into estimate_embedding_fit to calibrate the cheap screen.
        "implied_chars_per_token": (
            total_chars / total_tokens if total_tokens else None
        ),
        "chars": length_distribution(sorted(len(rendered[id(s)]) for s in sections)),
        "tokens": length_distribution(sorted(t for _, t in measured)),
        "over_limit_count": len(over),
        "over_limit": sorted(over, key=lambda row: -row["tokens"]),
    }
