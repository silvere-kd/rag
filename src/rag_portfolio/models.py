from dataclasses import dataclass



@dataclass(frozen=True)
class Source:
    corpus_id: str
    version: str
    edition_label: str
    filename: str
    sha256: str

@dataclass
class Section:
    text_id: str
    kind: str
    number: str | None
    text: str
    page_numbers: list[int]
    title_context: str | None
    source: Source


@dataclass(frozen=True)
class Citation:
    """One checked reference from an answer back to the passage it quoted.

    source_id is a label for a single answer (S1, S2), never a persistent id.

    excerpt was matched against section.text ignoring whitespace, which is
    weaker than verbatim: "nowhere" matches "now here". That is the deliberate
    price of repairing pypdf's intra-word splits ("est \u00e9 lu"), since both are
    the same operation. It establishes that the quote is present in the passage,
    never that the passage supports the answer.

    Holding the Section itself keeps the original passage available, so a caller
    can print the filename, edition and pages without the model supplying any of
    them, and a reader can judge the support for themselves.
    """

    source_id: str
    excerpt: str
    section: Section