"""Print one cleaned page, line by line, with repr() so whitespace is visible.

Usage: PYTHONPATH=src python scripts/show_page.py [page_number]
"""
import sys
from pathlib import Path

from rag_portfolio.ingest import clean_page_text, extract_pages

PATH = Path(__file__).resolve().parents[1]
PDF = PATH / "data" / "constitution" / "constitution.pdf"


def main() -> None:
    wanted = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    records = extract_pages(PDF)
    # Guard the index: page 0 would otherwise print the last page via negative indexing.
    if not 1 <= wanted <= len(records):
        sys.exit(f"page {wanted} is outside 1..{len(records)}")

    record = records[wanted - 1]
    text = clean_page_text(record["raw_text"], record["page_number"])
    print(f"page {record['page_number']} of {len(records)}")
    for index, line in enumerate(text.split("\n")):
        print(f"{index:>3} {line!r}")


if __name__ == "__main__":
    main()
