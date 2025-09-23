# dagsordener/app/parser.py
from pathlib import Path
from typing import NotRequired, TypedDict

from docling.document_converter import DocumentConverter


class Record(TypedDict):
    text: str
    type: NotRequired[str | None]
    page_start: NotRequired[int | None]
    page_end: NotRequired[int | None]
    title: NotRequired[str | None]
    source: str


def parse_pdf(pdf_path: str | Path) -> list[Record]:
    """
    Parse a PDF with Docling and return a list of records:
      [
        {
          "text": "...",
          "type": "Paragraph" | "Title" | "List" | "Table" | ...,
          "page_start": int | None,
          "page_end": int | None,
          "title": str | None,
          "source": "path/to/file.pdf",
        },
        ...
      ]

    If fine-grained elements aren't available, returns a single record with the
    whole document text in "text".
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))

    # Newer Docling returns an object with a .document; be defensive.
    doc = getattr(result, 'document', result)

    records: list[Record] = []

    # Try element-wise export first (preferred for RAG).
    elements = getattr(doc, 'elements', None)
    if elements:
        for el in elements:
            # Try to get text from the element; fall back to str(el) if needed.
            text = getattr(el, 'text', None)
            if not text and hasattr(el, 'to_text'):
                try:
                    text = el.to_text()
                except Exception:
                    text = None
            if not text:
                # Skip empty elements
                continue

            # Best-effort metadata
            el_type = getattr(el, 'category', el.__class__.__name__)
            page_start = getattr(el, 'start_page', None)
            page_end = getattr(el, 'end_page', None)
            title = getattr(el, 'title', None)

            records.append(
                {
                    'text': text,
                    'type': str(el_type) if el_type else None,
                    'page_start': int(page_start) if page_start is not None else None,
                    'page_end': int(page_end) if page_end is not None else None,
                    'title': title,
                    'source': str(pdf_path),
                }
            )

    # Fallback: one big text blob if we didn't collect anything above.
    if not records:
        # Try a full-document export method if present.
        whole_text: str | None = None
        for attr in ('export_to_text', 'to_text'):
            if hasattr(doc, attr):
                try:
                    whole_text = getattr(doc, attr)()
                    break
                except Exception:
                    pass
        if whole_text:
            records.append(
                {
                    'text': whole_text,
                    'type': 'Document',
                    'page_start': None,
                    'page_end': None,
                    'title': getattr(doc, 'title', None),
                    'source': str(pdf_path),
                }
            )

    return records
