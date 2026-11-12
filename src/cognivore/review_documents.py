"""Import document text and preserve its original citation locations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from cognivore.review import MAX_DOCUMENT_BYTES, MAX_LINES, Citation, text_citations

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
_MAX_DOCX_EXPANDED_BYTES = 32 * 1024 * 1024
_MAX_PDF_PAGES = 100
_HEADING_STYLE = re.compile(r"Heading [1-9]$")


class DocumentSupportUnavailable(RuntimeError):
    """The optional document parsers are unavailable."""


@dataclass
class ReviewDocument:
    citations: list[Citation]
    warnings: list[str] = field(default_factory=list)


def read_review_document(name: str, data: bytes) -> ReviewDocument:
    suffix = Path(name).suffix.lower()
    if suffix == ".docx":
        return _read_docx(name, data)
    if suffix == ".pdf":
        return _read_pdf(name, data)
    if suffix == ".doc":
        raise ValueError("Save legacy Word files as DOCX before uploading.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Upload a UTF-8 text, Markdown, DOCX, or text PDF file.") from exc
    return ReviewDocument(text_citations(name, text))


def _read_docx(name: str, data: bytes) -> ReviewDocument:
    try:
        from docx import Document
        from lxml.etree import XMLSyntaxError
    except ImportError as exc:
        raise DocumentSupportUnavailable(
            "Install cognivore[documents] to compare DOCX files."
        ) from exc

    try:
        with ZipFile(BytesIO(data)) as archive:
            if sum(member.file_size for member in archive.infolist()) > _MAX_DOCX_EXPANDED_BYTES:
                raise ValueError("The DOCX expands beyond 32 MiB.")
        document = Document(BytesIO(data))
    except (BadZipFile, KeyError, OSError, RuntimeError, XMLSyntaxError) as exc:
        raise ValueError("The DOCX could not be read.") from exc

    section = ""
    citations = []
    extracted_bytes = 0
    for index, paragraph in enumerate(document.paragraphs, 1):
        raw_text = paragraph.text
        extracted_bytes += len(raw_text.encode("utf-8"))
        if extracted_bytes > MAX_DOCUMENT_BYTES:
            raise ValueError("Each document must have at most 128 KiB of extracted text.")
        value = " ".join(raw_text.split())
        if not value:
            continue
        style = paragraph.style
        if style and (style.name == "Title" or _HEADING_STYLE.fullmatch(style.name or "")):
            section = value
            continue
        citations.append(
            Citation(source=name, line=index, paragraph=index, text=value, section=section)
        )
    if not citations:
        raise ValueError("The DOCX contains no readable body paragraphs.")
    if len(citations) > MAX_LINES:
        raise ValueError(f"Each document must have at most {MAX_LINES} comparison items.")
    warnings = [f"{name}: tables are excluded from comparison."] if document.tables else []
    return ReviewDocument(citations, warnings)


def _read_pdf(name: str, data: bytes) -> ReviewDocument:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:
        raise DocumentSupportUnavailable(
            "Install cognivore[documents] to compare PDF files."
        ) from exc

    citations = []
    skipped_pages = []
    extracted_bytes = 0
    try:
        pdf = PdfReader(BytesIO(data))
        if pdf.is_encrypted:
            raise ValueError("Upload a PDF without password protection.")
        if len(pdf.pages) > _MAX_PDF_PAGES:
            raise ValueError(f"Each PDF must have at most {_MAX_PDF_PAGES} pages.")
        for page_number, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            if not text.strip():
                skipped_pages.append(page_number)
                continue
            extracted_bytes += len(text.encode("utf-8"))
            if extracted_bytes > MAX_DOCUMENT_BYTES:
                raise ValueError("Each document must have at most 128 KiB of extracted text.")
            for line, value in enumerate(text.splitlines(), 1):
                if value.strip():
                    citations.append(
                        Citation(source=name, line=line, page=page_number, text=value.strip())
                    )
            if len(citations) > MAX_LINES:
                raise ValueError(f"Each document must have at most {MAX_LINES} comparison items.")
    except (PdfReadError, KeyError, OSError, RecursionError, TypeError) as exc:
        raise ValueError("The PDF could not be read.") from exc
    if not citations:
        raise ValueError("The PDF has no extractable text. Run OCR before uploading a scan.")
    warnings = []
    if skipped_pages:
        numbers = ", ".join(str(page) for page in skipped_pages)
        warnings.append(f"{name}: no text was extracted from PDF pages {numbers}.")
    return ReviewDocument(citations, warnings)
