from io import BytesIO
from pathlib import Path

from docx import Document
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from cognivore.config import Settings
from cognivore.web.app import create_app


def _docx() -> bytes:
    document = Document()
    document.add_heading("Power", level=1)
    document.add_paragraph()
    document.add_paragraph("PWR-1: The gateway shall run for 30 minutes.")
    document.add_table(rows=1, cols=1).cell(0, 0).text = "Excluded table text."
    stream = BytesIO()
    document.save(stream)
    return stream.getvalue()


def _pdf(*, blank: bool = False) -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    if not blank:
        page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        content = DecodedStreamObject()
        content.set_data(
            b"BT /F1 12 Tf 20 200 Td (PWR-1: The gateway shall run for 20 minutes.) Tj ET"
        )
        page[NameObject("/Contents")] = content
    stream = BytesIO()
    writer.write(stream)
    return stream.getvalue()


def _settings(path: Path) -> Settings:
    return Settings(
        data_dir=path, prefer_semantic_embedder=False, llm_provider="fake", seed_demo_kb=False
    )


def test_mixed_document_review_preserves_locations_and_import_notes(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        created = client.post(
            "/api/reviews",
            files={"before": ("v1.docx", _docx()), "after": ("v2.pdf", _pdf())},
        )
        assert created.status_code == 201, created.text
        review = created.json()
        assert len(review["findings"]) == 1
        finding = review["findings"][0]
        assert finding["signals"] == ["number or unit"]
        assert finding["before"]["paragraph"] == 3
        assert finding["before"]["section"] == "Power"
        assert finding["after"]["page"] == 2
        assert review["warnings"] == [
            "v1.docx: tables are excluded from comparison.",
            "v2.pdf: no text was extracted from PDF pages 1.",
        ]
        url = f"/api/reviews/{review['id']}"
        assert client.put(f"{url}/findings/0", json={"decision": "accepted"}).status_code == 200

    with TestClient(create_app(_settings(tmp_path))) as client:
        assert client.get(url).json()["warnings"] == review["warnings"]
        assert client.post(f"{url}/publish").status_code == 200
        report = client.get(f"{url}/report").text
        assert "v1.docx, paragraph 3" in report
        assert "v2.pdf, page 2" in report
        assert "Import notes" in report


def test_empty_pdf_and_invalid_docx_return_import_errors(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        for name, content, expected in [
            ("scan.pdf", _pdf(blank=True), "no extractable text"),
            ("broken.docx", b"not a ZIP archive", "could not be read"),
        ]:
            response = client.post(
                "/api/reviews", files={"before": (name, content), "after": ("new.txt", b"Text")}
            )
            assert response.status_code == 422
            assert expected in response.json()["detail"]
