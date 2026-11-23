import pymupdf
import pytest

from backend.services.pdf_processor import PDFProcessingError, extract_pdf
from tests.conftest import make_pdf


def test_extracts_text_from_every_page(leave_pdf):
    doc = extract_pdf(leave_pdf)
    assert doc.num_pages == 3
    assert "20 days of annual leave" in doc.pages[1].text
    assert "Carried forward days expire" in doc.pages[2].text


def test_page_numbers_are_preserved_and_one_based(leave_pdf):
    doc = extract_pdf(leave_pdf)
    assert [p.page_number for p in doc.pages] == [1, 2, 3]
    assert "Annual leave entitlement" in next(p for p in doc.pages if p.page_number == 2).text


def test_empty_page_is_recorded_not_fatal(handbook_pdf):
    doc = extract_pdf(handbook_pdf)
    assert doc.num_pages == 3
    assert doc.empty_pages == [2]
    assert [p.page_number for p in doc.pages] == [1, 3]  # numbering stays aligned with the PDF


@pytest.mark.parametrize(
    "payload,fragment",
    [
        (b"", "empty"),
        (b"this is just text, not a pdf", "not a valid PDF"),
        (b"%PDF-1.7\n garbage garbage garbage", "could not be opened"),
    ],
)
def test_invalid_pdfs_raise_clear_errors(payload, fragment):
    with pytest.raises(PDFProcessingError, match=fragment):
        extract_pdf(payload)


def test_truncated_pdf_is_flagged_not_silently_accepted(leave_pdf):
    truncated = leave_pdf[: len(leave_pdf) // 3]
    try:
        doc = extract_pdf(truncated)
    except PDFProcessingError:
        return  # rejecting is fine
    assert doc.repaired, "a damaged PDF must be reported as repaired"


def test_pdf_without_text_is_rejected():
    with pytest.raises(PDFProcessingError, match="No extractable text"):
        extract_pdf(make_pdf([None, None]))


def test_password_protected_pdf_is_rejected():
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret text")
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    with pytest.raises(PDFProcessingError, match="password"):
        extract_pdf(data)
