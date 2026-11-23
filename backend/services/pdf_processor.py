"""PDF text extraction with PyMuPDF. Preserves 1-based page numbers."""
from __future__ import annotations

import logging

import pymupdf

from backend.schemas.documents import ExtractedDocument, PageText

logger = logging.getLogger(__name__)


class PDFProcessingError(Exception):
    """Raised with a user-presentable reason when a PDF cannot be processed."""


def extract_pdf(content: bytes) -> ExtractedDocument:
    if not content:
        raise PDFProcessingError("File is empty.")
    if not content.lstrip()[:5] == b"%PDF-":
        raise PDFProcessingError("File is not a valid PDF (missing %PDF header).")

    try:
        doc = pymupdf.open(stream=content, filetype="pdf")
    except Exception as exc:  # corrupt / truncated file
        raise PDFProcessingError(f"PDF could not be opened: {_short(exc)}") from exc

    try:
        if doc.needs_pass:
            raise PDFProcessingError("PDF is password-protected.")
        if doc.page_count == 0:
            raise PDFProcessingError("PDF contains no pages.")

        pages: list[PageText] = []
        empty_pages: list[int] = []
        for index in range(doc.page_count):
            page_number = index + 1
            try:
                # Blocks keep paragraph structure; sort=True gives reading order.
                blocks = doc[index].get_text("blocks", sort=True)
                text = "\n\n".join(b[4].strip() for b in blocks if b[6] == 0 and b[4].strip())
            except Exception as exc:
                logger.warning("Page %s extraction failed: %s", page_number, exc)
                text = ""
            if text.strip():
                pages.append(PageText(page_number=page_number, text=text))
            else:
                empty_pages.append(page_number)

        if not pages:
            raise PDFProcessingError(
                "No extractable text found (the PDF may be scanned images; OCR is not supported)."
            )
        return ExtractedDocument(
            num_pages=doc.page_count, pages=pages, empty_pages=empty_pages,
            repaired=bool(getattr(doc, "is_repaired", False)),
        )
    except PDFProcessingError:
        raise
    except Exception as exc:
        raise PDFProcessingError(f"Unexpected extraction error: {_short(exc)}") from exc
    finally:
        doc.close()


def _short(exc: Exception) -> str:
    return str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__
