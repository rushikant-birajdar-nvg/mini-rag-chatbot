"""PDF text extraction with OCR fallback for scanned documents."""

from pathlib import Path
import logging
from pypdf import PdfReader

logger = logging.getLogger(__name__)


def get_ocr_unavailable_reason() -> str | None:
    """Return a reason when OCR is unavailable, otherwise None."""
    try:
        import pytesseract
    except ImportError:
        return "pytesseract is not installed"

    try:
        _ = pytesseract.get_tesseract_version()
    except Exception:
        return "tesseract binary is not available on PATH"
    return None


def extract_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract per-page text from a PDF, using OCR when direct text is missing."""
    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((idx, text))

    if pages:
        return pages

    # Fallback for scanned/image PDFs if OCR dependencies are available.
    try:
        import fitz
        import pytesseract
        from PIL import Image
    except ImportError:
        logger.warning("Skipping OCR for %s because OCR libraries are not installed", pdf_path)
        return []

    unavailable_reason = get_ocr_unavailable_reason()
    if unavailable_reason:
        logger.warning("Skipping OCR for %s: %s", pdf_path, unavailable_reason)
        return []

    doc = fitz.open(pdf_path)
    try:
        for idx in range(len(doc)):
            page = doc[idx]
            pix = page.get_pixmap(dpi=220)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(
                image,
                config="--psm 6",
            )
            cleaned = " ".join(ocr_text.split())
            if cleaned:
                pages.append((idx + 1, cleaned))
    finally:
        doc.close()

    return pages

