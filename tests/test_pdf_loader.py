import builtins
from pathlib import Path
from types import SimpleNamespace

import app.pdf_loader as pdf_loader


def test_get_ocr_unavailable_reason_when_pytesseract_missing(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pytesseract":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    reason = pdf_loader.get_ocr_unavailable_reason()
    assert reason == "pytesseract is not installed"


def test_get_ocr_unavailable_reason_when_tesseract_binary_missing(monkeypatch) -> None:
    class FakePyTesseract:
        @staticmethod
        def get_tesseract_version():
            raise RuntimeError("binary missing")

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pytesseract":
            return FakePyTesseract
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    reason = pdf_loader.get_ocr_unavailable_reason()
    assert reason == "tesseract binary is not available on PATH"


def test_extract_pdf_pages_text_based_pdf(monkeypatch) -> None:
    class FakePage:
        def __init__(self, text: str):
            self._text = text

        def extract_text(self):
            return self._text

    class FakeReader:
        def __init__(self, *_args, **_kwargs):
            self.pages = [FakePage("Page one text"), FakePage("Page two text")]

    monkeypatch.setattr(pdf_loader, "PdfReader", FakeReader)
    pages = pdf_loader.extract_pdf_pages(Path("dummy.pdf"))
    assert pages == [(1, "Page one text"), (2, "Page two text")]


def test_extract_pdf_pages_ocr_import_error_returns_empty(monkeypatch) -> None:
    class FakePage:
        @staticmethod
        def extract_text():
            return ""

    class FakeReader:
        def __init__(self, *_args, **_kwargs):
            self.pages = [FakePage()]

    monkeypatch.setattr(pdf_loader, "PdfReader", FakeReader)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"fitz", "pytesseract"}:
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    pages = pdf_loader.extract_pdf_pages(Path("dummy.pdf"))
    assert pages == []
