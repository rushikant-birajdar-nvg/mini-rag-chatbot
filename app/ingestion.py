from pathlib import Path
from typing import Any

from app.chunking import chunk_text
from app.config import get_settings
from app.embeddings import embed_texts
from app.pdf_loader import extract_pdf_pages, get_ocr_unavailable_reason
from app.vector_store import VectorStore

ACCESS_LEVEL_BY_FILE = {
    "hr/leave_policy.pdf": 1,
    "hr/code_of_conduct.pdf": 1,
    "hr/performance_review.pdf": 2,
    "finance/expense_policy.pdf": 1,
    "finance/travel_reimbursement.pdf": 1,
    "exec/compensation_committee.pdf": 3,
    "exec/strategic_plan.pdf": 3,
}


def ingest_documents(docs_root: Path) -> dict[str, Any]:
    settings = get_settings()
    store = VectorStore()
    chunk_count = 0
    doc_count = 0
    scanned_files_skipped = 0
    warnings: list[str] = []
    ocr_unavailable_reason = get_ocr_unavailable_reason()
    ocr_available = ocr_unavailable_reason is None

    texts: list[str] = []
    metadatas: list[dict] = []

    for pdf_path in sorted(docs_root.rglob("*.pdf")):
        rel = pdf_path.relative_to(docs_root).as_posix()
        department = rel.split("/", 1)[0]
        access_level = ACCESS_LEVEL_BY_FILE.get(rel, 1 if department != "exec" else 3)
        pages = extract_pdf_pages(pdf_path)
        if not pages:
            if not ocr_available:
                scanned_files_skipped += 1
                warnings.append(
                    f"Skipped OCR for {rel}: {ocr_unavailable_reason}"
                )
            continue

        doc_count += 1
        for page_num, page_text in pages:
            chunks = chunk_text(
                page_text, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap
            )
            for chunk in chunks:
                texts.append(chunk)
                metadatas.append(
                    {
                        "department": department,
                        "access_level": access_level,
                        "source_file": pdf_path.name,
                        "page": page_num,
                    }
                )
                chunk_count += 1

    vectors = embed_texts(texts) if texts else []
    store.upsert(texts=texts, vectors=vectors, metadatas=metadatas)
    return {
        "documents_ingested": doc_count,
        "chunks_upserted": chunk_count,
        "ocr_available": ocr_available,
        "scanned_files_skipped": scanned_files_skipped,
        "warnings": sorted(set(warnings)),
    }

