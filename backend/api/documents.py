from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from backend.schemas.documents import (
    DeleteResponse, DocumentListResponse, UploadResponse, UploadResult,
)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
def upload_documents(request: Request, files: list[UploadFile] = File(...)) -> UploadResponse:
    """Upload one or more PDFs. Each file gets its own result; one bad file never fails the batch."""
    c = request.app.state.container
    limit = c.settings.max_upload_bytes
    results: list[UploadResult] = []
    for f in files:
        name = f.filename or "unnamed"
        if not name.lower().endswith(".pdf"):
            results.append(UploadResult(filename=name, status="rejected", message="Only .pdf files are supported."))
            continue
        content = f.file.read(limit + 1)
        if len(content) > limit:
            results.append(UploadResult(
                filename=name, status="rejected",
                message=f"File exceeds the {c.settings.max_upload_mb} MB limit.",
            ))
            continue
        res = c.documents.ingest(name, content)
        results.append(UploadResult(
            filename=name, status=res.outcome, message=res.message,
            document_id=res.record.document_id, num_pages=res.record.num_pages or None,
            num_chunks=res.record.num_chunks or None,
        ))
    return UploadResponse(results=results)


@router.get("", response_model=DocumentListResponse)
def list_documents(request: Request) -> DocumentListResponse:
    docs = request.app.state.container.registry.list()
    return DocumentListResponse(documents=docs, total=len(docs))


@router.delete("/{document_id}", response_model=DeleteResponse)
def delete_document(document_id: str, request: Request) -> DeleteResponse:
    removed = request.app.state.container.documents.delete(document_id)
    if removed is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DeleteResponse(deleted=document_id, chunks_removed=removed)
