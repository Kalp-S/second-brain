import os
import json
from pathlib import Path
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select, func

from backend.app.db.database import get_db
from backend.app.db.models import Document, ParentChunk, ChildChunk
from backend.app.core.config import settings
from backend.app.core.dependencies import get_ingestion_pipeline, IngestionPipeline

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_documents(
    files: List[UploadFile] = File(...),
    session: AsyncSession = Depends(get_db),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline)
) -> Dict[str, Any]:
    """Upload and index multiple documents (Markdown, PDF, Code, Text)."""
    results = []
    for file in files:
        try:
            content_bytes = await file.read()
            res = await pipeline.ingest_bytes(
                filename=file.filename or "untitled.txt",
                raw_bytes=content_bytes,
                session=session
            )
            results.append(res)
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e)
            })

    return {"uploaded": len(files), "results": results}

@router.post("/sync-vault")
async def sync_sample_vault(
    session: AsyncSession = Depends(get_db),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline)
) -> Dict[str, Any]:
    """Scans the sample vault directory and indexes all Markdown and Code notes."""
    vault_dir = Path(settings.VAULT_PATH)
    if not vault_dir.exists():
        return {"status": "error", "message": f"Vault path {vault_dir} does not exist"}

    synced_files = []
    for file_path in vault_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in [".md", ".markdown", ".txt", ".py", ".pdf"]:
            try:
                raw_bytes = file_path.read_bytes()
                rel_name = file_path.name
                res = await pipeline.ingest_bytes(
                    filename=rel_name,
                    raw_bytes=raw_bytes,
                    session=session
                )
                synced_files.append(res)
            except Exception as e:
                synced_files.append({"filename": file_path.name, "status": "error", "message": str(e)})

    return {"status": "success", "synced_count": len(synced_files), "results": synced_files}

@router.get("")
async def list_documents(session: AsyncSession = Depends(get_db)) -> List[Dict[str, Any]]:
    """List all indexed documents in the Second Brain."""
    stmt = select(Document).order_by(Document.created_at.desc())
    res = await session.execute(stmt)
    docs = res.scalars().all()

    output = []
    for d in docs:
        p_count_stmt = select(func.count(ParentChunk.id)).where(ParentChunk.document_id == d.id)
        c_count_stmt = select(func.count(ChildChunk.id)).where(ChildChunk.document_id == d.id)
        
        p_count = (await session.execute(p_count_stmt)).scalar() or 0
        c_count = (await session.execute(c_count_stmt)).scalar() or 0

        tags = []
        try:
            tags = json.loads(d.tags) if d.tags else []
        except Exception:
            pass

        output.append({
            "id": d.id,
            "title": d.title,
            "filename": d.filename,
            "file_type": d.file_type,
            "byte_size": d.byte_size,
            "tags": tags,
            "num_parents": p_count,
            "num_children": c_count,
            "created_at": d.created_at.isoformat() if d.created_at else None,
            "updated_at": d.updated_at.isoformat() if d.updated_at else None
        })
    return output

@router.get("/{doc_id}")
async def get_document(doc_id: str, session: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Retrieve full document details and its hierarchical parent sections."""
    stmt = select(Document).where(Document.id == doc_id)
    res = await session.execute(stmt)
    doc = res.scalars().first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    p_stmt = select(ParentChunk).where(ParentChunk.document_id == doc_id).order_by(ParentChunk.chunk_index)
    p_res = await session.execute(p_stmt)
    parents = p_res.scalars().all()

    tags = []
    try:
        tags = json.loads(doc.tags) if doc.tags else []
    except Exception:
        pass

    return {
        "id": doc.id,
        "title": doc.title,
        "filename": doc.filename,
        "file_type": doc.file_type,
        "byte_size": doc.byte_size,
        "tags": tags,
        "raw_content": doc.raw_content,
        "parents": [
            {
                "id": p.id,
                "chunk_index": p.chunk_index,
                "header_path": p.header_path,
                "token_count": p.token_count,
                "content": p.content
            }
            for p in parents
        ]
    }

@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str,
    session: AsyncSession = Depends(get_db),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline)
) -> Dict[str, Any]:
    """Delete a document and purge all associated vector and lexical index entries."""
    success = await pipeline.delete_document(doc_id, session)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "success", "message": f"Document {doc_id} deleted successfully"}
