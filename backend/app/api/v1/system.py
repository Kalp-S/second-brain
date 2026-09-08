import httpx
from typing import Dict, Any
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select, func

from backend.app.core.config import settings
from backend.app.db.database import get_db
from backend.app.db.models import Document, ParentChunk, ChildChunk

router = APIRouter(prefix="/system", tags=["System & Health"])

@router.get("/status")
async def get_system_status(session: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    # Check Ollama connectivity
    ollama_status = "unavailable"
    available_models = []
    if settings.LLM_PROVIDER == "ollama":
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
                if r.status_code == 200:
                    ollama_status = "connected"
                    available_models = [m["name"] for m in r.json().get("models", [])]
        except Exception:
            ollama_status = "offline"

    doc_count = (await session.execute(select(func.count(Document.id)))).scalar() or 0
    parent_count = (await session.execute(select(func.count(ParentChunk.id)))).scalar() or 0
    child_count = (await session.execute(select(func.count(ChildChunk.id)))).scalar() or 0

    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "llm_provider": settings.LLM_PROVIDER,
        "ollama_status": ollama_status,
        "active_llm_model": settings.OLLAMA_MODEL if settings.LLM_PROVIDER == "ollama" else settings.OPENAI_MODEL,
        "available_ollama_models": available_models,
        "embedding_model": settings.EMBEDDING_MODEL,
        "reranker_model": settings.RERANKER_MODEL,
        "vault_statistics": {
            "documents": doc_count,
            "parent_chunks": parent_count,
            "child_chunks": child_count
        }
    }
