from typing import Dict, Any
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.database import get_db
from backend.app.services.graph.builder import KnowledgeGraphBuilder

router = APIRouter(prefix="/graph", tags=["Knowledge Graph"])

@router.get("")
async def get_knowledge_graph(session: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Returns nodes, edges, and cluster statistics for interactive knowledge graph rendering."""
    return await KnowledgeGraphBuilder.build_graph(session)
