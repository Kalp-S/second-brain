import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.app.core.config import settings
from backend.app.db.database import init_db, async_session_factory
from backend.app.core.dependencies import (
    get_dense_retriever,
    get_sparse_retriever,
    get_reranker,
    get_ingestion_pipeline,
    get_rag_retriever
)
from backend.app.api.v1.documents import router as documents_router
from backend.app.api.v1.rag import router as rag_router
from backend.app.api.v1.graph import router as graph_router
from backend.app.api.v1.evaluation import router as eval_router
from backend.app.api.v1.system import router as system_router

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize SQLite Database Tables
    await init_db()

    # 2. Warm up singletons (Embeddings, BM25, Reranker)
    dense = get_dense_retriever()
    sparse = get_sparse_retriever()
    get_reranker()
    get_rag_retriever()
    pipeline = get_ingestion_pipeline()

    # 3. Synchronize in-memory BM25 index from database
    async with async_session_factory() as session:
        count = await pipeline.reindex_all_from_db(session)
        print(f"[Lifespan] Second Brain initialized. Synced {count} chunks into BM25/Vector indices.")

        # 4. If DB is empty, auto-sync sample_vault notes
        if count == 0:
            vault_dir = Path(settings.VAULT_PATH)
            if vault_dir.exists():
                for f in vault_dir.glob("*.md"):
                    await pipeline.ingest_bytes(f.name, f.read_bytes(), session)
                print(f"[Lifespan] Seeded knowledge base with sample vault notes.")

    yield
    print("[Lifespan] Shutting down Second Brain.")

app = FastAPI(
    title="Second Brain RAG API",
    description="Enterprise-grade Personal Knowledge Management system using Advanced Hybrid RAG.",
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routers
app.include_router(documents_router, prefix="/api/v1")
app.include_router(rag_router, prefix="/api/v1")
app.include_router(graph_router, prefix="/api/v1")
app.include_router(eval_router, prefix="/api/v1")
app.include_router(system_router, prefix="/api/v1")

# Mount static frontend assets
if FRONTEND_DIR.exists():
    app.mount("/css", StaticFiles(directory=FRONTEND_DIR / "css"), name="css")
    app.mount("/js", StaticFiles(directory=FRONTEND_DIR / "js"), name="js")
    if (FRONTEND_DIR / "assets").exists():
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

    @app.api_route("/", methods=["GET", "HEAD"])
    async def serve_index():
        return FileResponse(FRONTEND_DIR / "index.html")
