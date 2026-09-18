import logging
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api.v1.documents import router as documents_router
from backend.app.api.v1.evaluation import router as eval_router
from backend.app.api.v1.graph import router as graph_router
from backend.app.api.v1.rag import router as rag_router
from backend.app.api.v1.system import router as system_router
from backend.app.core.config import settings
from backend.app.core.dependencies import (
    get_dense_retriever,
    get_ingestion_pipeline,
    get_rag_retriever,
    get_reranker,
    get_sparse_retriever,
)
from backend.app.db.database import async_session_factory, init_db

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"

logger = logging.getLogger("second_brain")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize SQLite Database Tables
    await init_db()

    # 2. Warm up singletons (Embeddings, BM25, Reranker)
    get_dense_retriever()
    get_sparse_retriever()
    get_reranker()
    get_rag_retriever()
    pipeline = get_ingestion_pipeline()

    # 3. Synchronize in-memory BM25 index from database
    async with async_session_factory() as session:
        count = await pipeline.reindex_all_from_db(session)
        print(
            f"[Lifespan] Second Brain initialized. Synced {count} chunks into BM25/Vector indices."
        )

        # 4. If DB is empty, auto-sync sample_vault notes
        if count == 0:
            vault_dir = Path(settings.VAULT_PATH)
            if vault_dir.exists():
                for f in vault_dir.glob("*.md"):
                    await pipeline.ingest_bytes(f.name, f.read_bytes(), session)
                print("[Lifespan] Seeded knowledge base with sample vault notes.")

    yield
    print("[Lifespan] Shutting down Second Brain.")


app = FastAPI(
    title="Second Brain RAG API",
    description="Enterprise-grade Personal Knowledge Management system using Advanced Hybrid RAG.",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)


SERVER_START_TIME = time.time()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Response-Time-Ms"],
)


# Observability Middleware: Correlation ID & Execution Timing
@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    start_time = time.perf_counter()

    response = await call_next(request)

    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = str(duration_ms)

    # Don't clutter logs with static asset queries
    if not request.url.path.startswith(("/css", "/js", "/assets")):
        logger.info(
            f"[{request.method}] {request.url.path} -> {response.status_code} ({duration_ms}ms) [req_id={request_id}]"
        )

    return response


# Sliding-window IP Rate Limiter for RAG inference
_request_history: defaultdict = defaultdict(list)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path.startswith("/api/v1/rag/"):
        # Detect true client IP behind Cloudflare or reverse proxy
        client_ip = (
            request.headers.get("cf-connecting-ip")
            or request.headers.get("x-forwarded-for", "").split(",")[0].strip()
            or (request.client.host if request.client else "127.0.0.1")
        )
        now = time.time()
        window_start = now - 60.0

        # Prune older records
        history = [ts for ts in _request_history[client_ip] if ts > window_start]
        if len(history) >= settings.RATE_LIMIT_PER_MINUTE:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded (demo mode). Please wait a moment before trying again."
                },
                headers={"Retry-After": "15"},
            )
        history.append(now)
        _request_history[client_ip] = history

    return await call_next(request)


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
