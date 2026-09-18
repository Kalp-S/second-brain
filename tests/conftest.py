from collections.abc import AsyncGenerator

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from backend.app.core.dependencies import (
    AdvancedRAGRetriever,
    DenseRetriever,
    IngestionPipeline,
    SparseBM25Retriever,
    get_dense_retriever,
    get_ingestion_pipeline,
    get_llm,
    get_rag_retriever,
    get_reranker,
    get_sparse_retriever,
    reset_dependencies,
)
from backend.app.db.database import get_db
from backend.app.main import _request_history, app
from backend.app.services.llm.provider import MockProvider

# In-memory SQLite for complete test isolation
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(
    TEST_DATABASE_URL, echo=False, future=True, connect_args={"check_same_thread": False}
)

test_session_factory = async_sessionmaker(
    bind=test_engine, class_=AsyncSession, expire_on_commit=False, autocommit=False, autoflush=False
)


async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
    async with test_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@pytest_asyncio.fixture(autouse=True)
async def setup_test_environment():
    """Sets up an isolated in-memory test database and Qdrant instance for each test."""
    _request_history.clear()

    # 1. Recreate tables in in-memory test SQLite
    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
        await conn.run_sync(SQLModel.metadata.create_all)

    # 2. Configure in-memory Qdrant and retrievers
    test_dense = DenseRetriever(storage_path=":memory:")
    test_sparse = SparseBM25Retriever()
    test_reranker = get_reranker()
    test_rag = AdvancedRAGRetriever(
        dense_retriever=test_dense, sparse_retriever=test_sparse, reranker=test_reranker
    )
    test_pipeline = IngestionPipeline(dense_retriever=test_dense, sparse_retriever=test_sparse)

    # 3. Override FastAPI dependencies
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_dense_retriever] = lambda: test_dense
    app.dependency_overrides[get_sparse_retriever] = lambda: test_sparse
    app.dependency_overrides[get_rag_retriever] = lambda: test_rag
    app.dependency_overrides[get_ingestion_pipeline] = lambda: test_pipeline
    app.dependency_overrides[get_llm] = lambda: MockProvider()

    yield

    # Clean up overrides after test
    app.dependency_overrides.clear()
    reset_dependencies()
