import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    APP_NAME: str = "Second Brain RAG"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    DEMO_MODE: bool = True
    RATE_LIMIT_PER_MINUTE: int = 20

    # Storage Paths
    BASE_PATH: Path = BASE_DIR
    DATA_PATH: Path = BASE_DIR / "data"
    VAULT_PATH: Path = BASE_DIR / "sample_vault"
    SQLITE_DB_URL: str = f"sqlite+aiosqlite:///{BASE_DIR / 'data' / 'second_brain.db'}"
    QDRANT_PATH: Path = BASE_DIR / "data" / "qdrant"

    # Embedding & Reranker Models (Local ONNX)
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384
    RERANKER_MODEL: str = "ms-marco-TinyBERT-L-2-v2"

    # Ingestion & Chunking parameters
    PARENT_CHUNK_SIZE: int = 1200
    CHILD_CHUNK_SIZE: int = 320
    CHUNK_OVERLAP: int = 50

    # Retrieval settings
    DENSE_TOP_K: int = 15
    BM25_TOP_K: int = 15
    RRF_K: int = 60  # Standard Reciprocal Rank Fusion constant
    RERANK_TOP_K: int = 5

    # LLM Settings
    LLM_PROVIDER: str = "ollama"  # "ollama", "openai", "gemini", "groq", "mock"
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "qwen2.5-coder:3b-instruct"

    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"

    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"


settings = Settings()
os.makedirs(settings.DATA_PATH, exist_ok=True)
os.makedirs(settings.VAULT_PATH, exist_ok=True)
os.makedirs(settings.QDRANT_PATH, exist_ok=True)
