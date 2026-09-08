from datetime import datetime, timezone
from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship
import uuid

def generate_uuid() -> str:
    return str(uuid.uuid4())

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class Document(SQLModel, table=True):
    __tablename__ = "documents"

    id: str = Field(default_factory=generate_uuid, primary_key=True)
    title: str = Field(index=True)
    filename: str = Field(index=True)
    file_type: str = Field(default="markdown")
    content_hash: str = Field(index=True)  # SHA-256 for deduplication
    byte_size: int = Field(default=0)
    tags: str = Field(default="[]")  # JSON string array
    raw_content: str = Field(default="")
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    parents: List["ParentChunk"] = Relationship(back_populates="document", cascade_delete=True)

class ParentChunk(SQLModel, table=True):
    __tablename__ = "parent_chunks"

    id: str = Field(default_factory=generate_uuid, primary_key=True)
    document_id: str = Field(foreign_key="documents.id", index=True)
    chunk_index: int = Field(default=0)
    header_path: str = Field(default="")  # e.g. "Overview > Architecture"
    content: str
    token_count: int = Field(default=0)
    created_at: datetime = Field(default_factory=utc_now)

    document: Optional[Document] = Relationship(back_populates="parents")
    children: List["ChildChunk"] = Relationship(back_populates="parent_chunk", cascade_delete=True)

class ChildChunk(SQLModel, table=True):
    __tablename__ = "child_chunks"

    id: str = Field(default_factory=generate_uuid, primary_key=True)
    document_id: str = Field(foreign_key="documents.id", index=True)
    parent_chunk_id: str = Field(foreign_key="parent_chunks.id", index=True)
    chunk_index: int = Field(default=0)
    contextual_content: str  # Prepended with Document Title + Header Path
    content: str  # Raw child text
    token_count: int = Field(default=0)
    created_at: datetime = Field(default_factory=utc_now)

    parent_chunk: Optional[ParentChunk] = Relationship(back_populates="children")

class ChatSession(SQLModel, table=True):
    __tablename__ = "chat_sessions"

    id: str = Field(default_factory=generate_uuid, primary_key=True)
    title: str = Field(default="New Second Brain Chat")
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    messages: List["ChatMessage"] = Relationship(back_populates="session", cascade_delete=True)

class ChatMessage(SQLModel, table=True):
    __tablename__ = "chat_messages"

    id: str = Field(default_factory=generate_uuid, primary_key=True)
    session_id: str = Field(foreign_key="chat_sessions.id", index=True)
    role: str = Field(default="user")  # "user", "assistant", "system"
    content: str
    strategy: str = Field(default="hybrid_reranked")
    citations: str = Field(default="[]")  # JSON string of source citations
    retrieval_trace: str = Field(default="{}")  # JSON string of retrieval steps
    faithfulness_score: Optional[float] = Field(default=None)
    created_at: datetime = Field(default_factory=utc_now)

    session: Optional[ChatSession] = Relationship(back_populates="messages")

class EvaluationRecord(SQLModel, table=True):
    __tablename__ = "evaluation_records"

    id: str = Field(default_factory=generate_uuid, primary_key=True)
    query: str
    strategy: str = Field(default="hybrid_reranked")
    context_relevance: float = Field(default=0.0)
    faithfulness: float = Field(default=0.0)
    answer_relevance: float = Field(default=0.0)
    latency_ms: float = Field(default=0.0)
    model_name: str = Field(default="")
    created_at: datetime = Field(default_factory=utc_now)
