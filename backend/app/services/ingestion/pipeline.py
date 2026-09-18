import json
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from backend.app.db.models import ChildChunk, Document, ParentChunk
from backend.app.services.ingestion.chunker import HierarchicalChunker
from backend.app.services.ingestion.parser import DocumentParser, ParsedDocument
from backend.app.services.retrieval.dense import DenseRetriever
from backend.app.services.retrieval.sparse import SparseBM25Retriever


class IngestionPipeline:
    """
    Orchestrates file ingestion, parsing, hierarchical chunking,
    database persistence, and vector + lexical indexing.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseBM25Retriever,
        chunker: HierarchicalChunker | None = None,
    ):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.chunker = chunker or HierarchicalChunker()

    async def ingest_bytes(
        self, filename: str, raw_bytes: bytes, session: AsyncSession
    ) -> dict[str, Any]:
        parsed_doc: ParsedDocument = DocumentParser.parse_bytes(filename, raw_bytes)

        # Idempotency check: Check if document with identical SHA-256 hash already exists
        existing_stmt = select(Document).where(Document.content_hash == parsed_doc.content_hash)
        existing_res = await session.execute(existing_stmt)
        existing_doc = existing_res.scalars().first()

        if existing_doc:
            return {
                "status": "already_exists",
                "document_id": existing_doc.id,
                "title": existing_doc.title,
                "message": f"Document '{existing_doc.title}' with identical hash is already indexed.",
            }

        # Check if filename exists with older content; if so, clean up old records & vectors
        name_stmt = select(Document).where(Document.filename == filename)
        name_res = await session.execute(name_stmt)
        old_doc = name_res.scalars().first()
        if old_doc:
            await self.delete_document(old_doc.id, session)

        # 1. Create Document in DB
        doc_id = str(uuid.uuid4())
        doc_record = Document(
            id=doc_id,
            title=parsed_doc.title,
            filename=filename,
            file_type=parsed_doc.file_type,
            content_hash=parsed_doc.content_hash,
            byte_size=parsed_doc.byte_size,
            tags=json.dumps(parsed_doc.tags),
            raw_content=parsed_doc.content,
        )
        session.add(doc_record)
        await session.flush()

        # 2. Hierarchical Chunking
        parent_dtos = self.chunker.chunk_document(parsed_doc.title, parsed_doc.content)

        dense_points: list[dict[str, Any]] = []
        sparse_records: list[dict[str, Any]] = []
        all_child_texts: list[str] = []

        for p_dto in parent_dtos:
            parent_id = str(uuid.uuid4())
            parent_record = ParentChunk(
                id=parent_id,
                document_id=doc_id,
                chunk_index=p_dto.chunk_index,
                header_path=p_dto.header_path,
                content=p_dto.content,
                token_count=p_dto.token_count,
            )
            session.add(parent_record)
            await session.flush()

            for c_dto in p_dto.children:
                child_id = str(uuid.uuid4())
                child_record = ChildChunk(
                    id=child_id,
                    document_id=doc_id,
                    parent_chunk_id=parent_id,
                    chunk_index=c_dto.chunk_index,
                    contextual_content=c_dto.contextual_content,
                    content=c_dto.content,
                    token_count=c_dto.token_count,
                )
                session.add(child_record)

                # Prepare for embedding
                all_child_texts.append(c_dto.contextual_content)
                dense_points.append(
                    {
                        "id": child_id,
                        "payload": {
                            "child_id": child_id,
                            "parent_id": parent_id,
                            "document_id": doc_id,
                            "doc_title": parsed_doc.title,
                            "header_path": p_dto.header_path,
                            "content": c_dto.content,
                        },
                    }
                )
                sparse_records.append(
                    {
                        "id": child_id,
                        "child_id": child_id,
                        "parent_id": parent_id,
                        "document_id": doc_id,
                        "doc_title": parsed_doc.title,
                        "header_path": p_dto.header_path,
                        "content": c_dto.content,
                    }
                )

        await session.commit()

        # 3. Vector Embeddings & Indexing
        if all_child_texts:
            embeddings = self.dense.embed_texts(all_child_texts)
            for idx, pt in enumerate(dense_points):
                pt["vector"] = embeddings[idx]
            self.dense.index_chunks(dense_points)

        # 4. Sparse BM25 Indexing
        if sparse_records:
            self.sparse.index_chunks(sparse_records)

        return {
            "status": "success",
            "document_id": doc_id,
            "title": parsed_doc.title,
            "filename": filename,
            "num_parents": len(parent_dtos),
            "num_children": len(dense_points),
            "tags": parsed_doc.tags,
            "links": parsed_doc.links,
        }

    async def delete_document(self, document_id: str, session: AsyncSession) -> bool:
        stmt = select(Document).where(Document.id == document_id)
        res = await session.execute(stmt)
        doc = res.scalars().first()
        if not doc:
            return False

        # Remove from vector index
        self.dense.delete_by_document(document_id)

        # Remove from sparse index
        self.sparse.remove_by_document(document_id)

        # Remove from DB (cascade deletes parents & children)
        await session.delete(doc)
        await session.commit()
        return True

    async def reindex_all_from_db(self, session: AsyncSession) -> int:
        """Rebuilds the in-memory BM25 index and ensures vector store is populated."""
        child_stmt = (
            select(ChildChunk, ParentChunk, Document)
            .join(ParentChunk, ChildChunk.parent_chunk_id == ParentChunk.id)
            .join(Document, ChildChunk.document_id == Document.id)
        )
        res = await session.execute(child_stmt)
        rows = res.all()

        sparse_records = []
        for child, parent, doc in rows:
            sparse_records.append(
                {
                    "id": child.id,
                    "child_id": child.id,
                    "parent_id": parent.id,
                    "document_id": doc.id,
                    "doc_title": doc.title,
                    "header_path": parent.header_path,
                    "content": child.content,
                }
            )

        # Rebuild BM25 in memory (takes <2ms)
        if sparse_records:
            self.sparse.index_chunks(sparse_records)

        return len(sparse_records)
