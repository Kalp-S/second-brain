import time
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from backend.app.core.config import settings
from backend.app.db.models import ParentChunk
from backend.app.services.retrieval.dense import DenseRetriever
from backend.app.services.retrieval.fusion import ReciprocalRankFusion
from backend.app.services.retrieval.reranker import CrossEncoderReranker
from backend.app.services.retrieval.sparse import SparseBM25Retriever


class RetrievalResult:
    def __init__(
        self,
        strategy: str,
        final_chunks: list[dict[str, Any]],
        context_text: str,
        citations: list[dict[str, Any]],
        trace: dict[str, Any],
    ):
        self.strategy = strategy
        self.final_chunks = final_chunks
        self.context_text = context_text
        self.citations = citations
        self.trace = trace


class AdvancedRAGRetriever:
    """
    Orchestrates the Advanced Multi-Stage RAG Pipeline:
    1. Parallel Dense & Sparse Retrieval
    2. Reciprocal Rank Fusion (RRF)
    3. Cross-Encoder Reranking
    4. Hierarchical Parent Context Expansion
    5. Detailed Retrieval Trace Recording
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: SparseBM25Retriever | None = None,
        reranker: CrossEncoderReranker | None = None,
    ):
        self.dense = dense_retriever or DenseRetriever()
        self.sparse = sparse_retriever or SparseBM25Retriever()
        self.fusion = ReciprocalRankFusion(k=settings.RRF_K)
        self.reranker = reranker or CrossEncoderReranker()

    async def retrieve(
        self,
        query: str,
        session: AsyncSession,
        strategy: str = "hybrid_reranked",
        top_k: int = settings.RERANK_TOP_K,
    ) -> RetrievalResult:
        trace: dict[str, Any] = {
            "query": query,
            "strategy": strategy,
            "latencies_ms": {},
            "dense_hits": [],
            "sparse_hits": [],
            "fused_hits": [],
            "reranked_hits": [],
        }

        t0_total = time.perf_counter()

        # Step 1: Dense Retrieval
        t0 = time.perf_counter()
        dense_hits = self.dense.search(query, top_k=settings.DENSE_TOP_K)
        t_dense = (time.perf_counter() - t0) * 1000
        trace["latencies_ms"]["dense"] = round(t_dense, 2)
        trace["dense_hits"] = [
            {
                "id": h["id"],
                "score": round(h["score"], 4),
                "rank": h["rank"],
                "title": h["payload"].get("doc_title", "Unknown"),
                "header": h["payload"].get("header_path", ""),
                "snippet": h["payload"].get("content", "")[:120] + "...",
            }
            for h in dense_hits
        ]

        # Step 2: Sparse BM25 Retrieval
        t0 = time.perf_counter()
        sparse_hits = self.sparse.search(query, top_k=settings.BM25_TOP_K)
        t_sparse = (time.perf_counter() - t0) * 1000
        trace["latencies_ms"]["sparse"] = round(t_sparse, 2)
        trace["sparse_hits"] = [
            {
                "id": h["id"],
                "score": round(h["score"], 4),
                "rank": h["rank"],
                "title": h["payload"].get("doc_title", "Unknown"),
                "header": h["payload"].get("header_path", ""),
                "snippet": h["payload"].get("content", "")[:120] + "...",
            }
            for h in sparse_hits
        ]

        # Determine candidates based on chosen strategy
        selected_candidates: list[dict[str, Any]] = []

        if strategy == "dense_only":
            selected_candidates = dense_hits[:top_k]
        elif strategy == "bm25_only":
            selected_candidates = sparse_hits[:top_k]
        elif strategy == "hybrid":
            t0 = time.perf_counter()
            fused = self.fusion.fuse(dense_hits, sparse_hits, top_k=top_k)
            t_fusion = (time.perf_counter() - t0) * 1000
            trace["latencies_ms"]["fusion"] = round(t_fusion, 2)
            trace["fused_hits"] = [
                {
                    "id": h["id"],
                    "rrf_score": h["score"],
                    "rank": h["rank"],
                    "title": h["payload"].get("doc_title", "Unknown"),
                    "snippet": h["payload"].get("content", "")[:120] + "...",
                }
                for h in fused
            ]
            selected_candidates = fused
        else:
            # Default: hybrid_reranked
            t0 = time.perf_counter()
            fused = self.fusion.fuse(
                dense_hits, sparse_hits, top_k=settings.DENSE_TOP_K + settings.BM25_TOP_K
            )
            t_fusion = (time.perf_counter() - t0) * 1000
            trace["latencies_ms"]["fusion"] = round(t_fusion, 2)
            trace["fused_hits"] = [
                {
                    "id": h["id"],
                    "rrf_score": h["score"],
                    "rank": h["rank"],
                    "title": h["payload"].get("doc_title", "Unknown"),
                    "snippet": h["payload"].get("content", "")[:120] + "...",
                }
                for h in fused
            ]

            t0 = time.perf_counter()
            reranked = self.reranker.rerank(query, fused, top_k=top_k)
            t_rerank = (time.perf_counter() - t0) * 1000
            trace["latencies_ms"]["rerank"] = round(t_rerank, 2)
            trace["reranked_hits"] = [
                {
                    "id": h["id"],
                    "rerank_score": h["score"],
                    "rank": h["rank"],
                    "title": h["payload"].get("doc_title", "Unknown"),
                    "snippet": h["payload"].get("content", "")[:120] + "...",
                }
                for h in reranked
            ]
            selected_candidates = reranked

        # Step 3: Hierarchical Parent Context Expansion
        # Map child chunk IDs back to parent chunks to retrieve full contextual sections
        parent_ids = list(
            dict.fromkeys(
                [
                    c["payload"].get("parent_id")
                    for c in selected_candidates
                    if c.get("payload", {}).get("parent_id")
                ]
            )
        )

        parent_map: dict[str, ParentChunk] = {}
        if parent_ids:
            statement = select(ParentChunk).where(ParentChunk.id.in_(parent_ids))
            db_res = await session.execute(statement)
            for p in db_res.scalars().all():
                parent_map[p.id] = p

        # Assemble Citations & Context String
        citations: list[dict[str, Any]] = []
        context_blocks: list[str] = []
        seen_parents = set()

        for idx, candidate in enumerate(selected_candidates, start=1):
            payload = candidate.get("payload", {})
            parent_id = payload.get("parent_id")
            doc_title = payload.get("doc_title", "Document")
            header_path = payload.get("header_path", "General")
            child_content = payload.get("content", "")

            # Use full parent context if available, otherwise fall back to child content
            full_content = child_content
            if parent_id and parent_id in parent_map:
                parent_obj = parent_map[parent_id]
                # Avoid duplicating same parent section if multiple children hit it
                if parent_id not in seen_parents:
                    full_content = parent_obj.content
                    seen_parents.add(parent_id)
                else:
                    full_content = child_content

            citation = {
                "citation_id": idx,
                "document_id": payload.get("document_id", ""),
                "doc_title": doc_title,
                "header_path": header_path,
                "score": candidate.get("score", 0.0),
                "snippet": child_content[:200] + "..."
                if len(child_content) > 200
                else child_content,
            }
            citations.append(citation)

            block = f"--- [Source {idx}]: {doc_title} > {header_path} ---\n{full_content}"
            context_blocks.append(block)

        t_total = (time.perf_counter() - t0_total) * 1000
        trace["latencies_ms"]["total_retrieval"] = round(t_total, 2)

        return RetrievalResult(
            strategy=strategy,
            final_chunks=selected_candidates,
            context_text="\n\n".join(context_blocks),
            citations=citations,
            trace=trace,
        )
