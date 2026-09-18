import json
import time
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from backend.app.core.dependencies import (
    AdvancedRAGRetriever,
    BaseLLMProvider,
    get_llm,
    get_rag_retriever,
)
from backend.app.db.database import get_db
from backend.app.db.models import ChatMessage, ChatSession, EvaluationRecord
from backend.app.services.llm.evaluator import RAGTriadEvaluator
from backend.app.services.llm.prompt import SYSTEM_PROMPT, build_rag_prompt

router = APIRouter(prefix="/rag", tags=["RAG"])


class QueryRequest(BaseModel):
    query: str = Field(..., description="User question to query against the Second Brain")
    strategy: str = Field(
        default="hybrid_reranked", description="dense_only | bm25_only | hybrid | hybrid_reranked"
    )
    session_id: str | None = Field(
        default=None, description="Optional chat session ID for multi-turn conversation"
    )
    top_k: int = Field(default=5, description="Number of top reranked passages to retrieve")


@router.get("/strategies")
async def list_strategies() -> list[dict[str, str]]:
    return [
        {
            "id": "hybrid_reranked",
            "name": "Hybrid + Cross-Encoder Rerank (Recommended)",
            "description": "Dense vectors + Sparse BM25 fused via RRF, then scored by a Cross-Encoder transformer. Highest precision.",
        },
        {
            "id": "hybrid",
            "name": "Hybrid RRF (Dense + BM25)",
            "description": "Dense semantic vectors + Sparse BM25 lexical search merged with Reciprocal Rank Fusion.",
        },
        {
            "id": "dense_only",
            "name": "Dense Vector Search Only",
            "description": "Pure semantic embedding similarity using BGE-small cosine distance.",
        },
        {
            "id": "bm25_only",
            "name": "Sparse BM25 Search Only",
            "description": "Pure lexical keyword search. Excels at exact technical tokens and identifiers.",
        },
    ]


@router.post("/query")
async def query_rag(
    req: QueryRequest,
    session: AsyncSession = Depends(get_db),
    retriever: AdvancedRAGRetriever = Depends(get_rag_retriever),
    llm: BaseLLMProvider = Depends(get_llm),
) -> dict[str, Any]:
    """Execute complete RAG pipeline with retrieval, generation, and automated evaluation."""
    t0_start = time.perf_counter()

    # 1. Advanced Multi-Stage Retrieval
    retrieval_res = await retriever.retrieve(
        query=req.query, session=session, strategy=req.strategy, top_k=req.top_k
    )

    # 2. Assemble Grounded Prompt
    rag_prompt = build_rag_prompt(req.query, retrieval_res.context_text)

    # 3. LLM Generation
    t0_gen = time.perf_counter()
    answer = await llm.generate(prompt=rag_prompt, system_prompt=SYSTEM_PROMPT)
    gen_time_ms = round((time.perf_counter() - t0_gen) * 1000, 2)
    retrieval_res.trace["latencies_ms"]["generation"] = gen_time_ms

    # 4. Automated RAG Triad Evaluation
    eval_metrics = RAGTriadEvaluator.evaluate(
        query=req.query, retrieved_context=retrieval_res.context_text, generated_answer=answer
    )

    total_time_ms = round((time.perf_counter() - t0_start) * 1000, 2)
    retrieval_res.trace["latencies_ms"]["end_to_end"] = total_time_ms

    # 5. Persist Session & Message History
    session_id = req.session_id
    if not session_id:
        chat_sess = ChatSession(title=req.query[:40] + "...")
        session.add(chat_sess)
        await session.flush()
        session_id = chat_sess.id

    user_msg = ChatMessage(
        session_id=session_id, role="user", content=req.query, strategy=req.strategy
    )
    assistant_msg = ChatMessage(
        session_id=session_id,
        role="assistant",
        content=answer,
        strategy=req.strategy,
        citations=json.dumps(retrieval_res.citations),
        retrieval_trace=json.dumps(retrieval_res.trace),
        faithfulness_score=eval_metrics["faithfulness"],
    )
    session.add(user_msg)
    session.add(assistant_msg)

    # Store Evaluation Record for benchmarking
    eval_rec = EvaluationRecord(
        query=req.query,
        strategy=req.strategy,
        context_relevance=eval_metrics["context_relevance"],
        faithfulness=eval_metrics["faithfulness"],
        answer_relevance=eval_metrics["answer_relevance"],
        latency_ms=total_time_ms,
    )
    session.add(eval_rec)
    await session.commit()

    return {
        "session_id": session_id,
        "answer": answer,
        "strategy": req.strategy,
        "citations": retrieval_res.citations,
        "evaluation": eval_metrics,
        "trace": retrieval_res.trace,
    }


@router.post("/stream")
async def stream_rag(
    req: QueryRequest,
    session: AsyncSession = Depends(get_db),
    retriever: AdvancedRAGRetriever = Depends(get_rag_retriever),
    llm: BaseLLMProvider = Depends(get_llm),
):
    """Server-Sent Events (SSE) streaming endpoint for low Time-to-First-Token (TTFT) interactions."""
    retrieval_res = await retriever.retrieve(
        query=req.query, session=session, strategy=req.strategy, top_k=req.top_k
    )

    rag_prompt = build_rag_prompt(req.query, retrieval_res.context_text)

    async def event_generator():
        # Step 1: Emit retrieval trace and candidates
        yield f"event: trace\ndata: {json.dumps(retrieval_res.trace)}\n\n"

        # Step 2: Stream tokens from LLM
        full_answer_parts = []
        try:
            async for token in llm.generate_stream(prompt=rag_prompt, system_prompt=SYSTEM_PROMPT):
                full_answer_parts.append(token)
                yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            fallback = f"Error during streaming generation: {str(e)}"
            yield f"event: token\ndata: {json.dumps({'token': fallback})}\n\n"
            full_answer_parts.append(fallback)

        full_answer = "".join(full_answer_parts)

        # Step 3: Run RAG Triad evaluation
        eval_metrics = RAGTriadEvaluator.evaluate(
            query=req.query,
            retrieved_context=retrieval_res.context_text,
            generated_answer=full_answer,
        )
        yield f"event: eval\ndata: {json.dumps(eval_metrics)}\n\n"

        # Step 4: Emit citations
        yield f"event: citations\ndata: {json.dumps(retrieval_res.citations)}\n\n"

        # Step 5: Finished
        yield f"event: done\ndata: {json.dumps({'status': 'finished'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sessions")
async def list_chat_sessions(session: AsyncSession = Depends(get_db)) -> list[dict[str, Any]]:
    stmt = select(ChatSession).order_by(ChatSession.updated_at.desc())
    res = await session.execute(stmt)
    sessions = res.scalars().all()
    return [
        {
            "id": s.id,
            "title": s.title,
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "updated_at": s.updated_at.isoformat() if s.updated_at else None,
        }
        for s in sessions
    ]


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str, session: AsyncSession = Depends(get_db)
) -> list[dict[str, Any]]:
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
    )
    res = await session.execute(stmt)
    messages = res.scalars().all()
    output = []
    for m in messages:
        citations = []
        try:
            citations = json.loads(m.citations) if m.citations else []
        except Exception:
            pass

        trace = {}
        try:
            trace = json.loads(m.retrieval_trace) if m.retrieval_trace else {}
        except Exception:
            pass

        output.append(
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "strategy": m.strategy,
                "citations": citations,
                "trace": trace,
                "faithfulness_score": m.faithfulness_score,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
        )
    return output
