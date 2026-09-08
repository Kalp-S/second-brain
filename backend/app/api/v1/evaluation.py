import time
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select, func

from backend.app.db.database import get_db
from backend.app.db.models import EvaluationRecord
from backend.app.core.dependencies import get_rag_retriever, get_llm, AdvancedRAGRetriever, BaseLLMProvider
from backend.app.services.llm.prompt import build_rag_prompt, SYSTEM_PROMPT
from backend.app.services.llm.evaluator import RAGTriadEvaluator

router = APIRouter(prefix="/evaluation", tags=["Evaluation & Benchmarking"])

class BenchmarkRequest(BaseModel):
    query: str = Field(..., description="Evaluation query to run against all 4 retrieval strategies")

@router.get("/records")
async def get_evaluation_records(
    limit: int = Query(default=20, ge=1, le=100),
    session: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    stmt = select(EvaluationRecord).order_by(EvaluationRecord.created_at.desc()).limit(limit)
    res = await session.execute(stmt)
    records = res.scalars().all()

    # Aggregate stats per strategy
    avg_stmt = select(
        EvaluationRecord.strategy,
        func.avg(EvaluationRecord.context_relevance),
        func.avg(EvaluationRecord.faithfulness),
        func.avg(EvaluationRecord.answer_relevance),
        func.avg(EvaluationRecord.latency_ms),
        func.count(EvaluationRecord.id)
    ).group_by(EvaluationRecord.strategy)

    avg_res = await session.execute(avg_stmt)
    strategy_stats = {}
    for strat, c_rel, faith, a_rel, lat, cnt in avg_res.all():
        strategy_stats[strat] = {
            "runs": cnt,
            "avg_context_relevance": round(c_rel or 0.0, 3),
            "avg_faithfulness": round(faith or 0.0, 3),
            "avg_answer_relevance": round(a_rel or 0.0, 3),
            "avg_latency_ms": round(lat or 0.0, 1)
        }

    return {
        "strategy_aggregates": strategy_stats,
        "recent_records": [
            {
                "id": r.id,
                "query": r.query,
                "strategy": r.strategy,
                "context_relevance": r.context_relevance,
                "faithfulness": r.faithfulness,
                "answer_relevance": r.answer_relevance,
                "latency_ms": r.latency_ms,
                "created_at": r.created_at.isoformat() if r.created_at else None
            }
            for r in records
        ]
    }

@router.post("/benchmark")
async def run_comparative_benchmark(
    req: BenchmarkRequest,
    session: AsyncSession = Depends(get_db),
    retriever: AdvancedRAGRetriever = Depends(get_rag_retriever),
    llm: BaseLLMProvider = Depends(get_llm)
) -> Dict[str, Any]:
    """Runs a head-to-head comparison across all 4 retrieval strategies for technical evaluation."""
    strategies = ["dense_only", "bm25_only", "hybrid", "hybrid_reranked"]
    comparison = {}

    for strat in strategies:
        t0 = time.perf_counter()
        retrieval_res = await retriever.retrieve(query=req.query, session=session, strategy=strat, top_k=5)
        
        prompt = build_rag_prompt(req.query, retrieval_res.context_text)
        answer = await llm.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)
        total_time_ms = round((time.perf_counter() - t0) * 1000, 2)

        eval_res = RAGTriadEvaluator.evaluate(
            query=req.query,
            retrieved_context=retrieval_res.context_text,
            generated_answer=answer
        )

        comparison[strat] = {
            "strategy": strat,
            "latency_ms": total_time_ms,
            "retrieval_latency_ms": retrieval_res.trace["latencies_ms"],
            "top_docs": [c["doc_title"] for c in retrieval_res.citations[:3]],
            "context_relevance": eval_res["context_relevance"],
            "faithfulness": eval_res["faithfulness"],
            "answer_relevance": eval_res["answer_relevance"],
            "composite_score": eval_res["composite_score"],
            "answer_preview": answer[:180] + "..." if len(answer) > 180 else answer
        }

    return {
        "query": req.query,
        "results": comparison
    }
