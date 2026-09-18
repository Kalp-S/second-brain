#!/usr/bin/env python3
"""
Second Brain PKM - Automated RAG Triad Benchmark & Evaluation CLI
==================================================================
Executes head-to-head empirical evaluations across all 4 retrieval strategies:
1. Dense Vector Only (BGE-small ONNX)
2. Sparse Lexical Only (Okapi BM25)
3. Hybrid RRF (Reciprocal Rank Fusion k=60)
4. Hybrid + Cross-Encoder Reranker (FlashRank ONNX)

Seamlessly queries the running Second Brain server if active, or runs standalone.
Outputs formatted terminal tables and exports `eval_benchmark_results.json`.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

DEFAULT_BENCHMARK_QUERIES = [
    "How does Raft leader election handle split votes and prevent split-brain?",
    "Explain the BESS control hierarchy and Modbus TCP polling constraints.",
    "Why is Reciprocal Rank Fusion preferable to score min-max normalization?",
    "Compare 2-Phase Locking vs Multi-Version Concurrency Control (MVCC).",
]

STRATEGIES = [
    ("dense_only", "Dense Vector Only (BGE-small)"),
    ("bm25_only", "Sparse BM25 Only (Lexical)"),
    ("hybrid", "Hybrid RRF (Dense + BM25)"),
    ("hybrid_reranked", "Hybrid + Cross-Encoder Reranker"),
]


async def check_server_available(base_url: str = "http://127.0.0.1:8000") -> bool:
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            res = await client.get(f"{base_url}/api/v1/system/status")
            return res.status_code == 200
    except Exception:
        return False


async def run_benchmark(
    queries: list[str],
    output_path: str = "eval_benchmark_results.json",
    base_url: str = "http://127.0.0.1:8000",
):
    print("=" * 88)
    print("🚀 SECOND BRAIN — ADVANCED HYBRID RAG BENCHMARK STUDIO")
    print("=" * 88)

    server_running = await check_server_available(base_url)
    print(
        f"Execution Mode  : {'Live Server (' + base_url + ')' if server_running else 'Standalone Local Engine'}"
    )

    benchmark_records = []

    if server_running:
        async with httpx.AsyncClient(timeout=90.0) as client:
            status_res = await client.get(f"{base_url}/api/v1/system/status")
            status = status_res.json()
            print(f"Embedding Model : {status.get('embedding_model')} (384-dim ONNX)")
            print(f"Reranker Model  : {status.get('reranker_model')} (Cross-Encoder ONNX)")
            print(
                f"LLM Provider    : {status.get('llm_provider')} ({status.get('active_llm_model')})"
            )
            print("=" * 88)

            for q_idx, query in enumerate(queries, 1):
                print(f'\n[Query {q_idx}/{len(queries)}] "{query}"')
                print("-" * 88)
                print(
                    f"{'Strategy':<34} | {'Total(ms)':<9} | {'R-Lat(ms)':<9} | {'C-Rel':<6} | {'Faith':<6} | {'Score':<6}"
                )
                print("-" * 88)

                bench_res = await client.post(
                    f"{base_url}/api/v1/evaluation/benchmark", json={"query": query}
                )
                data = bench_res.json()
                results = data.get("results", {})

                query_record = {"query": query, "strategies": {}}

                for strat_id, strat_label in STRATEGIES:
                    s_data = results.get(strat_id, {})
                    total_ms = s_data.get("latency_ms", 0.0)
                    r_lat = s_data.get("retrieval_latency_ms", {})
                    retrieval_ms = r_lat.get("total_retrieval", 0.0)
                    c_rel = s_data.get("context_relevance", 0.0)
                    faith = s_data.get("faithfulness", 0.0)
                    comp = s_data.get("composite_score", 0.0)

                    query_record["strategies"][strat_id] = {
                        "strategy_id": strat_id,
                        "label": strat_label,
                        "total_latency_ms": total_ms,
                        "retrieval_latency_ms": retrieval_ms,
                        "context_relevance": c_rel,
                        "faithfulness": faith,
                        "composite_score": comp,
                        "top_doc": s_data.get("top_docs", ["N/A"])[0]
                        if s_data.get("top_docs")
                        else "N/A",
                    }

                    print(
                        f"{strat_label:<34} | {total_ms:<9.1f} | {retrieval_ms:<9.1f} | {c_rel:<6.2f} | {faith:<6.2f} | {comp:<6.2f}"
                    )

                benchmark_records.append(query_record)

    else:
        # Fallback to local import if server is stopped
        from backend.app.core.config import settings
        from backend.app.core.dependencies import (
            get_ingestion_pipeline,
            get_llm,
            get_rag_retriever,
        )
        from backend.app.db.database import async_session_factory, init_db
        from backend.app.services.llm.evaluator import RAGTriadEvaluator
        from backend.app.services.llm.prompt import SYSTEM_PROMPT, build_rag_prompt

        await init_db()
        rag_retriever = get_rag_retriever()
        pipeline = get_ingestion_pipeline()
        llm = get_llm()

        async with async_session_factory() as session:
            count = await pipeline.reindex_all_from_db(session)
            if count == 0:
                vault_dir = Path(settings.VAULT_PATH)
                if vault_dir.exists():
                    for f in vault_dir.glob("*.md"):
                        await pipeline.ingest_bytes(f.name, f.read_bytes(), session)
                await pipeline.reindex_all_from_db(session)

            for q_idx, query in enumerate(queries, 1):
                print(f'\n[Query {q_idx}/{len(queries)}] "{query}"')
                print("-" * 88)
                print(
                    f"{'Strategy':<34} | {'Total(ms)':<9} | {'R-Lat(ms)':<9} | {'C-Rel':<6} | {'Faith':<6} | {'Score':<6}"
                )
                print("-" * 88)

                query_record = {"query": query, "strategies": {}}

                for strat_id, strat_label in STRATEGIES:
                    t0 = time.perf_counter()
                    retrieval_res = await rag_retriever.retrieve(
                        query=query, session=session, strategy=strat_id, top_k=5
                    )
                    retrieval_ms = round((time.perf_counter() - t0) * 1000, 1)

                    prompt = build_rag_prompt(query, retrieval_res.context_text)
                    answer = await llm.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)
                    total_ms = round((time.perf_counter() - t0) * 1000, 1)

                    eval_res = RAGTriadEvaluator.evaluate(
                        query=query,
                        retrieved_context=retrieval_res.context_text,
                        generated_answer=answer,
                    )

                    query_record["strategies"][strat_id] = {
                        "strategy_id": strat_id,
                        "label": strat_label,
                        "total_latency_ms": total_ms,
                        "retrieval_latency_ms": retrieval_ms,
                        "context_relevance": eval_res["context_relevance"],
                        "faithfulness": eval_res["faithfulness"],
                        "composite_score": eval_res["composite_score"],
                        "top_doc": retrieval_res.citations[0]["doc_title"]
                        if retrieval_res.citations
                        else "N/A",
                    }

                    c_rel = f"{eval_res['context_relevance']:.2f}"
                    faith = f"{eval_res['faithfulness']:.2f}"
                    comp = f"{eval_res['composite_score']:.2f}"

                    print(
                        f"{strat_label:<34} | {total_ms:<9.1f} | {retrieval_ms:<9.1f} | {c_rel:<6} | {faith:<6} | {comp:<6}"
                    )

                benchmark_records.append(query_record)

    # Compute Summary Averages
    summary = {}
    for strat_id, strat_label in STRATEGIES:
        all_comp = [r["strategies"][strat_id]["composite_score"] for r in benchmark_records]
        all_faith = [r["strategies"][strat_id]["faithfulness"] for r in benchmark_records]
        all_crel = [r["strategies"][strat_id]["context_relevance"] for r in benchmark_records]
        all_lat = [r["strategies"][strat_id]["total_latency_ms"] for r in benchmark_records]

        summary[strat_id] = {
            "strategy": strat_label,
            "avg_latency_ms": round(sum(all_lat) / len(all_lat), 1) if all_lat else 0.0,
            "avg_context_relevance": round(sum(all_crel) / len(all_crel), 3) if all_crel else 0.0,
            "avg_faithfulness": round(sum(all_faith) / len(all_faith), 3) if all_faith else 0.0,
            "avg_composite_score": round(sum(all_comp) / len(all_comp), 3) if all_comp else 0.0,
        }

    print("\n" + "=" * 88)
    print("📊 AGGREGATE STRATEGY PERFORMANCE SUMMARY")
    print("=" * 88)
    print(
        f"{'Strategy':<34} | {'Avg Latency':<11} | {'Avg C-Rel':<9} | {'Avg Faith':<9} | {'Composite Score'}"
    )
    print("-" * 88)
    for _strat_id, s in summary.items():
        print(
            f"{s['strategy']:<34} | {s['avg_latency_ms']:>8.1f} ms | {s['avg_context_relevance']:>9.2f} | {s['avg_faithfulness']:>9.2f} | {s['avg_composite_score']:>14.2f}"
        )
    print("=" * 88)

    # Save JSON Report
    output_file = Path(output_path)
    report_data = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "aggregate_summary": summary,
        "evaluations": benchmark_records,
    }
    output_file.write_text(json.dumps(report_data, indent=2))
    print(f"✨ Full benchmark evaluation report saved to: {output_file.resolve()}\n")


def main():
    parser = argparse.ArgumentParser(description="Second Brain RAG Triad Benchmark Studio")
    parser.add_argument("--query", "-q", type=str, help="Single query to evaluate")
    parser.add_argument(
        "--output", "-o", type=str, default="eval_benchmark_results.json", help="Output JSON path"
    )
    parser.add_argument(
        "--url", "-u", type=str, default="http://127.0.0.1:8000", help="Base URL of running server"
    )
    args = parser.parse_args()

    queries = [args.query] if args.query else [DEFAULT_BENCHMARK_QUERIES[0]]
    asyncio.run(run_benchmark(queries=queries, output_path=args.output, base_url=args.url))


if __name__ == "__main__":
    main()
