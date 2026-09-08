---
title: "Distributed Tracing, Latency Budgets, & RAG Observability"
tags: ["observability", "distributed-tracing", "rag-triad", "opentelemetry", "metrics"]
---

# Distributed Tracing, Latency Budgets, & RAG Observability

Deploying AI systems to production introduces non-deterministic failure modes that traditional APM (Application Performance Monitoring) tools cannot isolate. A comprehensive observability architecture requires distributed tracing coupled with automated evaluation metrics.

## 1. Latency Budgets in Multi-Stage Systems
In an advanced RAG pipeline, the total end-to-end response time budget is partitioned across asynchronous subsystems:
- **Query Embedding & Preprocessing**: 10ms - 25ms (FastEmbed ONNX CPU/GPU runtime).
- **Parallel Candidate Retrieval**:
  - Dense Vector ANN Search (Qdrant HNSW): 15ms - 35ms.
  - Sparse BM25 Lexical Scan: 10ms - 20ms.
- **Reciprocal Rank Fusion (RRF)**: $< 2\text{ms}$ in-memory ranking operation.
- **Cross-Encoder Reranking**: 30ms - 75ms (FlashRank MiniLM transformer cross-attention).
- **Time-to-First-Token (TTFT)**: 200ms - 600ms via Server-Sent Events (SSE) streaming.
- **Total Pipeline Execution**: $< 800\text{ms}$ to initial token stream.

## 2. Distributed Tracing & Span Hierarchy
Every user request generates a unique `trace_id` propagated across all internal function boundaries. The span hierarchy captures:
```
[HTTP POST /api/v1/rag/stream]
  ├── [Span: Dense Vector Search (Qdrant)] -> top_k=15, duration=24ms
  ├── [Span: Sparse BM25 Search]           -> top_k=15, duration=14ms
  ├── [Span: Reciprocal Rank Fusion]        -> merged_candidates=30, duration=1.2ms
  ├── [Span: Cross-Encoder Rerank]          -> top_k=5, duration=42ms
  ├── [Span: Parent Context DB Lookup]      -> num_parents=4, duration=4ms
  └── [Span: LLM Stream Generation]         -> token_count=184, duration=480ms
```

## 3. Automated RAG Triad Evaluation
Continuous evaluation audits the pipeline across three independent pillars:
1. **Context Relevance**: Evaluates whether retrieved passages are free of distracting noise.
2. **Faithfulness / Groundedness**: Verifies that every assertion made by the LLM is directly entailed by the source texts, guarding against hallucination.
3. **Answer Relevance**: Measures how completely the answer satisfies the user's explicit query.
