# System Architecture & Technical Deep-Dive

## 1. Executive Summary & Problem Statement

Naive Retrieval-Augmented Generation (RAG) systems implemented as simple wrappers around LangChain or LlamaIndex exhibit critical production failure modes:
1. **Vocabulary Mismatch & Keyword Blindness**: Dense vector embeddings map text into continuous latent space. While exceptional at semantic paraphrasing, vectors frequently fail to distinguish exact tokens, error codes (e.g. `ERR_503_UNAVAILABLE`), code identifiers, or domain acronyms.
2. **Context Fragmentation vs. Dilution**:
   - Small chunks (~100 tokens) yield high vector retrieval precision but starve the LLM of the surrounding context needed to synthesize complex architectural answers.
   - Large chunks (~1,500 tokens) preserve context but dilute vector representations into noisy semantic averages.
3. **Bi-Encoder False Positives**: Vector search relies on bi-encoder architectures that project query and document into separate embeddings without token-level cross-attention. Semantically adjacent but factually irrelevant documents introduce context contamination.
4. **Hallucination & Attribution Deficits**: Lack of strict grounding instructions and exact document provenance leaves models vulnerable to unchecked hallucinations.

**Second Brain** solves these challenges through an **Advanced Multi-Stage Hybrid RAG Architecture** incorporating Reciprocal Rank Fusion, Hierarchical Parent-Child Chunking, Cross-Encoder Reranking, and automated RAG Triad observability.

---

## 2. High-Level Architecture Diagram

```
+----------------------------------------------------------------------------------------------------+
|                                    INGESTION PIPELINE (Async)                                      |
|                                                                                                    |
|  [Raw Document]                                                                                    |
|  (MD, PDF, Code) ---> [SHA-256 Checksum] ---> [DocumentParser]                                     |
|                              | (Idempotency)           |                                           |
|                              v                         v                                           |
|                        [Skip/Update]        [Hierarchical Chunker]                                 |
|                                                        |                                           |
|                          +-----------------------------+-----------------------------+             |
|                          |                                                           |             |
|                          v                                                           v             |
|                   [Parent Chunks]                                             [Child Chunks]       |
|                   (~1,200 tokens)                                             (~300 tokens)        |
|                          |                                                    [Context Breadcrumbs]|
|                          v                                                           |             |
|                  [(SQLite Database)]                                                 v             |
|                                                              +-----------------------+----------+  |
|                                                              |                                  |  |
|                                                              v                                  v  |
|                                                    [FastEmbed ONNX]                      [BM25]    |
|                                                    (BGE-small 384d)                     (Lexical)  |
|                                                              |                                  |  |
|                                                              v                                  v  |
|                                                    [(Qdrant Vector DB)]                 [InMemory] |
+----------------------------------------------------------------------------------------------------+
```

---

## 3. Advanced Multi-Stage Retrieval Pipeline

```
[User Query]
     |
     +-----------------------------------+-----------------------------------+
     |                                                                       |
     v                                                                       v
[Dense Vector Search]                                               [Sparse BM25 Search]
- Model: BAAI/bge-small-en-v1.5 (ONNX)                               - Tokenized Okapi BM25
- Metric: Cosine Similarity                                          - Exact keyword & token match
- Top-K: 15 candidates                                               - Top-K: 15 candidates
     |                                                                       |
     +-----------------------------------+-----------------------------------+
                                         |
                                         v
                         [Reciprocal Rank Fusion (RRF)]
                                  k = 60
                         Score(d) = sum [ 1 / (60 + rank) ]
                                         |
                                         v
                            [Top 30 Merged Candidates]
                                         |
                                         v
                           [Cross-Encoder Reranking]
                         - Model: ms-marco-TinyBERT-L-2-v2
                         - Joint Attention: [CLS] Query [SEP] Doc
                         - Selects Top 5 Highest Signal Chunks
                                         |
                                         v
                       [Parent Context DB Expansion]
                         - Child Chunk -> Parent Chunk ID Lookup
                         - Injects full 1,200-token parent sections
                                         |
                                         v
                     [Grounded Prompt Assembly with [1],[2]]
                                         |
                                         v
                        [Streaming LLM Generation (SSE)]
                         - Local Ollama / OpenAI / Claude
                                         |
                                         v
                         [Automated RAG Triad Evaluation]
                         - Context Relevance
                         - Faithfulness / Groundedness
                         - Answer Relevance
```

---

## 4. Mathematical Foundations & Design Decisions

### 4.1. Why Reciprocal Rank Fusion (RRF) vs. Score Normalization?

When combining Dense and Lexical search, naive implementations attempt min-max normalization:
$$S_{norm} = \frac{S - S_{min}}{S_{max} - S_{min}}$$

**Why Min-Max Normalization Fails in Production:**
1. **Unbounded vs. Bounded Distributions**: BM25 scores are unbounded $[0, \infty)$ and vary drastically with document length and corpus frequency. Cosine similarity is bounded $[-1, 1]$ (or $[0, 1]$ for normalized vectors).
2. **Calibration Outliers**: A single document containing repetitive keywords can produce an anomalously high BM25 score, skewing the min-max distribution and suppressing genuine semantic matches.
3. **Rank Invariance**: **Reciprocal Rank Fusion (RRF)** relies solely on ranking order:
   $$RRF\_Score(d) = \sum_{m \in M} \frac{w_m}{k + rank_m(d)}$$
   where $k = 60$ (Cormack et al.). RRF ensures that items ranked highly across *both* modalities receive exponential priority, without being susceptible to score distribution anomalies.

### 4.2. Cross-Encoder Joint Attention vs. Bi-Encoder Dot Products

- **Bi-Encoder (Vector Embeddings)**:
  $$Sim(q, d) = \langle E(q), E(d) \rangle$$
  Fast ($O(1)$ lookup via HNSW index), but independent encoding means no token-to-token attention exists between query words and document words.
- **Cross-Encoder**:
  $$Score(q, d) = \text{Transformer}(q \circ d)$$
  Every query token attends to every passage token across all self-attention layers. This joint cross-attention filters out passages that share semantic keywords but fail to answer the query's relational premise.

### 4.3. Contextual Enrichment & Anthropic Contextual Retrieval

To eliminate pronoun ambiguity and contextual isolation in chunking, each child chunk is prepended with structured hierarchical breadcrumbs:
```markdown
[Document: {Document_Title} | Section: {Header_Path}]
{Child_Text}
```
This guarantees that vector embeddings capture the global context (e.g., that a sentence mentioning *"it enforces strict majority voting"* belongs to the *"Raft Consensus Protocol"*).

---

## 5. Failure Modes & Mitigations

| Failure Mode | Impact | Architectural Mitigation |
| :--- | :--- | :--- |
| **Out-of-Vocabulary / Acronyms** | Vector search returns random semantic neighbors | Parallel BM25 lexical retriever indexed on raw stems |
| **Hallucinated Citations** | LLM invents references or misattributes facts | Automated Faithfulness auditor checks token coverage |
| **Ingestion Duplication** | Duplicate vectors degrade index quality | SHA-256 hash comparison enforces idempotent upserts |
| **Backhaul / Offline Outage** | Cloud LLM dependency breaks edge systems | 100% offline local runtime via embedded Qdrant + Ollama |
| **High TTFT (Latency)** | Long wait times on user interface | Server-Sent Events (SSE) token streaming |

---

## 6. Scalability & Enterprise Roadmap

- **Vector Store**: Qdrant embedded mode can be swapped to distributed Qdrant cluster or AWS OpenSearch with zero API contract changes.
- **Relational Metadata**: SQLite database uses SQLAlchemy/SQLModel asynchronous sessions, allowing seamless migration to PostgreSQL (`asyncpg`).
- **Distributed Ingestion Workers**: Background tasks can be routed through Celery / Redis / AWS SQS for processing millions of pages concurrently.
