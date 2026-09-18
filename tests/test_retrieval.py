from backend.app.services.retrieval.fusion import ReciprocalRankFusion
from backend.app.services.retrieval.sparse import SparseBM25Retriever


def test_sparse_bm25_retriever():
    retriever = SparseBM25Retriever()

    records = [
        {
            "id": "c1",
            "child_id": "c1",
            "doc_title": "Raft Consensus",
            "header_path": "Leader Election",
            "content": "Followers start elections when their randomized election timeout expires.",
        },
        {
            "id": "c2",
            "child_id": "c2",
            "doc_title": "BESS Storage",
            "header_path": "Modbus Protocol",
            "content": "Modbus TCP communicates on port 502 using function codes 03 and 16.",
        },
        {
            "id": "c3",
            "child_id": "c3",
            "doc_title": "Database MVCC",
            "header_path": "Snapshot Isolation",
            "content": "Multi-version concurrency control prevents readers from blocking writers.",
        },
    ]

    retriever.index_chunks(records)

    # Query for Modbus
    results = retriever.search("Modbus TCP port", top_k=2)
    assert len(results) > 0
    assert results[0]["id"] == "c2"
    assert "Modbus" in results[0]["payload"]["header_path"]

    # Query for Raft election
    raft_results = retriever.search("election timeout", top_k=2)
    assert len(raft_results) > 0
    assert raft_results[0]["id"] == "c1"


def test_reciprocal_rank_fusion_math():
    fusion = ReciprocalRankFusion(k=60)

    dense_hits = [
        {"id": "doc_A", "rank": 1, "score": 0.95, "payload": {"doc_title": "Doc A"}},
        {"id": "doc_B", "rank": 2, "score": 0.88, "payload": {"doc_title": "Doc B"}},
    ]

    sparse_hits = [
        {"id": "doc_B", "rank": 1, "score": 12.4, "payload": {"doc_title": "Doc B"}},
        {"id": "doc_C", "rank": 2, "score": 8.1, "payload": {"doc_title": "Doc C"}},
    ]

    fused = fusion.fuse(dense_hits, sparse_hits, top_k=3)

    assert len(fused) == 3
    # doc_B has rank 2 in dense (1/(60+2) = 1/62) and rank 1 in sparse (1/(60+1) = 1/61)
    # total score = 1/62 + 1/61 ≈ 0.016129 + 0.016393 ≈ 0.032522
    # doc_A has rank 1 in dense only (1/61 ≈ 0.016393)
    # doc_B should be #1!
    assert fused[0]["id"] == "doc_B"
    assert fused[0]["rank"] == 1
    assert fused[0]["provenance"]["dense_rank"] == 2
    assert fused[0]["provenance"]["sparse_rank"] == 1
