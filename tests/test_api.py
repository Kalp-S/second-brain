import pytest
import httpx
from backend.app.main import app

@pytest.mark.asyncio
async def test_system_status():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/system/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "vault_statistics" in data

@pytest.mark.asyncio
async def test_list_strategies():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/rag/strategies")
        assert response.status_code == 200
        strategies = response.json()
        strat_ids = [s["id"] for s in strategies]
        assert "hybrid_reranked" in strat_ids
        assert "hybrid" in strat_ids
        assert "dense_only" in strat_ids
        assert "bm25_only" in strat_ids

@pytest.mark.asyncio
async def test_get_knowledge_graph():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/graph")
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "edges" in data

@pytest.mark.asyncio
async def test_end_to_end_rag_flow():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # 1. Sync the sample vault
        sync_res = await client.post("/api/v1/documents/sync-vault")
        assert sync_res.status_code == 200
        sync_data = sync_res.json()
        assert sync_data["status"] == "success"

        # 2. Query RAG with hybrid_reranked strategy
        query_payload = {
            "query": "How does Raft leader election handle split votes?",
            "strategy": "hybrid_reranked",
            "top_k": 3
        }
        rag_res = await client.post("/api/v1/rag/query", json=query_payload)
        assert rag_res.status_code == 200
        rag_data = rag_res.json()
        assert "answer" in rag_data
        assert "citations" in rag_data
        assert len(rag_data["citations"]) > 0
        assert "evaluation" in rag_data
        assert rag_data["evaluation"]["composite_score"] > 0.0

@pytest.mark.asyncio
async def test_demo_mode_blocks_upload_and_delete():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # Test upload blocked in demo mode
        upload_res = await client.post(
            "/api/v1/documents/upload",
            files={"files": ("test.txt", b"dummy content", "text/plain")}
        )
        assert upload_res.status_code == 403
        assert "disabled in public demo mode" in upload_res.json()["detail"]

        # Test delete blocked in demo mode
        delete_res = await client.delete("/api/v1/documents/dummy-doc-id")
        assert delete_res.status_code == 403
        assert "disabled in public demo mode" in delete_res.json()["detail"]
