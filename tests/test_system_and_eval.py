import httpx
import pytest

from backend.app.core.config import settings
from backend.app.main import _request_history, app


@pytest.mark.asyncio
async def test_system_status_runtime_telemetry():
    """Verify system status returns detailed runtime telemetry."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/system/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "runtime" in data
        runtime = data["runtime"]
        assert "python_version" in runtime
        assert "platform" in runtime
        assert "uptime_seconds" in runtime
        assert "memory_rss_mb" in runtime
        assert runtime["memory_rss_mb"] >= 0


@pytest.mark.asyncio
async def test_observability_middleware_headers():
    """Verify X-Request-ID and X-Response-Time-Ms headers are present on responses."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        custom_req_id = "test-custom-trace-id-1234"
        response = await client.get(
            "/api/v1/rag/strategies", headers={"X-Request-ID": custom_req_id}
        )
        assert response.status_code == 200
        assert response.headers.get("x-request-id") == custom_req_id
        assert "x-response-time-ms" in response.headers
        duration = float(response.headers["x-response-time-ms"])
        assert duration >= 0.0


@pytest.mark.asyncio
async def test_rate_limit_sliding_window():
    """Verify rate limiter blocks excessive requests once threshold is reached."""
    _request_history.clear()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Send up to the limit
        for _ in range(settings.RATE_LIMIT_PER_MINUTE):
            res = await client.get("/api/v1/rag/strategies")
            assert res.status_code == 200

        # Next request must be rate-limited (429)
        blocked_res = await client.get("/api/v1/rag/strategies")
        assert blocked_res.status_code == 429
        assert "Rate limit exceeded" in blocked_res.json()["detail"]
        assert "retry-after" in blocked_res.headers

    # Reset for other tests
    _request_history.clear()
