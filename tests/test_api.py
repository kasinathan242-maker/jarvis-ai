"""Tests for FastAPI Endpoints, Bearer Authentication, and REST Routes."""

import pytest
from httpx import ASGITransport, AsyncClient

from api.server import app
from security.secrets import get_or_create_api_token


@pytest.mark.asyncio
async def test_health_endpoint_public() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "JARVIS Lite"


@pytest.mark.asyncio
async def test_unauthorized_access_rejected() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # No header
        resp = await client.get("/api/v1/tools")
        assert resp.status_code == 401

        # Invalid token
        resp = await client.get(
            "/api/v1/tools",
            headers={"Authorization": "Bearer invalid_secret_token"},
        )
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_authenticated_tools_and_command() -> None:
    token = get_or_create_api_token()
    headers = {"Authorization": f"Bearer {token}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Tools list
        resp = await client.get("/api/v1/tools", headers=headers)
        assert resp.status_code == 200
        tools = resp.json()
        tool_names = [t["name"] for t in tools]
        assert "get_time" in tool_names
        assert "open_application" in tool_names
        assert "system_stats" in tool_names

        # Execute command
        cmd_resp = await client.post(
            "/api/v1/command",
            headers=headers,
            json={"text": "what time is it", "play_speech": False},
        )
        assert cmd_resp.status_code == 200
        res_data = cmd_resp.json()
        assert "time is" in res_data["response"].lower()
