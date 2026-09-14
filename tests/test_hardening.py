from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.core.deps import get_auth_settings
from app.core.rate_limit import rate_limit_auth, rate_limit_chat, rate_limit_upload
from app.core.security import create_access_token
from app.main import app


@pytest.fixture(autouse=True)
def reset_rate_limiters():
    """Ensure all rate limiters have a clean history before and after each test."""
    rate_limit_auth.reset()
    rate_limit_upload.reset()
    rate_limit_chat.reset()
    yield
    rate_limit_auth.reset()
    rate_limit_upload.reset()
    rate_limit_chat.reset()


@pytest.mark.anyio
async def test_security_headers_present_on_responses() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.headers.get("x-content-type-options") == "nosniff"
    assert response.headers.get("x-frame-options") == "DENY"
    assert response.headers.get("referrer-policy") == "strict-origin-when-cross-origin"


@pytest.mark.anyio
async def test_cors_allowed_origin_returns_headers() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Allowed origin
        response = await client.get("/health", headers={"Origin": "http://localhost:3000"})
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
        assert response.headers.get("access-control-allow-credentials") == "true"

        # Disallowed origin
        disallowed_resp = await client.get("/health", headers={"Origin": "https://malicious-attacker.com"})
        assert disallowed_resp.status_code == 200
        assert "access-control-allow-origin" not in disallowed_resp.headers


@pytest.mark.anyio
async def test_cors_preflight_options_request() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.options(
            "/chat",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Authorization,Content-Type",
            },
        )
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
        assert "POST" in response.headers.get("access-control-allow-methods", "")


@pytest.mark.anyio
async def test_readiness_probe_database_connected() -> None:
    fake_db = AsyncMock()
    fake_db.command.return_value = {"ok": 1.0}
    app.state.database = fake_db

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ready", "database": "connected"}
        fake_db.command.assert_awaited_once_with("ping")
    finally:
        app.state.database = None


@pytest.mark.anyio
async def test_readiness_probe_database_disconnected_or_error() -> None:
    # 1. State database is None
    app.state.database = None
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["detail"]["database"] == "disconnected"

    # 2. Database command raises exception
    failing_db = AsyncMock()
    failing_db.command.side_effect = RuntimeError("MongoDB connection timeout")
    app.state.database = failing_db

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["detail"]["database"] == "unreachable"
    finally:
        app.state.database = None


@pytest.mark.anyio
async def test_rate_limiting_enforcement_and_retry_after(tmp_path: Path) -> None:
    settings = Settings(
        jwt_secret_key="secret-key-for-testing-rate-limiting",
        rate_limit_enabled=True,
        rate_limit_auth_per_minute=2,
    )

    async def override_settings() -> Settings:
        return settings

    app.dependency_overrides[get_auth_settings] = override_settings

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # First request - rejected with 503 (no db), but passes rate limiter (1/2)
            resp1 = await client.post("/auth/login", json={"email": "u@test.com", "password": "pass"})
            assert resp1.status_code == 503

            # Second request - passes rate limiter (2/2)
            resp2 = await client.post("/auth/login", json={"email": "u@test.com", "password": "pass"})
            assert resp2.status_code == 503

            # Third request - rate limit exceeded!
            resp3 = await client.post("/auth/login", json={"email": "u@test.com", "password": "pass"})
            assert resp3.status_code == 429
            assert "Rate limit exceeded" in resp3.json()["detail"]
            assert "Retry-After" in resp3.headers
            assert int(resp3.headers["Retry-After"]) >= 1

            # Reset allows request through again
            rate_limit_auth.reset()
            resp4 = await client.post("/auth/login", json={"email": "u@test.com", "password": "pass"})
            assert resp4.status_code == 503
    finally:
        app.dependency_overrides.pop(get_auth_settings, None)


@pytest.mark.anyio
async def test_rate_limiting_disabled_mode() -> None:
    settings = Settings(
        jwt_secret_key="secret-key-for-testing-rate-limiting",
        rate_limit_enabled=False,
        rate_limit_auth_per_minute=1,
    )

    app.dependency_overrides[get_auth_settings] = lambda: settings

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            for _ in range(5):
                resp = await client.post("/auth/login", json={"email": "u@test.com", "password": "pass"})
                # Should not be 429 even though we made 5 requests with limit=1
                assert resp.status_code != 429
    finally:
        app.dependency_overrides.pop(get_auth_settings, None)


@pytest.mark.anyio
async def test_rate_limiting_authenticated_user_identity() -> None:
    settings = Settings(
        jwt_secret_key="test-secret-key-with-adequate-length",
        rate_limit_enabled=True,
        rate_limit_chat_per_minute=2,
    )
    app.dependency_overrides[get_auth_settings] = lambda: settings
    token_a = create_access_token("user-a", settings)
    token_b = create_access_token("user-b", settings)
    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # User A requests (rejected with 503 no db, but passes rate limiter)
            r1 = await client.post("/chat", json={"message": "hi"}, headers=headers_a)
            assert r1.status_code == 503
            r2 = await client.post("/chat", json={"message": "hi"}, headers=headers_a)
            assert r2.status_code == 503
            # User A exceeds limit (2/2) -> 429
            r3 = await client.post("/chat", json={"message": "hi"}, headers=headers_a)
            assert r3.status_code == 429
            assert "Rate limit exceeded" in r3.json()["detail"]

            # User B still has fresh quota
            r_b = await client.post("/chat", json={"message": "hi"}, headers=headers_b)
            assert r_b.status_code == 503
    finally:
        app.dependency_overrides.pop(get_auth_settings, None)


@pytest.mark.anyio
async def test_unhandled_exception_returns_clean_500() -> None:
    # Trigger an unhandled error to verify the global exception handler
    @app.get("/test-crash-endpoint", include_in_schema=False)
    async def crash():
        raise ZeroDivisionError("Simulated internal crash")

    try:
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/test-crash-endpoint")
        assert resp.status_code == 500
        assert resp.json() == {"detail": "An internal server error occurred."}
        # Verify stack trace / exception details are NOT exposed
        assert "ZeroDivisionError" not in resp.text
        assert "Simulated internal crash" not in resp.text
    finally:
        app.router.routes[:] = [
            route for route in app.router.routes if getattr(route, "path", None) != "/test-crash-endpoint"
        ]
