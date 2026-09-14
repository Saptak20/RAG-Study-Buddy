import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.main import app

CONFIG_ENV_VARS = (
    "GROQ_API_KEY",
    "MONGODB_URI",
    "MONGODB_DB_NAME",
    "JWT_SECRET_KEY",
    "JWT_ALGORITHM",
    "JWT_EXPIRE_MINUTES",
    "APP_ENV",
    "FAISS_INDEX_PATH",
)


@pytest.mark.anyio
async def test_health_returns_ok() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_settings_defaults_without_env(monkeypatch):
    for var in CONFIG_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    settings = Settings(_env_file=None)

    assert settings.groq_api_key is None
    assert settings.mongodb_uri is None
    assert settings.jwt_secret_key is None
    assert settings.mongodb_db_name == "rag_study_buddy"
    assert settings.jwt_algorithm == "HS256"
    assert settings.jwt_expire_minutes == 1440
    assert settings.app_env == "development"
    assert settings.faiss_index_path == "./data/faiss_index"
