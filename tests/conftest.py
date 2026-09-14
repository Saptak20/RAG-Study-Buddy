from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.database import get_database
from app.core.deps import get_auth_settings, get_embeddings, get_llm
from app.main import app
from app.rag.embeddings import FakeEmbeddings
from app.rag.llm import FakeLLMService
from tests.test_documents import FakeDatabase


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the project's installed asyncio backend only."""
    return "asyncio"


@pytest.fixture
def chat_setup(tmp_path: Path):
    database = FakeDatabase()
    settings = Settings(
        jwt_secret_key="test-secret-key-with-adequate-length",
        document_storage_path=str(tmp_path / "documents"),
        faiss_index_path=str(tmp_path / "faiss_index"),
        retriever_top_k=3,
        rag_max_context_chars=4000,
    )
    embedder = FakeEmbeddings(dimension=64)
    fake_llm = FakeLLMService(canned_response="Grounded study response based on notes.")

    async def override_get_database() -> AsyncIterator[FakeDatabase]:
        yield database

    async def override_get_settings() -> Settings:
        return settings

    def override_get_embeddings() -> FakeEmbeddings:
        return embedder

    def override_get_llm() -> FakeLLMService:
        return fake_llm

    app.dependency_overrides[get_database] = override_get_database
    app.dependency_overrides[get_auth_settings] = override_get_settings
    app.dependency_overrides[get_embeddings] = override_get_embeddings
    app.dependency_overrides[get_llm] = override_get_llm

    yield database, settings, embedder, fake_llm

    app.dependency_overrides.clear()
