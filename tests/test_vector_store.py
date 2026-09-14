import hashlib
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.core.database import get_database
from app.core.deps import get_auth_settings, get_embeddings
from app.main import app
from app.rag.embeddings import FakeEmbeddings
from app.rag.vector_store import (
    create_and_persist_index,
    delete_index,
    index_exists,
    load_index,
    search_document_index,
)
from tests.test_documents import FakeDatabase, make_token


@pytest.fixture
def vector_test_setup(tmp_path: Path):
    database = FakeDatabase()
    settings = Settings(
        jwt_secret_key="test-secret-key-with-adequate-length",
        document_storage_path=str(tmp_path / "documents"),
        faiss_index_path=str(tmp_path / "faiss_index"),
        max_upload_size_bytes=10 * 1024 * 1024,
    )
    fake_embedder = FakeEmbeddings(dimension=64)

    async def override_get_database() -> AsyncIterator[FakeDatabase]:
        yield database

    async def override_get_settings() -> Settings:
        return settings

    def override_get_embeddings() -> FakeEmbeddings:
        return fake_embedder

    app.dependency_overrides[get_database] = override_get_database
    app.dependency_overrides[get_auth_settings] = override_get_settings
    app.dependency_overrides[get_embeddings] = override_get_embeddings

    yield database, settings, fake_embedder

    app.dependency_overrides.clear()


def test_embedding_generation_dimensionality() -> None:
    embedder64 = FakeEmbeddings(dimension=64)
    embedder128 = FakeEmbeddings(dimension=128)

    assert embedder64.dimension == 64
    assert embedder128.dimension == 128

    texts = ["First study chunk", "Second study chunk", "Third study chunk"]
    vectors64 = embedder64.embed_texts(texts)
    assert len(vectors64) == 3
    for vec in vectors64:
        assert len(vec) == 64
        # Verify L2 normalized (magnitude approx 1.0)
        norm_sq = sum(x * x for x in vec)
        assert abs(norm_sq - 1.0) < 1e-4

    query_vec = embedder64.embed_query("Query question")
    assert len(query_vec) == 64


def test_vector_store_indexing_lifecycle_and_metadata_recovery(vector_test_setup) -> None:
    _database, settings, embedder = vector_test_setup
    user_id = "student-101"
    doc_id = "doc-alpha"

    chunks = [
        {
            "chunk_id": "c-1",
            "document_id": doc_id,
            "user_id": user_id,
            "filename": "biology_notes.pdf",
            "page": 1,
            "text": "Cell theory states that all living organisms are composed of cells.",
        },
        {
            "chunk_id": "c-2",
            "document_id": doc_id,
            "user_id": user_id,
            "filename": "biology_notes.pdf",
            "page": 2,
            "text": "Mitochondria are the powerhouse of the cell generating ATP.",
        },
        {
            "chunk_id": "c-3",
            "document_id": doc_id,
            "user_id": user_id,
            "filename": "biology_notes.pdf",
            "page": 3,
            "text": "Photosynthesis converts light energy into chemical sugars.",
        },
    ]

    embeddings = embedder.embed_texts([c["text"] for c in chunks])
    index_dir = create_and_persist_index(
        settings=settings,
        user_id=user_id,
        document_id=doc_id,
        chunks=chunks,
        embeddings=embeddings,
        dimension=embedder.dimension,
    )

    # 1. Index and metadata files exist
    assert index_dir.exists()
    assert (index_dir / "index.faiss").is_file()
    assert (index_dir / "metadata.json").is_file()
    assert index_exists(settings, user_id, doc_id)

    # 2. Index can be loaded again
    loaded = load_index(settings, user_id, doc_id)
    assert loaded is not None
    index, metadata = loaded

    # 3. Vector count equals chunk count
    assert index.ntotal == len(chunks) == 3
    assert metadata["total_chunks"] == 3
    assert metadata["dimension"] == embedder.dimension
    assert metadata["metric"] == "inner_product"

    # 4. Vector-position -> chunk metadata mapping and text recovery
    loaded_chunks = metadata["chunks"]
    assert len(loaded_chunks) == 3
    for i, orig_chunk in enumerate(chunks):
        mapped = loaded_chunks[i]
        assert mapped["vector_id"] == i
        assert mapped["chunk_id"] == orig_chunk["chunk_id"]
        assert mapped["document_id"] == orig_chunk["document_id"]
        assert mapped["user_id"] == orig_chunk["user_id"]
        assert mapped["filename"] == orig_chunk["filename"]
        assert mapped["page"] == orig_chunk["page"]
        assert mapped["text"] == orig_chunk["text"]

    # 5. Search retrieves matching chunk with metadata preserved
    query_vec = embedder.embed_query("Mitochondria ATP")
    results = search_document_index(
        settings=settings,
        user_id=user_id,
        document_id=doc_id,
        query_embedding=query_vec,
        top_k=2,
    )
    assert len(results) == 2
    assert "score" in results[0]
    assert results[0]["document_id"] == doc_id
    assert results[0]["filename"] == "biology_notes.pdf"
    assert results[0]["page"] in (1, 2, 3)


def test_independent_indexes_and_user_isolation(vector_test_setup) -> None:
    _database, settings, embedder = vector_test_setup
    user_a = "user-alice"
    user_b = "user-bob"
    doc_1 = "doc-shared-id"
    doc_2 = "doc-alice-only"

    chunks_a1 = [
        {
            "chunk_id": "ca1",
            "document_id": doc_1,
            "user_id": user_a,
            "filename": "alice_notes.txt",
            "page": 1,
            "text": "Alice confidential notes on thermodynamics.",
        }
    ]
    chunks_a2 = [
        {
            "chunk_id": "ca2",
            "document_id": doc_2,
            "user_id": user_a,
            "filename": "alice_lab.txt",
            "page": 1,
            "text": "Alice confidential lab findings.",
        }
    ]
    chunks_b1 = [
        {
            "chunk_id": "cb1",
            "document_id": doc_1,
            "user_id": user_b,
            "filename": "bob_notes.txt",
            "page": 1,
            "text": "Bob public notes on literature.",
        }
    ]

    create_and_persist_index(
        settings=settings,
        user_id=user_a,
        document_id=doc_1,
        chunks=chunks_a1,
        embeddings=embedder.embed_texts([c["text"] for c in chunks_a1]),
        dimension=embedder.dimension,
    )
    create_and_persist_index(
        settings=settings,
        user_id=user_a,
        document_id=doc_2,
        chunks=chunks_a2,
        embeddings=embedder.embed_texts([c["text"] for c in chunks_a2]),
        dimension=embedder.dimension,
    )
    create_and_persist_index(
        settings=settings,
        user_id=user_b,
        document_id=doc_1,
        chunks=chunks_b1,
        embeddings=embedder.embed_texts([c["text"] for c in chunks_b1]),
        dimension=embedder.dimension,
    )

    # User A and User B have separate indexes for doc_1
    loaded_a1 = load_index(settings, user_a, doc_1)
    loaded_b1 = load_index(settings, user_b, doc_1)
    assert loaded_a1 is not None and loaded_b1 is not None
    assert loaded_a1[1]["chunks"][0]["filename"] == "alice_notes.txt"
    assert loaded_b1[1]["chunks"][0]["filename"] == "bob_notes.txt"

    # User B cannot access User A's doc_2
    loaded_b2 = load_index(settings, user_b, doc_2)
    assert loaded_b2 is None

    # Searching User B's context for doc_2 returns empty list
    search_res = search_document_index(
        settings=settings,
        user_id=user_b,
        document_id=doc_2,
        query_embedding=embedder.embed_query("lab findings"),
    )
    assert search_res == []


def test_vector_store_deletion(vector_test_setup) -> None:
    _database, settings, embedder = vector_test_setup
    user_id = "user-carol"
    doc_id = "doc-temp"

    chunks = [
        {
            "chunk_id": "c-del",
            "document_id": doc_id,
            "user_id": user_id,
            "filename": "temp.txt",
            "page": 1,
            "text": "Temporary notes.",
        }
    ]
    create_and_persist_index(
        settings=settings,
        user_id=user_id,
        document_id=doc_id,
        chunks=chunks,
        embeddings=embedder.embed_texts([c["text"] for c in chunks]),
        dimension=embedder.dimension,
    )
    assert index_exists(settings, user_id, doc_id)

    deleted = delete_index(settings, user_id, doc_id)
    assert deleted is True
    assert not index_exists(settings, user_id, doc_id)

    # Deleting again returns False safely
    assert delete_index(settings, user_id, doc_id) is False


@pytest.mark.anyio
async def test_failed_indexing_cleans_up_and_records_failed_status(vector_test_setup) -> None:
    database, settings, _ = vector_test_setup
    token = make_token("user-err", settings)
    headers = {"Authorization": f"Bearer {token}"}

    class FailingEmbedder:
        @property
        def dimension(self) -> int:
            return 64

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("Simulated embedding engine explosion")

    app.dependency_overrides[get_embeddings] = lambda: FailingEmbedder()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/documents/upload",
            files={"file": ("failing_doc.txt", b"Valid content that fails during embedding.", "text/plain")},
            headers=headers,
        )

    # Endpoint must return 500 without leaking stack traces or internal implementation
    assert resp.status_code == 500
    assert resp.json()["detail"] == "Document indexing failed."

    # Verify that the document in DB is marked as failed, NOT falsely "indexed"
    failed_doc = next(iter(database.documents.records.values()))
    assert failed_doc["status"] == "failed"

    # Verify that vector index directory is clean and does not exist
    user_segment = hashlib.sha256(b"user-err").hexdigest()
    faiss_dir = Path(settings.faiss_index_path) / user_segment / failed_doc["document_id"]
    assert not faiss_dir.exists()


@pytest.mark.anyio
async def test_end_to_end_smoke_ingest_index_reload_delete(vector_test_setup) -> None:
    database, settings, embedder = vector_test_setup
    user_id = "user-smoke"
    token = make_token(user_id, settings)
    headers = {"Authorization": f"Bearer {token}"}

    content = b"Integration smoke test: verifying upload, chunking, embeddings, FAISS, and deletion."

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Upload
        upload_resp = await client.post(
            "/documents/upload",
            files={"file": ("smoke.txt", content, "text/plain")},
            headers=headers,
        )
        assert upload_resp.status_code == 201
        data = upload_resp.json()
        doc_id = data["document_id"]
        assert data["status"] == "indexed"
        assert data["chunk_count"] >= 1

        # 2. Chunks exist on disk
        db_doc = database.documents.records[doc_id]
        chunks_file = Path(db_doc["stored_path"]).parent / "chunks.json"
        assert chunks_file.is_file()
        chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
        assert len(chunks) == data["chunk_count"]

        # 3. FAISS index exists on disk and can be loaded
        assert index_exists(settings, user_id, doc_id)
        loaded = load_index(settings, user_id, doc_id)
        assert loaded is not None
        index, meta = loaded
        assert index.ntotal == len(chunks)
        assert meta["chunks"][0]["text"] == chunks[0]["text"]

        # 4. Search document index directly
        hits = search_document_index(
            settings, user_id, doc_id, embedder.embed_query("Integration smoke"), top_k=1
        )
        assert len(hits) == 1
        assert "score" in hits[0]

        # 5. Delete document
        del_resp = await client.delete(f"/documents/{doc_id}", headers=headers)
        assert del_resp.status_code == 204

        # 6. Verify MongoDB record, document files, and FAISS index are completely gone
        assert doc_id not in database.documents.records
        assert not Path(db_doc["stored_path"]).parent.exists()
        assert not index_exists(settings, user_id, doc_id)
