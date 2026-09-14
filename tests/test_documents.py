import hashlib
import io
import json
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import jwt
import pytest
from httpx import ASGITransport, AsyncClient
from pymongo.errors import DuplicateKeyError
from pypdf import PdfReader, PdfWriter

from app.core.config import Settings
from app.core.database import get_database
from app.core.deps import get_auth_settings, get_embeddings
from app.main import app
from app.rag.embeddings import FakeEmbeddings


class FakeDocumentsCollection:
    def __init__(self) -> None:
        self.records: dict[str, dict] = {}

    async def insert_one(self, record: dict) -> object:
        user_id = record["user_id"]
        sha256 = record["sha256"]
        for existing in self.records.values():
            if existing.get("user_id") == user_id and existing.get("sha256") == sha256:
                raise DuplicateKeyError("duplicate (user_id, sha256)")
        self.records[record["document_id"]] = dict(record)
        return type("Result", (), {"inserted_id": record["document_id"]})()

    async def find_one(self, query: dict, projection: dict | None = None) -> dict | None:
        for doc in self.records.values():
            match = True
            for k, v in query.items():
                val = doc.get("_id") if k == "_id" else doc.get(k)
                if isinstance(v, dict) and "$in" in v:
                    if val not in v["$in"]:
                        match = False
                        break
                elif val != v:
                    match = False
                    break
            if match:
                return dict(doc)
        return None

    def find(self, query: dict, projection: dict | None = None) -> object:
        matched = []
        for doc in self.records.values():
            match = True
            for k, v in query.items():
                val = doc.get("_id") if k == "_id" else doc.get(k)
                if isinstance(v, dict) and "$in" in v:
                    if val not in v["$in"]:
                        match = False
                        break
                elif val != v:
                    match = False
                    break
            if match:
                matched.append(dict(doc))

        class FakeCursor:
            def __init__(self, items: list[dict]) -> None:
                self.items = items

            def sort(self, field: str, direction: int) -> "FakeCursor":
                reverse = direction == -1
                self.items.sort(key=lambda x: x.get(field), reverse=reverse)
                return self

            def __aiter__(self) -> "FakeCursor":
                self._iter = iter(self.items)
                return self

            async def __anext__(self) -> dict:
                try:
                    return next(self._iter)
                except StopIteration:
                    raise StopAsyncIteration

        return FakeCursor(matched)

    async def delete_one(self, query: dict) -> object:
        to_del = None
        for doc_id, doc in self.records.items():
            match = True
            for k, v in query.items():
                if k == "_id" and doc.get("_id") != v or k != "_id" and doc.get(k) != v:
                    match = False
                    break
            if match:
                to_del = doc_id
                break
        if to_del:
            del self.records[to_del]
            return type("Result", (), {"deleted_count": 1})()
        return type("Result", (), {"deleted_count": 0})()

    async def update_one(self, query: dict, update: dict) -> object:
        matched = None
        for doc_id, doc in self.records.items():
            match = True
            for k, v in query.items():
                if k == "_id" and doc.get("_id") != v or k != "_id" and doc.get(k) != v:
                    match = False
                    break
            if match:
                matched = doc_id
                break
        if matched is not None:
            if "$set" in update:
                self.records[matched].update(update["$set"])
            return type("Result", (), {"matched_count": 1, "modified_count": 1})()
        return type("Result", (), {"matched_count": 0, "modified_count": 0})()


class FakeMessagesCursor:
    def __init__(self, items: list[dict]) -> None:
        self.items = list(items)

    def sort(self, field: str | list[tuple[str, int]], direction: int = 1) -> "FakeMessagesCursor":
        if isinstance(field, list):
            for f, d in reversed(field):
                self.items.sort(key=lambda x: x.get(f), reverse=(d == -1))
        else:
            self.items.sort(key=lambda x: x.get(field), reverse=(direction == -1))
        return self

    def skip(self, n: int) -> "FakeMessagesCursor":
        self.items = self.items[n:]
        return self

    def limit(self, n: int) -> "FakeMessagesCursor":
        self.items = self.items[:n]
        return self

    def __aiter__(self) -> "FakeMessagesCursor":
        self._iter = iter(self.items)
        return self

    async def __anext__(self) -> dict:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class FakeMessagesCollection:
    def __init__(self) -> None:
        self.records: list[dict] = []

    async def insert_one(self, record: dict) -> object:
        doc = dict(record)
        if "_id" not in doc:
            doc["_id"] = doc.get("message_id") or str(uuid.uuid4())
        self.records.append(doc)
        return type("Result", (), {"inserted_id": doc["_id"]})()

    def find(self, query: dict, projection: dict | None = None) -> FakeMessagesCursor:
        matched = []
        for doc in self.records:
            match = True
            for k, v in query.items():
                val = doc.get("_id") if k == "_id" else doc.get(k)
                if val != v:
                    match = False
                    break
            if match:
                matched.append(dict(doc))
        return FakeMessagesCursor(matched)

    async def count_documents(self, query: dict) -> int:
        matched = 0
        for doc in self.records:
            match = True
            for k, v in query.items():
                val = doc.get("_id") if k == "_id" else doc.get(k)
                if val != v:
                    match = False
                    break
            if match:
                matched += 1
        return matched

    async def delete_many(self, query: dict) -> object:
        original_count = len(self.records)
        remaining = []
        for doc in self.records:
            match = True
            for k, v in query.items():
                val = doc.get("_id") if k == "_id" else doc.get(k)
                if val != v:
                    match = False
                    break
            if not match:
                remaining.append(doc)
        self.records = remaining
        deleted = original_count - len(remaining)
        return type("Result", (), {"deleted_count": deleted})()


class FakeUsersCollection:
    def __init__(self) -> None:
        self.records: dict[str, dict] = {}

    async def insert_one(self, document: dict) -> object:
        email = document["email"].lower()
        if email in self.records:
            raise DuplicateKeyError("duplicate")
        identifier = document.get("_id") or f"user-{len(self.records) + 1}"
        self.records[email] = {**document, "email": email, "_id": identifier}
        return type("Result", (), {"inserted_id": identifier})()

    async def find_one(self, query: dict) -> dict | None:
        if "email" in query:
            return self.records.get(query["email"].lower())
        if "_id" in query:
            for u in self.records.values():
                if u.get("_id") == query["_id"]:
                    return dict(u)
        return None


class FakeDatabase:
    def __init__(self) -> None:
        self.users = FakeUsersCollection()
        self.documents = FakeDocumentsCollection()
        self.messages = FakeMessagesCollection()


def make_token(user_id: str, settings: Settings) -> str:
    payload = {
        "sub": user_id,
        "exp": datetime.now(UTC) + timedelta(minutes=60),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def make_sample_pdf(pages_text: list[str]) -> bytes:
    writer = PdfWriter()
    for text in pages_text:
        content_stream = f"BT /F1 18 Tf 50 700 Td ({text}) Tj ET".encode("latin-1")
        minimal_page_pdf = (
            b"%PDF-1.4\n"
            b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
            b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
            b"3 0 obj << /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >> endobj\n"
            b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
            b"5 0 obj << /Length " + str(len(content_stream)).encode() + b" >> stream\n"
            + content_stream + b"\nendstream\nendobj\n"
            b"xref\n0 6\n"
            b"0000000000 65535 f \n"
            b"0000000009 00000 n \n"
            b"0000000058 00000 n \n"
            b"0000000115 00000 n \n"
            b"0000000227 00000 n \n"
            b"0000000305 00000 n \n"
            b"trailer << /Size 6 /Root 1 0 R >>\n"
            b"startxref\n400\n%%EOF\n"
        )
        page_reader = PdfReader(io.BytesIO(minimal_page_pdf))
        writer.add_page(page_reader.pages[0])
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


@pytest.fixture
def test_setup(tmp_path: Path):
    database = FakeDatabase()
    settings = Settings(
        jwt_secret_key="test-secret-key-with-adequate-length",
        document_storage_path=str(tmp_path / "documents"),
        faiss_index_path=str(tmp_path / "faiss_index"),
        max_upload_size_bytes=10 * 1024 * 1024,
    )

    async def override_get_database() -> AsyncIterator[FakeDatabase]:
        yield database

    async def override_get_settings() -> Settings:
        return settings

    app.dependency_overrides[get_database] = override_get_database
    app.dependency_overrides[get_auth_settings] = override_get_settings
    fake_embedder = FakeEmbeddings(dimension=64)
    app.dependency_overrides[get_embeddings] = lambda: fake_embedder

    yield database, settings

    app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_unauthenticated_access_rejected(test_setup) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        upload = await client.post("/documents/upload", files={"file": ("notes.txt", b"content", "text/plain")})
        list_docs = await client.get("/documents")
        get_doc = await client.get("/documents/doc-123")
        del_doc = await client.delete("/documents/doc-123")

    assert upload.status_code == 401
    assert list_docs.status_code == 401
    assert get_doc.status_code == 401
    assert del_doc.status_code == 401


@pytest.mark.anyio
async def test_pdf_upload_and_metadata_preservation(test_setup) -> None:
    database, settings = test_setup
    user_id = "user-alice"
    token = make_token(user_id, settings)
    headers = {"Authorization": f"Bearer {token}"}

    pdf_bytes = make_sample_pdf(["Biology Chapter 1 Overview", "Cellular Respiration Details"])
    files = {"file": ("biology_notes.pdf", pdf_bytes, "application/pdf")}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/documents/upload", files=files, headers=headers)

    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "biology_notes.pdf"
    assert data["user_id"] == user_id
    assert data["file_type"] == "pdf"
    assert data["chunk_count"] >= 2
    assert data["status"] == "indexed"

    # Verify document in DB
    db_doc = database.documents.records[data["document_id"]]
    assert db_doc["document_id"] == data["document_id"]
    assert db_doc["stored_path"] is not None

    # Verify FAISS index and metadata.json created on disk
    user_segment = hashlib.sha256(user_id.encode()).hexdigest()
    faiss_dir = Path(settings.faiss_index_path) / user_segment / data["document_id"]
    assert (faiss_dir / "index.faiss").exists()
    assert (faiss_dir / "metadata.json").exists()

    # Verify chunk metadata and page numbers preserved on disk
    chunks_path = Path(db_doc["stored_path"]).parent / "chunks.json"
    assert chunks_path.exists()
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    assert len(chunks) >= 2
    for chunk in chunks:
        assert set(chunk.keys()) == {"chunk_id", "document_id", "user_id", "filename", "page", "text"}
        assert chunk["document_id"] == data["document_id"]
        assert chunk["user_id"] == user_id
        assert chunk["filename"] == "biology_notes.pdf"
    pages = [c["page"] for c in chunks]
    assert 1 in pages
    assert 2 in pages


@pytest.mark.anyio
async def test_txt_and_md_uploads(test_setup) -> None:
    _database, settings = test_setup
    token = make_token("user-1", settings)
    headers = {"Authorization": f"Bearer {token}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        txt_resp = await client.post(
            "/documents/upload",
            files={"file": ("lecture.txt", b"Simple plain text study guide.", "text/plain")},
            headers=headers,
        )
        md_resp = await client.post(
            "/documents/upload",
            files={"file": ("guide.md", b"# Markdown Title\nSome notes here.", "text/markdown")},
            headers=headers,
        )

    assert txt_resp.status_code == 201
    assert txt_resp.json()["file_type"] == "txt"
    assert md_resp.status_code == 201
    assert md_resp.json()["file_type"] == "md"


@pytest.mark.anyio
async def test_unsupported_extension_rejected(test_setup) -> None:
    _database, settings = test_setup
    token = make_token("user-1", settings)
    headers = {"Authorization": f"Bearer {token}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/documents/upload",
            files={"file": ("malicious.exe", b"binary content", "application/octet-stream")},
            headers=headers,
        )

    assert resp.status_code == 400
    assert "Only PDF, TXT, and Markdown files are supported." in resp.json()["detail"]


@pytest.mark.anyio
async def test_empty_file_rejected(test_setup) -> None:
    _database, settings = test_setup
    token = make_token("user-1", settings)
    headers = {"Authorization": f"Bearer {token}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/documents/upload",
            files={"file": ("empty.txt", b"", "text/plain")},
            headers=headers,
        )

    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


@pytest.mark.anyio
async def test_oversized_file_rejected(test_setup) -> None:
    _database, settings = test_setup
    token = make_token("user-1", settings)
    headers = {"Authorization": f"Bearer {token}"}

    settings.max_upload_size_bytes = 100
    oversized = b"A" * 150

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/documents/upload",
            files={"file": ("large.txt", oversized, "text/plain")},
            headers=headers,
        )

    assert resp.status_code == 413
    assert "exceeds" in resp.json()["detail"].lower()


@pytest.mark.anyio
async def test_malformed_pdf_rejected(test_setup) -> None:
    _database, settings = test_setup
    token = make_token("user-1", settings)
    headers = {"Authorization": f"Bearer {token}"}

    corrupt_pdf = b"%PDF-corrupted-and-incomplete-bytes"

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/documents/upload",
            files={"file": ("corrupt.pdf", corrupt_pdf, "application/pdf")},
            headers=headers,
        )

    assert resp.status_code == 400
    assert "could not be processed" in resp.json()["detail"]


@pytest.mark.anyio
async def test_duplicate_upload_per_user_and_cross_user_allowed(test_setup) -> None:
    _database, settings = test_setup
    token1 = make_token("user-1", settings)
    token2 = make_token("user-2", settings)

    file_content = b"Deterministic identical content for test."
    files1 = {"file": ("notes.txt", file_content, "text/plain")}
    files2 = {"file": ("notes.txt", file_content, "text/plain")}
    files3 = {"file": ("notes.txt", file_content, "text/plain")}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp1 = await client.post("/documents/upload", files=files1, headers={"Authorization": f"Bearer {token1}"})
        resp2 = await client.post("/documents/upload", files=files2, headers={"Authorization": f"Bearer {token1}"})
        resp3 = await client.post("/documents/upload", files=files3, headers={"Authorization": f"Bearer {token2}"})

    assert resp1.status_code == 201
    assert resp2.status_code == 409
    assert resp3.status_code == 201


@pytest.mark.anyio
async def test_list_get_and_delete_user_isolation(test_setup) -> None:
    database, settings = test_setup
    token_alice = make_token("alice", settings)
    token_bob = make_token("bob", settings)
    headers_alice = {"Authorization": f"Bearer {token_alice}"}
    headers_bob = {"Authorization": f"Bearer {token_bob}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Alice uploads a document
        alice_upload = await client.post(
            "/documents/upload",
            files={"file": ("alice_notes.txt", b"Alice confidential study material.", "text/plain")},
            headers=headers_alice,
        )
        alice_doc_id = alice_upload.json()["document_id"]

        # Bob uploads a document
        bob_upload = await client.post(
            "/documents/upload",
            files={"file": ("bob_notes.txt", b"Bob private physics notes.", "text/plain")},
            headers=headers_bob,
        )
        bob_doc_id = bob_upload.json()["document_id"]

        # Verify list isolation
        alice_list = await client.get("/documents", headers=headers_alice)
        bob_list = await client.get("/documents", headers=headers_bob)

        assert len(alice_list.json()["documents"]) == 1
        assert alice_list.json()["documents"][0]["document_id"] == alice_doc_id
        assert len(bob_list.json()["documents"]) == 1
        assert bob_list.json()["documents"][0]["document_id"] == bob_doc_id

        # Verify get isolation: Bob cannot GET Alice's doc
        bob_get_alice = await client.get(f"/documents/{alice_doc_id}", headers=headers_bob)
        assert bob_get_alice.status_code == 404

        # Verify delete isolation: Bob cannot DELETE Alice's doc
        bob_del_alice = await client.delete(f"/documents/{alice_doc_id}", headers=headers_bob)
        assert bob_del_alice.status_code == 404

        # Verify Alice's doc still exists
        alice_get = await client.get(f"/documents/{alice_doc_id}", headers=headers_alice)
        assert alice_get.status_code == 200

        # Alice deletes her doc
        alice_del = await client.delete(f"/documents/{alice_doc_id}", headers=headers_alice)
        assert alice_del.status_code == 204

        # Alice's doc is gone
        alice_get_after = await client.get(f"/documents/{alice_doc_id}", headers=headers_alice)
        assert alice_get_after.status_code == 404

        # Verify files on disk are completely deleted
        stored_path = database.documents.records.get(alice_doc_id, {}).get("stored_path")
        assert stored_path is None or not Path(stored_path).exists()
