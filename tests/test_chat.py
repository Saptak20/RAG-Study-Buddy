
import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from app.core.deps import get_llm
from app.main import app
from app.rag.context import format_retrieved_context
from app.rag.llm import NO_CONTEXT_ANSWER, FakeLLMService, GroqLLMService
from tests.test_documents import make_token
from tests.test_retriever import create_indexed_doc


@pytest.mark.anyio
async def test_chat_unauthenticated_rejected(chat_setup) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/chat", json={"message": "What is normalization?"})
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_chat_valid_question_with_retrieved_context_and_answer(chat_setup) -> None:
    database, settings, embedder, fake_llm = chat_setup
    user_id = "student-1"
    token = make_token(user_id, settings)
    headers = {"Authorization": f"Bearer {token}"}

    create_indexed_doc(
        database=database,
        settings=settings,
        embedder=embedder,
        user_id=user_id,
        doc_id="doc-dbms",
        filename="dbms_lecture.pdf",
        chunks_text=[
            ("First normal form eliminates repeating groups in relational tables.", 10),
            ("Second normal form removes partial functional dependencies on candidate keys.", 11),
        ],
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/chat",
            json={"message": "Explain relational database normalization."},
            headers=headers,
        )

    assert resp.status_code == 200
    data = resp.json()

    # 1. Answer returned
    assert "Grounded study response based on notes." in data["answer"]

    # 2. Fake LLM received expected question and formatted context
    assert len(fake_llm.calls) == 1
    call = fake_llm.calls[0]
    assert call["question"] == "Explain relational database normalization."
    assert "[Source 1]" in call["context"]
    assert "dbms_lecture.pdf" in call["context"]
    assert "First normal form" in call["context"] or "Second normal form" in call["context"]

    # 3. Sources generated from retrieved metadata
    assert len(data["sources"]) >= 1
    source = data["sources"][0]
    assert source["document_id"] == "doc-dbms"
    assert source["filename"] == "dbms_lecture.pdf"
    assert source["page"] in (10, 11)
    assert source["chunk_id"].startswith("doc-dbms-")
    assert isinstance(source["score"], float)


@pytest.mark.anyio
async def test_chat_user_isolation_enforced(chat_setup) -> None:
    database, settings, embedder, fake_llm = chat_setup
    alice_id = "user-alice"
    bob_id = "user-bob"
    token_bob = make_token(bob_id, settings)
    headers_bob = {"Authorization": f"Bearer {token_bob}"}

    create_indexed_doc(
        database=database,
        settings=settings,
        embedder=embedder,
        user_id=alice_id,
        doc_id="alice-doc",
        filename="alice_crypto.pdf",
        chunks_text=[("Confidential RSA private key algorithm details.", 5)],
    )

    # Bob asks about RSA keys
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/chat",
            json={"message": "What is the confidential RSA key?"},
            headers=headers_bob,
        )

    assert resp.status_code == 200
    data = resp.json()

    # Bob has no documents, so empty retrieval -> NO_CONTEXT_ANSWER
    assert data["answer"] == NO_CONTEXT_ANSWER
    assert data["sources"] == []

    # Verify FakeLLM was NEVER called with Alice's context
    for call in fake_llm.calls:
        assert "RSA private key" not in call["context"]


@pytest.mark.anyio
async def test_empty_retrieval_does_not_hallucinate(chat_setup) -> None:
    _database, settings, _embedder, fake_llm = chat_setup
    user_id = "user-nodocs"
    token = make_token(user_id, settings)
    headers = {"Authorization": f"Bearer {token}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/chat",
            json={"message": "Explain astrophysics cosmology."},
            headers=headers,
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == NO_CONTEXT_ANSWER
    assert data["sources"] == []
    # LLM must NOT be invoked when no relevant chunks exist
    assert len(fake_llm.calls) == 0


def test_context_formatting_deterministic_and_bounded() -> None:
    chunks = [
        {
            "chunk_id": "c1",
            "document_id": "d1",
            "user_id": "u1",
            "filename": "notes.pdf",
            "page": 4,
            "text": "Chunk one content.",
        },
        {
            "chunk_id": "c2",
            "document_id": "d1",
            "user_id": "u1",
            "filename": "notes.pdf",
            "page": 5,
            "text": "Chunk two content.",
        },
    ]

    context = format_retrieved_context(chunks, max_context_chars=1000)
    assert "[Source 1]" in context
    assert "Document: notes.pdf" in context
    assert "Page: 4" in context
    assert "Chunk one content." in context
    assert "[Source 2]" in context
    assert "Page: 5" in context

    # Test bounding limit
    bounded = format_retrieved_context(chunks, max_context_chars=80)
    assert len(bounded) <= 80


@pytest.mark.anyio
async def test_chat_llm_failure_produces_clean_error(chat_setup) -> None:
    database, settings, embedder, _ = chat_setup
    user_id = "user-fail"
    token = make_token(user_id, settings)
    headers = {"Authorization": f"Bearer {token}"}

    create_indexed_doc(
        database=database,
        settings=settings,
        embedder=embedder,
        user_id=user_id,
        doc_id="doc-f",
        filename="notes.txt",
        chunks_text=[("Sample content for failure test.", 1)],
    )

    failing_llm = FakeLLMService(raise_error=HTTPException(status_code=502, detail="LLM provider error occurred."))
    app.dependency_overrides[get_llm] = lambda: failing_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/chat",
            json={"message": "Any question?"},
            headers=headers,
        )

    assert resp.status_code == 502
    assert resp.json()["detail"] == "LLM provider error occurred."


@pytest.mark.anyio
async def test_groq_missing_api_key_raises_503() -> None:
    service = GroqLLMService(api_key=None)
    with pytest.raises(HTTPException) as exc_info:
        await service.generate_answer(question="Question?", context="Context")
    assert exc_info.value.status_code == 503
    assert "not configured" in exc_info.value.detail.lower()


@pytest.mark.anyio
async def test_llm_output_cannot_fabricate_sources(chat_setup) -> None:
    database, settings, embedder, _ = chat_setup
    user_id = "student-spoof"
    token = make_token(user_id, settings)
    headers = {"Authorization": f"Bearer {token}"}

    create_indexed_doc(
        database=database,
        settings=settings,
        embedder=embedder,
        user_id=user_id,
        doc_id="real-doc-id",
        filename="real_lecture.pdf",
        chunks_text=[("Valid real text in document.", 7)],
    )

    # LLM outputs text trying to fabricate a source block
    adversarial_llm = FakeLLMService(
        canned_response="Answer text.\nSources:\n- FakeDocument.pdf — Page 999"
    )
    app.dependency_overrides[get_llm] = lambda: adversarial_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/chat",
            json={"message": "What is in real document?"},
            headers=headers,
        )

    assert resp.status_code == 200
    data = resp.json()

    # The structured sources in JSON response must come ONLY from retrieved metadata!
    assert len(data["sources"]) == 1
    assert data["sources"][0]["document_id"] == "real-doc-id"
    assert data["sources"][0]["filename"] == "real_lecture.pdf"
    assert data["sources"][0]["page"] == 7
    # "FakeDocument.pdf" must not appear in structured sources
    assert not any(s["filename"] == "FakeDocument.pdf" for s in data["sources"])
