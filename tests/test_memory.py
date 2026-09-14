import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from app.core.deps import get_llm
from app.main import app
from app.rag.llm import NO_CONTEXT_ANSWER, FakeLLMService
from tests.test_documents import make_token
from tests.test_retriever import create_indexed_doc


@pytest.mark.anyio
async def test_chat_persists_user_and_assistant_messages(chat_setup) -> None:
    database, settings, embedder, _ = chat_setup
    user_id = "student-persist"
    token = make_token(user_id, settings)
    headers = {"Authorization": f"Bearer {token}"}

    create_indexed_doc(
        database=database,
        settings=settings,
        embedder=embedder,
        user_id=user_id,
        doc_id="doc-history-1",
        filename="distributed_systems.pdf",
        chunks_text=[("Raft is a consensus algorithm designed to be easy to understand.", 3)],
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/chat",
            json={"message": "What is the Raft consensus algorithm?"},
            headers=headers,
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "Grounded study response" in data["answer"]
    assert len(data["sources"]) == 1

    # Verify messages in MongoDB
    saved = database.messages.records
    user_msgs = [m for m in saved if m["user_id"] == user_id and m["role"] == "user"]
    asst_msgs = [m for m in saved if m["user_id"] == user_id and m["role"] == "assistant"]

    assert len(user_msgs) == 1
    assert user_msgs[0]["content"] == "What is the Raft consensus algorithm?"
    assert user_msgs[0]["citations"] == []
    assert isinstance(user_msgs[0]["created_at"], datetime)

    assert len(asst_msgs) == 1
    assert asst_msgs[0]["content"] == data["answer"]
    assert len(asst_msgs[0]["citations"]) == 1
    assert asst_msgs[0]["citations"][0]["document_id"] == "doc-history-1"
    assert asst_msgs[0]["citations"][0]["filename"] == "distributed_systems.pdf"
    assert asst_msgs[0]["citations"][0]["page"] == 3


@pytest.mark.anyio
async def test_empty_retrieval_persists_both_turns(chat_setup) -> None:
    database, settings, _, _ = chat_setup
    user_id = "user-empty-mem"
    token = make_token(user_id, settings)
    headers = {"Authorization": f"Bearer {token}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/chat",
            json={"message": "What is quantum computing?"},
            headers=headers,
        )

    assert resp.status_code == 200
    assert resp.json()["answer"] == NO_CONTEXT_ANSWER

    user_msgs = [m for m in database.messages.records if m["user_id"] == user_id and m["role"] == "user"]
    asst_msgs = [m for m in database.messages.records if m["user_id"] == user_id and m["role"] == "assistant"]

    assert len(user_msgs) == 1
    assert user_msgs[0]["content"] == "What is quantum computing?"
    assert len(asst_msgs) == 1
    assert asst_msgs[0]["content"] == NO_CONTEXT_ANSWER
    assert asst_msgs[0]["citations"] == []


@pytest.mark.anyio
async def test_chat_history_injected_into_subsequent_llm_calls(chat_setup) -> None:
    database, settings, embedder, fake_llm = chat_setup
    user_id = "student-multi-turn"
    token = make_token(user_id, settings)
    headers = {"Authorization": f"Bearer {token}"}

    create_indexed_doc(
        database=database,
        settings=settings,
        embedder=embedder,
        user_id=user_id,
        doc_id="doc-os",
        filename="os_threads.pdf",
        chunks_text=[
            ("A mutex provides mutual exclusion for critical sections.", 12),
            ("A semaphore maintains an integer count for signaling.", 14),
        ],
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Turn 1
        resp1 = await client.post(
            "/chat",
            json={"message": "What is a mutex?"},
            headers=headers,
        )
        assert resp1.status_code == 200
        # For turn 1, history sent to LLM should be empty
        assert len(fake_llm.calls) == 1
        assert fake_llm.calls[0]["chat_history"] == []

        # Turn 2
        resp2 = await client.post(
            "/chat",
            json={"message": "How does a semaphore differ?"},
            headers=headers,
        )
        assert resp2.status_code == 200

        # For turn 2, fake_llm should have received turn 1's history
        assert len(fake_llm.calls) == 2
        history = fake_llm.calls[1]["chat_history"]
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "What is a mutex?"
        assert history[1]["role"] == "assistant"
        assert "Grounded study response" in history[1]["content"]


@pytest.mark.anyio
async def test_chat_history_windowing_respects_limit(chat_setup) -> None:
    database, settings, embedder, fake_llm = chat_setup
    user_id = "user-windowing"
    token = make_token(user_id, settings)
    headers = {"Authorization": f"Bearer {token}"}

    # Set history limit to 4
    settings.chat_history_limit = 4

    create_indexed_doc(
        database=database,
        settings=settings,
        embedder=embedder,
        user_id=user_id,
        doc_id="doc-window",
        filename="window_notes.txt",
        chunks_text=[("Windowing memory tests content.", 1)],
    )

    # Pre-populate 8 messages (4 turns) with timestamps in ascending order
    base_time = datetime.now(UTC) - timedelta(minutes=10)
    for i in range(8):
        role = "user" if i % 2 == 0 else "assistant"
        content = f"Message {i}"
        database.messages.records.append({
            "_id": str(uuid.uuid4()),
            "message_id": f"msg-{i}",
            "user_id": user_id,
            "role": role,
            "content": content,
            "citations": [],
            "created_at": base_time + timedelta(minutes=i),
        })

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/chat",
            json={"message": "What is the new question?"},
            headers=headers,
        )

    assert resp.status_code == 200
    assert len(fake_llm.calls) == 1
    history = fake_llm.calls[0]["chat_history"]
    # Exactly 4 most recent messages passed (Messages 4, 5, 6, 7)
    assert len(history) == 4
    assert [m["content"] for m in history] == ["Message 4", "Message 5", "Message 6", "Message 7"]


@pytest.mark.anyio
async def test_get_chat_history_endpoint_and_pagination(chat_setup) -> None:
    database, settings, embedder, _ = chat_setup
    user_id = "user-history-get"
    token = make_token(user_id, settings)
    headers = {"Authorization": f"Bearer {token}"}

    create_indexed_doc(
        database=database,
        settings=settings,
        embedder=embedder,
        user_id=user_id,
        doc_id="doc-hist",
        filename="history.pdf",
        chunks_text=[("History fact.", 1)],
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Perform 2 chat interactions (4 messages: 2 user + 2 assistant)
        await client.post("/chat", json={"message": "First query"}, headers=headers)
        await client.post("/chat", json={"message": "Second query"}, headers=headers)

        # GET full history
        resp = await client.get("/chat/history", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 4
        assert len(data["messages"]) == 4

        # Verify chronological ordering
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "First query"
        assert data["messages"][1]["role"] == "assistant"
        assert data["messages"][2]["role"] == "user"
        assert data["messages"][2]["content"] == "Second query"
        assert data["messages"][3]["role"] == "assistant"

        # Check message response schema fields
        first_msg = data["messages"][0]
        assert "message_id" in first_msg
        assert "id" in first_msg
        assert first_msg["id"] == first_msg["message_id"]
        assert "created_at" in first_msg

        # Assistant messages should have citation list
        asst_msg = data["messages"][1]
        assert len(asst_msg["citations"]) == 1
        assert asst_msg["citations"][0]["filename"] == "history.pdf"

        # GET paginated history: limit 2, skip 1
        resp_paged = await client.get("/chat/history?limit=2&skip=1", headers=headers)
        assert resp_paged.status_code == 200
        paged_data = resp_paged.json()
        assert paged_data["total"] == 4
        assert len(paged_data["messages"]) == 2
        assert paged_data["messages"][0]["content"] == data["messages"][1]["content"]
        assert paged_data["messages"][1]["content"] == data["messages"][2]["content"]


@pytest.mark.anyio
async def test_clear_chat_history_endpoint(chat_setup) -> None:
    database, settings, embedder, _ = chat_setup
    user_id = "user-clear-mem"
    token = make_token(user_id, settings)
    headers = {"Authorization": f"Bearer {token}"}

    create_indexed_doc(
        database=database,
        settings=settings,
        embedder=embedder,
        user_id=user_id,
        doc_id="doc-c",
        filename="clear.txt",
        chunks_text=[("Clear content.", 1)],
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.post("/chat", json={"message": "Hello"}, headers=headers)
        await client.post("/chat", json={"message": "World"}, headers=headers)

        # Confirm 4 messages
        hist_before = await client.get("/chat/history", headers=headers)
        assert hist_before.json()["total"] == 4

        # DELETE history
        del_resp = await client.delete("/chat/history", headers=headers)
        assert del_resp.status_code == 200
        assert del_resp.json()["deleted_count"] == 4
        assert "cleared successfully" in del_resp.json()["detail"].lower()

        # Confirm 0 messages remain
        hist_after = await client.get("/chat/history", headers=headers)
        assert hist_after.status_code == 200
        assert hist_after.json()["total"] == 0
        assert hist_after.json()["messages"] == []


@pytest.mark.anyio
async def test_memory_strict_user_isolation(chat_setup) -> None:
    database, settings, embedder, fake_llm = chat_setup
    alice_id = "user-alice-isolated"
    bob_id = "user-bob-isolated"
    token_alice = make_token(alice_id, settings)
    token_bob = make_token(bob_id, settings)
    headers_alice = {"Authorization": f"Bearer {token_alice}"}
    headers_bob = {"Authorization": f"Bearer {token_bob}"}

    create_indexed_doc(
        database=database,
        settings=settings,
        embedder=embedder,
        user_id=alice_id,
        doc_id="alice-doc",
        filename="alice.txt",
        chunks_text=[("Alice secret knowledge.", 1)],
    )
    create_indexed_doc(
        database=database,
        settings=settings,
        embedder=embedder,
        user_id=bob_id,
        doc_id="bob-doc",
        filename="bob.txt",
        chunks_text=[("Bob secret knowledge.", 1)],
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Alice chats
        await client.post("/chat", json={"message": "Alice question"}, headers=headers_alice)
        # Bob chats
        await client.post("/chat", json={"message": "Bob question"}, headers=headers_bob)

        # Alice gets history -> only Alice's 2 messages
        alice_hist = await client.get("/chat/history", headers=headers_alice)
        assert alice_hist.json()["total"] == 2
        assert all("Alice" in m["content"] or "Grounded" in m["content"] for m in alice_hist.json()["messages"])

        # Bob gets history -> only Bob's 2 messages
        bob_hist = await client.get("/chat/history", headers=headers_bob)
        assert bob_hist.json()["total"] == 2
        assert all("Bob" in m["content"] or "Grounded" in m["content"] for m in bob_hist.json()["messages"])

        # Alice clears her history
        del_alice = await client.delete("/chat/history", headers=headers_alice)
        assert del_alice.status_code == 200
        assert del_alice.json()["deleted_count"] == 2

        # Bob's history is completely intact!
        bob_hist_after = await client.get("/chat/history", headers=headers_bob)
        assert bob_hist_after.json()["total"] == 2

        # Alice chats again -> her history sent to LLM is empty (cleared)
        fake_llm.calls.clear()
        await client.post("/chat", json={"message": "Alice question 2"}, headers=headers_alice)
        assert len(fake_llm.calls) == 1
        assert fake_llm.calls[0]["chat_history"] == []


@pytest.mark.anyio
async def test_unauthenticated_memory_endpoints_rejected() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        get_resp = await client.get("/chat/history")
        assert get_resp.status_code == 401

        del_resp = await client.delete("/chat/history")
        assert del_resp.status_code == 401


@pytest.mark.anyio
async def test_chat_llm_failure_does_not_persist_orphaned_messages(chat_setup) -> None:
    database, settings, embedder, _ = chat_setup
    user_id = "user-fail-mem"
    token = make_token(user_id, settings)
    headers = {"Authorization": f"Bearer {token}"}

    create_indexed_doc(
        database=database,
        settings=settings,
        embedder=embedder,
        user_id=user_id,
        doc_id="doc-err",
        filename="err.txt",
        chunks_text=[("Sample content for failure test.", 1)],
    )

    failing_llm = FakeLLMService(raise_error=HTTPException(status_code=502, detail="LLM error"))
    app.dependency_overrides[get_llm] = lambda: failing_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/chat", json={"message": "Fail question"}, headers=headers)

    assert resp.status_code == 502

    # Verify no orphaned messages were persisted
    user_records = [m for m in database.messages.records if m["user_id"] == user_id]
    assert len(user_records) == 0
