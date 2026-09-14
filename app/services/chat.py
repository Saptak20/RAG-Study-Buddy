"""Chat memory and message persistence services."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pymongo import ASCENDING, DESCENDING
from pymongo.asynchronous.database import AsyncDatabase

from app.schemas.chat import Citation


async def save_message(
    database: AsyncDatabase,
    user_id: str,
    role: str,
    content: str,
    citations: list[Citation] | list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Persist a single chat message (user or assistant) in MongoDB."""
    clean_citations: list[dict[str, Any]] = []
    if citations:
        for c in citations:
            if hasattr(c, "model_dump"):
                clean_citations.append(c.model_dump())
            elif isinstance(c, dict):
                clean_citations.append(dict(c))

    msg_id = str(uuid.uuid4())
    doc: dict[str, Any] = {
        "message_id": msg_id,
        "user_id": user_id,
        "role": role,
        "content": content,
        "citations": clean_citations,
        "created_at": datetime.now(UTC),
    }

    await database.messages.insert_one(doc)
    return doc


async def get_recent_history(
    database: AsyncDatabase,
    user_id: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Retrieve the most recent messages for a user in chronological order."""
    if limit <= 0:
        return []

    cursor = (
        database.messages.find({"user_id": user_id})
        .sort("created_at", DESCENDING)
        .limit(limit)
    )
    messages: list[dict[str, Any]] = []
    async for msg in cursor:
        messages.append(msg)

    # Reverse so oldest of the recent messages comes first (chronological order)
    messages.reverse()
    return messages


async def get_chat_history(
    database: AsyncDatabase,
    user_id: str,
    limit: int = 50,
    skip: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Retrieve paginated chat history for a user in chronological order."""
    total = await database.messages.count_documents({"user_id": user_id})

    cursor = (
        database.messages.find({"user_id": user_id})
        .sort("created_at", ASCENDING)
        .skip(skip)
        .limit(limit)
    )

    messages: list[dict[str, Any]] = []
    async for msg in cursor:
        messages.append(msg)

    return messages, total


async def clear_chat_history(
    database: AsyncDatabase,
    user_id: str,
) -> int:
    """Clear all chat history for a specific user. Returns count of deleted messages."""
    result = await database.messages.delete_many({"user_id": user_id})
    return getattr(result, "deleted_count", 0)
