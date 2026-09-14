"""MongoDB connection lifecycle and collection index setup."""

from collections.abc import AsyncIterator

from fastapi import HTTPException, Request, status
from pymongo import ASCENDING
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.asynchronous.mongo_client import AsyncMongoClient

from app.core.config import Settings


async def create_indexes(database: AsyncDatabase) -> None:
    """Create the indexes required for data integrity and common lookups."""
    await database.users.create_index("email", unique=True, name="users_email_unique")
    await database.documents.create_index(
        "user_id", name="documents_user_id"
    )
    await database.documents.create_index(
        [("user_id", ASCENDING), ("sha256", ASCENDING)],
        unique=True,
        name="documents_user_sha256_unique",
    )
    await database.messages.create_index(
        [("user_id", ASCENDING), ("created_at", ASCENDING)],
        name="messages_user_id_created_at",
    )


async def open_database(settings: Settings) -> tuple[AsyncMongoClient | None, AsyncDatabase | None]:
    """Connect to MongoDB and provision indexes when MongoDB is configured."""
    if not settings.mongodb_uri:
        return None, None

    client = AsyncMongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=5_000)
    database = client[settings.mongodb_db_name]
    await client.admin.command("ping")
    await create_indexes(database)
    return client, database


async def close_database(client: AsyncMongoClient | None) -> None:
    if client is not None:
        await client.close()


async def get_database(request: Request) -> AsyncIterator[AsyncDatabase]:
    database = getattr(request.app.state, "database", None)
    if database is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is not configured.",
        )
    yield database
