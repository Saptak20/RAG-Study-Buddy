import pytest

from app.core.config import Settings
from app.core.database import create_indexes, open_database


class FakeCollection:
    def __init__(self) -> None:
        self.calls: list[tuple[object, dict[str, object]]] = []

    async def create_index(self, keys: object, **kwargs: object) -> None:
        self.calls.append((keys, kwargs))


class FakeDatabase:
    def __init__(self) -> None:
        self.users = FakeCollection()
        self.documents = FakeCollection()
        self.messages = FakeCollection()


@pytest.mark.anyio
async def test_create_indexes_uses_expected_collections() -> None:
    database = FakeDatabase()

    await create_indexes(database)  # type: ignore[arg-type]

    assert database.users.calls == [
        ("email", {"unique": True, "name": "users_email_unique"})
    ]
    assert database.documents.calls == [
        ("user_id", {"name": "documents_user_id"}),
        (
            [("user_id", 1), ("sha256", 1)],
            {"name": "documents_user_sha256_unique", "unique": True},
        ),
    ]
    assert database.messages.calls == [
        (
            [("user_id", 1), ("created_at", 1)],
            {"name": "messages_user_id_created_at"},
        )
    ]


@pytest.mark.anyio
async def test_open_database_skips_connection_without_uri() -> None:
    settings = Settings(mongodb_uri=None)

    client, database = await open_database(settings)

    assert client is None
    assert database is None
