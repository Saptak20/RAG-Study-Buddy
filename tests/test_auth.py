from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from httpx import ASGITransport, AsyncClient
from pymongo.errors import DuplicateKeyError

from app.core.config import Settings
from app.core.database import get_database
from app.core.deps import get_auth_settings
from app.core.security import decode_access_token, verify_password
from app.main import app

SETTINGS = Settings(jwt_secret_key="test-secret-key-with-adequate-length")


class Users:
    def __init__(self) -> None:
        self.records: dict[str, dict] = {}

    async def insert_one(self, document: dict):
        if document["email"] in self.records:
            raise DuplicateKeyError("duplicate")
        identifier = str(len(self.records) + 1)
        self.records[document["email"]] = {**document, "_id": identifier}
        return type("Result", (), {"inserted_id": identifier})()

    async def find_one(self, query: dict) -> dict | None:
        return self.records.get(query["email"])


class Database:
    def __init__(self) -> None:
        self.users = Users()


def configure(database: Database) -> None:
    async def database_dependency() -> AsyncIterator[Database]:
        yield database

    async def settings_dependency() -> Settings:
        return SETTINGS

    app.dependency_overrides[get_database] = database_dependency
    app.dependency_overrides[get_auth_settings] = settings_dependency


@pytest.mark.anyio
async def test_register_duplicate_validation_and_hashing() -> None:
    database = Database()
    configure(database)
    payload = {"email": "Student@Example.com", "password": "secure-pass"}
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            registered = await client.post("/auth/register", json=payload)
            duplicate = await client.post("/auth/register", json=payload)
            malformed = await client.post("/auth/register", json={"email": "bad", "password": "short"})
        stored = database.users.records["student@example.com"]
        assert registered.status_code == 201
        assert duplicate.status_code == 409
        assert malformed.status_code == 422
        assert stored["password_hash"] != payload["password"]
        assert verify_password(payload["password"], stored["password_hash"])
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_login_and_protected_endpoint() -> None:
    database = Database()
    configure(database)
    payload = {"email": "student@example.com", "password": "secure-pass"}
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post("/auth/register", json=payload)
            login = await client.post("/auth/login", json=payload)
            wrong = await client.post("/auth/login", json={**payload, "password": "wrong-password"})
            unknown = await client.post("/auth/login", json={"email": "none@example.com", "password": "secure-pass"})
            token = login.json()["access_token"]
            authenticated = await client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
            missing = await client.get("/auth/me")
            malformed = await client.get("/auth/me", headers={"Authorization": "Bearer bad"})
        assert login.status_code == 200
        assert decode_access_token(token, SETTINGS) == "1"
        assert wrong.status_code == unknown.status_code == 401
        assert authenticated.json() == {"id": "1"}
        assert missing.status_code == malformed.status_code == 401
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_expired_and_invalid_subject_tokens_are_rejected() -> None:
    database = Database()
    configure(database)
    expired = jwt.encode({"sub": "1", "exp": datetime.now(UTC) - timedelta(minutes=1)}, SETTINGS.jwt_secret_key, algorithm="HS256")
    invalid_subject = jwt.encode({"sub": 1, "exp": datetime.now(UTC) + timedelta(minutes=1)}, SETTINGS.jwt_secret_key, algorithm="HS256")
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            expired_response = await client.get("/auth/me", headers={"Authorization": f"Bearer {expired}"})
            subject_response = await client.get("/auth/me", headers={"Authorization": f"Bearer {invalid_subject}"})
        assert expired_response.status_code == subject_response.status_code == 401
    finally:
        app.dependency_overrides.clear()
