from datetime import UTC, datetime

from fastapi import HTTPException, status
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import DuplicateKeyError

from app.core.security import hash_password, verify_password


async def register_user(database: AsyncDatabase, email: str, password: str) -> dict:
    user = {
        "email": email,
        "password_hash": hash_password(password),
        "created_at": datetime.now(UTC),
    }
    try:
        result = await database.users.insert_one(user)
    except DuplicateKeyError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        ) from error
    return {"id": str(result.inserted_id), "email": email}


async def authenticate_user(
    database: AsyncDatabase, email: str, password: str
) -> dict:
    user = await database.users.find_one({"email": email})
    if user is None or not verify_password(password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"id": str(user["_id"]), "email": user["email"]}
