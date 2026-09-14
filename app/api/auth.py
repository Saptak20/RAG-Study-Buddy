from fastapi import APIRouter, Depends, status
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings
from app.core.database import get_database
from app.core.deps import get_auth_settings, get_current_user_id
from app.core.rate_limit import rate_limit_auth
from app.core.security import create_access_token
from app.schemas.auth import (
    CurrentUserResponse,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from app.services.users import authenticate_user, register_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
    description="Registers a new user with an email and password. Passwords are securely hashed with bcrypt.",
    responses={
        status.HTTP_201_CREATED: {"description": "User registered successfully."},
        status.HTTP_409_CONFLICT: {"description": "Email address already registered."},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error in request payload."},
        status.HTTP_429_TOO_MANY_REQUESTS: {"description": "Rate limit exceeded."},
    },
    dependencies=[Depends(rate_limit_auth)],
)
async def register(
    payload: RegisterRequest, database: AsyncDatabase = Depends(get_database)
) -> UserResponse:
    return UserResponse(**await register_user(database, payload.email, payload.password))


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Authenticate user and obtain JWT token",
    description="Authenticates email and password, returning a signed HS256 JWT access token.",
    responses={
        status.HTTP_200_OK: {"description": "Authentication successful."},
        status.HTTP_401_UNAUTHORIZED: {"description": "Invalid email or password."},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error in request payload."},
        status.HTTP_429_TOO_MANY_REQUESTS: {"description": "Rate limit exceeded."},
    },
    dependencies=[Depends(rate_limit_auth)],
)
async def login(
    payload: LoginRequest,
    database: AsyncDatabase = Depends(get_database),
    settings: Settings = Depends(get_auth_settings),
) -> TokenResponse:
    user = await authenticate_user(database, payload.email, payload.password)
    return TokenResponse(access_token=create_access_token(user["id"], settings))


@router.get(
    "/me",
    response_model=CurrentUserResponse,
    summary="Get current user identity",
    description="Returns the authenticated user's ID extracted from the validated JWT bearer token.",
    responses={
        status.HTTP_200_OK: {"description": "Authenticated user identity."},
        status.HTTP_401_UNAUTHORIZED: {"description": "Missing, invalid, or expired JWT token."},
    },
)
async def current_user(
    user_id: str = Depends(get_current_user_id),
) -> CurrentUserResponse:
    return CurrentUserResponse(id=user_id)
