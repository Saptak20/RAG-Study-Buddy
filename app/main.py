import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.core.config import get_settings
from app.core.database import close_database, open_database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag_study_buddy")


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info("Initializing application, storage directories, and database connection...")
    try:
        Path(settings.document_storage_path).mkdir(parents=True, exist_ok=True)
        Path(settings.faiss_index_path).mkdir(parents=True, exist_ok=True)
    except OSError as err:
        logger.warning("Could not pre-create storage directories during startup: %s", err)

    client, database = await open_database(settings)
    application.state.database_client = client
    application.state.database = database
    yield
    logger.info("Closing database connection on application shutdown...")
    await close_database(client)


app = FastAPI(
    title="RAG Study Buddy",
    version="0.1.0",
    description=(
        "AI-powered study assistant built with Retrieval-Augmented Generation (RAG): "
        "upload study materials and ask questions with source citations."
    ),
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=getattr(exc, "headers", None),
        )
    logger.exception(
        "Unhandled server error processing %s %s: %s",
        request.method,
        request.url.path,
        type(exc).__name__,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred."},
    )


class SecurityHeadersMiddleware:
    """Pure ASGI middleware applying defensive HTTP security headers and unhandled error shielding."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_security_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-content-type-options", b"nosniff"))
                headers.append((b"x-frame-options", b"DENY"))
                headers.append((b"referrer-policy", b"strict-origin-when-cross-origin"))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_security_headers)
        except Exception as exc:
            logger.exception("Unhandled server error in ASGI pipeline: %s", type(exc).__name__)
            response = JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "An internal server error occurred."},
                headers={
                    "X-Content-Type-Options": "nosniff",
                    "X-Frame-Options": "DENY",
                    "Referrer-Policy": "strict-origin-when-cross-origin",
                },
            )
            await response(scope, receive, send)


app.add_middleware(SecurityHeadersMiddleware)


# Configure CORS with explicit origins, disallowed wildcard with credentials
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(chat_router)


@app.get(
    "/health",
    tags=["health"],
    summary="Liveness probe",
    description="Verifies the ASGI web application is responsive and accepting HTTP requests.",
    responses={status.HTTP_200_OK: {"description": "ASGI server is live and healthy."}},
)
async def health() -> dict[str, str]:
    """Liveness probe: verifies the ASGI server is running and accepting requests."""
    return {"status": "ok"}


@app.get(
    "/health/ready",
    tags=["health"],
    summary="Readiness probe",
    description="Verifies external dependencies (MongoDB database connectivity) are reachable and healthy.",
    responses={
        status.HTTP_200_OK: {"description": "Service and all external dependencies are ready."},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Database disconnected or unreachable."},
    },
)
async def readiness(request: Request) -> dict[str, str]:
    """Readiness probe: verifies external dependencies (MongoDB) are reachable."""
    database = getattr(request.app.state, "database", None)
    if database is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not_ready", "database": "disconnected"},
        )
    try:
        # Ping the MongoDB server
        await database.command("ping")
        return {"status": "ready", "database": "connected"}
    except Exception as exc:
        logger.warning("Readiness probe database ping failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not_ready", "database": "unreachable"},
        ) from exc
