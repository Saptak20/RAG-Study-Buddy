"""In-process sliding-window rate limiting for expensive and sensitive endpoints."""

import time
from collections import defaultdict
from collections.abc import Callable

import jwt
from fastapi import Depends, HTTPException, Request, status

from app.core.config import Settings
from app.core.deps import get_auth_settings


class InMemoryRateLimiter:
    """In-process sliding window rate limiter.

    Note: Because state is stored in process memory, this rate limits per application
    instance/worker process. For distributed multi-worker architectures in the future,
    an external shared cache (like Redis) can be adopted.
    """

    def __init__(
        self,
        endpoint_name: str,
        limit_getter: Callable[[Settings], int],
        window_seconds: int = 60,
    ) -> None:
        self.endpoint_name = endpoint_name
        self.limit_getter = limit_getter
        self.window_seconds = window_seconds
        self._history: dict[str, list[float]] = defaultdict(list)

    def _get_client_key(self, request: Request, settings: Settings) -> str:
        # Prioritize stable authenticated user identity derived from validated JWT subject
        auth = request.headers.get("Authorization")
        if auth and auth.startswith("Bearer "):
            token = auth[7:].strip()
            if token and settings.jwt_secret_key:
                try:
                    payload = jwt.decode(
                        token,
                        settings.jwt_secret_key,
                        algorithms=[settings.jwt_algorithm],
                    )
                    user_id = payload.get("sub")
                    if isinstance(user_id, str) and user_id:
                        return f"user:{user_id}"
                except jwt.PyJWTError:
                    # Invalid/expired tokens fallback to client IP
                    pass

        # For unauthenticated requests, derive from direct socket client IP
        # (do not blindly trust spoofable arbitrary X-Forwarded-For headers in untrusted deployments)
        client = request.client
        ip = client.host if client else "unknown"
        return f"ip:{ip}"

    def _prune(self, key: str, now: float) -> None:
        threshold = now - self.window_seconds
        timestamps = self._history[key]
        # Timestamps are monotonically increasing; prune old entries
        idx = 0
        while idx < len(timestamps) and timestamps[idx] <= threshold:
            idx += 1
        if idx > 0:
            del timestamps[:idx]
        if not timestamps:
            self._history.pop(key, None)

    async def __call__(
        self,
        request: Request,
        settings: Settings = Depends(get_auth_settings),
    ) -> None:
        if not settings.rate_limit_enabled:
            return

        limit = self.limit_getter(settings)
        if limit <= 0:
            return

        now = time.monotonic()
        key = f"{self.endpoint_name}:{self._get_client_key(request, settings)}"

        self._prune(key, now)
        timestamps = self._history[key]

        if len(timestamps) >= limit:
            retry_after = int(self.window_seconds - (now - timestamps[0])) + 1
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
                headers={"Retry-After": str(max(1, retry_after))},
            )

        timestamps.append(now)

    def reset(self) -> None:
        """Clear state (useful for test isolation)."""
        self._history.clear()


rate_limit_auth = InMemoryRateLimiter(
    endpoint_name="auth",
    limit_getter=lambda s: s.rate_limit_auth_per_minute,
    window_seconds=60,
)

rate_limit_upload = InMemoryRateLimiter(
    endpoint_name="upload",
    limit_getter=lambda s: s.rate_limit_upload_per_minute,
    window_seconds=60,
)

rate_limit_chat = InMemoryRateLimiter(
    endpoint_name="chat",
    limit_getter=lambda s: s.rate_limit_chat_per_minute,
    window_seconds=60,
)
