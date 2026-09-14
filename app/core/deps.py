from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import Settings, get_settings
from app.core.security import decode_access_token
from app.rag.embeddings import Embeddings, get_embedding_generator
from app.rag.llm import LLMService, get_llm_service

bearer_scheme = HTTPBearer(auto_error=False)


async def get_auth_settings() -> Settings:
    return get_settings()


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    settings: Settings = Depends(get_auth_settings),
) -> str:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing access token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_access_token(credentials.credentials, settings)


def get_embeddings(
    settings: Settings = Depends(get_auth_settings),
) -> Embeddings:
    return get_embedding_generator(settings.embedding_model_name)


def get_llm(
    settings: Settings = Depends(get_auth_settings),
) -> LLMService:
    return get_llm_service(settings)
