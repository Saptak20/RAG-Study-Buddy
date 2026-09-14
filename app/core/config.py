from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str | None = None
    mongodb_uri: str | None = None
    jwt_secret_key: str | None = None

    mongodb_db_name: str = "rag_study_buddy"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440
    app_env: str = "development"
    faiss_index_path: str = "./data/faiss_index"
    document_storage_path: str = "./data/documents"
    max_upload_size_bytes: int = 10 * 1024 * 1024
    chunk_size: int = 1000
    chunk_overlap: int = 150
    embedding_model_name: str = "all-MiniLM-L6-v2"
    retriever_top_k: int = 3
    groq_model: str = "llama-3.3-70b-versatile"
    rag_max_context_chars: int = 4000
    chat_history_limit: int = 10
    max_chunks_per_document: int = 5000
    port: int = 8000
    cors_origins: str = "http://localhost:3000,http://localhost:5173,http://localhost:8000"
    rate_limit_enabled: bool = True
    rate_limit_auth_per_minute: int = 30
    rate_limit_upload_per_minute: int = 15
    rate_limit_chat_per_minute: int = 45

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
