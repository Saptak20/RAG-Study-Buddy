"""Schemas for chat/question request and response."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Citation(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    document_id: str
    filename: str
    page: int | None = None
    chunk_id: str
    similarity_score: float
    score: float = 0.0
    snippet: str = ""


SourceMetadata = Citation


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000, description="User question or query")
    document_ids: list[str] | None = Field(default=None, max_length=50, description="Optional target document filter")

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Message cannot be empty or whitespace-only.")
        return stripped

    @field_validator("document_ids")
    @classmethod
    def validate_document_ids(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            v = [doc_id.strip() for doc_id in v if doc_id.strip()]
        return v


class ChatResponse(BaseModel):
    answer: str
    sources: list[Citation] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class ChatMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    message_id: str = Field(..., description="Unique message identifier")
    id: str | None = Field(default=None, description="Convenience identifier alias")
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message text content")
    citations: list[Citation] = Field(default_factory=list, description="Citations attached to message")
    created_at: datetime = Field(..., description="UTC creation timestamp")

    @model_validator(mode="before")
    @classmethod
    def populate_ids(cls, data: Any) -> Any:
        if isinstance(data, dict):
            msg_id = data.get("message_id") or str(data.get("_id", ""))
            data["message_id"] = msg_id
            if not data.get("id"):
                data["id"] = msg_id
        return data


class ChatHistoryResponse(BaseModel):
    messages: list[ChatMessageResponse] = Field(default_factory=list)
    total: int = 0


class ClearHistoryResponse(BaseModel):
    detail: str = "Chat history cleared successfully."
    deleted_count: int = 0
