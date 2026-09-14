"""Chat and grounded RAG question answering endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, status
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings
from app.core.database import get_database
from app.core.deps import (
    get_auth_settings,
    get_current_user_id,
    get_embeddings,
    get_llm,
)
from app.core.rate_limit import rate_limit_chat
from app.rag.citations import build_citations
from app.rag.context import format_retrieved_context
from app.rag.embeddings import Embeddings
from app.rag.llm import NO_CONTEXT_ANSWER, LLMService
from app.rag.retriever import retrieve_relevant_chunks
from app.schemas.chat import (
    ChatHistoryResponse,
    ChatMessageResponse,
    ChatRequest,
    ChatResponse,
    ClearHistoryResponse,
)
from app.services.chat import (
    clear_chat_history,
    get_chat_history,
    get_recent_history,
    save_message,
)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "",
    response_model=ChatResponse,
    summary="Ask a question grounded in study materials",
    description=(
        "Retrieves semantically relevant document chunks from the user's FAISS indexes, "
        "constructs grounded context with strict isolation, injects recent conversation history, "
        "and generates a grounded answer via Groq (LLaMA 3.3 70B) with backend-verified academic citations."
    ),
    responses={
        status.HTTP_200_OK: {"description": "Grounded answer with citations."},
        status.HTTP_401_UNAUTHORIZED: {"description": "Unauthorized access."},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error in question or filter IDs."},
        status.HTTP_429_TOO_MANY_REQUESTS: {"description": "Chat rate limit exceeded."},
        status.HTTP_502_BAD_GATEWAY: {"description": "Upstream LLM provider failure."},
    },
    dependencies=[Depends(rate_limit_chat)],
)
@router.post(
    "/",
    response_model=ChatResponse,
    include_in_schema=False,
    dependencies=[Depends(rate_limit_chat)],
)
async def chat(
    payload: ChatRequest,
    user_id: str = Depends(get_current_user_id),
    database: AsyncDatabase = Depends(get_database),
    settings: Settings = Depends(get_auth_settings),
    embedder: Embeddings = Depends(get_embeddings),
    llm: LLMService = Depends(get_llm),
) -> ChatResponse:
    question = payload.message.strip()

    recent_history = await get_recent_history(
        database=database,
        user_id=user_id,
        limit=settings.chat_history_limit,
    )
    formatted_history = [
        {"role": str(msg.get("role", "")), "content": str(msg.get("content", ""))}
        for msg in recent_history
        if msg.get("role") and msg.get("content")
    ]

    chunks = await retrieve_relevant_chunks(
        user_id=user_id,
        query=question,
        settings=settings,
        embedder=embedder,
        database=database,
        top_k=settings.retriever_top_k,
        document_ids=payload.document_ids,
    )

    if not chunks:
        await save_message(
            database=database,
            user_id=user_id,
            role="user",
            content=question,
            citations=[],
        )
        await save_message(
            database=database,
            user_id=user_id,
            role="assistant",
            content=NO_CONTEXT_ANSWER,
            citations=[],
        )
        return ChatResponse(
            answer=NO_CONTEXT_ANSWER,
            sources=[],
            citations=[],
        )

    context = format_retrieved_context(
        chunks=chunks,
        max_context_chars=settings.rag_max_context_chars,
    )

    answer = await llm.generate_answer(
        question=question,
        context=context,
        chat_history=formatted_history,
    )

    citations = build_citations(chunks)

    await save_message(
        database=database,
        user_id=user_id,
        role="user",
        content=question,
        citations=[],
    )
    await save_message(
        database=database,
        user_id=user_id,
        role="assistant",
        content=answer,
        citations=citations,
    )

    return ChatResponse(
        answer=answer,
        sources=citations,
        citations=citations,
    )


@router.get(
    "/history",
    response_model=ChatHistoryResponse,
    summary="Retrieve paginated chat conversation history",
    description="Returns previous conversation turns (questions, answers, and citations) for the authenticated user.",
    responses={
        status.HTTP_200_OK: {"description": "Paginated list of chat messages."},
        status.HTTP_401_UNAUTHORIZED: {"description": "Unauthorized access."},
    },
)
@router.get("/history/", response_model=ChatHistoryResponse, include_in_schema=False)
async def chat_history(
    limit: int = Query(default=50, ge=1, le=100, description="Maximum messages to return"),
    skip: int = Query(default=0, ge=0, description="Messages to skip"),
    user_id: str = Depends(get_current_user_id),
    database: AsyncDatabase = Depends(get_database),
) -> ChatHistoryResponse:
    messages, total = await get_chat_history(
        database=database,
        user_id=user_id,
        limit=limit,
        skip=skip,
    )
    return ChatHistoryResponse(
        messages=[ChatMessageResponse.model_validate(msg) for msg in messages],
        total=total,
    )


@router.delete(
    "/history",
    response_model=ClearHistoryResponse,
    summary="Clear chat conversation history",
    description="Permanently removes all conversation history turns for the authenticated user.",
    responses={
        status.HTTP_200_OK: {"description": "Chat history cleared successfully."},
        status.HTTP_401_UNAUTHORIZED: {"description": "Unauthorized access."},
    },
)
@router.delete("/history/", response_model=ClearHistoryResponse, include_in_schema=False)
async def clear_history(
    user_id: str = Depends(get_current_user_id),
    database: AsyncDatabase = Depends(get_database),
) -> ClearHistoryResponse:
    deleted_count = await clear_chat_history(database=database, user_id=user_id)
    return ClearHistoryResponse(
        detail="Chat history cleared successfully.",
        deleted_count=deleted_count,
    )
