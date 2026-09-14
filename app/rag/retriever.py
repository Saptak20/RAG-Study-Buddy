"""User-scoped multi-document vector retrieval and ranking."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings
from app.rag.embeddings import Embeddings
from app.rag.vector_store import index_exists, search_document_index


async def get_user_indexed_document_ids(
    database: AsyncDatabase | None,
    settings: Settings,
    user_id: str,
    target_document_ids: list[str] | None = None,
) -> list[str]:
    """Return the list of indexed document IDs owned by the authenticated user."""
    if database is not None:
        query: dict[str, Any] = {"user_id": user_id, "status": "indexed"}
        if target_document_ids:
            query["document_id"] = {"$in": target_document_ids}

        cursor = database.documents.find(query, {"_id": 1, "document_id": 1})
        return [
            doc.get("document_id", str(doc["_id"]))
            async for doc in cursor
        ]

    user_segment = hashlib.sha256(user_id.encode()).hexdigest()
    user_dir = Path(settings.faiss_index_path).expanduser().resolve() / user_segment
    if not user_dir.is_dir():
        return []

    doc_ids = []
    for item in user_dir.iterdir():
        if (
            item.is_dir()
            and index_exists(settings, user_id, item.name)
            and (target_document_ids is None or item.name in target_document_ids)
        ):
            doc_ids.append(item.name)
    return doc_ids


async def retrieve_relevant_chunks(
    user_id: str,
    query: str,
    settings: Settings,
    embedder: Embeddings,
    database: AsyncDatabase | None = None,
    top_k: int | None = None,
    document_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve, merge, rank, and select the top-K most relevant chunks across a user's documents."""
    clean_query = query.strip()
    if not clean_query:
        return []

    effective_top_k = top_k if top_k is not None else settings.retriever_top_k
    if effective_top_k <= 0:
        return []

    doc_ids = await get_user_indexed_document_ids(
        database=database,
        settings=settings,
        user_id=user_id,
        target_document_ids=document_ids,
    )

    if not doc_ids:
        return []

    query_embedding = embedder.embed_query(clean_query)

    all_candidates: list[dict[str, Any]] = []
    for doc_id in doc_ids:
        chunks = search_document_index(
            settings=settings,
            user_id=user_id,
            document_id=doc_id,
            query_embedding=query_embedding,
            top_k=effective_top_k,
        )
        all_candidates.extend(chunks)

    # Rank candidates across all documents by similarity score in descending order
    all_candidates.sort(key=lambda c: c.get("score", -float("inf")), reverse=True)

    return all_candidates[:effective_top_k]
