"""Per-user, per-document FAISS vector store management and persistence."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import Settings


def get_index_directory(settings: Settings, user_id: str, document_id: str) -> Path:
    """Return safe directory path for a user's document FAISS index."""
    user_segment = hashlib.sha256(user_id.encode()).hexdigest()
    root = Path(settings.faiss_index_path).expanduser().resolve()
    user_dir = root / user_segment
    directory = (user_dir / document_id).resolve()
    if not directory.is_relative_to(user_dir) or directory == user_dir:
        raise RuntimeError("Resolved index path is outside user faiss root")
    return directory


def create_and_persist_index(
    settings: Settings,
    user_id: str,
    document_id: str,
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    dimension: int,
) -> Path:
    """Create a FAISS index, populate it with chunk vectors, and persist with metadata."""
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunk count ({len(chunks)}) does not match embedding count ({len(embeddings)})"
        )
    if not chunks:
        raise ValueError("Cannot create an index for zero chunks")

    import faiss

    # IndexFlatIP with normalized vectors computes exact cosine similarity
    index = faiss.IndexFlatIP(dimension)
    vectors = np.array(embeddings, dtype=np.float32)
    if vectors.shape[1] != dimension:
        raise ValueError(f"Vector dimension {vectors.shape[1]} does not match expected {dimension}")

    index.add(vectors)

    directory = get_index_directory(settings, user_id, document_id)
    directory.mkdir(parents=True, exist_ok=True)
    index_file = directory / "index.faiss"
    metadata_file = directory / "metadata.json"

    try:
        faiss.write_index(index, str(index_file))
        metadata = {
            "document_id": document_id,
            "user_id": user_id,
            "dimension": dimension,
            "metric": "inner_product",
            "total_chunks": len(chunks),
            "chunks": [
                {
                    "vector_id": i,
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "user_id": chunk["user_id"],
                    "filename": chunk["filename"],
                    "page": chunk.get("page"),
                    "text": chunk["text"],
                }
                for i, chunk in enumerate(chunks)
            ],
        }
        metadata_file.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise

    return directory


def load_index(
    settings: Settings,
    user_id: str,
    document_id: str,
) -> tuple[Any, dict[str, Any]] | None:
    """Load the FAISS index and chunk metadata for a given user and document."""
    import faiss

    try:
        directory = get_index_directory(settings, user_id, document_id)
    except RuntimeError:
        return None

    index_file = directory / "index.faiss"
    metadata_file = directory / "metadata.json"

    if not index_file.is_file() or not metadata_file.is_file():
        return None

    try:
        index = faiss.read_index(str(index_file))
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        return index, metadata
    except (OSError, json.JSONDecodeError, ValueError, RuntimeError):
        return None


def search_document_index(
    settings: Settings,
    user_id: str,
    document_id: str,
    query_embedding: list[float],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Search a single document's FAISS index and return top_k matching chunks with scores."""
    loaded = load_index(settings, user_id, document_id)
    if loaded is None:
        return []

    index, metadata = loaded
    query_vec = np.array([query_embedding], dtype=np.float32)
    k = min(top_k, index.ntotal)
    if k <= 0:
        return []

    distances, indices = index.search(query_vec, k)

    results: list[dict[str, Any]] = []
    chunks = metadata.get("chunks", [])
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk_data = dict(chunks[idx])
        chunk_data["score"] = float(dist)
        results.append(chunk_data)

    return results


def delete_index(settings: Settings, user_id: str, document_id: str) -> bool:
    """Delete the FAISS index directory for a document."""
    try:
        directory = get_index_directory(settings, user_id, document_id)
    except RuntimeError:
        return False
    root = Path(settings.faiss_index_path).expanduser().resolve()
    if directory.exists() and directory != root and directory.is_relative_to(root):
        shutil.rmtree(directory, ignore_errors=True)
        return True
    return False


def index_exists(settings: Settings, user_id: str, document_id: str) -> bool:
    """Check whether a valid FAISS index exists for the specified user and document."""
    try:
        directory = get_index_directory(settings, user_id, document_id)
        return (directory / "index.faiss").is_file() and (directory / "metadata.json").is_file()
    except (RuntimeError, OSError):
        return False
