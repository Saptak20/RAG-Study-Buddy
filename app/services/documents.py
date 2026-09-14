"""Document validation, extraction, chunking, embedding, and vector persistence."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import DuplicateKeyError
from pypdf import PdfReader
from pypdf.errors import PdfReadError, PyPdfError

from app.core.config import Settings, get_settings
from app.rag.embeddings import Embeddings, get_embedding_generator
from app.rag.vector_store import create_and_persist_index, delete_index

logger = logging.getLogger("rag_study_buddy.documents")

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
CHUNK_METADATA_KEYS = ("document_id", "user_id", "filename", "page", "chunk_id")


def _safe_storage_root(settings: Settings) -> Path:
    return Path(settings.document_storage_path).expanduser().resolve()


def _document_directory(settings: Settings, user_id: str, document_id: str) -> Path:
    # Hashing the JWT subject prevents even an unusual subject claim from becoming a path.
    user_segment = hashlib.sha256(user_id.encode()).hexdigest()
    root = _safe_storage_root(settings)
    user_dir = root / user_segment
    directory = (user_dir / document_id).resolve()
    if not directory.is_relative_to(user_dir) or directory == user_dir:
        raise RuntimeError("Resolved document path is outside user storage root")
    return directory


def _validate_filename(filename: str | None) -> tuple[str, str]:
    if not filename:
        raise HTTPException(status_code=422, detail="A filename is required.")
    name = Path(filename).name.strip()
    extension = Path(name).suffix.lower()
    if not name or extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, TXT, and Markdown files are supported.",
        )
    return name, extension


async def _save_upload(
    upload: UploadFile, destination: Path, max_size: int
) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    try:
        with destination.open("wb") as target:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > max_size:
                    raise HTTPException(status_code=413, detail="File exceeds the 10 MB limit.")
                digest.update(chunk)
                target.write(chunk)
    except Exception:
        shutil.rmtree(destination.parent, ignore_errors=True)
        raise
    if size == 0:
        shutil.rmtree(destination.parent, ignore_errors=True)
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    return size, digest.hexdigest()


def _extract_pdf(path: Path) -> list[tuple[str, int | None]]:
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported.")
        pages = [(page.extract_text() or "", index + 1) for index, page in enumerate(reader.pages)]
    except HTTPException:
        raise
    except (PdfReadError, PyPdfError, ValueError, OSError, Exception) as error:
        raise HTTPException(status_code=400, detail="The PDF could not be processed.") from error
    return pages


def _extract_text(path: Path) -> list[tuple[str, int | None]]:
    try:
        return [(path.read_text(encoding="utf-8"), None)]
    except (UnicodeDecodeError, OSError) as error:
        raise HTTPException(status_code=400, detail="The text file encoding is not supported.") from error


def extract_text(path: Path, extension: str) -> list[tuple[str, int | None]]:
    pages = _extract_pdf(path) if extension == ".pdf" else _extract_text(path)
    if not any(text.strip() for text, _ in pages):
        raise HTTPException(status_code=400, detail="The document contains no readable text.")
    return pages


def chunk_pages(
    pages: Iterable[tuple[str, int | None]],
    *,
    document_id: str,
    user_id: str,
    filename: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    max_chunks: int = 5000,
) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks: list[dict] = []
    for text, page in pages:
        for chunk_text in splitter.split_text(text):
            chunk_id = str(uuid.uuid4())
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "user_id": user_id,
                    "filename": filename,
                    "page": page,
                    "text": chunk_text,
                }
            )
            if len(chunks) > max_chunks:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"The document produced too many chunks (exceeds limit of {max_chunks}).",
                )
    if not chunks:
        raise HTTPException(status_code=400, detail="The document contains no readable text.")
    return chunks


async def ingest_document(
    database: AsyncDatabase,
    upload: UploadFile,
    user_id: str,
    settings: Settings,
    embedder: Embeddings | None = None,
) -> dict:
    if embedder is None:
        embedder = get_embedding_generator(settings.embedding_model_name)

    filename, extension = _validate_filename(upload.filename)
    document_id = str(uuid.uuid4())
    directory = _document_directory(settings, user_id, document_id)
    original_path = directory / "original_file"
    try:
        size, digest = await _save_upload(upload, original_path, settings.max_upload_size_bytes)
        duplicate = await database.documents.find_one(
            {"user_id": user_id, "sha256": digest}, {"_id": 1, "document_id": 1}
        )
        if duplicate:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"message": "This document was already uploaded.", "document_id": duplicate.get("document_id", str(duplicate["_id"]))},
            )
        pages = extract_text(original_path, extension)
        chunks = chunk_pages(
            pages,
            document_id=document_id,
            user_id=user_id,
            filename=filename,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            max_chunks=settings.max_chunks_per_document,
        )
        chunks_path = directory / "chunks.json"
        chunks_path.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        now = datetime.now(UTC)
        record = {
            "_id": document_id,
            "document_id": document_id,
            "user_id": user_id,
            "filename": filename,
            "file_type": extension.lstrip("."),
            "content_type": upload.content_type or "application/octet-stream",
            "size_bytes": size,
            "sha256": digest,
            "status": "processing",
            "chunk_count": len(chunks),
            "created_at": now,
            "updated_at": now,
            "stored_path": str(original_path),
        }
        try:
            await database.documents.insert_one(record)
        except DuplicateKeyError as error:
            raise HTTPException(status_code=409, detail="This document was already uploaded.") from error

        try:
            chunk_texts = [c["text"] for c in chunks]
            vectors = embedder.embed_texts(chunk_texts)
            create_and_persist_index(
                settings=settings,
                user_id=user_id,
                document_id=document_id,
                chunks=chunks,
                embeddings=vectors,
                dimension=embedder.dimension,
            )
            indexed_time = datetime.now(UTC)
            await database.documents.update_one(
                {"_id": document_id, "user_id": user_id},
                {"$set": {"status": "indexed", "updated_at": indexed_time}},
            )
            record["status"] = "indexed"
            record["updated_at"] = indexed_time
            return record
        except Exception as error:
            delete_index(settings, user_id, document_id)
            failed_time = datetime.now(UTC)
            await database.documents.update_one(
                {"_id": document_id, "user_id": user_id},
                {"$set": {"status": "failed", "updated_at": failed_time}},
            )
            record["status"] = "failed"
            record["updated_at"] = failed_time
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Document indexing failed.",
            ) from error
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise


async def list_documents(database: AsyncDatabase, user_id: str) -> list[dict]:
    cursor = database.documents.find({"user_id": user_id}).sort("created_at", -1)
    return [document async for document in cursor]


async def get_document(database: AsyncDatabase, user_id: str, document_id: str) -> dict:
    document = await database.documents.find_one(
        {"_id": document_id, "user_id": user_id}
    )
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return document


async def delete_document(
    database: AsyncDatabase,
    user_id: str,
    document_id: str,
    settings: Settings | None = None,
) -> None:
    if settings is None:
        settings = get_settings()

    document = await get_document(database, user_id, document_id)
    stored_path = Path(document["stored_path"]).resolve()
    result = await database.documents.delete_one(
        {"_id": document_id, "user_id": user_id}
    )
    if result.deleted_count != 1:
        raise HTTPException(status_code=404, detail="Document not found.")

    storage_root = _safe_storage_root(settings)
    doc_dir = stored_path.parent
    if doc_dir.exists() and doc_dir != storage_root and doc_dir.is_relative_to(storage_root):
        shutil.rmtree(doc_dir, ignore_errors=True)

    delete_index(settings, user_id, document_id)
