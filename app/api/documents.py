from fastapi import APIRouter, Depends, File, UploadFile, status
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings
from app.core.database import get_database
from app.core.deps import get_auth_settings, get_current_user_id, get_embeddings
from app.core.rate_limit import rate_limit_upload
from app.rag.embeddings import Embeddings
from app.schemas.documents import DocumentListResponse, DocumentResponse
from app.services.documents import (
    delete_document,
    get_document,
    ingest_document,
    list_documents,
)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a study document",
    description=(
        "Uploads a PDF, TXT, or MD study document (max 10MB). "
        "Extracts text, chunks content, computes local sentence embeddings, "
        "and creates an isolated FAISS vector index for retrieval."
    ),
    responses={
        status.HTTP_201_CREATED: {"description": "Document successfully ingested and indexed."},
        status.HTTP_400_BAD_REQUEST: {"description": "Unsupported file format, empty document, or corrupt file."},
        status.HTTP_409_CONFLICT: {"description": "Identical document already uploaded by user."},
        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: {"description": "Document exceeds 10MB size limit."},
        status.HTTP_429_TOO_MANY_REQUESTS: {"description": "Upload rate limit exceeded."},
    },
    dependencies=[Depends(rate_limit_upload)],
)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
    database: AsyncDatabase = Depends(get_database),
    settings: Settings = Depends(get_auth_settings),
    embedder: Embeddings = Depends(get_embeddings),
) -> DocumentResponse:
    return DocumentResponse(
        **await ingest_document(database, file, user_id, settings, embedder)
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all uploaded documents",
    description="Retrieves metadata for all study documents belonging to the authenticated user.",
    responses={
        status.HTTP_200_OK: {"description": "List of documents owned by user."},
        status.HTTP_401_UNAUTHORIZED: {"description": "Unauthorized access."},
    },
)
async def documents(
    user_id: str = Depends(get_current_user_id),
    database: AsyncDatabase = Depends(get_database),
) -> DocumentListResponse:
    return DocumentListResponse(documents=await list_documents(database, user_id))


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document details",
    description="Retrieves metadata for a specific document by its ID for the authenticated user.",
    responses={
        status.HTTP_200_OK: {"description": "Document metadata details."},
        status.HTTP_404_NOT_FOUND: {"description": "Document not found."},
        status.HTTP_401_UNAUTHORIZED: {"description": "Unauthorized access."},
    },
)
async def document(
    document_id: str,
    user_id: str = Depends(get_current_user_id),
    database: AsyncDatabase = Depends(get_database),
) -> DocumentResponse:
    return DocumentResponse(**await get_document(database, user_id, document_id))


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a study document",
    description="Deletes a document record and purges its chunk storage and FAISS vector index.",
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Document and associated index purged."},
        status.HTTP_404_NOT_FOUND: {"description": "Document not found."},
        status.HTTP_401_UNAUTHORIZED: {"description": "Unauthorized access."},
    },
)
async def remove_document(
    document_id: str,
    user_id: str = Depends(get_current_user_id),
    database: AsyncDatabase = Depends(get_database),
    settings: Settings = Depends(get_auth_settings),
) -> None:
    await delete_document(database, user_id, document_id, settings)
