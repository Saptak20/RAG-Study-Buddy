from pathlib import Path

import pytest

from app.core.config import Settings
from app.rag.embeddings import FakeEmbeddings
from app.rag.retriever import retrieve_relevant_chunks
from app.rag.vector_store import create_and_persist_index
from tests.test_documents import FakeDatabase


@pytest.fixture
def retriever_setup(tmp_path: Path):
    database = FakeDatabase()
    settings = Settings(
        jwt_secret_key="test-secret-key-with-adequate-length",
        document_storage_path=str(tmp_path / "documents"),
        faiss_index_path=str(tmp_path / "faiss_index"),
        retriever_top_k=3,
    )
    embedder = FakeEmbeddings(dimension=64)
    yield database, settings, embedder


def create_indexed_doc(
    database: FakeDatabase,
    settings: Settings,
    embedder: FakeEmbeddings,
    user_id: str,
    doc_id: str,
    filename: str,
    chunks_text: list[tuple[str, int]],
    status: str = "indexed",
) -> None:
    # 1. Add record to fake database
    database.documents.records[doc_id] = {
        "_id": doc_id,
        "document_id": doc_id,
        "user_id": user_id,
        "filename": filename,
        "status": status,
        "chunk_count": len(chunks_text),
    }

    # 2. Add FAISS index if indexed
    if status == "indexed":
        chunks = [
            {
                "chunk_id": f"{doc_id}-c{i}",
                "document_id": doc_id,
                "user_id": user_id,
                "filename": filename,
                "page": page,
                "text": text,
            }
            for i, (text, page) in enumerate(chunks_text)
        ]
        embeddings = embedder.embed_texts([c["text"] for c in chunks])
        create_and_persist_index(
            settings=settings,
            user_id=user_id,
            document_id=doc_id,
            chunks=chunks,
            embeddings=embeddings,
            dimension=embedder.dimension,
        )


@pytest.mark.anyio
async def test_multi_document_retrieval_ranking_and_citations(retriever_setup) -> None:
    database, settings, embedder = retriever_setup
    user_id = "student-alice"

    # Create Document 1 (Biology Chapter 1 - PDF with pages)
    create_indexed_doc(
        database=database,
        settings=settings,
        embedder=embedder,
        user_id=user_id,
        doc_id="doc-bio-1",
        filename="biology_ch1.pdf",
        chunks_text=[
            ("Photosynthesis light reactions absorb solar photons via chlorophyll.", 12),
            ("Chloroplasts contain thylakoid membranes organized into grana stacks.", 13),
        ],
    )

    # Create Document 2 (Biology Chapter 2 - PDF with pages)
    create_indexed_doc(
        database=database,
        settings=settings,
        embedder=embedder,
        user_id=user_id,
        doc_id="doc-bio-2",
        filename="biology_ch2.pdf",
        chunks_text=[
            ("Calvin cycle synthesizes carbohydrates using ATP and NADPH in dark reactions.", 24),
            ("Carbon fixation utilizes RuBisCO enzymes in the stroma.", 25),
        ],
    )

    # Create Document 3 (Physics notes - TXT)
    create_indexed_doc(
        database=database,
        settings=settings,
        embedder=embedder,
        user_id=user_id,
        doc_id="doc-phys",
        filename="physics.txt",
        chunks_text=[
            ("Newtonian mechanics describes gravitational attraction between masses.", 1),
        ],
    )

    # Query for photosynthesis concepts
    results = await retrieve_relevant_chunks(
        user_id=user_id,
        query="Explain light reactions and Calvin cycle photosynthesis",
        settings=settings,
        embedder=embedder,
        database=database,
        top_k=3,
    )

    # Verify top-K count
    assert len(results) == 3

    # Verify ranking: sorted strictly descending by similarity score
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)

    # Verify citation-ready metadata is preserved on every chunk
    for chunk in results:
        assert chunk["user_id"] == user_id
        assert chunk["document_id"] in ("doc-bio-1", "doc-bio-2", "doc-phys")
        assert chunk["filename"] in ("biology_ch1.pdf", "biology_ch2.pdf", "physics.txt")
        assert chunk["page"] is not None
        assert len(chunk["text"]) > 0
        assert isinstance(chunk["score"], float)


@pytest.mark.anyio
async def test_strict_user_isolation_in_retrieval(retriever_setup) -> None:
    database, settings, embedder = retriever_setup
    alice_id = "user-alice"
    bob_id = "user-bob"

    # Alice has secret document
    create_indexed_doc(
        database=database,
        settings=settings,
        embedder=embedder,
        user_id=alice_id,
        doc_id="alice-confidential",
        filename="confidential.txt",
        chunks_text=[("Secret exam answers: 1A, 2B, 3C, 4D.", 1)],
    )

    # Bob has public document
    create_indexed_doc(
        database=database,
        settings=settings,
        embedder=embedder,
        user_id=bob_id,
        doc_id="bob-public",
        filename="public.txt",
        chunks_text=[("General study guide for history exams.", 1)],
    )

    # Bob queries for "Secret exam answers"
    bob_results = await retrieve_relevant_chunks(
        user_id=bob_id,
        query="Secret exam answers: 1A, 2B",
        settings=settings,
        embedder=embedder,
        database=database,
        top_k=5,
    )

    # Bob must NEVER see Alice's document
    assert len(bob_results) == 1
    assert bob_results[0]["document_id"] == "bob-public"
    assert bob_results[0]["user_id"] == bob_id
    assert "Secret exam answers" not in bob_results[0]["text"]

    # Alice queries for the same and receives her document
    alice_results = await retrieve_relevant_chunks(
        user_id=alice_id,
        query="Secret exam answers",
        settings=settings,
        embedder=embedder,
        database=database,
        top_k=5,
    )
    assert len(alice_results) == 1
    assert alice_results[0]["document_id"] == "alice-confidential"
    assert alice_results[0]["user_id"] == alice_id


@pytest.mark.anyio
async def test_filter_by_target_document_ids(retriever_setup) -> None:
    database, settings, embedder = retriever_setup
    user_id = "user-filter"

    create_indexed_doc(
        database=database,
        settings=settings,
        embedder=embedder,
        user_id=user_id,
        doc_id="doc-1",
        filename="notes1.txt",
        chunks_text=[("Quantum mechanics wavefunctions and probability amplitudes.", 1)],
    )
    create_indexed_doc(
        database=database,
        settings=settings,
        embedder=embedder,
        user_id=user_id,
        doc_id="doc-2",
        filename="notes2.txt",
        chunks_text=[("Thermodynamics entropy and the second law.", 1)],
    )

    # Filter to only doc-2
    results = await retrieve_relevant_chunks(
        user_id=user_id,
        query="wavefunctions and entropy",
        settings=settings,
        embedder=embedder,
        database=database,
        document_ids=["doc-2"],
    )

    assert len(results) == 1
    assert results[0]["document_id"] == "doc-2"
    assert results[0]["filename"] == "notes2.txt"


@pytest.mark.anyio
async def test_unindexed_and_failed_documents_excluded(retriever_setup) -> None:
    database, settings, embedder = retriever_setup
    user_id = "user-status"

    create_indexed_doc(
        database=database,
        settings=settings,
        embedder=embedder,
        user_id=user_id,
        doc_id="ready-doc",
        filename="ready.txt",
        chunks_text=[("Cleanly indexed biology notes.", 1)],
        status="indexed",
    )
    create_indexed_doc(
        database=database,
        settings=settings,
        embedder=embedder,
        user_id=user_id,
        doc_id="failed-doc",
        filename="failed.txt",
        chunks_text=[("Corrupt data that failed indexing.", 1)],
        status="failed",
    )
    create_indexed_doc(
        database=database,
        settings=settings,
        embedder=embedder,
        user_id=user_id,
        doc_id="proc-doc",
        filename="processing.txt",
        chunks_text=[("Data currently still in processing.", 1)],
        status="processing",
    )

    results = await retrieve_relevant_chunks(
        user_id=user_id,
        query="biology notes",
        settings=settings,
        embedder=embedder,
        database=database,
    )

    assert len(results) == 1
    assert results[0]["document_id"] == "ready-doc"


@pytest.mark.anyio
async def test_empty_query_and_zero_documents(retriever_setup) -> None:
    database, settings, embedder = retriever_setup
    user_id = "user-empty"

    # User with no documents
    res_empty_user = await retrieve_relevant_chunks(
        user_id=user_id,
        query="Any question",
        settings=settings,
        embedder=embedder,
        database=database,
    )
    assert res_empty_user == []

    # Add a document
    create_indexed_doc(
        database=database,
        settings=settings,
        embedder=embedder,
        user_id=user_id,
        doc_id="doc-ok",
        filename="ok.txt",
        chunks_text=[("Valid study material.", 1)],
    )

    # Empty query strings
    assert await retrieve_relevant_chunks(
        user_id=user_id,
        query="",
        settings=settings,
        embedder=embedder,
        database=database,
    ) == []

    assert await retrieve_relevant_chunks(
        user_id=user_id,
        query="   \n\t  ",
        settings=settings,
        embedder=embedder,
        database=database,
    ) == []

    # top_k <= 0
    assert await retrieve_relevant_chunks(
        user_id=user_id,
        query="Valid material",
        settings=settings,
        embedder=embedder,
        database=database,
        top_k=0,
    ) == []


@pytest.mark.anyio
async def test_retrieval_without_database_fallback_to_filesystem(retriever_setup) -> None:
    _database, settings, embedder = retriever_setup
    user_id = "user-fs"

    chunks = [
        {
            "chunk_id": "c-fs",
            "document_id": "doc-fs-1",
            "user_id": user_id,
            "filename": "filesystem_only.txt",
            "page": 1,
            "text": "Filesystem-backed chunk content.",
        }
    ]
    create_and_persist_index(
        settings=settings,
        user_id=user_id,
        document_id="doc-fs-1",
        chunks=chunks,
        embeddings=embedder.embed_texts([c["text"] for c in chunks]),
        dimension=embedder.dimension,
    )

    # Query with database=None
    results = await retrieve_relevant_chunks(
        user_id=user_id,
        query="Filesystem-backed chunk",
        settings=settings,
        embedder=embedder,
        database=None,
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["document_id"] == "doc-fs-1"
    assert results[0]["filename"] == "filesystem_only.txt"
    assert results[0]["page"] == 1
